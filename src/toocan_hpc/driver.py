"""Chunked TOOCAN driver: same chunk/overlap/label-hand-off logic as toocan/run_toocan.py.

Only software engineering differs from the original: parallel/prefetched input, asynchronous
output, resumable state, structured logging.  The labelling arithmetic is in libtoocan_core.so.
"""
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np

from .core import CoreState, Params, zero_i32
from .loader import VolumeLoader
from .navigation import crop_navigation
from .output import DayWriter, _atomic_nc
from .params import parse_param_file
from .readers import scan_inputs

log = logging.getLogger("toocan_hpc")


def plan_chunks(start, end, res_min, zsize, overlap):
    """Exactly the window arithmetic of the original driver.  Returns [(pre_overlap, post_overlap)]."""
    res = timedelta(minutes=res_min)
    chunk = res * zsize
    cur, out = start, []
    while cur < end:
        pre = max(cur - res * overlap, start)
        post = min(pre + chunk - res, end)
        out.append((pre, post))
        cur = post + res
    return out


class Runner:
    def __init__(self, toocan_cfg, geo_cfg, start, end, box, outroot, *, nthreads=8, nread=8, nwrite=3,
                 state_dir=None, resume=True, quant_mode=1, var=None, log_dir=None, fast=2):
        self.pt = parse_param_file(toocan_cfg)
        self.pg = parse_param_file(geo_cfg)
        self.start, self.end, self.box = start, end, tuple(box)
        self.outroot = outroot
        self.nthreads, self.nread, self.resume, self.quant_mode, self.var = nthreads, nread, resume, quant_mode, var
        self.fast = int(fast); self.nwrite = int(nwrite)
        self.state_dir = state_dir or os.path.join(outroot, ".state")
        self.log_dir = log_dir or self.state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.Z = int(self.pt["VolumeImage"]); self.ov = int(self.pt["overlap_window_size"])
        self.res = int(self.pg["temporalresolution"])
        sig = dict(pt=self.pt, pg={k: v for k, v in self.pg.items() if k != "file_list"},
                   start=str(start), end=str(end), box=list(self.box), q=quant_mode)
        self.sig = hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ checkpoint
    # State needed to START chunk k: the labels of its first `overlap` slots (= last slots of chunk k-1), the global
    # label counter and labelMin.  One small sparse file per chunk; after a restart the run resumes from the first
    # chunk whose finalised slots belong to a day that is not yet complete on disk (<= 2 chunks are recomputed).
    def _state_path(self, k):
        return os.path.join(self.state_dir, f"start_chunk_{k:05d}.npz")

    def _save_start_state(self, k, nb, labelMin, tail):
        flat = tail.ravel(); idx = np.flatnonzero(flat)
        tmp = self._state_path(k) + f".tmp{os.getpid()}.npz"
        np.savez(tmp, idx=idx, val=flat[idx], shape=np.array(tail.shape), nb=nb, labelMin=labelMin, sig=self.sig)
        os.replace(tmp, self._state_path(k))
        # long (multi-year) chains: keep only the most recent start states (a restart needs at most ~5 chunks back)
        keep = int(os.environ.get("TOOCAN_HPC_KEEP_STATES", "12"))
        if k - keep >= 1 and os.path.exists(self._state_path(k - keep)):
            os.unlink(self._state_path(k - keep))

    def _load_start_state(self, k):
        p = self._state_path(k)
        if not os.path.exists(p):
            return None
        d = np.load(p)
        if str(d["sig"]) != self.sig:
            return None
        tail = np.zeros(int(np.prod(d["shape"])), dtype=np.int32); tail[d["idx"]] = d["val"]
        return int(d["nb"]), int(d["labelMin"]), tail.reshape(tuple(d["shape"]))

    # ------------------------------------------------------------------ main
    def run(self):
        t_run = time.perf_counter()
        nav = crop_navigation(self.pg["file_navigation"], *self.box)  # box = lonmin lonmax latmin latmax
        Y, X = nav["area2d"].shape
        nbox = (nav["ymin"], nav["ymax"], nav["xmin"], nav["xmax"])
        recs = scan_inputs(self.pg["path_ir"], self.start, self.end)
        log.info("inputs: %d time records, grid %dx%d, domain %s", len(recs), Y, X, self.box)
        if not recs:
            raise RuntimeError("no input records")
        times = [r[1] for r in recs]
        plan = plan_chunks(self.start, self.end, self.res, self.Z, self.ov)
        # slots of every chunk
        tarr = np.array(times, dtype="datetime64[us]")
        chunks = []
        for pre, post in plan:
            sel = np.nonzero((tarr >= np.datetime64(pre)) & (tarr <= np.datetime64(post)))[0]
            chunks.append(sel)
        n = len(chunks)
        # ---- resume: first expected day whose file is missing -> restart at the chunk that finalises its first slot
        k0, nb, labelMin, prev_tail, skip_before = 0, int(self.pt["firstlabel"]), int(self.pt["firstlabel"]), None, None
        out_probe = DayWriter(self.outroot, self.pt.get("version", "2.08"), self.pg.get("REGION", "UNKNOWN"),
                              nav["lat2d"], nav["lon2d"], self.pg.get("GEOplatform", "UNKNOWN"), 0, times, nworkers=0)
        if self.resume:
            days = sorted({t.date() for t in times})
            missing = [d for d in days if not out_probe.day_exists(d)]
            if not missing:
                log.info("nothing to do (all %d day files exist)", len(days)); return
            if len(missing) < len(days):
                d0 = missing[0]
                s0 = next(i for i, t in enumerate(times) if t.date() == d0)
                kmin = None
                for k, sel in enumerate(chunks):
                    nfin = len(sel) if k == n - 1 else len(sel) - self.ov
                    if len(sel) and sel[0] <= s0 < sel[0] + nfin:
                        kmin = k; break
                st0 = self._load_start_state(kmin) if kmin else None
                if kmin is not None and (kmin == 0 or st0 is not None):
                    k0 = kmin
                    if st0: nb, labelMin, prev_tail = st0
                    skip_before = d0
                    log.info("RESUME: %d/%d days complete -> restart at chunk %d (first incomplete day %s)", len(days) - len(missing), len(days), k0, d0)
                else:
                    log.warning("no usable start state for chunk %s -> recomputing from the beginning", kmin)
        writer = DayWriter(self.outroot, self.pt.get("version", "2.08"), self.pg.get("REGION", "UNKNOWN"),
                           nav["lat2d"], nav["lon2d"], self.pg.get("GEOplatform", "UNKNOWN"),
                           int(self.pt.get("netcdf_compression_level", 4)), times, nworkers=self.nwrite, skip_days_before=skip_before)
        # validate chunk lengths like the original (missing slot inside a non-terminal chunk = error)
        for k, sel in enumerate(chunks):
            if len(sel) != self.Z and plan[k][1] < self.end:
                raise RuntimeError(f"Missing IR slot inside non-terminal chunk {k}: {len(sel)}/{self.Z} "
                                   f"({plan[k][0]} -> {plan[k][1]})")
        loader = VolumeLoader(self.Z, Y, X, nbuf=2, nworkers=self.nread, var=self.var)
        labs = [np.zeros((self.Z, Y, X), dtype=np.int32) for _ in range(3)]
        state = CoreState()
        wpool = cf.ThreadPoolExecutor(max_workers=1)
        wjobs = {}
        area = nav["area2d"]
        statf = open(os.path.join(self.log_dir, "chunk_stats.jsonl"), "a")

        def chunk_records(k):
            return [(recs[i][0], recs[i][2]) for i in chunks[k]]

        def writer_job(k, lab_view, nfinal, nb_, lmin_, tail_):
            t0 = time.perf_counter()
            if tail_ is not None and k < n - 1:
                self._save_start_state(k + 1, nb_, lmin_, tail_)       # state to START chunk k+1 (before its days are written)
            tsel = [times[i] for i in chunks[k][:nfinal]]
            writer.add_slots(tsel, lab_view[:nfinal])
            writer.flush_complete(force_all=(k == n - 1))
            return time.perf_counter() - t0

        futs = {k0: loader.submit(k0 % 2, chunk_records(k0), nbox)}
        try:
            for k in range(k0, n):
                tw = time.perf_counter()
                loader.wait(futs.pop(k))
                t_io_wait = time.perf_counter() - tw
                if k + 1 < n:
                    futs[k + 1] = loader.submit((k + 1) % 2, chunk_records(k + 1), nbox)
                T = len(chunks[k])
                if k - 2 >= k0 and (k - 2) in wjobs:         # label-buffer ring protection
                    wjobs.pop(k - 2).result()
                lab = labs[k % 3][:T]
                lab_full = labs[k % 3]
                tp0 = time.perf_counter()
                zero_i32(lab_full, self.nthreads)
                if prev_tail is not None:
                    lab_full[:self.ov] = prev_tail
                    labels_present = np.unique(prev_tail[prev_tail > 0])
                    if labels_present.size:
                        labelMin = int(labels_present.min())
                        state.begin_chunk(labels_present, labelMin - 1)
                        labelMin -= 1
                else:
                    state.begin_chunk(np.zeros(0, np.int32), labelMin)
                bt = loader.arrs[k % 2][:T]
                t_prep = time.perf_counter() - tp0
                p = Params(T, Y, X, int(self.pt["minBT_threshold"]), int(self.pt["maxBT_threshold"]),
                           int(self.pt["stepBT_threshold"]), int(self.pt.get("deltaBT_Spread", 1)), 10,
                           int(self.pt["minAreaSeed"]), int(self.pt["minLifetime"]),
                           int(self.pt["nbMaxCluster"]), self.quant_mode, self.fast)
                t0 = time.perf_counter()
                nb, st = state.run_chunk(p, bt, area, lab, nb, labelMin, self.nthreads)
                t_comp = time.perf_counter() - t0
                tq0 = time.perf_counter()
                tail = np.array(lab_full[T - self.ov:T], copy=True) if T >= self.ov else None
                last = (k == n - 1)
                nfinal = T if last else T - self.ov
                wjobs[k] = wpool.submit(writer_job, k, lab_full, nfinal, nb, labelMin, tail)
                prev_tail = tail
                t_post = time.perf_counter() - tq0
                rec = dict(chunk=k, T=T, t_io_wait=t_io_wait, t_prep=t_prep, t_post=t_post, t_compute=t_comp, nb_ConvSeeds=nb,
                           labelMin=labelMin, **{f"c_{a}": b for a, b in st.items()})
                statf.write(json.dumps(rec) + "\n"); statf.flush()
                log.info("chunk %d/%d  %s..%s  T=%d  io_wait=%.2fs compute=%.2fs  nb_ConvSeeds=%d",
                         k + 1, n, plan[k][0], plan[k][1], T, t_io_wait, t_comp, nb)
            for k, f in sorted(wjobs.items()):
                f.result()
            writer.finish()
        finally:
            statf.close()
            wpool.shutdown(wait=True)
            loader.close()
        log.info("DONE in %.1fs", time.perf_counter() - t_run)
        return nb
