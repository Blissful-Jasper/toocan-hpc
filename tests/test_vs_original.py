#!/usr/bin/env python3
"""Regression test without any data files: synthetic convective fields, ORIGINAL TOOCAN core vs toocan_hpc core.

For several random seeds a 3-D (t,y,x) brightness-temperature field with moving, merging and splitting cold
cores, warm noise and NaN holes is generated.  The chunk chain of the original driver (overlap hand-off,
clean_clusters, labelMin bookkeeping) is replayed with
   (a) the original  toocan.lib.detection_spreading.detect_and_spread  (C Label_region + numba spread)
   (b) toocan_hpc.core in literal mode (fast=0) and in restricted/frontier mode (fast=1), 1 and 4 threads
and every final label volume must be IDENTICAL (same label values, not just same partition).

Run:  python tests/test_vs_original.py [--seeds 6]
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

ORIG = os.path.abspath(os.environ.get("TOOCAN_ORIG_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "toocan-python")))  # https://gitlab.in2p3.fr/thomas.fiolleau1/toocan-python
sys.path.insert(0, ORIG)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", os.environ.get("TOOCAN_HPC_SRC", "src")))

from toocan.lib.struct.data_param import DataParam                       # noqa: E402
from toocan.lib.struct.blob import make_Blob_class                       # noqa: E402
from toocan.lib.detection_spreading.detect_and_spread import detect_and_spread  # noqa: E402
from toocan.lib.detection_spreading.cluster_tools import clean_clusters  # noqa: E402
from toocan_hpc.core import CoreState, Params                            # noqa: E402

PT = dict(minBT_threshold=190, maxBT_threshold=235, stepBT_threshold=2, deltaBT_Spread=1,
          minAreaSeed=625, minLifetime=3, firstlabel=1, VolumeImage=24, overlap_window_size=4, nbMaxCluster=100000)


def synth(seed, T, Y, X):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float32)
    bt = 285.0 + 6.0 * rng.standard_normal((T, Y, X)).astype(np.float32)
    ncore = 14
    cy, cx = rng.uniform(10, Y - 10, ncore), rng.uniform(10, X - 10, ncore)
    vy, vx = rng.uniform(-0.6, 0.6, ncore), rng.uniform(-1.0, 1.0, ncore)
    amp = rng.uniform(60, 110, ncore); sig = rng.uniform(3.0, 7.0, ncore)
    t0 = rng.integers(0, T // 2, ncore); dur = rng.integers(8, T, ncore)
    for t in range(T):
        for k in range(ncore):
            if not (t0[k] <= t < t0[k] + dur[k]):
                continue
            life = (t - t0[k]) / max(1, dur[k]); env = np.sin(np.pi * life) ** 0.7
            y0, x0 = cy[k] + vy[k] * t, cx[k] + vx[k] * t
            bt[t] -= (amp[k] * env * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sig[k] ** 2))).astype(np.float32)
    bt += 1.5 * rng.standard_normal(bt.shape).astype(np.float32)
    bt[rng.random(bt.shape) < 0.0005] = np.nan                       # missing pixels
    bt = np.round(bt, 2).astype(np.float32)                          # many values on the 0.01 K grid (quantisation test)
    return bt


def chunks(total_T, Z, ov):
    out, s = [], 0
    while True:
        out.append((s, min(s + Z, total_T)))
        if s + Z >= total_T:
            break
        s += Z - ov
    return out


def run_original(bt_all, area, Z, ov):
    Y, X = area.shape
    dp = DataParam()
    dp.minBT, dp.maxBT, dp.stepBT = PT["minBT_threshold"], PT["maxBT_threshold"], PT["stepBT_threshold"]
    dp.deltaDetect = dp.deltaSpread = 1.0
    dp.timin, dp.lifemin, dp.labelFirstMCS = PT["minAreaSeed"], PT["minLifetime"], PT["firstlabel"]
    dp.ZSIZE, dp.overlap_window_size, dp.nbMaxCluster = Z, ov, PT["nbMaxCluster"]
    dp.XSIZE, dp.YSIZE = X, Y
    Blob = make_Blob_class()
    clusters = (Blob * dp.nbMaxCluster)()
    for i in range(dp.nbMaxCluster):
        clusters[i].seed_area_perFrame = (ctypes.c_int * Z)()
        clusters[i].labelVoisin = (ctypes.c_int * 1000)()
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    nb, labelMin = PT["firstlabel"], PT["firstlabel"]
    results = []
    for ci, (s, e) in enumerate(chunks(bt_all.shape[0], Z, ov)):
        if ci > 0:
            prev = lab[-ov:].copy(); lab[:] = 0; lab[:ov] = prev
            present = np.unique(lab[lab > 0])
            if present.size:
                labelMin = int(present.min())
                clusters = clean_clusters(dp, clusters, labels_present=present, labelMin=labelMin, nbMax=dp.nbMaxCluster)
                labelMin -= 1
        T = e - s
        if T != lab.shape[0]:
            lab = lab[:T]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            lab, nb = detect_and_spread(dp, clusters, np.ascontiguousarray(bt_all[s:e]), area, None, None, lab, PT, nb, labelMin)
        results.append(np.array(lab, copy=True))
        lab = np.ascontiguousarray(lab)
        if T != Z:
            break
        # keep the full-size buffer for the next chunk like the driver does
    return results, nb


def run_hpc(bt_all, area, Z, ov, fast, nth):
    Y, X = area.shape
    st = CoreState()
    nb, labelMin = PT["firstlabel"], PT["firstlabel"]
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    results = []
    for ci, (s, e) in enumerate(chunks(bt_all.shape[0], Z, ov)):
        T = e - s
        full = np.zeros((Z, Y, X), dtype=np.int32)
        if ci > 0:
            full[:ov] = lab[-ov:]
            present = np.unique(full[:ov][full[:ov] > 0])
            if present.size:
                labelMin = int(present.min()); st.begin_chunk(present, labelMin - 1); labelMin -= 1
        else:
            st.begin_chunk(np.zeros(0, np.int32), labelMin)
        lab = full
        p = Params(T, Y, X, PT["minBT_threshold"], PT["maxBT_threshold"], PT["stepBT_threshold"], PT["deltaBT_Spread"], 10,
                   PT["minAreaSeed"], PT["minLifetime"], PT["nbMaxCluster"], 1, fast)
        nb, _ = st.run_chunk(p, np.ascontiguousarray(bt_all[s:e]), area, lab[:T], nb, labelMin, nth)
        results.append(np.array(lab[:T], copy=True))
        if T != Z:
            break
    return results, nb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=6); a = ap.parse_args()
    Z, ov, total, Y, X = PT["VolumeImage"], PT["overlap_window_size"], 62, 70, 90
    area = (100.0 + 40.0 * np.random.default_rng(1).random((Y, X))).astype(np.float32)     # km2, non-integer
    ok_all = True
    for seed in range(a.seeds):
        bt = synth(seed, total, Y, X)
        t = time.time(); ref, nb_ref = run_original(bt, area, Z, ov); t_ref = time.time() - t
        nobj = max(int(r.max()) for r in ref)
        for fast, nth in ((0, 1), (1, 1), (1, 4), (2, 4), (2, 7)):
            t = time.time(); new, nb_new = run_hpc(bt, area, Z, ov, fast, nth); t_new = time.time() - t
            same = len(ref) == len(new) and all(np.array_equal(r, n) for r, n in zip(ref, new)) and nb_ref == nb_new
            ok_all &= same
            print(f"seed {seed} chunks={len(ref)} max_label={nobj:5d}  fast={fast} threads={nth}  "
                  f"{'IDENTICAL' if same else '*** DIFFERENT ***'}   (orig {t_ref:.1f}s, hpc {t_new:.2f}s)", flush=True)
            if not same:
                for ci, (r, n) in enumerate(zip(ref, new)):
                    d = int((r != n).sum())
                    if d: print(f"      chunk {ci}: {d} voxels differ")
    print("ALL TESTS PASSED" if ok_all else "TESTS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
