"""Catalog (FileTracking) stage: masks + IR  ->  per-DCS catalog.  Parallel over day files."""
import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta

import numpy as np

from . import legacy_format as lf
from .catalog import OBS_FIELDS, catalog_block, mask_blocks
from .navigation import crop_navigation
from .params import parse_param_file
from .readers import scan_inputs

log = logging.getLogger("toocan_hpc.catalog")


def _mask_day_files(mask_root, version, region, start, end):
    d = start.date()
    out = {}
    while d <= end.date():
        p = os.path.join(mask_root, f"toocan_{version}", region, f"{d:%Y}", f"{d:%Y_%m_%d}", f"ToocanCloudMask_{d:%Y%m%d}.nc")
        if os.path.exists(p):
            out[d] = p
        d += timedelta(days=1)
    return out


def build_obs(epochs, slot_dicts):
    """Concatenate per-slot dicts -> time-ordered observation table."""
    ep_col, parts = [], {k: [] for k in OBS_FIELDS}
    for e, d in zip(epochs, slot_dicts):
        n = len(d["label"])
        if n == 0:
            continue
        ep_col.append(np.full(n, e))
        for k in OBS_FIELDS:
            parts[k].append(d[k])
    if not ep_col:
        return None
    obs = {k: np.concatenate(v) for k, v in parts.items()}
    obs["epoch"] = np.concatenate(ep_col)
    obs["label"] = obs["label"].astype(np.int64)
    return obs


def run_catalog(toocan_cfg, geo_cfg, start, end, box, mask_root, outdir, nproc=16, ellipse=False,
                legacy=True, dense_limit=2e8, block=6):
    t00 = time.perf_counter()
    pt, pg = parse_param_file(toocan_cfg), parse_param_file(geo_cfg)
    nav = crop_navigation(pg["file_navigation"], *box)
    recs = scan_inputs(pg["path_ir"], start, end)
    EP = datetime(1970, 1, 1)
    ir_by_day = {}
    for f, t, ti in recs:
        ir_by_day.setdefault(t.date(), {})[float((t - EP).total_seconds())] = (f, ti)
    masks = _mask_day_files(mask_root, str(pt.get("version", "2.08")), pg.get("REGION", "UNKNOWN"), start, end)
    days = sorted(set(masks) & set(ir_by_day))
    log.info("catalog: %d days, %d workers", len(days), nproc)
    tasks = [(masks[d], ir_by_day[d], nav, ellipse, i0, i1) for d in days for (i0, i1) in mask_blocks(masks[d], block)]
    ctx = mp.get_context("fork")
    epochs, slots = [], []
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as ex:
        for e, s in ex.map(catalog_block, tasks):             # ordered -> time ordered
            epochs += e; slots += s
    t_slots = time.perf_counter() - t0
    obs = build_obs(epochs, slots)
    os.makedirs(outdir, exist_ok=True)
    res = dict(n_slots=len(epochs), t_slots_s=t_slots)
    if obs is None:
        log.warning("no DCS found"); return res
    tracks = lf.tracks_from_obs(obs)
    dt = int(pg["temporalresolution"])
    INT = lf.compute_INT_variables(tracks, dt_minutes=dt)
    global_times = sorted({t for tr in tracks.values() for t in tr["times"]})
    res.update(n_dcs=len(tracks), n_obs=int(obs["label"].size))
    if legacy and len(tracks) * len(global_times) <= dense_limit:
        LC = lf.build_LC_variables(tracks, global_times)
        out = os.path.join(outdir, f"TOOCAN-{pg['REGION']}-{start:%Y%m%d}-{end:%Y%m%d}.nc")
        lf.write_tracking_nc(out, INT, LC, global_times, region=pg["REGION"], platform=pg.get("GEOplatform", "UNKNOWN"),
                             version=pt["version"])
        res["legacy_file"] = out
    elif legacy:
        log.warning("dense LC (%d x %d) exceeds dense_limit -> legacy file skipped (use compact catalog)",
                    len(tracks), len(global_times))
    # compact, scalable catalog (ragged observation table)
    import netCDF4
    cpath = os.path.join(outdir, f"catalog_compact-{pg['REGION']}-{start:%Y%m%d}-{end:%Y%m%d}.nc")
    order = np.lexsort((obs["epoch"], obs["label"]))
    with netCDF4.Dataset(cpath, "w") as ds:
        ds.createDimension("obs", order.size)
        for k, v in obs.items():
            dtp = "i8" if k in ("label", "area", "pix_235", "pix_220", "pix_210", "pix_200") else "f8"
            var = ds.createVariable(f"obs_{k}", dtp, ("obs",), zlib=True, complevel=4)
            var[:] = v[order].astype(np.int64 if dtp == "i8" else np.float64)
        ds.title = "TOOCAN-HPC compact catalog (one row per DCS per time slot, sorted by label then time)"
        ds.units_epoch = "seconds since 1970-01-01 UTC"
    res["compact_file"] = cpath
    res["t_total_s"] = time.perf_counter() - t00
    return res
