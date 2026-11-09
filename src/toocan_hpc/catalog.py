"""Per-slot DCS statistics (the expensive part of toocan.run_write_filetracking), exact and vectorised.

The original (tracking_DCS.track_month_from_df) loops over every label of every slot and, for each label,
recomputes `mask == lab` over the WHOLE grid ~10 times (cold surfaces x4, stats, 2 ellipses ...).
Here the labelled pixels of a slot are grouped ONCE (stable sort by label, ascending raster order inside a
label) and every statistic is computed on that label's pixel subset.  The subset arrays are element-for-
element identical (same values, same order, same dtype) to the original's boolean-indexed arrays, so every
numpy reduction (sum, mean, nanpercentile, cov, eig) returns bit-identical results.

Slots are independent given the masks -> embarrassingly parallel (process pool over day files).
"""
import os
from datetime import datetime, timedelta

import netCDF4
import numpy as np

THR_COLD = (235, 220, 210, 200)
OBS_FIELDS = (
    "label", "area", "lon_cm", "lat_cm", "tbmin", "tb90", "avg235", "avg208", "avg200",
    "pix_235", "pix_220", "pix_210", "pix_200", "km2_235", "km2_220", "km2_210", "km2_200",
    "ell235_a", "ell235_b", "ell235_e", "ell235_ang", "ell220_a", "ell220_b", "ell220_e", "ell220_ang",
)


def _ellipse(yy, xx, lat2d, lon2d):
    """Verbatim numerics of tracking_utils.compute_equivalent_ellipse on pre-selected cold pixels."""
    if yy.size < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    y0, x0 = yy.mean(), xx.mean()
    lat0 = float(lat2d[int(y0), int(x0)])
    lon0 = float(lon2d[int(y0), int(x0)])
    km_lat = 111.0
    km_lon = 111.0 * np.cos(np.deg2rad(lat0))
    x_km = (lon2d[yy, xx] - lon0) * km_lon
    y_km = (lat2d[yy, xx] - lat0) * km_lat
    coords = np.vstack([x_km, y_km])
    cov = np.cov(coords)
    eigvals, eigvecs = np.linalg.eig(cov)
    a = 2 * np.sqrt(eigvals.max())
    b = 2 * np.sqrt(eigvals.min())
    vec = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return float(a), float(b), float(b / a), float(angle)


def slot_catalog(mask, bt, area2d, lat2d, lon2d, ellipse=False):
    """mask:int32(Y,X) bt:float32(Y,X) area2d:float32(Y,X). Returns dict of per-label arrays (ascending label)."""
    Y, X = mask.shape
    flat_m = mask.ravel()
    flat_idx = np.flatnonzero(flat_m > 0)
    out = {k: [] for k in OBS_FIELDS}
    if flat_idx.size == 0:
        return {k: np.zeros(0) for k in OBS_FIELDS}
    lb = flat_m[flat_idx]
    order = np.argsort(lb, kind="stable")          # keeps ascending raster order inside each label
    idx_sorted = flat_idx[order]
    lb_sorted = lb[order]
    cut = np.flatnonzero(np.diff(lb_sorted)) + 1
    starts = np.concatenate(([0], cut)); ends = np.concatenate((cut, [lb_sorted.size]))
    btf, arf = bt.ravel(), area2d.ravel()
    for s, e in zip(starts, ends):
        idxs = idx_sorted[s:e]
        lab = int(lb_sorted[s])
        yy, xx = idxs // X, idxs % X                # == np.nonzero(mask == lab) (raster order)
        vals = btf[idxs]                            # == bt[mask == lab]
        ar = arf[idxs]
        out["label"].append(lab)
        out["area"].append(int(idxs.size))
        y_cm, x_cm = int(yy.mean()), int(xx.mean())
        out["lon_cm"].append(float(lon2d[y_cm, x_cm]))
        out["lat_cm"].append(float(lat2d[y_cm, x_cm]))
        for thr in THR_COLD:
            sel = vals <= thr
            out[f"pix_{thr}"].append(int(sel.sum()))
            out[f"km2_{thr}"].append(float(ar[sel].sum()))
        out["tbmin"].append(float(np.nanmin(vals)))
        out["tb90"].append(float(np.nanpercentile(vals, 90)))
        for thr in (235, 208, 200):
            v = vals[vals <= thr]
            out[f"avg{thr}"].append(float(v.mean()) if v.size else np.nan)
        for thr in (235, 220):
            if ellipse:
                sel = vals <= thr
                a, b, ecc, ang = _ellipse(yy[sel], xx[sel], lat2d, lon2d)
            else:
                a = b = ecc = ang = np.nan
            out[f"ell{thr}_a"].append(a); out[f"ell{thr}_b"].append(b)
            out[f"ell{thr}_e"].append(ecc); out[f"ell{thr}_ang"].append(ang)
    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- day-level worker
def _read_day_mask(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        t = np.asarray(ds.variables["time"][:], dtype=np.float64)
        m = np.asarray(ds.variables["DCS_number"][:])
    return t, m


def catalog_block(args):
    """Worker: slots [i0,i1) of one daily mask file.  -> (epochs, [slot dict,...])  (slots are independent)."""
    from .readers import read_slot_bt
    mask_path, ir_by_epoch, nav, ellipse, i0, i1 = args
    with netCDF4.Dataset(mask_path) as ds:
        ds.set_auto_mask(False)
        ep = np.asarray(ds.variables["time"][i0:i1], dtype=np.float64)
        masks = np.asarray(ds.variables["DCS_number"][i0:i1])
    res_e, res = [], []
    for e, m in zip(ep, masks):
        rec = ir_by_epoch.get(float(e))
        if rec is None:
            continue
        bt = read_slot_bt(rec[0], rec[1], nav["ymin"], nav["ymax"], nav["xmin"], nav["xmax"])
        res.append(slot_catalog(m.astype(np.int32), bt, nav["area2d"], nav["lat2d"], nav["lon2d"], ellipse))
        res_e.append(float(e))
    return res_e, res


def mask_blocks(mask_path, block):
    with netCDF4.Dataset(mask_path) as ds:
        n = len(ds.dimensions["time"])
    return [(i, min(i + block, n)) for i in range(0, n, block)]
