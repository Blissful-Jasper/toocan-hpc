#!/usr/bin/env python3
"""Object-level scientific comparison of two TOOCAN label sets (label numbering may differ).

The (reference object, test object) voxel-overlap table is accumulated slot by slot (vectorised).  Reported:
  1. number of objects, objects per slot (equal?), voxels labelled in one run only
  2. objects whose voxel SET is identical in both runs (membership identity, independent of numbering)
  3. best-match IoU of reference objects (mean, min, 5th percentile); unmatched objects
  4. split / merge discrepancies:  ref object overlapping >=2 test objects (>5 % each) and vice versa
  5. lifecycle of matched pairs: birth/death slot differences, duration
  6. distributions (size in voxels, duration in slots): Kolmogorov-Smirnov distance
  7. optional breakdown by 'day within a segment' (--seg-days N): fraction of labelled voxels that belong to
     membership-identical objects, per day index (shows the warm-up/boundary effect of segmented runs)

Usage: compare_objects.py REF_ROOT TEST_ROOT [--start YYYYMMDD] [--end YYYYMMDD] [--seg-days 10] [--ref-segment-days 10] [--json out.json]
"""
import argparse, glob, json, os
import numpy as np
import netCDF4

OFF = 1000          # labels are < 2**27, background -998 -> non-negative after +1000
B = 27


def days(root):
    return {os.path.basename(p)[-11:-3]: p for p in sorted(glob.glob(os.path.join(root, "**", "ToocanCloudMask_*.nc"), recursive=True))}


def load(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        return np.asarray(ds.variables["time"][:]), np.asarray(ds.variables["DCS_number"][:])


def ks(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    x = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, x, "right") / a.size - np.searchsorted(b, x, "right") / b.size).max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--json"); ap.add_argument("--seg-days", type=int, default=0)
    ap.add_argument("--ref-segment-days", type=int, default=0,
                    help="reference labels restart in every N-day segment (production segmented run): make them unique as (segment, label)")
    a = ap.parse_args()
    dr, dt = days(a.ref), days(a.test)
    keys = sorted(set(dr) & set(dt))
    if a.start: keys = [k for k in keys if k >= a.start]
    if a.end: keys = [k for k in keys if k <= a.end]
    codes, counts, dayidx = [], [], []
    nobj_r, nobj_t = [], []
    first_r, last_r, first_t, last_t = {}, {}, {}, {}
    for di, k in enumerate(keys):
        tr, ar = load(dr[k]); tt, at = load(dt[k])
        assert np.array_equal(tr, tt), f"time axes differ on {k}"
        if a.ref_segment_days:
            from datetime import datetime as _dt
            seg = (_dt.strptime(k, "%Y%m%d").timetuple().tm_yday - 1) // a.ref_segment_days
            ar = np.where(ar > 0, ar + np.int32(seg * (1 << 20)), ar).astype(np.int32)
        for s in range(ar.shape[0]):
            gs = int(tr[s]) // 1200
            x, y = ar[s], at[s]
            m = (x != -998) | (y != -998)
            ux, uy = np.unique(x[x > 0]), np.unique(y[y > 0])
            nobj_r.append(ux.size); nobj_t.append(uy.size)
            for l in ux.tolist(): first_r.setdefault(l, gs); last_r[l] = gs
            for l in uy.tolist(): first_t.setdefault(l, gs); last_t[l] = gs
            if not m.any(): continue
            c = ((x[m].astype(np.int64) + OFF) << B) | (y[m].astype(np.int64) + OFF)
            u, n = np.unique(c, return_counts=True)
            codes.append(u); counts.append(n); dayidx.append(np.full(u.size, di, np.int32))
    codes = np.concatenate(codes); counts = np.concatenate(counts); dayidx = np.concatenate(dayidx)
    r = (codes >> B) - OFF
    # global pair table
    pk, inv = np.unique(codes, return_inverse=True)
    pc = np.bincount(inv, weights=counts).astype(np.int64)
    pr = (pk >> B) - OFF; pt = (pk & ((1 << B) - 1)) - OFF
    vox_ref_only = int(pc[(pr > 0) & (pt <= 0)].sum()); vox_test_only = int(pc[(pt > 0) & (pr <= 0)].sum())
    both = (pr > 0) & (pt > 0)
    br, bt, bc = pr[both], pt[both], pc[both]
    ur, ir = np.unique(pr[pr > 0], return_inverse=True); rsz_arr = np.bincount(ir, weights=pc[pr > 0]).astype(np.int64)
    ut, it = np.unique(pt[pt > 0], return_inverse=True); tsz_arr = np.bincount(it, weights=pc[pt > 0]).astype(np.int64)
    rsz = dict(zip(ur.tolist(), rsz_arr.tolist())); tsz = dict(zip(ut.tolist(), tsz_arr.tolist()))
    # best match per ref object
    order = np.lexsort((-bc, br))
    br_s, bt_s, bc_s = br[order], bt[order], bc[order]
    first = np.concatenate(([True], br_s[1:] != br_s[:-1]))
    best_t = dict(zip(br_s[first].tolist(), zip(bt_s[first].tolist(), bc_s[first].tolist())))
    second = {}
    nxt = np.nonzero(first)[0] + 1
    for i in nxt:
        if i < br_s.size and br_s[i] == br_s[i - 1]:
            second[int(br_s[i])] = int(bc_s[i])
    ident = {}; iou = []; unmatched = 0; split_ref = 0
    for l, sz in rsz.items():
        if l not in best_t:
            unmatched += 1; iou.append(0.0); continue
        tl, v = best_t[l]
        iou.append(v / (sz + tsz[tl] - v))
        ident[l] = (sz == v == tsz[tl])
        if second.get(l, 0) / sz > 0.05: split_ref += 1
    # test objects overlapping >=2 ref objects (merge in test)
    order = np.lexsort((-bc, bt)); bt_s, bc_s = bt[order], bc[order]
    firstt = np.concatenate(([True], bt_s[1:] != bt_s[:-1]))
    sec_t = {}
    for i in np.nonzero(firstt)[0] + 1:
        if i < bt_s.size and bt_s[i] == bt_s[i - 1]:
            sec_t[int(bt_s[i])] = int(bc_s[i])
    merge_test = sum(1 for l, v in sec_t.items() if v / tsz[l] > 0.05)
    n_ident = sum(ident.values())
    dur_r = np.array([last_r[l] - first_r[l] + 1 for l in rsz]); dur_t = np.array([last_t[l] - first_t[l] + 1 for l in tsz])
    life_diff = 0
    for l, (tl, v) in best_t.items():
        if ident.get(l): continue
        if first_r[l] != first_t[tl] or last_r[l] != last_t[tl]: life_diff += 1
    res = dict(days=len(keys), slots=len(nobj_r), ref_objects=len(rsz), test_objects=len(tsz),
               objects_identical_membership=n_ident, frac_ref_objects_identical=n_ident / max(1, len(rsz)),
               unmatched_ref_objects=unmatched, ref_split_in_test=split_ref, test_merge_of_ref=merge_test,
               matched_nonidentical_with_different_birth_or_death=life_diff,
               voxels_ref_only=vox_ref_only, voxels_test_only=vox_test_only, voxels_ref_total=int(pc[pr > 0].sum()),
               iou_mean=float(np.mean(iou)) if iou else None, iou_min=float(np.min(iou)) if iou else None,
               iou_p05=float(np.percentile(iou, 5)) if iou else None,
               objects_per_slot_equal=bool(np.array_equal(nobj_r, nobj_t)),
               max_abs_diff_objects_per_slot=int(np.abs(np.array(nobj_r) - np.array(nobj_t)).max()),
               KS_size_voxels=ks(list(rsz.values()), list(tsz.values())), KS_duration_slots=ks(dur_r, dur_t))
    if a.seg_days:
        isid = np.array([ident.get(int(x), False) for x in r])
        sel = r > 0
        per = {}
        for di, k in enumerate(keys):
            mm = sel & (dayidx == di)
            tot = counts[mm].sum()
            per[k] = float(counts[mm & isid].sum() / tot) if tot else None
        # day index inside a segment: (day-of-year-1) % seg_days
        from datetime import datetime
        agg = {}
        for k, v in per.items():
            if v is None: continue
            doy = datetime.strptime(k, "%Y%m%d").timetuple().tm_yday
            agg.setdefault((doy - 1) % a.seg_days, []).append(v)
        res["frac_voxels_in_identical_objects_by_day_in_segment"] = {int(i): float(np.mean(v)) for i, v in sorted(agg.items())}
    print(json.dumps(res, indent=1))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
