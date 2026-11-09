#!/usr/bin/env python3
"""Compare two TOOCAN mask trees (daily NetCDF 'DCS_number').

Level 1  exact: identical label values voxel by voxel.
Level 2  membership: identical partition of voxels into objects up to a label renaming
         (bijection between the label sets) -- the scientifically meaningful test.
Reports, per day and in total: voxels that differ, slots that differ, objects that differ.
"""
import argparse, glob, os, sys, json
import numpy as np
import netCDF4


def day_files(root):
    return {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(root, "**", "ToocanCloudMask_*.nc"), recursive=True))}


def load(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        return np.asarray(ds.variables["time"][:]), np.asarray(ds.variables["DCS_number"][:])


def membership_equal(a, b):
    """True iff a and b define the same partition (label values may differ by a 1:1 renaming)."""
    m = (a != -998) | (b != -998)
    if not m.any():
        return True, 0
    la, lb = a[m].astype(np.int64), b[m].astype(np.int64)
    key = la * (1 << 32) + (lb + (1 << 31))
    uk = np.unique(key)
    ua = uk >> 32
    ub = (uk & 0xFFFFFFFF) - (1 << 31)
    ok = (np.unique(ua).size == ua.size) and (np.unique(ub).size == ub.size)  # each a-label pairs with exactly one b-label
    return bool(ok), int(uk.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fr, fn = day_files(a.ref), day_files(a.new)
    missing = sorted(set(fr) - set(fn)); extra = sorted(set(fn) - set(fr))
    tot = dict(days=0, exact_days=0, member_days=0, voxels_diff=0, slots_diff=0, days_missing=len(missing), days_extra=len(extra),
               labels_max_ref=0, labels_max_new=0)
    bad = []
    for name in sorted(set(fr) & set(fn)):
        tr, ar = load(fr[name]); tn, an = load(fn[name])
        tot["days"] += 1
        if tr.shape != tn.shape or not np.array_equal(tr, tn) or ar.shape != an.shape:
            bad.append((name, "shape/time mismatch", tr.shape, tn.shape)); continue
        ne = ar != an
        nv = int(ne.sum())
        tot["labels_max_ref"] = max(tot["labels_max_ref"], int(ar.max())); tot["labels_max_new"] = max(tot["labels_max_new"], int(an.max()))
        if nv == 0:
            tot["exact_days"] += 1; tot["member_days"] += 1
        else:
            tot["voxels_diff"] += nv
            tot["slots_diff"] += int(ne.reshape(ne.shape[0], -1).any(axis=1).sum())
            ok, npairs = membership_equal(ar, an)
            if ok: tot["member_days"] += 1
            bad.append((name, f"{nv} voxels differ in {int(ne.reshape(ne.shape[0],-1).any(axis=1).sum())} slots; same partition up to renaming: {ok}"))
    print(json.dumps(tot, indent=1))
    for b in bad[:30]:
        print("DIFF", *b)
    if missing: print("MISSING in new:", missing[:10])
    if extra: print("EXTRA in new:", extra[:10])
    if a.json:
        json.dump(dict(totals=tot, diffs=[list(map(str, b)) for b in bad]), open(a.json, "w"), indent=1)
    ok = tot["voxels_diff"] == 0 and not missing and not bad
    print("RESULT:", "IDENTICAL (bit-for-bit labels)" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
