#!/usr/bin/env python3
"""Compare two FileTracking NetCDF files (original vs TOOCAN-HPC): every INT_* / LC_* variable."""
import argparse, sys, json
import numpy as np
import netCDF4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("new"); ap.add_argument("--json")
    a = ap.parse_args()
    r, n = netCDF4.Dataset(a.ref), netCDF4.Dataset(a.new)
    for d in (r, n): d.set_auto_mask(False)
    res = {}; allexact = True
    names = sorted(set(r.variables) | set(n.variables))
    for k in names:
        if k not in r.variables or k not in n.variables:
            res[k] = "MISSING in " + ("new" if k in r.variables else "ref"); allexact = False; continue
        x, y = np.asarray(r.variables[k][:]), np.asarray(n.variables[k][:])
        if x.shape != y.shape:
            res[k] = f"SHAPE {x.shape} vs {y.shape}"; allexact = False; continue
        if np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y):
            res[k] = "exact"
        else:
            allexact = False
            if x.dtype.kind == "f":
                m = ~(np.isnan(x) & np.isnan(y))
                diff = np.abs(x.astype(np.float64) - y.astype(np.float64))[m]
                rel = diff / np.maximum(np.abs(x.astype(np.float64))[m], 1e-300)
                res[k] = f"DIFF n={int((diff>0).sum())}/{diff.size} maxabs={diff.max():.3e} maxrel={rel.max():.3e}"
            else:
                res[k] = f"DIFF n={int((x!=y).sum())}/{x.size}"
    ok = {k: v for k, v in res.items() if v == "exact"}
    bad = {k: v for k, v in res.items() if v != "exact"}
    print(f"{len(ok)}/{len(res)} variables bit-identical")
    for k, v in bad.items(): print("  ", k, v)
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
    print("RESULT:", "ALL VARIABLES BIT-IDENTICAL" if allexact else "DIFFERENCES (see above)")
    sys.exit(0 if allexact else 1)

if __name__ == "__main__": main()
