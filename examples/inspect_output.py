#!/usr/bin/env python3
"""Print a short summary of the TOOCAN-HPC output tree:  python examples/inspect_output.py <out dir>"""
import glob
import sys
import numpy as np
import xarray as xr

root = sys.argv[1]
masks = sorted(glob.glob(f"{root}/toocan_*/*/*/*/ToocanCloudMask_*.nc"))
print(f"{len(masks)} daily mask files; first: {masks[0] if masks else None}")
if masks:
    m = xr.open_dataset(masks[0])
    lab = m.DCS_number.values
    print(f"{masks[0].split('/')[-1]}: DCS_number {lab.shape}, {np.unique(lab[lab > 0]).size} DCS present, "
          f"{(lab > 0).mean() * 100:.1f}% of pixels inside a DCS")
for f in sorted(glob.glob(f"{root}/catalog/catalog_compact-*.nc")):
    c = xr.open_dataset(f)
    lab = c.obs_label.values
    u, first = np.unique(lab, return_index=True)
    n_slots = np.bincount(np.unique(lab, return_inverse=True)[1])
    print(f"{f.split('/')[-1]}: {u.size} DCS, {lab.size} (DCS,time) rows, median lifetime {np.median(n_slots) * 20:.0f} min, "
          f"max area {np.nanmax(c.obs_km2_235.values):.0f} km2")
