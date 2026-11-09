#!/usr/bin/env python3
"""Write a small SYNTHETIC input data set (daily OLR files + navigation file) so the whole pipeline can be tried
without any real data.  Moving / merging cold-cloud blobs on a 0.25-degree grid, 30S-30N x 60E-120E, 20-min cadence.

    python tools/make_demo_data.py --out demo_data --days 3

Creates   demo_data/olr/YYYY/olr_YYYYMMDD.nc   (variable rlut, W m-2, dims time/lat/lon)
          demo_data/navigation.nc
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import netCDF4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_navigation import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data"); ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start", default="2000-01-01"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    lat = np.arange(-15, 15.01, 0.25); lon = np.arange(60, 120.01, 0.25)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(a.seed)
    nslot = a.days * 72
    nb = 40                                             # number of convective systems
    t0 = rng.integers(0, nslot, nb); life = rng.integers(8, 60, nb)
    y0 = rng.uniform(-12, 12, nb); x0 = rng.uniform(65, 115, nb)
    vy = rng.normal(0, 0.02, nb); vx = rng.normal(-0.08, 0.04, nb)        # deg / slot, westward drift
    rad = rng.uniform(1.0, 3.0, nb); depth = rng.uniform(100, 190, nb)    # deg, W m-2
    d0 = datetime.strptime(a.start, "%Y-%m-%d")
    build(lat, lon, os.path.join(a.out, "navigation.nc"))
    for d in range(a.days):
        day = d0 + timedelta(days=d)
        p = os.path.join(a.out, "olr", f"{day:%Y}", f"olr_{day:%Y%m%d}.nc"); os.makedirs(os.path.dirname(p), exist_ok=True)
        olr = np.empty((72, lat.size, lon.size), np.float32)
        for s in range(72):
            k = d * 72 + s
            f = 270 + 8 * rng.standard_normal(yy.shape)
            for b in np.where((k >= t0) & (k < t0 + life))[0]:
                age = (k - t0[b]) / life[b]; amp = depth[b] * np.sin(np.pi * age)      # grow then decay
                cy, cx = y0[b] + vy[b] * (k - t0[b]), x0[b] + vx[b] * (k - t0[b])
                f -= amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (rad[b] * (0.5 + age)) ** 2))
            olr[s] = np.clip(f, 60, 330)
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("time", 72); ds.createDimension("lat", lat.size); ds.createDimension("lon", lon.size)
            t = ds.createVariable("time", "f8", ("time",)); t.units = "minutes since 1970-01-01 00:00:00"; t.calendar = "standard"
            t[:] = netCDF4.date2num([day + timedelta(minutes=20 * s) for s in range(72)], t.units)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat; ds["lat"].units = "degrees_north"
            ds.createVariable("lon", "f8", ("lon",))[:] = lon; ds["lon"].units = "degrees_east"
            v = ds.createVariable("rlut", "f4", ("time", "lat", "lon"), zlib=True, complevel=1); v[:] = olr; v.units = "W m-2"
        print("wrote", p)


if __name__ == "__main__":
    main()
