#!/usr/bin/env python3
"""Create the TOOCAN *navigation file* (lat, lon, pixel area in km^2) for a regular lat/lon grid.

TOOCAN applies `minAreaSeed` in km^2, so it needs the area of every pixel.  Area of a cell on a sphere (R = 6371 km):
    A = R^2 * dlon * (sin(lat_north) - sin(lat_south))        (cell edges = midpoints between grid centres)

Either take the grid from one of your input files:
    python tools/make_navigation.py --from-file my_olr_day.nc --out navigation.nc
or specify it (cell CENTRES):
    python tools/make_navigation.py --lat -30 30 --lon 0 359.9 --res 0.1 --out navigation.nc
(Reproduces, to float32 precision, the 0.1-degree tropics file used for the published C5 runs.)
"""
import argparse
import os
import numpy as np
import netCDF4

R_EARTH_KM = 6371.0


def edges(c):
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e


def build(lat, lon, out):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    la = np.deg2rad(np.clip(edges(lat), -90, 90)); lo = np.deg2rad(edges(lon))
    area = R_EARTH_KM ** 2 * np.diff(lo)[None, :] * (np.sin(la[1:]) - np.sin(la[:-1]))[:, None]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with netCDF4.Dataset(out, "w") as ds:
        ds.title = "TOOCAN navigation grid"; ds.area_method = "spherical quadrilateral"
        ds.createDimension("y", lat.size); ds.createDimension("x", lon.size)
        for n, d, dims in (("lat", lat, ("y",)), ("lon", lon, ("x",))):
            ds.createVariable(n, "f8", dims)[:] = d
        ds.createVariable("mat_latitude", "f4", ("y", "x"))[:] = np.broadcast_to(lat[:, None], area.shape)
        ds.createVariable("mat_longitude", "f4", ("y", "x"))[:] = np.broadcast_to(lon[None, :], area.shape)
        v = ds.createVariable("mat_surfacePix", "f4", ("y", "x")); v[:] = area; v.units = "km2"
    print(f"wrote {out}: {lat.size} x {lon.size} pixels, area {area.min():.1f}..{area.max():.1f} km2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file"); ap.add_argument("--lat-name", default="lat"); ap.add_argument("--lon-name", default="lon")
    ap.add_argument("--lat", nargs=2, type=float); ap.add_argument("--lon", nargs=2, type=float)
    ap.add_argument("--res", type=float); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_file:
        with netCDF4.Dataset(a.from_file) as ds:
            lat, lon = ds[a.lat_name][:].data, ds[a.lon_name][:].data
    elif a.lat and a.lon and a.res:
        lat = np.arange(a.lat[0], a.lat[1] + a.res / 2, a.res); lon = np.arange(a.lon[0], a.lon[1] + a.res / 2, a.res)
    else:
        ap.error("give --from-file, or --lat --lon --res")
    build(lat, lon, a.out)


if __name__ == "__main__":
    main()
