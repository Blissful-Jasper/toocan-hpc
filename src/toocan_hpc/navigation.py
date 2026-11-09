"""Navigation crop: same selection rule as toocan.lib.io.open_navigationFile.crop_navigation_file."""
import numpy as np
import netCDF4


def crop_navigation(navfile, lonmin, lonmax, latmin, latmax):
    """Returns dict(lat2d, lon2d, area2d(float32 C-contig), ymin,ymax,xmin,xmax)."""
    with netCDF4.Dataset(navfile) as ds:
        ds.set_auto_mask(False)
        def pick(cands):
            for k in cands:
                if k in ds.variables:
                    return np.asarray(ds.variables[k][:])
            return None
        lat = pick(['lat', 'latitude', 'mat_latitude', 'matlat'])
        lon = pick(['lon', 'longitude', 'mat_longitude', 'matlon'])
        area = pick(['mat_surfacePix', 'surface', 'area'])
    if lat is None or lon is None:
        raise ValueError("no lat/lon in navigation file")
    if lat.ndim == 1 and lon.ndim == 1:
        lat_tol = 1.0e-6 * max(1.0, float(np.nanmax(np.abs(lat))))
        lon_tol = 1.0e-6 * max(1.0, float(np.nanmax(np.abs(lon))))
        yi = np.where((lat >= latmin - lat_tol) & (lat <= latmax + lat_tol))[0]
        xi = np.where((lon >= lonmin - lon_tol) & (lon <= lonmax + lon_tol))[0]
        if yi.size == 0 or xi.size == 0:
            raise ValueError("No pixel found inside requested domain")
        ymin, ymax, xmin, xmax = int(yi[0]), int(yi[-1]), int(xi[0]), int(xi[-1])
        LAT = lat[ymin:ymax + 1][:, None] * np.ones((1, xmax - xmin + 1))
        LON = lon[xmin:xmax + 1][None, :] * np.ones((ymax - ymin + 1, 1))
        area = np.ones_like(LAT) if area is None else area[ymin:ymax + 1, xmin:xmax + 1]
    else:
        LATf, LONf = lat, lon
        lat_tol = 1.0e-6 * max(1.0, float(np.nanmax(np.abs(LATf))))
        lon_tol = 1.0e-6 * max(1.0, float(np.nanmax(np.abs(LONf))))
        m = ((LATf >= latmin - lat_tol) & (LATf <= latmax + lat_tol) &
             (LONf >= lonmin - lon_tol) & (LONf <= lonmax + lon_tol))
        if not m.any():
            raise ValueError("No pixel found inside requested domain")
        yy, xx = np.where(m)
        ymin, ymax, xmin, xmax = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
        LAT = LATf[ymin:ymax + 1, xmin:xmax + 1]
        LON = LONf[ymin:ymax + 1, xmin:xmax + 1]
        area = np.ones_like(LAT) if area is None else area[ymin:ymax + 1, xmin:xmax + 1]
    return dict(lat2d=np.ascontiguousarray(LAT), lon2d=np.ascontiguousarray(LON),
                area2d=np.ascontiguousarray(area, dtype=np.float32),
                ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
