"""Compatibility layer: original TOOCAN FileTracking output (INT_* per DCS, LC_* DCS x time).

compute_INT_variables / build_LC_variables / write_tracking_nc are the original functions
(toocan-python/toocan/lib/postprocessing/tracking/{int_builder,lc_builder,nc_writer}.py, GPL-3.0-or-later,
(C) Thomas Fiolleau), reproduced so the numbers are the original ones.  `tracks_from_obs` converts the
compact per-observation arrays produced by catalog.py into the original `tracks` dict-of-lists.
"""
from datetime import datetime, timedelta

import numpy as np

EPOCH = datetime(1970, 1, 1)


def _utc(t):
    """Seconds since 1970-01-01 UTC of a naive-UTC datetime.  Timezone independent.

    The original used `int(t.timestamp())`, which interprets a naive datetime in the *host's local timezone*
    (CET on the Levante login nodes, UTC on others) -> INT_UTC_*/LC_UTC/time depended on the host.  This is
    identical to the original on a UTC host."""
    return int((t - EPOCH).total_seconds())


def tracks_from_obs(obs):
    """obs: dict of 1-D arrays, time-ordered ('epoch' float seconds). -> original tracks dict."""
    lab = obs["label"]
    order = np.argsort(lab, kind="stable")          # stable: keeps time order within a label
    ls = lab[order]
    cut = np.flatnonzero(np.diff(ls)) + 1
    starts = np.concatenate(([0], cut)); ends = np.concatenate((cut, [ls.size]))
    tracks = {}
    for s, e in zip(starts, ends):
        ix = order[s:e]
        t = {"times": [EPOCH + timedelta(seconds=float(x)) for x in obs["epoch"][ix]],
             "area": [int(x) for x in obs["area"][ix]],
             "lon_cm": obs["lon_cm"][ix].tolist(), "lat_cm": obs["lat_cm"][ix].tolist(),
             "tbmin": obs["tbmin"][ix].tolist(), "tb90": obs["tb90"][ix].tolist(),
             "avg235": obs["avg235"][ix].tolist(), "avg208": obs["avg208"][ix].tolist(),
             "avg200": obs["avg200"][ix].tolist()}
        for thr in (235, 220, 210, 200):
            t[f"pix_{thr}"] = [int(x) for x in obs[f"pix_{thr}"][ix]]
            t[f"km2_{thr}"] = obs[f"km2_{thr}"][ix].tolist()
        for thr in (235, 220):
            t[f"ellipse_{thr}"] = list(zip(obs[f"ell{thr}_a"][ix].tolist(), obs[f"ell{thr}_b"][ix].tolist(),
                                           obs[f"ell{thr}_e"][ix].tolist(), obs[f"ell{thr}_ang"][ix].tolist()))
        tracks[np.int32(ls[s])] = t
    return tracks


def compute_INT_variables(tracks, dt_minutes=30):
    INT = {}
    R = 6371.0
    dt_hours = dt_minutes / 60.0
    for lab, tr in tracks.items():
        times = tr["times"]
        lon = np.array(tr["lon_cm"])
        lat = np.array(tr["lat_cm"])
        duration_hours = len(times) * dt_hours
        lon0, lat0 = lon[0], lat[0]
        lon1, lat1 = lon[-1], lat[-1]
        dlon = np.deg2rad(lon1 - lon0)
        dlat = np.deg2rad(lat1 - lat0)
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat0)) * np.cos(np.deg2rad(lat1)) * np.sin(dlon / 2) ** 2)
        dist_km = 2 * R * np.sqrt(a)
        vel_ms = (dist_km * 1000) / (duration_hours * 3600) if duration_hours > 0 else np.nan
        INT[lab] = {
            "DCS_number": int(lab), "duration_hours": duration_hours,
            "UTC_init": _utc(times[0]), "UTC_end": _utc(times[-1]),
            "lon_init": float(lon0), "lat_init": float(lat0), "lon_end": float(lon1), "lat_end": float(lat1),
            "lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)),
            "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat)),
            "tb_min_global": float(np.nanmin(tr["tbmin"])),
            "surfmaxkm2_235": float(np.nanmax(tr["km2_235"])), "surfmaxkm2_220": float(np.nanmax(tr["km2_220"])),
            "surfmaxkm2_210": float(np.nanmax(tr["km2_210"])), "surfmaxkm2_200": float(np.nanmax(tr["km2_200"])),
            "surfcumkm2_235": float(np.nansum(tr["km2_235"])),
            "distance_km": float(dist_km), "velocity_ms": float(vel_ms),
            "qltyDCS": 11100, "classif_Roca": -999, "classif_JIRAK": -999, "classif_MADDOX": -999,
            "TS_number_IBTRACS": -999, "TS_nature_IBTRACS": -999, "TS_mindistance_IBTRACS": -999.0,
        }
    return INT


def build_LC_variables(tracks, global_times):
    dcs_ids = sorted(tracks.keys())
    n_dcs, n_time = len(dcs_ids), len(global_times)
    index_time = {t: i for i, t in enumerate(global_times)}
    fill_int, fill_float = -999, -999.0
    f32 = lambda: np.full((n_dcs, n_time), fill_float, dtype=np.float32)
    LC = {
        "DCS_ids": dcs_ids,
        "LC_tbmin": f32(), "LC_tb90th": f32(), "LC_tbavg_235K": f32(), "LC_tbavg_208K": f32(), "LC_tbavg_200K": f32(),
        "LC_lon": f32(), "LC_lat": f32(),
        "LC_UTC": np.full((n_dcs, n_time), fill_int, dtype=np.int32),
        "LC_velocity": f32(),
        "LC_surfPix_235K": np.full((n_dcs, n_time), fill_int, dtype=np.int32),
        "LC_surfkm2_235K": f32(), "LC_surfkm2_220K": f32(), "LC_surfkm2_210K": f32(), "LC_surfkm2_200K": f32(),
        "LC_semiminor_235K": np.full((n_dcs, n_time), fill_float), "LC_semimajor_235K": np.full((n_dcs, n_time), fill_float),
        "LC_ecc_235K": np.full((n_dcs, n_time), fill_float), "LC_orientation_235K": np.full((n_dcs, n_time), fill_float),
        "LC_semiminor_220K": np.full((n_dcs, n_time), fill_float), "LC_semimajor_220K": np.full((n_dcs, n_time), fill_float),
        "LC_ecc_220K": np.full((n_dcs, n_time), fill_float), "LC_orientation_220K": np.full((n_dcs, n_time), fill_float),
    }
    for i, lab in enumerate(dcs_ids):
        tr = tracks[lab]
        times, lon, lat = tr["times"], tr["lon_cm"], tr["lat_cm"]
        prev_lon = prev_lat = prev_t = None
        for idx in range(len(times)):
            t = times[idx]
            j = index_time[t]
            LC["LC_lon"][i, j] = lon[idx]; LC["LC_lat"][i, j] = lat[idx]
            LC["LC_tbmin"][i, j] = tr["tbmin"][idx]; LC["LC_tb90th"][i, j] = tr["tb90"][idx]
            LC["LC_tbavg_235K"][i, j] = tr["avg235"][idx]; LC["LC_tbavg_208K"][i, j] = tr["avg208"][idx]
            LC["LC_tbavg_200K"][i, j] = tr["avg200"][idx]
            LC["LC_surfPix_235K"][i, j] = tr["pix_235"][idx]
            LC["LC_surfkm2_235K"][i, j] = tr["km2_235"][idx]; LC["LC_surfkm2_220K"][i, j] = tr["km2_220"][idx]
            LC["LC_surfkm2_210K"][i, j] = tr["km2_210"][idx]; LC["LC_surfkm2_200K"][i, j] = tr["km2_200"][idx]
            LC["LC_UTC"][i, j] = _utc(t)
            if prev_lon is not None:
                R = 6371.0
                dlon = np.deg2rad(lon[idx] - prev_lon); dlat = np.deg2rad(lat[idx] - prev_lat)
                a = (np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(prev_lat)) * np.cos(np.deg2rad(lat[idx])) * np.sin(dlon / 2) ** 2)
                dist_km = 2 * R * np.sqrt(a)
                dt_sec = (t - prev_t).total_seconds()
                LC["LC_velocity"][i, j] = (dist_km * 1000) / dt_sec if dt_sec > 0 else -999
            prev_lon, prev_lat, prev_t = lon[idx], lat[idx], t
    return LC


def write_tracking_nc(outfile, INT, LC, global_times, region="AFRICA", platform="MSG",
                      title="TOOCAN Tracking", version="3.0.0"):
    import xarray as xr
    dcs_ids = LC["DCS_ids"]
    time_seconds = np.array([_utc(t) for t in global_times], dtype=np.int32)
    ds = xr.Dataset(coords={"DCS": ("DCS", np.array(dcs_ids, dtype=np.int32)), "time": ("time", time_seconds)})
    for key in INT[dcs_ids[0]]:
        ds["INT_" + key] = ("DCS", np.array([INT[lab][key] for lab in dcs_ids]))
    for key in LC:
        if key == "DCS_ids":
            continue
        ds[key] = (("DCS", "time"), LC[key])
    ds.attrs.update({"title": title, "tracker": "TOOCAN", "version": version, "region": region, "platform": platform})
    ds.to_netcdf(outfile)
