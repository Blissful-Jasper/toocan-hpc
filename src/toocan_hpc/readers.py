"""Input readers.  Numerically identical to toocan.lib.io.open_IRdata.read_sam_olr (OLR -> BT)."""
import glob
import os
import re
from datetime import datetime

import netCDF4
import numpy as np
import pandas as pd

SIGMA = 5.670374419e-8  # Stefan-Boltzmann


def convert_olr_to_irbt(olr):
    """EXACT same float32 operation sequence as the original (numpy-2 weak python scalars)."""
    olr = olr.astype(np.float32)
    tb = (olr / SIGMA) ** 0.25
    return tb ** 2 * 0.00443828 - tb * 0.543846 + 129.544


_DATE_RE = re.compile(r"(\d{8})")


def _file_date(name):
    m = _DATE_RE.findall(name)
    for s in m:
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            continue
    return None


def scan_inputs(path_ir, start, end):
    """[(file, datetime, time_index)] sorted by time; mirrors build_ir_filelist(expand_time_records=True)."""
    files = sorted(set(glob.glob(os.path.join(path_ir, "**", "*.nc"), recursive=True)))
    out = []
    for f in files:
        d = _file_date(os.path.basename(f))
        if d is not None and (d < start.date() or d > end.date()):
            continue
        with netCDF4.Dataset(f) as ds:
            tv = ds.variables["time"]
            times = pd.to_datetime(netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False,
                                                    only_use_python_datetimes=True))
        for ti, t in enumerate(times):
            t = pd.Timestamp(t).to_pydatetime()
            if start <= t <= end:
                out.append((f, t, ti))
    out.sort(key=lambda x: x[1])
    return out


def read_slot_bt(path, time_index, ymin, ymax, xmin, xmax, varname=None):
    """One time record -> float32 BT(K) of the cropped domain."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        v = varname or ("rlut" if "rlut" in ds.variables else "LWNTA")
        a = ds.variables[v][time_index, ymin:ymax + 1, xmin:xmax + 1]
    return convert_olr_to_irbt(np.asarray(a))
