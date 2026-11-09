"""Daily-NetCDF mask writer: asynchronous, parallel (process pool), atomic installs.

Output layout/variables are the ones of the original `save_labels_daily` (mask_file_layout = daily):
  <root>/toocan_<version>/<REGION>/<YYYY>/<YYYY_MM_DD>/ToocanCloudMask_<YYYYMMDD>.nc
  DCS_number(time,y,x) int32, fill -998 (== "no DCS"), latitude(y,x), longitude(y,x), time [s since 1970]

Finalised slots are gathered per UTC day; a day is handed to a worker process (shared memory, no pickling of the
array) as soon as all of its slots are final.  The file is written under a temporary name, fsync'ed and renamed,
so an interrupted job can never leave a half-written day file under the final name.
"""
import concurrent.futures as cf
import multiprocessing as mp
import os
import uuid
from datetime import datetime, timedelta
from multiprocessing import shared_memory

import netCDF4
import numpy as np

EPOCH = datetime(1970, 1, 1)
FILL = -998
_W = {}          # fork-inherited worker context (lat/lon/attributes)


def _atomic_nc(path, write_fn):
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        open(tmp, "wb").close()          # pre-create: avoids a spurious HDF5 'file check' diagnostic
        write_fn(tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_nc(path, epochs, arr, lat, lon, platform, version, complevel):
    Y, X = lat.shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(epochs)); ds.createDimension("y", Y); ds.createDimension("x", X)
        tv = ds.createVariable("time", "f8", ("time",)); tv[:] = np.asarray(epochs, dtype=np.float64)
        tv.units = "seconds since 1970-01-01 00:00:00 UTC"
        la = ds.createVariable("latitude", "f8", ("y", "x")); la[:] = lat
        lo = ds.createVariable("longitude", "f8", ("y", "x")); lo[:] = lon
        v = ds.createVariable("DCS_number", "i4", ("time", "y", "x"), zlib=complevel > 0, complevel=complevel,
                              shuffle=False, chunksizes=(1, Y, X), fill_value=FILL)
        v.long_name = "number of Deep Cloud Systems"
        v.standard_name = "number_of_Deep_Cloud_Systems"
        v.missing_value = np.int32(FILL)
        v[:] = arr
        ds.title = "TOOCAN segmented daily labels"; ds.conventions = "CF-1.6"
        ds.platform = platform; ds.version = str(version)


def _worker_write(args):
    shm_name, n, Y, X, epochs, path = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray((n, Y, X), dtype=np.int32, buffer=shm.buf)
        c = _W
        _atomic_nc(path, lambda tmp: _write_nc(tmp, epochs, arr, c["lat"], c["lon"], c["platform"], c["version"], c["complevel"]))
    finally:
        shm.close()
    return path


class DayWriter:
    def __init__(self, root, version, region, lat2d, lon2d, platform, complevel=4, expected_times=None, nworkers=2,
                 skip_days_before=None):
        self.outdir = os.path.join(root, f"toocan_{version}", region)
        self.expected = {}
        for t in (expected_times or []):
            self.expected.setdefault(t.date(), []).append(t)
        self.skip_before = skip_days_before          # days < this date are already complete (resume)
        self.buf = {}                                  # date -> {epoch: int32 (Y,X)}
        self.written = []
        self.shape = lat2d.shape
        _W.update(lat=lat2d, lon=lon2d, platform=platform, version=version, complevel=int(complevel))
        self.pool = cf.ProcessPoolExecutor(max_workers=nworkers, mp_context=mp.get_context("fork")) if nworkers > 0 else None
        self.futs = []

    def day_path(self, d):
        return os.path.join(self.outdir, f"{d:%Y}", f"{d:%Y_%m_%d}", f"ToocanCloudMask_{d:%Y%m%d}.nc")

    def day_exists(self, d):
        return os.path.exists(self.day_path(d))

    def add_slots(self, times, labels):
        """labels: (n,Y,X) int32 with 0 for background (converted to -998 here). Arrays are COPIED."""
        for t, lab in zip(times, labels):
            if self.skip_before is not None and t.date() < self.skip_before:
                continue
            ep = float((t - EPOCH).total_seconds())
            a = np.array(lab, dtype=np.int32, copy=True)
            a[a == 0] = FILL
            self.buf.setdefault(t.date(), {})[ep] = a

    def _dispatch(self, d):
        have = self.buf.pop(d)
        eps = sorted(have)
        Y, X = self.shape
        path = self.day_path(d)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shm = shared_memory.SharedMemory(create=True, size=len(eps) * Y * X * 4)
        arr = np.ndarray((len(eps), Y, X), dtype=np.int32, buffer=shm.buf)
        for i, e in enumerate(eps):
            arr[i] = have[e]
        del arr
        if self.pool is None:
            _worker_write((shm.name, len(eps), Y, X, eps, path)); shm.close(); shm.unlink()
            self.written.append(d); return
        f = self.pool.submit(_worker_write, (shm.name, len(eps), Y, X, eps, path))
        self.futs.append((f, shm, d))

    def _reap(self, block=False):
        keep = []
        for f, shm, d in self.futs:
            if block or f.done():
                f.result()                                  # raises on failure
                shm.close(); shm.unlink(); self.written.append(d)
            else:
                keep.append((f, shm, d))
        self.futs = keep

    def flush_complete(self, force_all=False):
        for d in sorted(self.buf):
            exp = self.expected.get(d)
            if (exp is not None and len(self.buf[d]) >= len(exp)) or force_all:
                self._dispatch(d)
        self._reap(block=False)
        while len(self.futs) > 3:                          # back-pressure: bound shared-memory use
            self.futs[0][0].result(); self._reap(block=False)

    def finish(self):
        self._reap(block=True)
        if self.pool:
            self.pool.shutdown(wait=True)
