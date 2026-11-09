"""Parallel, shared-memory loader of IR volumes (one process pool, one task per time slot)."""
import os
from multiprocessing import shared_memory
import concurrent.futures as cf
import multiprocessing as mp

import numpy as np

from .readers import read_slot_bt

_G = {}


def _attach(name, shape):
    shm = shared_memory.SharedMemory(name=name)
    return shm, np.ndarray(shape, dtype=np.float32, buffer=shm.buf)


def _worker(args):
    name, shape, slot, path, ti, box, var = args
    key = (name, shape)
    if key not in _G:
        _G[key] = _attach(name, shape)
    _, arr = _G[key]
    arr[slot] = read_slot_bt(path, ti, *box, varname=var)
    return slot


class VolumeLoader:
    """Owns `nbuf` shared-memory volumes (T,Y,X) float32; loads slot lists asynchronously."""

    def __init__(self, T, Y, X, nbuf=2, nworkers=8, var=None):
        self.shape = (T, Y, X)
        self.var = var
        self.shms = [shared_memory.SharedMemory(create=True, size=T * Y * X * 4) for _ in range(nbuf)]
        self.arrs = [np.ndarray(self.shape, dtype=np.float32, buffer=s.buf) for s in self.shms]
        ctx = mp.get_context("fork")
        self.pool = cf.ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx)

    def submit(self, buf_index, records, box):
        """records: [(path, time_index)] for slots 0..n-1.  Returns list of futures."""
        s = self.shms[buf_index].name
        return [self.pool.submit(_worker, (s, self.shape, k, p, ti, box, self.var))
                for k, (p, ti) in enumerate(records)]

    @staticmethod
    def wait(futs):
        for f in futs:
            f.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for s in self.shms:
            s.close(); s.unlink()
