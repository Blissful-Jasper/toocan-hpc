#!/usr/bin/env python3
"""Which arithmetic does the ORIGINAL numba kernel use for  (volume_bt*100).astype(np.int16) ?
Compares numba (same expression as spreading.py) with the two candidate C semantics."""
import numpy as np
from numba import njit

@njit
def q_numba(volume_bt):          # verbatim expression from spread_labels_fast
    return (volume_bt * 100).astype(np.int16)

rng = np.random.default_rng(0)
bt = (rng.uniform(180.0, 240.0, 5_000_000)).astype(np.float32).reshape(50, 1000, 100)
bt.ravel()[:100000] = np.round(bt.ravel()[:100000], 2)       # many values sitting on the 0.01 K grid
qn = q_numba(bt)
q_double = (bt.astype(np.float64) * 100.0).astype(np.int32).astype(np.int16)      # mode 0
q_float = (bt * np.float32(100.0)).astype(np.int32).astype(np.int16)             # mode 1 (float32 multiply)
print("numba == mode0 (double multiply):", np.array_equal(qn, q_double), " mismatches:", int((qn != q_double).sum()))
print("numba == mode1 (float32 multiply):", np.array_equal(qn, q_float), " mismatches:", int((qn != q_float).sum()))
