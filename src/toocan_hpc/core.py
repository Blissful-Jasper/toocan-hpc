"""ctypes binding to libtoocan_core.so (exact port of Label_region + spread + threshold loop)."""
import ctypes as C
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


class Params(C.Structure):
    _fields_ = [("T", C.c_int), ("Y", C.c_int), ("X", C.c_int),
                ("minBT", C.c_int), ("maxBT", C.c_int), ("stepBT", C.c_int),
                ("delta_spread", C.c_int), ("final_delta", C.c_int),
                ("timin", C.c_int), ("lifemin", C.c_int),
                ("nbMaxCluster", C.c_int), ("quant_mode", C.c_int), ("fast", C.c_int)]


class Stats(C.Structure):
    _fields_ = [("t_quant", C.c_double), ("t_spread1", C.c_double), ("t_spread2", C.c_double),
                ("t_spread3", C.c_double), ("t_coldlist", C.c_double), ("t_scan", C.c_double),
                ("t_merge", C.c_double), ("t_accum", C.c_double), ("t_filter", C.c_double),
                ("t_remap", C.c_double),
                ("n_cold_total", C.c_longlong), ("n_label_calls", C.c_longlong),
                ("n_spread_queue_total", C.c_longlong), ("n_spread_claims_total", C.c_longlong),
                ("n_raw_labels_total", C.c_longlong), ("n_new_mcs_total", C.c_longlong),
                ("t_spread_q", C.c_double), ("t_spread_bfs", C.c_double), ("t_bins", C.c_double),
                ("t_ulist", C.c_double), ("t_vset", C.c_double),
                ("n_U0_total", C.c_longlong), ("n_V_total", C.c_longlong), ("n_asg_total", C.c_longlong),
                ("n_spread_skipped", C.c_longlong),
                ("err", C.c_int)]

    def as_dict(self):
        return {n: getattr(self, n) for n, _ in self._fields_}


def _find_lib():
    for p in (os.environ.get("TOOCAN_HPC_LIB", ""),
              os.path.join(_here, "libtoocan_core.so"),
              os.path.join(_here, "..", "libtoocan_core.so")):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("libtoocan_core.so not found; run src/build.sh")


_lib = C.CDLL(_find_lib())
_lib.tc_state_new.restype = C.c_void_p
_lib.tc_state_free.argtypes = [C.c_void_p]
_lib.tc_begin_chunk.argtypes = [C.c_void_p, C.c_void_p, C.c_long, C.c_int]
_lib.tc_begin_chunk.restype = C.c_int
_lib.tc_run_chunk.argtypes = [C.c_void_p, C.POINTER(Params), C.c_void_p, C.c_void_p, C.c_void_p,
                              C.c_int, C.c_int, C.c_int, C.POINTER(Stats)]
_lib.tc_run_chunk.restype = C.c_int
_lib.tc_zero_i32.argtypes = [C.c_void_p, C.c_int64, C.c_int]


def zero_i32(a, nthreads):
    assert a.dtype == np.int32 and a.flags.c_contiguous
    _lib.tc_zero_i32(a.ctypes.data, a.size, int(nthreads))


assert _lib.tc_stats_size() == C.sizeof(Stats) and _lib.tc_params_size() == C.sizeof(Params)

ERR = {-2: "ptr_seed init", -3: "label table overflow (original: exit(0))", -4: "neighbour index",
       -5: "relabel", -6: "label index", -7: "p[] overflow (original: heap overflow, undefined)",
       -100: "alloc"}


class CoreState:
    """Persistent table of already-identified MCS (the original's `clusters` array)."""

    def __init__(self):
        self._h = C.c_void_p(_lib.tc_state_new())

    def __del__(self):
        try:
            _lib.tc_state_free(self._h)
        except Exception:
            pass

    def begin_chunk(self, labels_present, labelMin):
        lp = np.ascontiguousarray(labels_present, dtype=np.int32)
        rc = _lib.tc_begin_chunk(self._h, lp.ctypes.data, lp.size, int(labelMin))
        if rc:
            raise RuntimeError("tc_begin_chunk failed")

    def run_chunk(self, p: Params, bt, area2d, labels, nseeds, labelMin, nthreads):
        assert bt.dtype == np.float32 and bt.flags.c_contiguous
        assert labels.dtype == np.int32 and labels.flags.c_contiguous
        assert area2d.dtype == np.float32 and area2d.flags.c_contiguous
        assert bt.shape == labels.shape == (p.T, p.Y, p.X) and area2d.shape == (p.Y, p.X)
        st = Stats()
        r = _lib.tc_run_chunk(self._h, C.byref(p), bt.ctypes.data, area2d.ctypes.data, labels.ctypes.data,
                              int(nseeds), int(labelMin), int(nthreads), C.byref(st))
        if r < 0:
            raise RuntimeError(f"tc_run_chunk error {r}: {ERR.get(r, '?')}")
        return r, st.as_dict()
