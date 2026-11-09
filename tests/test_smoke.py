#!/usr/bin/env python3
"""Self-contained check (no real data, no original TOOCAN needed), ~30 s:  python tests/test_smoke.py

Builds the C core, writes synthetic data, then verifies that
  * --fast-level 0 (literal port), 1 and 2, with 1 and 4 threads, produce IDENTICAL label volumes;
  * a run interrupted by deleting one day file and re-submitted (resume) reproduces the same masks;
  * the catalog stage runs and reports the same number of DCS as there are distinct labels in the masks.
For bit-identity against the ORIGINAL toocan-python use tests/test_vs_original.py.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
ENV = dict(os.environ, TZ="UTC", HDF5_USE_FILE_LOCKING="FALSE", OMP_NUM_THREADS="1")
BOX = ["60", "120", "-15", "15"]; T0, T1 = "2000-01-01T00:00", "2000-01-03T23:40"


def run(*a):
    subprocess.run([str(x) for x in a], check=True, env=ENV, stdout=subprocess.DEVNULL)


def labels(out):
    fs = sorted(glob.glob(f"{out}/toocan_*/*/*/*/ToocanCloudMask_*.nc"))
    assert len(fs) == 3, f"expected 3 daily masks in {out}, found {len(fs)}"
    arr = []
    for f in fs:
        with netCDF4.Dataset(f) as d:
            arr.append(np.asarray(d["DCS_number"][:]))
    return np.concatenate(arr)


def main():
    tmp = tempfile.mkdtemp(prefix="toocan_smoke_")
    try:
        run("bash", f"{ROOT}/src/build.sh")
        run(PY, f"{ROOT}/tools/make_demo_data.py", "--out", f"{tmp}/data", "--days", 3)
        with open(f"{ROOT}/config/geo_template.dat") as f:
            geo = f.read().replace("/path/to/olr_archive", f"{tmp}/data/olr").replace("/path/to/navigation.nc", f"{tmp}/data/navigation.nc")
        open(f"{tmp}/geo.dat", "w").write(geo)
        seg = lambda out, *x: run(PY, f"{ROOT}/src/run_toocan_hpc.py", f"{ROOT}/config/toocan.dat", f"{tmp}/geo.dat", T0, T1,
                                  *BOX, "--out", out, "--readers", 2, "--writers", 1, *x)
        ref = None
        for fast, thr in ((0, 1), (1, 1), (2, 1), (2, 4)):
            out = f"{tmp}/out_f{fast}_t{thr}"; seg(out, "--fast-level", fast, "--threads", thr)
            lab = labels(out); ref = lab if ref is None else ref
            assert np.array_equal(lab, ref), f"fast-level {fast}, {thr} threads differs from fast-level 0"
            print(f"ok  fast-level {fast} threads {thr}: {np.unique(lab[lab > 0]).size} DCS, identical")
        # resume: delete day 2 and re-run the same command
        out = f"{tmp}/out_f2_t4"
        shutil.rmtree(glob.glob(f"{out}/toocan_*/*/*/2000_01_02")[0])
        seg(out, "--fast-level", 2, "--threads", 4)
        assert np.array_equal(labels(out), ref); print("ok  resume after deleting a day reproduces the masks")
        run(PY, f"{ROOT}/src/run_catalog_hpc.py", f"{ROOT}/config/toocan.dat", f"{tmp}/geo.dat", T0, T1, *BOX,
            "--masks", out, "--out", f"{out}/catalog", "--nproc", 2)
        with netCDF4.Dataset(glob.glob(f"{out}/catalog/catalog_compact-*.nc")[0]) as d:
            n_cat = np.unique(np.asarray(d["obs_label"][:])).size
        assert n_cat == np.unique(ref[ref > 0]).size, (n_cat, np.unique(ref[ref > 0]).size)
        print(f"ok  catalog: {n_cat} DCS == distinct mask labels")
        print("ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
