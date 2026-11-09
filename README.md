# TOOCAN-HPC

A fast, **bit-for-bit identical** re-implementation of the TOOCAN deep-convective-system (DCS) segmentation and
catalog stages for HPC systems. [TOOCAN](https://gitlab.in2p3.fr/thomas.fiolleau1/toocan-python)
(Fiolleau & Roca 2013) tracks DCS by 3-D (lon × lat × time) hierarchical segmentation of brightness temperature (BT).
Here the heavy part is one OpenMP C library driven by a thin Python pipeline; the labels are the same as the original's,
voxel by voxel.

| | original `toocan-python` | TOOCAN-HPC |
|---|---|---|
| segmentation core | Python + `label.c` (ctypes) + Numba, 1 core | one C library, OpenMP |
| one 72-slot chunk, 601×3600 grid | ≈130 s | 2.1 s (16 threads) |
| one year (400 chunks) | ≈20 h | **19 min** on one node |
| catalog (FileTracking), 1 global day | 1998 s | 31 s (1 proc) / 11.7 s (32 procs) |
| restart after a kill | rerun the whole segment | resumes from the last finished day |

Validation (full-domain production data, synthetic fields, catalog variables) and timings: [docs/](docs/)
(Chinese; tables are numeric). Language: [中文版 README](README_zh.md).

## 1. Install

Requirements: Linux, `gcc` with OpenMP, Python ≥ 3.9 with `numpy netCDF4 pandas xarray` (`mpi4py` only for the MPI catalog).
No Numba, Dask or CuPy.

```bash
git clone https://github.com/Blissful-Jasper/toocan-hpc.git && cd toocan-hpc
conda env create -f environment.yml && conda activate toocan-hpc     # or: pip install -r requirements.txt
bash src/build.sh                                                     # -> src/libtoocan_core.so
python tests/test_smoke.py                                            # ~30 s, no data needed
```

`build.sh` compiles with `-O3 -fopenmp -ffp-contract=off` and **without** `-ffast-math`; changing these flags
breaks bit-identity. Set `CC=...` to use another compiler. Re-running `build.sh` is safe while jobs are running
(atomic rename).

## 2. Try it on synthetic data (1 min)

```bash
PY=python bash examples/demo.sh
```
It builds the library, writes 3 days of fake OLR + a navigation file to `demo_run/data`, runs segmentation and
catalog, and prints a summary. Read the script: it *is* the minimal tutorial for your own data (next section).

## 3. Run on your own data

### 3.1 Prepare the input

TOOCAN-HPC needs two things.

**(a) OLR (or BT) files**, one NetCDF per day, under any directory tree (`path_ir` is searched recursively for `*.nc`):

| requirement | detail |
|---|---|
| file name | contains `YYYYMMDD` (used to pick files in the requested period) |
| variable | `rlut` (or `LWNTA`), **OLR in W m⁻²**, dims `(time, lat, lon)`; converted to BT with the original TOOCAN formula |
| time | CF `time` axis; equally spaced at `temporalresolution` minutes; no gaps inside the period |
| grid | regular lat/lon, same for all files |

**(b) a navigation file** with latitude, longitude and the **area of each pixel in km²** (TOOCAN's `minAreaSeed` is in km²):

```bash
python tools/make_navigation.py --from-file my_olr_day.nc --out navigation.nc            # grid taken from your data
# or  python tools/make_navigation.py --lat -30 30 --lon 0 359.9 --res 0.1 --out navigation.nc
```

### 3.2 Configure

```bash
cp config/geo_template.dat config/geo.dat      # edit: REGION, temporalresolution, path_ir, file_navigation
$EDITOR config/toocan.dat                      # science parameters; defaults = original TOOCAN v2.08 C5 setup
```
Keep `VolumeImage = 72`, `overlap_window_size = 6` and the BT thresholds unless you know why (they affect labels).
Comments in `.dat` files must be on their own line (start with `;`).

### 3.3 Segment, then build the catalog

```bash
export TZ=UTC HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1

# 1) segmentation: one continuous chain over the whole period (start/end inclusive, then lonmin lonmax latmin latmax)
python src/run_toocan_hpc.py config/toocan.dat config/geo.dat 2000-01-01T00:00 2000-12-31T23:40  0 359.9 -30 30 \
       --out outputs/exp1 --threads 24 --readers 12 --writers 3

# 2) catalog: one row per (DCS, time slot); parallel over days
python src/run_catalog_hpc.py config/toocan.dat config/geo.dat 2000-01-01T00:00 2000-01-31T23:40  0 359.9 -30 30 \
       --masks outputs/exp1 --out outputs/exp1/catalog --nproc 32
```

| option | meaning |
|---|---|
| `--threads N` | OpenMP threads of the C core (benefit flattens above ~32) |
| `--readers N` / `--writers N` | input decode / output compression processes (hidden behind compute) |
| `--fast-level {0,1,2}` | 0 literal port, 1 restricted scan, 2 (default) + parallel BFS; all give identical labels |
| `--no-resume` | ignore existing output and start over |
| `--ellipse` (catalog) | also compute equivalent ellipses (not in the original output) |

**Restart:** after a crash/timeout run the *same command* again. Finished day files are skipped, at most 2 chunks are
recomputed. Do not change parameters or period between the runs.

### 3.4 SLURM (e.g. one year)

```bash
mkdir -p logs                                  # run all of the following from the repo root
export PY=$(which python) GEO=$PWD/config/geo.dat     # optional: CFG, LON0 LON1 LAT0 LAT1, OUTROOT, THREADS
slurm/submit_pipeline.sh exp1 2000-01-01T00:00 2000-12-31T23:40 -A <account> -p <partition>
```
This submits the chain job, then (after success) a 12-task catalog array. Individual scripts and their variables are
documented at the top of each file in [slurm/](slurm/). Site options (`-A`, `-p`, `--time`, …) go on the command line;
do not request more memory than needed — on partitions with per-CPU memory limits `--mem` silently multiplies the CPUs.

### 3.5 Output

* Masks: `<out>/toocan_<version>/<REGION>/<YYYY>/<YYYY_MM_DD>/ToocanCloudMask_<YYYYMMDD>.nc` — `DCS_number(time,y,x)` int32;
  background −998, DCS labels ≥ 1, unique over the whole run. `time` in seconds since 1970.
* Catalog `catalog_compact-<REGION>-<start>-<end>.nc`: one row per (DCS, slot): `obs_label, obs_epoch, obs_area,
  obs_lon_cm, obs_lat_cm, obs_tbmin, obs_tb90, obs_avg235/208/200, obs_pix_*, obs_km2_{235,220,210,200}`.
* `TOOCAN-<REGION>-<start>-<end>.nc`: the original TOOCAN catalog format (`INT_*`, `LC_*`); skipped if the dense
  DCS×time matrix would exceed `dense_limit` (2×10⁸ elements).
* `<out>/.state/`: restart state and per-chunk timing (`chunk_stats.jsonl`).

```python
import xarray as xr, numpy as np
c = xr.open_dataset("catalog_compact-DEMO-20000101-20000103.nc")
lab = c.obs_label.values
u, first = np.unique(lab, return_index=True)
print(len(u), "DCS; max 235 K area per DCS:", np.maximum.reduceat(c.obs_km2_235.values, first)[:5])
```
`python examples/inspect_output.py <out>` prints a summary.

## 4. Verify against the original

```bash
python tests/test_smoke.py                                         # self-contained: fast levels/threads/resume agree
TOOCAN_ORIG_ROOT=/path/to/toocan-python python tests/test_vs_original.py --seeds 3   # core vs original, synthetic
python validation/compare_masks.py   REF_ROOT NEW_ROOT             # your own runs: bit-for-bit labels
python validation/compare_objects.py REF_ROOT NEW_ROOT            # object-level statistics
python validation/compare_catalog.py REF.nc NEW.nc                # every INT_/LC_ variable
```

## 5. Differences from the original (corrections only)

1. **Time zone**: the original derives `INT_UTC_*`/`LC_UTC` with `datetime.timestamp()`, which depends on the host TZ.
   TOOCAN-HPC always uses UTC (set `TZ=UTC`); this equals the original on a UTC host.
2. The original writes out of bounds when more than `nbMaxCluster` clusters occur; TOOCAN-HPC stops with an error.
3. Each daily file is opened once instead of once per time slot.

## Troubleshooting

* `KeyError: 'minAreaSeed'` — inline comment in a `.dat` file; put comments on their own line.
* `Missing IR slot inside non-terminal chunk` — a gap in the input time series (only the last chunk may be short).
* `tc_run_chunk error -7` — more clusters than `nbMaxCluster`; raise it and check the overlap region.
* Labels differ from the original — check compiler flags, `--quant-mode` (keep 1), `VolumeImage`/`overlap_window_size`.
* `libtoocan_core.so not found` — run `bash src/build.sh`.

## Layout

```
src/        toocan_core.c (C core), build.sh, run_toocan_hpc.py, run_catalog_hpc.py, run_catalog_mpi.py, toocan_hpc/ (Python)
config/     toocan.dat (parameters), geo_template.dat (input description)
tools/      make_navigation.py, make_demo_data.py
examples/   demo.sh, inspect_output.py
slurm/      chain, catalog array, MPI catalog, pipeline submit
tests/ validation/   smoke test, comparison with the original
docs/       algorithm analysis, parallel design, performance report (Chinese)
```

## License and citation

GPL-3.0 (as the original TOOCAN; see [LICENSE](LICENSE) and [NOTICE](NOTICE)). If you use the results, cite the TOOCAN papers:
Fiolleau & Roca (2013) IEEE TGRS 51(7), [10.1109/TGRS.2012.2227762](https://doi.org/10.1109/TGRS.2012.2227762);
Fiolleau & Roca (2024) ESSD 16, [10.5194/essd-16-4021-2024](https://doi.org/10.5194/essd-16-4021-2024).
