# Common settings sourced by the TOOCAN-HPC SLURM scripts.
# SLURM runs a copy of the batch script from a spool dir, so the repo location cannot be derived from $0:
# submit from the repo root (TOOCAN_HPC_HOME defaults to the submit directory) or export TOOCAN_HPC_HOME.
export TOOCAN_HPC_HOME=${TOOCAN_HPC_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "$TOOCAN_HPC_HOME/src/toocan_core.c" ] || { echo "TOOCAN_HPC_HOME=$TOOCAN_HPC_HOME is not the repo root; submit from there or export TOOCAN_HPC_HOME" >&2; exit 2; }
# Python with numpy, netCDF4, pandas, xarray (and mpi4py for the MPI catalog).  Export PY=/path/to/env/bin/python to override.
export PY=${PY:-python}
# The labelling is a serial chain (docs/TOOCAN_parallel_design.md): threads are used INSIDE the C core and are set by
# --threads, not by OMP_NUM_THREADS.  Everything else stays single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export TZ=UTC                       # INT_UTC/LC_UTC must not depend on the host timezone
export PYTHONUNBUFFERED=1
# Parameter + data-description files (see config/).  CFG: science parameters.  GEO: where the input data are.
export CFG=${CFG:-$TOOCAN_HPC_HOME/config/toocan.dat}
export GEO=${GEO:-$TOOCAN_HPC_HOME/config/geo.dat}
[ -f "$GEO" ] || { echo "GEO file $GEO not found: cp config/geo_template.dat config/geo.dat and edit it (or export GEO=...)" >&2; exit 2; }
# Domain (lon0 lon1 lat0 lat1) -- must lie inside the navigation file
export LON0=${LON0:-0} LON1=${LON1:-359.9} LAT0=${LAT0:--30} LAT1=${LAT1:-30}
mkdir -p "$TOOCAN_HPC_HOME/logs"
