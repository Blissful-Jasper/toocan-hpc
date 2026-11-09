#!/usr/bin/env bash
#SBATCH --job-name=toocan_cat_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/catmpi_%x_%j.out
#SBATCH --error=logs/catmpi_%x_%j.err
#
# Multi-node catalog stage with MPI (mpi4py): one rank per core, static round-robin over 6-slot blocks, one gather at
# the end.  Equivalent to run_catalog_array.sbatch but in ONE job (useful for a few months at once).
#   sbatch -A myproj -p <partition> --export=ALL,RUN=exp1,START=2000-01-01T00:00,END=2000-03-31T23:40,MASKS=<chain out dir> \
#          slurm/run_catalog_mpi.sbatch
# The dense DCS x time matrix of the legacy file lives on rank 0 -> keep the period to a few months.
# MPI launcher: use `srun` with the PMI flavour your site's MPI needs (e.g. --mpi=pmix); a plain `mpirun` from a different
# MPI than mpi4py's can silently start N independent single-rank copies.
set -euo pipefail
source "${TOOCAN_HPC_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
RUN=${RUN:-${EXPERIMENT:?set RUN=<name>}}; START=${START:?}; END=${END:?}; MASKS=${MASKS:?}
OUT=${CATOUT:-$(dirname "$MASKS")/catalog_mpi_$RUN}
cd "$TOOCAN_HPC_HOME/src"
srun ${SRUN_OPTS:---mpi=pmix} $PY -W ignore run_catalog_mpi.py "$CFG" "$GEO" "$START" "$END" $LON0 $LON1 $LAT0 $LAT1 \
     --masks "$MASKS" --out "$OUT"
