#!/usr/bin/env bash
#SBATCH --job-name=toocan_cat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/cat_%x_%A_%a.out
#SBATCH --error=logs/cat_%x_%A_%a.err
#
# Catalog (FileTracking) stage.  One array task = one calendar month of YEAR; each task uses NPROC worker processes
# over the days of its month.  Tasks are independent -> spread over any number of nodes.
#   sbatch -A myproj --array=0-11 --export=ALL,RUN=exp1,YEAR=2000,MASKS=<chain out dir> slurm/run_catalog_array.sbatch
# (array index 0..11 = Jan..Dec).  Output: <CATOUT>/<YYYYMM>/ ; CATOUT defaults to <MASKS>/../catalog_<RUN>.
set -euo pipefail
source "${TOOCAN_HPC_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
RUN=${RUN:-${EXPERIMENT:?set RUN=<name>}}; YEAR=${YEAR:?}; MASKS=${MASKS:?mask root (= the chain out dir)}
M=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
LAST=$($PY -c "import calendar;print(calendar.monthrange($YEAR,$M)[1])")
START=$(printf '%04d-%02d-01T00:00' $YEAR $M); END=$(printf '%04d-%02d-%02dT23:40' $YEAR $M $LAST)
OUT=${CATOUT:-$(dirname "$MASKS")/catalog_$RUN}/$(printf '%04d%02d' $YEAR $M)
echo "month $YEAR-$M  $START..$END -> $OUT"
cd "$TOOCAN_HPC_HOME/src"
$PY -W ignore run_catalog_hpc.py "$CFG" "$GEO" "$START" "$END" $LON0 $LON1 $LAT0 $LAT1 --masks "$MASKS" --out "$OUT" \
    --nproc ${NPROC:-28} ${CATALOG_OPTS:-}
