#!/usr/bin/env bash
# Submit the two-stage pipeline for one run and period within ONE calendar year:
#   segmentation chain  ->  catalog array (one task per month, starts after the chain succeeded)
#   slurm/submit_pipeline.sh exp1 2000-01-01T00:00 2000-12-31T23:40  [extra sbatch args, e.g. -A myproj -p compute]
# Run from the repo root.  CFG / GEO / LON0.. / OUTROOT are passed on via the environment (see slurm/common.sh).
set -euo pipefail
RUN=${1:?run name}; START=${2:?}; END=${3:?}; shift 3
H=${TOOCAN_HPC_HOME:-$PWD}
YEAR=${START:0:4}; [ "${END:0:4}" = "$YEAR" ] || { echo "START and END must be in the same year (submit one pipeline per year)"; exit 2; }
MONTH0=$((10#${START:5:2} - 1)); MONTH1=$((10#${END:5:2} - 1))
OUT=${OUTROOT:-$H/outputs}/$RUN/chain_${START%%T*}_${END%%T*}
J1=$(sbatch --parsable "$@" --job-name=chain_$RUN --export=ALL,RUN=$RUN,START=$START,END=$END "$H/slurm/run_chain_single_node.sbatch")
echo "chain job $J1"
J2=$(sbatch --parsable "$@" --dependency=afterok:$J1 --array=$MONTH0-$MONTH1 --job-name=cat_$RUN \
      --export=ALL,RUN=$RUN,YEAR=$YEAR,MASKS=$OUT "$H/slurm/run_catalog_array.sbatch")
echo "catalog array $J2 (months $MONTH0-$MONTH1, 0-based) after $J1"
