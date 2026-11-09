#!/usr/bin/env bash
#SBATCH --job-name=toocan_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --output=logs/chain_%x_%j.out
#SBATCH --error=logs/chain_%x_%j.err
#
# Segmentation stage: ONE exact continuous chain for one run name and period (bit-identical to original TOOCAN).
# C core uses THREADS OpenMP threads, READERS processes decode/convert input, 3 processes write output.
# Resumable: re-submit the same command and it continues from the last completed day.  Multi-year periods are one chain.
#
# Submit from the repo root (site options on the command line, e.g. -A <account> -p <partition>):
#   sbatch -A myproj --export=ALL,RUN=exp1,START=2000-01-01T00:00,END=2000-12-31T23:40 slurm/run_chain_single_node.sbatch
# Optional env: CFG GEO (config files), LON0 LON1 LAT0 LAT1, THREADS, READERS, OUTROOT (default $TOOCAN_HPC_HOME/outputs), PY
set -euo pipefail
source "${TOOCAN_HPC_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
RUN=${RUN:-${EXPERIMENT:?set RUN=<name of this run>}}
START=${START:?set START=YYYY-MM-DDTHH:MM}; END=${END:?set END=YYYY-MM-DDTHH:MM}
THREADS=${THREADS:-24}; READERS=${READERS:-12}
OUT=${OUTROOT:-$TOOCAN_HPC_HOME/outputs}/$RUN/chain_${START%%T*}_${END%%T*}
mkdir -p "$OUT"
# per-run copy of the parameter file recording the output path
sed "s#^pathout_TOOCAN.*#pathout_TOOCAN = $OUT#" "$CFG" > "$OUT/toocan.dat"
grep -q '^mask_file_layout' "$OUT/toocan.dat" || echo "mask_file_layout = daily" >> "$OUT/toocan.dat"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local}  $RUN  $START .. $END  threads=$THREADS readers=$READERS"; date
cd "$TOOCAN_HPC_HOME/src"
$PY -W ignore run_toocan_hpc.py "$OUT/toocan.dat" "$GEO" "$START" "$END" $LON0 $LON1 $LAT0 $LAT1 \
    --out "$OUT" --threads "$THREADS" --readers "$READERS" --writers 3
date; echo "chain finished: $OUT"
