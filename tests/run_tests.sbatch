#!/usr/bin/env bash
#SBATCH --job-name=toocan_unit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=logs/unit_%j.out
# Regression of the C core against the ORIGINAL toocan-python (needs it: export TOOCAN_ORIG_ROOT=/path/to/toocan-python).
#   sbatch -A myproj --export=ALL,PY=/path/to/python,TOOCAN_ORIG_ROOT=... tests/run_tests.sbatch   (from the repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
${PY:-python} -W ignore tests/test_vs_original.py --seeds ${SEEDS:-3}
