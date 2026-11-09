#!/usr/bin/env bash
# End-to-end demo on synthetic data (~1 min, no real data needed).  Run from the repo root:  bash examples/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export TZ=UTC HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1
D=${DEMO_DIR:-demo_run}; mkdir -p "$D"

bash src/build.sh                                              # 1) compile the C core
$PY tools/make_demo_data.py --out "$D/data" --days 3           # 2) synthetic OLR + navigation file

# 3) config: copy the template, point it to the demo data
sed -e "s#^REGION.*#REGION = DEMO#" -e "s#^spatialresolution.*#spatialresolution = 0.25#" \
    -e "s#^path_ir.*#path_ir = $D/data/olr#" -e "s#^file_navigation.*#file_navigation = $D/data/navigation.nc#" \
    config/geo_template.dat > "$D/geo.dat"

# 4) segmentation (3-D labelling of all time slots) and 5) catalog (one row per DCS and time slot)
$PY src/run_toocan_hpc.py config/toocan.dat "$D/geo.dat" 2000-01-01T00:00 2000-01-03T23:40  60 120 -15 15 \
    --out "$D/out" --threads 4 --readers 2 --writers 1
$PY src/run_catalog_hpc.py config/toocan.dat "$D/geo.dat" 2000-01-01T00:00 2000-01-03T23:40  60 120 -15 15 \
    --masks "$D/out" --out "$D/out/catalog" --nproc 4

$PY examples/inspect_output.py "$D/out"
