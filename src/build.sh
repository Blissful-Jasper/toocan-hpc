#!/usr/bin/env bash
# Build the exact C core.  No -ffast-math, FMA contraction OFF: results must match the original bit for bit.
# The library is installed with an atomic rename so that running jobs keep their (old) inode.
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-gcc}
$CC -O3 -fopenmp -fPIC -shared -ffp-contract=off -fno-fast-math -Wall -Wextra \
    -o libtoocan_core.so.new toocan_core.c
mv -f libtoocan_core.so.new libtoocan_core.so
cp -f libtoocan_core.so toocan_hpc/libtoocan_core.so.new && mv -f toocan_hpc/libtoocan_core.so.new toocan_hpc/libtoocan_core.so
echo "built $(pwd)/libtoocan_core.so"
