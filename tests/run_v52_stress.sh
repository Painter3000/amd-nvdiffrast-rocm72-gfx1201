#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "===== v52 AA forward statistical stress ====="
python ./aa_matrix_stat_probe.py --runs "${RUNS:-20}" \
  --shapes single,grid1,grid4,grid16 \
  --res-list 160,180,182,192,224,256 \
  --colors interp \
  --hashes explicit \
  --label "final v52 forward AA"

echo
echo "===== v52 many-triangle stress ====="
AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" TORCH_DISABLE_ADDR2LINE=1 \
python ./test_many_triangles_stress_v1.py

