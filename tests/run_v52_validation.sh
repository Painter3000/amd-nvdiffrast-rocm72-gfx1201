#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"

echo "===== v52 path probe ====="
python ./nvdiffrast_path_probe_v1.py --timeout 30

echo
echo "===== v52 AA forward matrix ====="
python ./nvdiffrast_aa_matrix_probe_v4.py \
  --timeout 30 \
  --shapes single,grid1,grid4,grid16 \
  --res-list 160,180,182,192,224,256 \
  --colors interp \
  --hashes explicit

echo
echo "===== v52 AA backward / grad matrix ====="
python ./test_antialias_backward_matrix_v52.py \
  --timeout 30 \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages call,sync,finite,diff \
  --verbose

echo
echo "v52 validation completed."

