#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"

python ./aa_backward_pos_stat_probe_v52.py --runs "${RUNS:-20}" \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages backward_pos \
  --label "final v52 AA backward_pos statistical probe"

