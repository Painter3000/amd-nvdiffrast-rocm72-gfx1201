#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" \
TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}" \
python ./test_advanced_pipeline.py
