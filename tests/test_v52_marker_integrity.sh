#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

SRC="$REPO/csrc/torch/torch_antialias.cpp"
HIP="$REPO/csrc/torch/torch_antialias_hip.cpp"

if [[ ! -f "$SRC" || ! -f "$HIP" ]]; then
  echo "ERROR: antialias source files not found."
  echo "Set REPO=/path/to/patched/nvdiffrast when running this test."
  echo "Current REPO=$REPO"
  exit 1
fi

echo "===== v47 marker check ====="
grep -c "v47 FIX" "$SRC" "$HIP"

echo "===== v51 marker check ====="
grep -c "v51 FIX" "$SRC" "$HIP"

v47_src=$(grep -c "v47 FIX" "$SRC" || true)
v47_hip=$(grep -c "v47 FIX" "$HIP" || true)
v51_src=$(grep -c "v51 FIX" "$SRC" || true)
v51_hip=$(grep -c "v51 FIX" "$HIP" || true)

[[ "$v47_src" == "1" && "$v47_hip" == "1" ]] || {
  echo "ERROR: v47 must be present exactly once in both files."
  exit 1
}

[[ "$v51_src" == "1" && "$v51_hip" == "1" ]] || {
  echo "ERROR: v51 must be present exactly once in both files."
  exit 1
}

echo "===== diagnostic marker scan ====="
if grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" "$SRC" "$HIP"; then
  echo "ERROR: diagnostic marker/sync found in active source files."
  exit 1
fi

echo "OK: v52 markers are clean."
