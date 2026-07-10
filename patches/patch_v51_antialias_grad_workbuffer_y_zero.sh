#!/usr/bin/env bash
set -euo pipefail

# patch_v51_antialias_grad_workbuffer_y_zero.sh
#
# v51 FIX:
# Rebuild-stable Backward/Grad counterpart to v47.
#
# v47 fixed antialias_fwd() by replacing a raw cuda/hipMemsetAsync on
# p.workBuffer[0].x/header with torch-native work_buffer.narrow(...).zero_()
# in BOTH torch_antialias.cpp and torch_antialias_hip.cpp.
#
# v51 does the same for antialias_grad():
#   raw cuda/hipMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)
#   ->
#   work_buffer.narrow(0, 1, 1).zero_()
#
# It patches BOTH files so the fix survives clean rebuilds where hipify may
# regenerate torch_antialias_hip.cpp from torch_antialias.cpp.
#
# It also removes the stale German sync comment left behind by older diagnostics.
#
# Usage:
#   REPO=/home/oem/therock_test/nvdiffrast ./patch_v51_antialias_grad_workbuffer_y_zero.sh

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
SRC_FILE="$REPO/csrc/torch/torch_antialias.cpp"
HIP_FILE="$REPO/csrc/torch/torch_antialias_hip.cpp"

for f in "$SRC_FILE" "$HIP_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

echo "Checking current source state..."
if grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" "$SRC_FILE" "$HIP_FILE" 2>/dev/null; then
  echo
  echo "ERROR: stale diagnostic markers/syncs found in active antialias files." >&2
  echo "Restore to clean v47-only state before applying v51." >&2
  exit 1
fi

if ! grep -q "v47 FIX" "$SRC_FILE" || ! grep -q "v47 FIX" "$HIP_FILE"; then
  echo "ERROR: v47 FIX marker must be present in BOTH torch_antialias.cpp and torch_antialias_hip.cpp." >&2
  exit 1
fi

python3 - "$SRC_FILE" "$HIP_FILE" <<'PY'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1])
hip = Path(sys.argv[2])

def patch_one(path: Path, api: str) -> None:
    text = path.read_text()
    old = text

    if "v51 FIX: replace raw gradient workBuffer y-counter memset" in text:
        print(f"[skip] {path}: v51 already applied")
        return

    # Remove stale diagnostic comment if present.
    text = text.replace(
        "\n    // damit normale Laeufe nicht durch den zusaetzlichen Sync verlangsamt werden.\n",
        "\n"
    )

    memset_name = "cudaMemsetAsync" if api == "cuda" else "hipMemsetAsync"

    pattern = re.compile(
        rf'^[ \t]*NVDR_CHECK_CUDA_ERROR\s*\(\s*{memset_name}\s*\(\s*&p\.workBuffer\[0\]\.y\s*,\s*0\s*,\s*sizeof\s*\(\s*int\s*\)\s*,\s*stream\s*\)\s*\)\s*;\s*$',
        re.M
    )

    replacement = f"""    #if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // v51 FIX: replace raw gradient workBuffer y-counter memset
    // with torch-native zero_() so PyTorch owns stream/allocator ordering.
    work_buffer.narrow(0, 1, 1).zero_();
    #else
    NVDR_CHECK_CUDA_ERROR({memset_name}(&p.workBuffer[0].y, 0, sizeof(int), stream));
    #endif"""

    text, n = pattern.subn(replacement, text, count=1)

    if n != 1:
        print(f"ERROR: {path}: did not find exactly one raw {memset_name}(&p.workBuffer[0].y, ...) line.")
        print("Relevant candidates:")
        for ln, line in enumerate(old.splitlines(), 1):
            if "workBuffer[0].y" in line or "work_buffer.narrow" in line or "v47" in line or "v51" in line or "MemsetAsync" in line:
                print(f"{ln}: {line}")
        raise SystemExit(1)

    backup = path.with_suffix(path.suffix + ".before_v51_grad_workbuffer_y_zero")
    if not backup.exists():
        backup.write_text(old)

    path.write_text(text)
    print(f"[ok] patched {path}")

patch_one(src, "cuda")
patch_one(hip, "hip")
PY

echo
echo "v51 verification:"
grep -n "v47 FIX\|v51 FIX\|work_buffer.narrow\|workBuffer\[0\]\.y\|MemsetAsync\|zusaetzlichen Sync" "$SRC_FILE" "$HIP_FILE" || true

echo
echo "Strict active diagnostic marker scan:"
if grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" "$SRC_FILE" "$HIP_FILE" 2>/dev/null; then
  echo "WARNING: diagnostic marker/sync still present. Inspect before rebuild." >&2
else
  echo "OK: no stale diagnostic markers/syncs in active files."
fi

echo
echo "Next:"
echo "  1) Clean rebuild"
echo "  2) Re-check v47/v51 markers in both files after rebuild"
echo "  3) Run antialias backward/grad tests"
