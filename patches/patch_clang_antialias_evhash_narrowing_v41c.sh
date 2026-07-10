#!/usr/bin/env bash
set -euo pipefail

# v41c BUILD FIX:
# ROCm clang++ rejects narrowing conversion in torch::zeros({uint64_t_expr}, opts)
# in torch_antialias_hip.cpp:
#
#   error: non-constant-expression cannot be narrowed from type 'uint64_t'
#
# GCC accepted this, clang++ correctly rejects it in C++20 initializer-list context.
#
# Fix:
#   compute the hash tensor length as int64_t explicitly and pass that variable
#   to torch::zeros({evHashElements}, opts).
#
# Applies to both CUDA and HIP mirror files.

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"

python - "$REPO" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])

files = [
    repo / "csrc/torch/torch_antialias.cpp",
    repo / "csrc/torch/torch_antialias_hip.cpp",
]

old = "    torch::Tensor ev_hash = torch::zeros({(uint64_t)p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * 4}, opts);"

new = """    int64_t evHashElements = static_cast<int64_t>(
        static_cast<uint64_t>(p.allocTriangles) *
        static_cast<uint64_t>(AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles)) *
        4ull);
    torch::Tensor ev_hash = torch::zeros({evHashElements}, opts);"""

patched = False

for path in files:
    if not path.exists():
        print(f"skip missing {path}")
        continue

    s = path.read_text()

    if "evHashElements" in s:
        print(f"{path}: already patched")
        continue

    if old not in s:
        print(f"{path}: target line not found")
        print("Nearby candidates:")
        for n, line in enumerate(s.splitlines(), 1):
            if "ev_hash" in line or "AA_HASH_ELEMENTS_PER_TRIANGLE" in line:
                print(f"{n}: {line}")
        continue

    backup = path.with_suffix(path.suffix + ".before_clang_evhash_narrowing_v41c")
    if not backup.exists():
        backup.write_text(s)

    s = s.replace(old, new)
    path.write_text(s)
    patched = True
    print(f"patched {path}")

if not patched:
    print("No new changes made.")
PY

echo
echo "Verify:"
grep -R -n "evHashElements\|ev_hash = torch::zeros\|AA_HASH_ELEMENTS_PER_TRIANGLE" \
  "$REPO"/csrc/torch/torch_antialias.cpp \
  "$REPO"/csrc/torch/torch_antialias_hip.cpp | head -120
