#!/usr/bin/env bash
set -euo pipefail

# v38: ROCm/RDNA gfx1201 diagnostic/stability patch for nvdiffrast antialias backward.
#
# Why:
#   antialias forward passes, but antialias backward crashes even for constant
#   color-only gradients. In antialias.cu, only AntialiasGradKernel uses
#   __ballot_sync(0xffffffffffffffffull, ...), while the forward analysis kernel
#   does not. On RDNA/gfx1201 this is suspicious because the CUDA-style 64-lane
#   mask model does not match the native Wave32 execution.
#
# What this patch does for HIP/ROCm only:
#   - replaces AntialiasGradKernel ballot narrowing with a fixed Wave32 row mask
#   - bypasses CA_SET_GROUP_MASK / caAtomicAdd3_xyw and uses direct atomicAdd
#     for gradPos on HIP
#
# This is intentionally conservative and may be slower. It is a diagnostic
# stability patch first, not a performance optimization.

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
AA="$REPO/csrc/common/antialias.cu"

if [[ ! -f "$AA" ]]; then
  echo "ERROR: antialias.cu not found: $AA" >&2
  exit 1
fi

cp -n "$AA" "$AA.before_rocm_antialias_grad_v38" || true

python - "$AA" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text()
orig = s

repls = [
(
"""        unsigned long long amask = __ballot_sync(0xffffffffffffffffull, item.w);""",
"""#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // RDNA/gfx1201: native Wave32. Avoid CUDA-style 64-lane ballot here.
        unsigned long long amask = 0x00000000ffffffffull;
#else
        unsigned long long amask = __ballot_sync(0xffffffffffffffffull, item.w);
#endif"""
),
(
"""        amask = __ballot_sync(amask, !triFail);""",
"""#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
        amask = __ballot_sync(amask, !triFail);
#endif"""
),
(
"""        amask = __ballot_sync(amask, !noGrad);""",
"""#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
        amask = __ballot_sync(amask, !noGrad);
#endif"""
),
(
"""        amask = __ballot_sync(amask, !vtxFail);""",
"""#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
        amask = __ballot_sync(amask, !vtxFail);
#endif"""
),
(
"""        CA_SET_GROUP_MASK(tri ^ (di << 30), amask);

        // Accumulate gradients.
        caAtomicAdd3_xyw(p.gradPos + 4 * vi1, gp1x, gp1y, gp1w);
        caAtomicAdd3_xyw(p.gradPos + 4 * vi2, gp2x, gp2y, gp2w);""",
"""#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // ROCm/RDNA diagnostic path: avoid CUDA warp-match/coalesced atomics.
        // Direct atomics are slower but avoid CA_* assumptions about warp masks.
        atomicAdd(p.gradPos + 4 * vi1 + 0, gp1x);
        atomicAdd(p.gradPos + 4 * vi1 + 1, gp1y);
        atomicAdd(p.gradPos + 4 * vi1 + 3, gp1w);
        atomicAdd(p.gradPos + 4 * vi2 + 0, gp2x);
        atomicAdd(p.gradPos + 4 * vi2 + 1, gp2y);
        atomicAdd(p.gradPos + 4 * vi2 + 3, gp2w);
#else
        CA_SET_GROUP_MASK(tri ^ (di << 30), amask);

        // Accumulate gradients.
        caAtomicAdd3_xyw(p.gradPos + 4 * vi1, gp1x, gp1y, gp1w);
        caAtomicAdd3_xyw(p.gradPos + 4 * vi2, gp2x, gp2y, gp2w);
#endif"""
),
]

missing = []
for old, new in repls:
    if old not in s:
        missing.append(old.splitlines()[0])
    else:
        s = s.replace(old, new)

if missing:
    print("ERROR: expected source fragments not found:")
    for m in missing:
        print("  -", m)
    print("\nMaybe antialias.cu was already patched or source formatting differs.")
    raise SystemExit(1)

if s == orig:
    raise SystemExit("ERROR: no changes made")

path.write_text(s)
print(f"patched {path}")

# Print quick verification.
for needle in [
    "0xffffffffffffffffull",
    "__ballot_sync(amask",
    "CA_SET_GROUP_MASK(tri ^ (di << 30), amask)",
    "atomicAdd(p.gradPos + 4 * vi1 + 0, gp1x)",
]:
    print(f"{needle}: {path.read_text().count(needle)}")
PY

echo
echo "Rebuild:"
cat <<'EOF'
source ~/therock_test/venv/bin/activate
cd ~/therock_test/nvdiffrast

rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast

export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v
EOF
