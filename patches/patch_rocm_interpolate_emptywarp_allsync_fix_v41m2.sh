#!/usr/bin/env bash
set -euo pipefail

# v41m2 FINAL-CANDIDATE:
# ROCm/RDNA-safe fix for InterpolateFwdKernelTemplate empty-warp shortcut,
# preserving the exact original CUDA mask:
#
#   if (__all_sync(0xffffffffffffffffull, !triValid))
#
# Background:
#   v41l proved that disabling this one collective fixes interpolate_fwd crashes
#   at resolutions 164/172/180.
#
# Root cause:
#   The kernel returns edge lanes before this full-mask collective:
#
#       if (px >= p.width || py >= p.height || pz >= p.depth)
#           return;
#       ...
#       if (__all_sync(0xffffffffffffffffull, !triValid))
#
#   On HIP/RDNA, using a full-mask collective after some lanes may have exited is
#   unsafe. Disable only this shortcut for HIP/ROCm.
#
# Semantics:
#   For triValid=false, the normal path below still writes zeros:
#     b0=b1=b2=0 -> out=0
#     and for ENABLE_DA, a0=a1=a2 -> dsdu=dsdv=0 -> outDA=(0,0)
#
# Applies to:
#   csrc/common/interpolate.cu
#   csrc/common/interpolate.hip

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"

python - "$REPO" <<'PY'
from pathlib import Path
import re
import sys

repo = Path(sys.argv[1])
files = [
    repo / "csrc/common/interpolate.cu",
    repo / "csrc/common/interpolate.hip",
]

replacement = """#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // v41m2 ROCm/RDNA fix:
    // Do not use the full-mask empty-warp shortcut after edge lanes may have returned.
    // The normal triValid=false path below still writes zero output for live lanes.
    if (false)
#else
    if (__all_sync(0xffffffffffffffffull, !triValid))
#endif"""

patched_any = False

for path in files:
    if not path.exists():
        print(f"skip missing {path}")
        continue

    s = path.read_text()
    orig = s

    # If a previous v41m with 32-bit mask is present, replace the whole block.
    if "v41m ROCm/RDNA fix" in s or "v41m2 ROCm/RDNA fix" in s:
        pat_existing = re.compile(
            r'#if defined\(__HIP_PLATFORM_AMD__\) \|\| defined\(__HIPCC__\)\n'
            r'    // v41m[^\n]*\n'
            r'.*?'
            r'    if\s*\(false\)\n'
            r'#else\n'
            r'    if\s*\(__all_sync\s*\([^)]*!\s*triValid\s*\)\)\n'
            r'#endif',
            re.DOTALL
        )
        s2, n = pat_existing.subn(replacement, s, count=1)
    else:
        # Current v41l state: the original location contains if (false).
        # Original state: it contains if (__all_sync(0xffffffffffffffffull, !triValid)).
        pat_shortcut = re.compile(
            r'(?P<prefix>    // If no geometry in entire warp, zero the output and exit\.\n'
            r'    // Otherwise force barys to zero and output with live threads\.\n)'
            r'    if\s*\(\s*(?:false|__all_sync\s*\([^)]*!\s*triValid\s*\))\s*\)',
            re.MULTILINE
        )
        s2, n = pat_shortcut.subn(lambda m: m.group("prefix") + replacement, s, count=1)

    if n == 0:
        print(f"{path}: target not found; candidates:")
        for ln, line in enumerate(s.splitlines(), 1):
            if "If no geometry" in line or "Otherwise force" in line or "__all_sync" in line or "if (false)" in line or "v41m" in line:
                print(f"{ln}: {line}")
        continue

    backup = path.with_suffix(path.suffix + ".before_v41m2_rocm_interpolate_emptywarp_fix")
    if not backup.exists():
        backup.write_text(orig)

    path.write_text(s2)
    patched_any = True
    print(f"patched {path}")

if not patched_any:
    raise SystemExit("ERROR: v41m2 patched nothing")
PY

echo
echo "Verify v41m2 block:"
for f in "$REPO"/csrc/common/interpolate.cu "$REPO"/csrc/common/interpolate.hip; do
  echo "===== $f ====="
  nl -ba "$f" | sed -n '34,58p'
done

echo
echo "Rebuild with ROCm clang:"
cat <<'EOF'
source ~/therock_test/venv/bin/activate
cd ~/therock_test/nvdiffrast

rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast

export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++
export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v
EOF
