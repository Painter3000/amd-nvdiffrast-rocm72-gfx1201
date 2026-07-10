#!/usr/bin/env bash
set -euo pipefail

# v40f:
# Patch BOTH csrc/common/antialias.cu and csrc/common/antialias.hip with the
# v40e forward-analysis bounds guards.
#
# Why:
#   During pip build, antialias.hip may be regenerated/overwritten from
#   antialias.cu or by a hipify/copy step. If only antialias.hip is patched,
#   the markers disappear after build.
#
# This patch makes antialias.cu the patched source-of-truth too, so regenerated
# antialias.hip should keep the guard logic.

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"

patch_one() {
  local AA="$1"
  if [[ ! -f "$AA" ]]; then
    echo "skip missing: $AA"
    return 0
  fi

  echo "patching $AA"
  cp -n "$AA" "$AA.before_rocm_antialias_fwd_bounds_v40f" || true

  python - "$AA" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text()
orig = s

# Remove existing v40e/v40f blocks if partially applied? We do not attempt a
# destructive cleanup here; we only insert if markers are absent.

# 1) WorkCount guard after workCount load in AntialiasFwdAnalysisKernel.
if "v40f ROCm/RDNA workCount bounds guard" not in s and "v40e ROCm/RDNA workCount bounds guard" not in s:
    needle = """    __shared__ int s_base;
    int workCount = p.workBuffer[0].x;
    for(;;)
"""
    repl = """    __shared__ int s_base;
    int workCount = p.workBuffer[0].x;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // v40f ROCm/RDNA workCount bounds guard:
    // Discontinuity pass can emit at most two work items per pixel.
    int maxWorkCount = p.n * p.width * p.height * 2;
    if (workCount < 0 || workCount > maxWorkCount)
        return;
#endif
    for(;;)
"""
    if needle not in s:
        raise SystemExit(f"ERROR: workCount insertion point not found in {path}")
    s = s.replace(needle, repl, 1)
    print("  patched workCount guard")
else:
    print("  workCount guard already present")

# 2) WorkItem and pixel0/pixel1 guard.
if "v40f ROCm/RDNA pixel0/pixel1 bounds guard" not in s and "v40e ROCm/RDNA pixel0/pixel1 bounds guard" not in s:
    needle = """        int px = item.x;
        int py = item.y;
        int pz = (int)(((unsigned int)item.z) >> 16);
        int d  = (item.z >> AAWorkItem::FLAG_DOWN_BIT) & 1;

        int pixel0 = px + p.width * (py + p.height * pz);
        int pixel1 = pixel0 + (d ? p.width : 1);
        float2 zt0 = ((float2*)p.rasterOut)[(pixel0 << 1) + 1];
        float2 zt1 = ((float2*)p.rasterOut)[(pixel1 << 1) + 1];
"""
    repl = """        int px = item.x;
        int py = item.y;
        int pz = (int)(((unsigned int)item.z) >> 16);
        int d  = (item.z >> AAWorkItem::FLAG_DOWN_BIT) & 1;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // v40f ROCm/RDNA WorkItem bounds guard.
        if (px < 0 || py < 0 || px >= p.width || py >= p.height)
            continue;
        if (pz < 0 || pz >= p.n)
            continue;
        if (d != 0 && d != 1)
            continue;
        // d == 0 -> neighbor x+1. d == 1 -> neighbor y+1.
        if ((d == 0 && px >= p.width  - 1) ||
            (d == 1 && py >= p.height - 1))
            continue;
#endif

        int pixel0 = px + p.width * (py + p.height * pz);
        int pixel1 = pixel0 + (d ? p.width : 1);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // v40f ROCm/RDNA pixel0/pixel1 bounds guard.
        int totalPixels = p.n * p.width * p.height;
        if (pixel0 < 0 || pixel1 < 0 || pixel0 >= totalPixels || pixel1 >= totalPixels)
            continue;
#endif

        float2 zt0 = ((float2*)p.rasterOut)[(pixel0 << 1) + 1];
        float2 zt1 = ((float2*)p.rasterOut)[(pixel1 << 1) + 1];
"""
    if needle not in s:
        # If v40d changed this exact block, add a helpful error instead of silently doing nothing.
        if "v40d ROCm/RDNA WorkItem/pixel bounds guard" in s:
            raise SystemExit(f"ERROR: v40d block already present in {path}; reset or manually merge v40f.")
        raise SystemExit(f"ERROR: pixel guard insertion point not found in {path}")
    s = s.replace(needle, repl, 1)
    print("  patched pixel guards")
else:
    print("  pixel guards already present")

if s != orig:
    path.write_text(s)
    print("  wrote changes")
else:
    print("  no changes")

text = path.read_text()
print("  verify markers:")
for marker in [
    "v40f ROCm/RDNA workCount bounds guard",
    "v40f ROCm/RDNA WorkItem bounds guard",
    "v40f ROCm/RDNA pixel0/pixel1 bounds guard",
    "v40e ROCm/RDNA workCount bounds guard",
    "v40e ROCm/RDNA WorkItem bounds guard",
    "v40e ROCm/RDNA pixel0/pixel1 bounds guard",
]:
    print(f"    {marker}: {text.count(marker)}")
PY
}

patch_one "$REPO/csrc/common/antialias.cu"
patch_one "$REPO/csrc/common/antialias.hip"

echo
echo "Search for build-time regeneration/copy sources:"
grep -R "antialias.hip\|hipify\|antialias.cu" -n "$REPO/setup.py" "$REPO/csrc" "$REPO" 2>/dev/null | head -120 || true

echo
echo "Before rebuild, verify both files:"
grep -n "v40f ROCm/RDNA\|v40e ROCm/RDNA" "$REPO/csrc/common/antialias.cu" "$REPO/csrc/common/antialias.hip" || true

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

echo "After rebuild, verify both files:"
grep -n "v40f ROCm/RDNA\|v40e ROCm/RDNA" csrc/common/antialias.cu csrc/common/antialias.hip || true
EOF
