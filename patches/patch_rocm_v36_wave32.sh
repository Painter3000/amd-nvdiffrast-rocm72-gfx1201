#!/usr/bin/env bash
set -euo pipefail

# patch_current_nvdiffrast_fineraster_wave64_runtime_fix_v36_rdna_wave32.sh
# ----------------------------------------------------------------
# EXPERIMENTAL RUNTIME FIX for ROCm 7.2 / gfx1201.
#
# IMPORTANT:
#   Apply this on a clean runtime source tree, not on a diagnostic v19..v29 tree.
#
# Recommended sequence:
#   cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4
#   MODE=runtime ./reinstall_nvdiffrast_rocm72_gfx1201.sh
#
#   cd ~/therock_test/nvdiffrast
#   bash /mnt/data/patch_current_nvdiffrast_fineraster_wave64_runtime_fix_v36_rdna_wave32.sh
#   rm -rf build/ dist/ ./*.egg-info
#   pip uninstall -y nvdiffrast
#   export PYTORCH_ROCM_ARCH=gfx1201 FORCE_CUDA=1 MAX_JOBS=1
#   export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"
#   python -m pip install . --no-build-isolation -v
#
# What it fixes:
#   CUDA original assumes threadIdx.y rows are independent warp32 lanes.
#   On AMD Wave64, one hardware wave can contain two such rows.
#
#   This patch:
#     - adds per-row Wave64 masks:
#         y even -> lower 32 lanes
#         y odd  -> upper 32 lanes
#     - makes scan32_value() take a mask
#     - makes updateTileZMax() take a mask
#     - makes executeROP() take U64 exact participation masks
#     - converts per-row ballots to logical U32 masks by shifting rowMask output
#     - computes ropMask from the exact doROP participant lanes
#
# This is the RDNA/gfx1201 Wave32 runtime fix candidate: all logical threadIdx.y rows use the same lower-32 mask inside their own Wave32.

FR="csrc/common/hipraster/impl/FineRaster.inl"

if [[ ! -f "$FR" ]]; then
  echo "FEHLER: $FR nicht gefunden. Bitte im nvdiffrast-Repo ausführen."
  exit 1
fi

if grep -q "NVDR_ROCM_FINERASTER_" "$FR"; then
  echo "FEHLER: FineRaster.inl enthält noch einen Diagnose-Probe-Marker."
  echo "Bitte erst auf Runtime-Basis zurücksetzen:"
  echo "  cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4"
  echo "  MODE=runtime ./reinstall_nvdiffrast_rocm72_gfx1201.sh"
  echo "Dann diesen Patch erneut ausführen."
  grep -n "NVDR_ROCM_FINERASTER_" "$FR" || true
  exit 1
fi

cp -v "$FR" "$FR.before_wave64_runtime_fix_v36_rdna_wave32.$(date +%Y%m%d_%H%M%S)"

python - <<'PY'
from pathlib import Path
import re

p = Path("csrc/common/hipraster/impl/FineRaster.inl")
s = p.read_text()
orig = s

def replace_func(s, name, new_text):
    sig = re.search(
        r'__device__\s+__inline__\s+[A-Za-z0-9_:<>\*&\s]+\s+' + re.escape(name) + r'\s*\(',
        s,
        flags=re.S,
    )
    if not sig:
        raise SystemExit(f"FEHLER: Funktion {name} nicht gefunden.")

    open_brace = s.find("{", sig.end())
    if open_brace < 0:
        raise SystemExit(f"FEHLER: opening brace für {name} nicht gefunden.")

    depth = 0
    close_brace = None
    for i in range(open_brace, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                close_brace = i
                break

    if close_brace is None:
        raise SystemExit(f"FEHLER: closing brace für {name} nicht gefunden.")

    return s[:sig.start()] + new_text.rstrip() + s[close_brace+1:]

# 1) Add helper for logical warp32 ballot from AMD Wave64 row mask.
helper = r"""
//------------------------------------------------------------------------
// ROCm/gfx1201 Wave64 compatibility helper.
// Converts an actual Wave64 ballot mask to the logical CUDA-style warp32 mask
// used by the original FineRaster ring-buffer math.
__device__ __inline__ U32 nvdr_ballot32_from_rowmask(U64 rowMask, U32 rowShift, bool pred)
{
    return (U32)(__ballot_sync(rowMask, pred) >> rowShift);
}
"""
if "nvdr_ballot32_from_rowmask" not in s:
    m = re.search(r'__device__\s+__inline__\s+S32\s+findBit\s*\(', s)
    if not m:
        raise SystemExit("FEHLER: findBit() nicht gefunden; kann Helper nicht einfügen.")
    open_brace = s.find("{", m.end())
    depth = 0
    close_brace = None
    for i in range(open_brace, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                close_brace = i
                break
    if close_brace is None:
        raise SystemExit("FEHLER: findBit() Ende nicht gefunden.")
    s = s[:close_brace+1] + helper + s[close_brace+1:]
    print("OK: inserted nvdr_ballot32_from_rowmask helper")

# 2) Replace updateTileZMax with mask-aware version.
new_update = r"""
__device__ __inline__ void updateTileZMax(U32& tileZMax, bool& tileZUpd, volatile U32* tileDepth, volatile U32* temp, U64 warpMask)
{
    // Entry is coherent over the logical CUDA warp32 row.
    if (__any_sync(warpMask, tileZUpd))
    {
        U32 z = ::max(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]);

        __syncwarp(warpMask);
        temp[threadIdx.x + 16] = z; __syncwarp(warpMask);
        z = ::max(z, temp[threadIdx.x + 16 -  1]); __syncwarp(warpMask); temp[threadIdx.x + 16] = z; __syncwarp(warpMask);
        z = ::max(z, temp[threadIdx.x + 16 -  2]); __syncwarp(warpMask); temp[threadIdx.x + 16] = z; __syncwarp(warpMask);
        z = ::max(z, temp[threadIdx.x + 16 -  4]); __syncwarp(warpMask); temp[threadIdx.x + 16] = z; __syncwarp(warpMask);
        z = ::max(z, temp[threadIdx.x + 16 -  8]); __syncwarp(warpMask); temp[threadIdx.x + 16] = z; __syncwarp(warpMask);
        z = ::max(z, temp[threadIdx.x + 16 - 16]); __syncwarp(warpMask); temp[threadIdx.x + 16] = z; __syncwarp(warpMask);

        tileZMax = temp[47];
        tileZUpd = false;
    }
}
"""
s = replace_func(s, "updateTileZMax", new_update)
print("OK: updateTileZMax mask-aware")

# 3) Replace scan32_value with mask-aware version.
new_scan = r"""
__device__ __inline__ U32 scan32_value(U32 value, volatile U32* temp, U64 warpMask)
{
    __syncwarp(warpMask);
    temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    value += temp[threadIdx.x + 16 -  1]; __syncwarp(warpMask); temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    value += temp[threadIdx.x + 16 -  2]; __syncwarp(warpMask); temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    value += temp[threadIdx.x + 16 -  4]; __syncwarp(warpMask); temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    value += temp[threadIdx.x + 16 -  8]; __syncwarp(warpMask); temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    value += temp[threadIdx.x + 16 - 16]; __syncwarp(warpMask); temp[threadIdx.x + 16] = value; __syncwarp(warpMask);
    return value;
}
"""
s = replace_func(s, "scan32_value", new_scan)
print("OK: scan32_value mask-aware")

# 4) Replace executeROP with U64 exact-mask version.
new_rop = r"""
__device__ __inline__ void executeROP(U32 color, U32 depth, volatile U32* pColor, volatile U32* pDepth, U64 ropMask)
{
    // ropMask must be the exact set of lanes that enter executeROP().
    atomicMin((U32*)pDepth, depth);
    __syncwarp(ropMask);

    bool act = (depth == *pDepth);
    __syncwarp(ropMask);

    U64 actMask = __ballot_sync(ropMask, act);

    if (act)
    {
        *pDepth = 0;
        __syncwarp(actMask);

        atomicMax((U32*)pDepth, threadIdx.x);
        __syncwarp(actMask);

        if (*pDepth == threadIdx.x)
        {
            *pDepth = depth;
            *pColor = color;
        }

        __syncwarp(actMask);
    }
}
"""
s = replace_func(s, "executeROP", new_rop)
print("OK: executeROP U64 exact-mask")

# 5) Patch fineRasterImpl body.
m = re.search(r'__device__\s+__inline__\s+void\s+fineRasterImpl\s*\(\s*const\s+CRParams\s+p\s*\)', s, flags=re.S)
if not m:
    raise SystemExit("FEHLER: fineRasterImpl nicht gefunden.")

open_brace = s.find("{", m.end())
depth = 0
close_brace = None
for i in range(open_brace, len(s)):
    if s[i] == "{":
        depth += 1
    elif s[i] == "}":
        depth -= 1
        if depth == 0:
            close_brace = i
            break
if close_brace is None:
    raise SystemExit("FEHLER: fineRasterImpl Ende nicht gefunden.")

prefix = s[:open_brace+1]
body = s[open_brace+1:close_brace]
suffix = s[close_brace:]

if "nvdr_rowMask" not in body:
    body = body.replace(
        "cover8x8_setupLUT(s_cover8x8_lut);\n    __syncthreads();",
        """cover8x8_setupLUT(s_cover8x8_lut);
    __syncthreads();

    // AMD RDNA/gfx1201 Wave32 safety:
    // On RDNA/gfx1201 HIP kernels use native Wave32 unless explicitly compiled
    // with a wave64 mode. nvdiffrast's threadIdx.y rows are independent
    // logical warp32 rows, so every row uses the same lower-32 participation
    // mask inside its own wave.
    const U32 nvdr_rowShift = 0u;
    const U64 nvdr_rowMask  = 0x00000000ffffffffull;"""
    )
    print("OK: inserted nvdr_rowMask/nvdr_rowShift in fineRasterImpl")

body = body.replace(
    "updateTileZMax(tileZMax, tileZUpd, tileDepth, temp);",
    "updateTileZMax(tileZMax, tileZUpd, tileDepth, temp, nvdr_rowMask);"
)
body = body.replace(
    "U32 frag = scan32_value(pop, temp);",
    "U32 frag = scan32_value(pop, temp, nvdr_rowMask);"
)
# Source may contain bare __syncwarp() or compatibility-expanded __syncwarp(~0ull).
body = body.replace("__syncwarp();", "__syncwarp(nvdr_rowMask);")
body = body.replace("__syncwarp(~0u);", "__syncwarp(nvdr_rowMask);")
body = body.replace("__syncwarp(~0ull);", "__syncwarp(nvdr_rowMask);")
# Source variants use either ~0u or ~0ull depending on prior ROCm cleanup.
for full_mask_token in ["~0u", "~0ull"]:
    body = body.replace(
        f"U32 goodMask = __ballot_sync({full_mask_token}, pop != 0);",
        "U32 goodMask = nvdr_ballot32_from_rowmask(nvdr_rowMask, nvdr_rowShift, pop != 0);"
    )
    body = body.replace(
        f"U32 boundaryMask = __ballot_sync({full_mask_token}, temp[ropLaneIdx + 16]);",
        "U32 boundaryMask = nvdr_ballot32_from_rowmask(nvdr_rowMask, nvdr_rowShift, temp[ropLaneIdx + 16] != 0);"
    )
    body = body.replace(
        f"U32 fragmentMask = __ballot_sync({full_mask_token}, hasFragment);",
        "U64 fragmentMask64 = __ballot_sync(nvdr_rowMask, hasFragment);\n            U32 fragmentMask = (U32)(fragmentMask64 >> nvdr_rowShift);"
    )

body = body.replace(
    "U32 ropMask = __ballot_sync(fragmentMask, !zkill);",
    "U64 ropMask = __ballot_sync(fragmentMask64, !zkill);"
)

s = prefix + body + suffix

problems = []
if "scan32_value(pop, temp)" in s:
    problems.append("unpatched scan32_value(pop, temp)")
if "updateTileZMax(tileZMax, tileZUpd, tileDepth, temp)" in s:
    problems.append("unpatched updateTileZMax(..., temp)")
if "__ballot_sync(~0u" in s:
    problems.append("__ballot_sync(~0u")
if "__any_sync(~0u" in s:
    problems.append("__any_sync(~0u")
if "__syncwarp();" in s:
    problems.append("bare __syncwarp()")
if "__syncwarp(~0u" in s:
    problems.append("__syncwarp(~0u/~0ull)")
if "__ballot_sync(~0ull" in s:
    problems.append("__ballot_sync(~0ull")
if "__any_sync(~0ull" in s:
    problems.append("__any_sync(~0ull")
if "U32 ropMask = __ballot_sync" in s:
    problems.append("U32 ropMask ballot remains")
if "U32 fragmentMask = __ballot_sync" in s:
    problems.append("U32 fragmentMask direct ballot remains")

if "0xffffffff00000000ull" in s:
    problems.append("upper-half Wave64 mask remains")
if "((threadIdx.y & 1) << 5)" in s:
    problems.append("row parity shift remains")

if problems:
    print("FEHLER: v36 validation failed:")
    for item in problems:
        print("  -", item)
    for i, line in enumerate(s.splitlines(), 1):
        if any(tok in line for tok in [
            "scan32_value(pop, temp)",
            "updateTileZMax(tileZMax, tileZUpd, tileDepth, temp)",
            "__ballot_sync(~0u",
            "__any_sync(~0u",
            "__syncwarp();",
            "__syncwarp(~0u",
            "__ballot_sync(~0ull",
            "__any_sync(~0ull",
            "U32 ropMask = __ballot_sync",
            "U32 fragmentMask = __ballot_sync",
        ]):
            print(f"{i}: {line}")
    raise SystemExit(1)

if s == orig:
    raise SystemExit("FEHLER: Keine Änderung vorgenommen.")

p.write_text(s)
print("OK: v36 Wave64 runtime fix applied.")
PY

echo
echo "=== v36 patch markers / critical lines ==="
grep -n "nvdr_rowMask\|nvdr_rowShift\|nvdr_ballot32_from_rowmask\|scan32_value(pop, temp, nvdr_rowMask)\|updateTileZMax(tileZMax, tileZUpd, tileDepth, temp, nvdr_rowMask)\|fragmentMask64\|U64 ropMask\|executeROP" "$FR" | head -240

echo
echo "=== remaining dangerous tokens check ==="
if grep -n "__ballot_sync(~0u\|__ballot_sync(~0ull\|__any_sync(~0u\|__any_sync(~0ull\|__syncwarp();\|__syncwarp(~0u\|__syncwarp(~0ull" "$FR"; then
  echo "FEHLER: dangerous token remains"
  exit 1
else
  echo "OK: no bare CUDA warp32 sync/ballot tokens remain in FineRaster.inl"
fi

echo
echo "OK: v36 runtime fix patch ready. Rebuild nvdiffrast now."
