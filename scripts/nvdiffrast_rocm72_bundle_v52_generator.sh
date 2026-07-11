#!/usr/bin/env bash
set -euo pipefail

# nvdiffrast_rocm72_bundle_v52_generator.sh
# --------------------------------------------
# Creates the ROCm runtime-baseline reinstall bundle used by the final v52
# nvdiffrast ROCm 7.2 / RDNA4 gfx1201 patch stack.
#
# This generator is still the low-level v4 runtime-baseline phase. The final
# v52 setup applies patches/apply_final_rocm72_gfx1201_v52.sh after this phase.

BUNDLE_DIR="${BUNDLE_DIR:-$HOME/therock_test/nvdiffrast_rocm72_gfx1201_final_v52_bundle}"

mkdir -p "$BUNDLE_DIR"

cat > "$BUNDLE_DIR/reinstall_nvdiffrast_rocm72_gfx1201.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

# reinstall_nvdiffrast_rocm72_gfx1201.sh
# --------------------------------------
# Runtime-baseline reinstall used before the final v52 patch stack.
#
# Modes:
#   MODE=coverage_probe  -> replaces FineRaster with a safe rebaseline probe
#   MODE=runtime         -> normal FineRaster plus known ROCm compile/runtime hygiene

ROOT="${ROOT:-$HOME/therock_test}"
REPO="${REPO:-$ROOT/nvdiffrast}"
VENV="${VENV:-$ROOT/venv}"
MODE="${MODE:-coverage_probe}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "$REPO" ]]; then
  echo "FEHLER: Repo nicht gefunden: $REPO"
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "FEHLER: Venv nicht gefunden: $VENV"
  exit 1
fi

if [[ ! -d "$ROCM_PATH" ]]; then
  if [[ -d /opt/rocm ]]; then
    ROCM_PATH="/opt/rocm"
  else
    echo "FEHLER: ROCm nicht gefunden: $ROCM_PATH oder /opt/rocm"
    exit 1
  fi
fi

cd "$REPO"

echo "=== nvdiffrast ROCm72 gfx1201 reinstall ==="
echo "REPO=$REPO"
echo "VENV=$VENV"
echo "MODE=$MODE"
echo "ROCM_PATH=$ROCM_PATH"
echo

source "$VENV/bin/activate"

python - <<'PY'
import torch, sys
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

CUDA_ROOT="csrc/common/cudaraster"
HIP_ROOT="csrc/common/hipraster"
HIP_IMPL="$HIP_ROOT/impl"
CUDA_IMPL="$CUDA_ROOT/impl"

if [[ ! -d "$CUDA_IMPL" ]]; then
  echo "FEHLER: $CUDA_IMPL fehlt."
  exit 1
fi

for f in FineRaster.inl BinRaster.inl CoarseRaster.inl TriangleSetup.inl Util.inl; do
  if [[ ! -f "$CUDA_IMPL/$f" ]]; then
    echo "FEHLER: $CUDA_IMPL/$f fehlt."
    exit 1
  fi
done

echo
echo "=== Ensure hipraster tree exists ==="
if [[ ! -d "$HIP_IMPL" ]]; then
  echo "INFO: $HIP_IMPL fehlt. Erzeuge HIP-Struktur aus $CUDA_ROOT ..."
  rm -rf "$HIP_ROOT"
  mkdir -p "$(dirname "$HIP_ROOT")"
  cp -a "$CUDA_ROOT" "$HIP_ROOT"

  HIPIFY_BIN=""
  for cand in     "$ROCM_PATH/bin/hipify-perl"     "$ROCM_PATH/hip/bin/hipify-perl"     "/opt/rocm/bin/hipify-perl"     "/opt/rocm/hip/bin/hipify-perl"
  do
    if [[ -x "$cand" ]]; then
      HIPIFY_BIN="$cand"
      break
    fi
  done

  if [[ -n "$HIPIFY_BIN" ]]; then
    echo "INFO: hipify-perl gefunden: $HIPIFY_BIN"
    while IFS= read -r -d "" f; do
      "$HIPIFY_BIN" -inplace "$f" >/dev/null
    done < <(find "$HIP_ROOT" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.inl" -o -name "*.cu" \) -print0)
  else
    echo "WARNUNG: hipify-perl nicht gefunden. Verwende einfache CUDA->HIP Text-Migration."
    while IFS= read -r -d "" f; do
      sed -i \
        -e 's/cuda_runtime.h/hip\/hip_runtime.h/g' \
        -e 's/cuda_runtime_api.h/hip\/hip_runtime_api.h/g' \
        -e 's/cudaMemcpyAsync/hipMemcpyAsync/g' \
        -e 's/cudaMemcpy/hipMemcpy/g' \
        -e 's/cudaMalloc/hipMalloc/g' \
        -e 's/cudaFree/hipFree/g' \
        -e 's/cudaStream_t/hipStream_t/g' \
        -e 's/cudaEvent_t/hipEvent_t/g' \
        -e 's/cudaSuccess/hipSuccess/g' \
        -e 's/cudaGetLastError/hipGetLastError/g' \
        -e 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' \
        -e 's/cudaMemcpyHostToDevice/hipMemcpyHostToDevice/g' \
        -e 's/cudaMemcpyDeviceToHost/hipMemcpyDeviceToHost/g' \
        -e 's/cudaMemcpyDeviceToDevice/hipMemcpyDeviceToDevice/g' \
        "$f"
    done < <(find "$HIP_ROOT" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.inl" -o -name "*.cu" \) -print0)
  fi

  if [[ ! -d "$HIP_IMPL" ]]; then
    echo "FEHLER: $HIP_IMPL konnte nicht erzeugt werden."
    exit 1
  fi
else
  echo "OK: vorhandene HIP-Struktur gefunden: $HIP_IMPL"
fi

ts="$(date +%Y%m%d_%H%M%S)"

echo
echo "=== Backup current hipraster/impl/*.inl ==="
mkdir -p "$HIP_IMPL/_bundle_backups_$ts"
shopt -s nullglob
for f in "$HIP_IMPL"/*.inl; do
  cp -v "$f" "$HIP_IMPL/_bundle_backups_$ts/$(basename "$f")"
done

echo
echo "=== Reset hipraster impl from cudaraster impl ==="
for src in "$CUDA_IMPL"/*.inl; do
  base="$(basename "$src")"
  cp -v "$src" "$HIP_IMPL/$base"
done

echo
echo "=== Apply ROCm72/gfx1201 source patches ==="
MODE="$MODE" python - <<'PY'
from pathlib import Path
import re, os

impl = Path("csrc/common/hipraster/impl")
mode = os.environ.get("MODE", "coverage_probe")

# -------------------------------------------------------------------------
# 1) General HIP sync-mask compatibility.
# -------------------------------------------------------------------------
def hip_mask_compat(path: Path):
    s = path.read_text()
    orig = s

    repl = {
        "__ballot_sync(~0u,": "__ballot_sync(~0ull,",
        "__any_sync(~0u,": "__any_sync(~0ull,",
        "__all_sync(~0u,": "__all_sync(~0ull,",
        "__syncwarp(~0u)": "__syncwarp(~0ull)",
        "__ballot_sync(0xffffffff,": "__ballot_sync(0xffffffffull,",
        "__ballot_sync(0xffffffffu,": "__ballot_sync(0xffffffffull,",
        "__any_sync(0xffffffff,": "__any_sync(0xffffffffull,",
        "__any_sync(0xffffffffu,": "__any_sync(0xffffffffull,",
        "__all_sync(0xffffffff,": "__all_sync(0xffffffffull,",
        "__all_sync(0xffffffffu,": "__all_sync(0xffffffffull,",
        "__syncwarp(0xffffffff)": "__syncwarp(0xffffffffull)",
        "__syncwarp(0xffffffffu)": "__syncwarp(0xffffffffull)",
    }
    for a,b in repl.items():
        s = s.replace(a,b)

    s = re.sub(
        r'(executeROP\s*\(\s*U32\s+color\s*,\s*U32\s+depth\s*,\s*volatile\s+U32\*\s*pColor\s*,\s*volatile\s+U32\*\s*pDepth\s*,\s*)U32(\s+ropMask\s*\))',
        r'\1U64\2',
        s,
    )

    s = re.sub(
        r'\bU32\s+(actMask|ropMask)\s*=\s*__ballot_sync\s*\(',
        r'U64 \1 = __ballot_sync(',
        s,
    )

    s = re.sub(
        r'__(ballot_sync|any_sync|all_sync)\(\s*([A-Za-z_]\w*Mask)\s*,',
        r'__\1((U64)\2,',
        s,
    )
    s = re.sub(
        r'__syncwarp\(\s*([A-Za-z_]\w*Mask)\s*\)',
        r'__syncwarp((U64)\1)',
        s,
    )

    path.write_text(s)
    print(f"OK mask compat: {path.name} changed={s != orig}")

for p in sorted(impl.glob("*.inl")):
    hip_mask_compat(p)

# -------------------------------------------------------------------------
# 1b) antialias.cu: HIP requires 64-bit masks for __ballot_sync().
#     Upstream declares the antialias-grad active mask as 32-bit:
#         unsigned int amask = __ballot_sync(0xffffffffu, item.w);
#     hipcc rejects this (static_assert: mask must be 64-bit), so the very
#     first baseline build fails before any v52 stack patch can run.
#     Widening it here also establishes the exact 64-bit baseline form that
#     patch_antialias_grad_rocm_wave32_v38.sh expects later in the stack.
#     Marker: NVDR_ROCM_AA_BALLOT64
# -------------------------------------------------------------------------
aa_targets = [
    Path("csrc/common/antialias.cu"),
    Path("csrc/common/antialias.hip"),  # stale hipify output from a previous build
]
for p in aa_targets:
    if not p.exists():
        continue
    s = p.read_text()
    orig = s
    s = s.replace(
        "        unsigned int amask = __ballot_sync(0xffffffffu, item.w);",
        "        // NVDR_ROCM_AA_BALLOT64: HIP requires a 64-bit mask for __ballot_sync().\n"
        "        unsigned long long amask = __ballot_sync(0xffffffffffffffffull, item.w);",
    )
    if "unsigned long long amask" not in s:
        print(f"FEHLER: 64-bit amask baseline missing in {p} "
              f"(upstream formatting changed? expected 'unsigned int amask = __ballot_sync(0xffffffffu, item.w);')")
        raise SystemExit(1)
    if s != orig:
        p.write_text(s)
    print(f"OK antialias 64-bit ballot mask (NVDR_ROCM_AA_BALLOT64): {p} changed={s != orig}")

# -------------------------------------------------------------------------
# 1c) interpolate.cu: same HIP 64-bit mask requirement for __all_sync().
#     Upstream: if (__all_sync(0xffffffffu, !triValid))
#     hipcc rejects the 32-bit mask, so the baseline build fails on
#     interpolate.hip before patch_rocm_interpolate_emptywarp_allsync_fix_v41m2.sh
#     can run. Widening the literal to 64-bit produces exactly the baseline
#     form that v41m2 later expects and replaces with the #if HIP block.
#     Marker: NVDR_ROCM_INTERP_ALLSYNC64
# -------------------------------------------------------------------------
interp_targets = [
    Path("csrc/common/interpolate.cu"),
    Path("csrc/common/interpolate.hip"),  # stale hipify output from a previous build
]
for p in interp_targets:
    if not p.exists():
        continue
    s = p.read_text()
    orig = s
    s = s.replace(
        "    if (__all_sync(0xffffffffu, !triValid))",
        "    if (__all_sync(0xffffffffffffffffull, !triValid)) // NVDR_ROCM_INTERP_ALLSYNC64",
    )
    if "0xffffffffffffffffull, !triValid" not in s:
        print(f"FEHLER: 64-bit __all_sync baseline missing in {p} "
              f"(upstream formatting changed? expected 'if (__all_sync(0xffffffffu, !triValid))')")
        raise SystemExit(1)
    if s != orig:
        p.write_text(s)
    print(f"OK interpolate 64-bit all_sync mask (NVDR_ROCM_INTERP_ALLSYNC64): {p} changed={s != orig}")

# -------------------------------------------------------------------------
# 1d) texture_kernel.cu: HIP has no directed-rounding reciprocal (__frcp_rz).
#     ROCm's __clang_hip_math.h only provides __frcp_rn (round-to-nearest), so
#     the baseline build fails with "use of undeclared identifier '__frcp_rz'".
#     Mapping to __frcp_rn is NOT bit-exact (differs by 1 ULP in ~50% of cases),
#     so we emulate round-toward-zero exactly: IEEE divide, then step one ULP
#     toward zero if the magnitude overshot. This preserves the upstream
#     invariant |x * m| <= 0.5 used by the cubemap face mapping.
#     Implemented as a macro so the call sites stay untouched and the CUDA
#     path keeps using the native intrinsic.
#     Marker: NVDR_ROCM_FRCP_RZ
# -------------------------------------------------------------------------
frcp_shim = '''
// NVDR_ROCM_FRCP_RZ: HIP/ROCm provides no round-toward-zero reciprocal.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
static __device__ __forceinline__ float nvdr_frcp_rz(float x)
{
    float r = 1.0f / x;                 // IEEE divide, round-to-nearest.
    if (fmaf(-x, r, 1.0f) < 0.0f)       // Magnitude overshot the exact quotient?
        r = nextafterf(r, 0.0f);        // Step one ULP toward zero.
    return r;
}
#define __frcp_rz(x) nvdr_frcp_rz(x)
#endif
'''

tex_targets = [
    Path("csrc/common/texture_kernel.cu"),
    Path("csrc/common/texture_kernel.hip"),  # stale hipify output from a previous build
]
for p in tex_targets:
    if not p.exists():
        continue
    s = p.read_text()
    orig = s
    if "NVDR_ROCM_FRCP_RZ" not in s:
        anchor = '#include "texture.h"\n'
        if anchor not in s:
            print(f"FEHLER: include anchor not found in {p} (upstream layout changed?)")
            raise SystemExit(1)
        s = s.replace(anchor, anchor + frcp_shim, 1)
        p.write_text(s)
    if "__frcp_rz" in s and "NVDR_ROCM_FRCP_RZ" not in s:
        print(f"FEHLER: __frcp_rz still unshimmed in {p}")
        raise SystemExit(1)
    print(f"OK texture __frcp_rz shim (NVDR_ROCM_FRCP_RZ): {p} changed={s != orig}")

# -------------------------------------------------------------------------
# 1e) torch_antialias.cpp: clang++ rejects the narrowing conversion in
#         torch::zeros({(uint64_t)... * ... * 4}, opts)
#     because torch::zeros takes IntArrayRef (int64_t) and C++20 forbids the
#     implicit uint64_t -> int64_t narrowing inside a braced-init-list. GCC
#     accepted it, ROCm clang++ does not:
#         error: non-constant-expression cannot be narrowed from 'uint64_t' to 'long'
#     This is the same fix as patches/patch_clang_antialias_evhash_narrowing_v41c.sh,
#     but it must run BEFORE the baseline build -- the patch stack only runs
#     afterwards, so the build would never get that far.
#     Note we patch the .cpp source, not torch_antialias_hip.cpp: the latter is
#     produced from it by hipify during the build and inherits the fix.
#     v41c is idempotent (it checks for evHashElements), so it later reports
#     "already patched" instead of conflicting.
#     Marker: evHashElements (kept identical to v41c on purpose)
# -------------------------------------------------------------------------
evhash_old = "    torch::Tensor ev_hash = torch::zeros({(uint64_t)p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * 4}, opts);"
evhash_new = """    int64_t evHashElements = static_cast<int64_t>(
        static_cast<uint64_t>(p.allocTriangles) *
        static_cast<uint64_t>(AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles)) *
        4ull);
    torch::Tensor ev_hash = torch::zeros({evHashElements}, opts);"""

evhash_targets = [
    Path("csrc/torch/torch_antialias.cpp"),
    Path("csrc/torch/torch_antialias_hip.cpp"),  # stale hipify output from a previous build
]
for p in evhash_targets:
    if not p.exists():
        continue
    s = p.read_text()
    orig = s
    if "evHashElements" in s:
        print(f"OK evhash narrowing fix: {p} already patched")
        continue
    if evhash_old not in s:
        print(f"FEHLER: evhash target line not found in {p} (upstream formatting changed?)")
        raise SystemExit(1)
    s = s.replace(evhash_old, evhash_new)
    p.write_text(s)
    print(f"OK evhash narrowing fix (evHashElements): {p} changed={s != orig}")

# -------------------------------------------------------------------------
# 2) Util.inl: replace CUDA/PTX-only inline asm with portable HIP/C++ helpers.
# -------------------------------------------------------------------------
p = impl / "Util.inl"
s = p.read_text()
orig = s

lane_helpers = {
r'__device__\s+__inline__\s+U32\s+getLaneMaskLt\s*\(void\)\s*\{[^{}]*?%lanemask_lt;[^{}]*?\}':
"__device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 lane = (U32)(threadIdx.x & 31); return (lane == 0u) ? 0u : ((1u << lane) - 1u); }",
r'__device__\s+__inline__\s+U32\s+getLaneMaskLe\s*\(void\)\s*\{[^{}]*?%lanemask_le;[^{}]*?\}':
"__device__ __inline__ U32   getLaneMaskLe           (void)                  { U32 lane = (U32)(threadIdx.x & 31); return (lane == 31u) ? 0xffffffffu : ((1u << (lane + 1u)) - 1u); }",
r'__device__\s+__inline__\s+U32\s+getLaneMaskGt\s*\(void\)\s*\{[^{}]*?%lanemask_gt;[^{}]*?\}':
"__device__ __inline__ U32   getLaneMaskGt           (void)                  { return ~getLaneMaskLe(); }",
r'__device__\s+__inline__\s+U32\s+getLaneMaskGe\s*\(void\)\s*\{[^{}]*?%lanemask_ge;[^{}]*?\}':
"__device__ __inline__ U32   getLaneMaskGe           (void)                  { return ~getLaneMaskLt(); }",
}
for pat, repl in lane_helpers.items():
    s, n = re.subn(pat, repl, s)
    if n:
        print(f"OK Util lane helper patched: {n}")

repls = [
(
r'__device__\s+__inline__\s+S32\s+f32_to_s32_sat\s*\(\s*F32\s+a\s*\)\s*\{[^{}]*?cvt\.rni\.sat\.s32\.f32[^{}]*?\}',
"__device__ __inline__ S32   f32_to_s32_sat          (F32 a)                 { a = fminf(fmaxf(a, -2147483648.0f), 2147483520.0f); return (S32)rintf(a); }"
),
(
r'__device__\s+__inline__\s+U32\s+f32_to_u32_sat\s*\(\s*F32\s+a\s*\)\s*\{[^{}]*?cvt\.rni\.sat\.u32\.f32[^{}]*?\}',
"__device__ __inline__ U32   f32_to_u32_sat          (F32 a)                 { a = fminf(fmaxf(a, 0.0f), 4294967040.0f); return (U32)rintf(a); }"
),
(
r'__device__\s+__inline__\s+U32\s+f32_to_u32_sat_rmi\s*\(\s*F32\s+a\s*\)\s*\{[^{}]*?cvt\.rmi\.sat\.u32\.f32[^{}]*?\}',
"__device__ __inline__ U32   f32_to_u32_sat_rmi      (F32 a)                 { a = fminf(fmaxf(a, 0.0f), 4294967040.0f); return (U32)floorf(a); }"
),
(
r'__device__\s+__inline__\s+U32\s+f32_to_u8_sat\s*\(\s*F32\s+a\s*\)\s*\{[^{}]*?cvt\.rni\.sat\.u8\.f32[^{}]*?\}',
"__device__ __inline__ U32   f32_to_u8_sat           (F32 a)                 { a = fminf(fmaxf(a, 0.0f), 255.0f); return (U32)rintf(a); }"
),
(
r'__device__\s+__inline__\s+S64\s+f32_to_s64\s*\(\s*F32\s+a\s*\)\s*\{[^{}]*?cvt\.rni\.s64\.f32[^{}]*?\}',
"__device__ __inline__ S64   f32_to_s64              (F32 a)                 { return (S64)llrintf(a); }"
),
(
r'__device__\s+__inline__\s+F32\s+slct\s*\(\s*F32\s+a\s*,\s*F32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?slct\.f32\.s32[^{}]*?\}',
"__device__ __inline__ F32   slct                    (F32 a, F32 b, S32 c)   { return (c >= 0) ? a : b; }"
),
(
r'__device__\s+__inline__\s+F64\s+rcp_approx\s*\(\s*F64\s+a\s*\)\s*\{[^{}]*?rcp\.approx\.ftz\.f64[^{}]*?\}',
"__device__ __inline__ F64   rcp_approx              (F64 a)                 { return 1.0 / a; }"
),
(
r'__device__\s+__inline__\s+F32\s+fma_rm\s*\(\s*F32\s+a\s*,\s*F32\s+b\s*,\s*F32\s+c\s*\)\s*\{[^{}]*?fma\.rm\.f32[^{}]*?\}',
"__device__ __inline__ F32   fma_rm                  (F32 a, F32 b, F32 c)   { return floorf(fmaf(a, b, c)); }"
),
]
for pat, repl in repls:
    s, n = re.subn(pat, repl, s)
    if n:
        print(f"OK Util asm fallback patched: {n}")

# Extra CUDA/PTX video-instruction helpers that HIP/ROCm rejects on gfx1201.
# These are conservative C++ equivalents. The original CUDA code uses PTX
# video ops for speed; correctness is more important for the ROCm bring-up.
extra_repls = [
(
r'__device__\s+__inline__\s+S32\s+max_max\s*\(\s*S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?vmax\.s32\.s32\.s32\.max[^{}]*?\}',
"__device__ __inline__ S32   max_max                                     (S32 a, S32 b, S32 c)   { return max(max(a, b), c); }"
),
(
r'__device__\s+__inline__\s+S32\s+min_min\s*\(\s*S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?vmin\.s32\.s32\.s32\.min[^{}]*?\}',
"__device__ __inline__ S32   min_min                                     (S32 a, S32 b, S32 c)   { return min(min(a, b), c); }"
),
(
r'__device__\s+__inline__\s+U32\s+add_add\s*\(\s*U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c\s*\)\s*\{[^{}]*?vadd\.u32\.u32\.u32\.add[^{}]*?\}',
"__device__ __inline__ U32   add_add                                     (U32 a, U32 b, U32 c)   { return a + b + c; }"
),
(
r'__device__\s+__inline__\s+U32\s+add_sub\s*\(\s*U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c\s*\)\s*\{[^{}]*?vsub\.u32\.u32\.u32\.add[^{}]*?\}',
"__device__ __inline__ U32   add_sub                                     (U32 a, U32 b, U32 c)   { return a + b - c; }"
),
(
r'__device__\s+__inline__\s+S32\s+max_min\s*\(\s*S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?vmax\.s32\.s32\.s32\.min[^{}]*?\}',
"__device__ __inline__ S32   max_min                                     (S32 a, S32 b, S32 c)   { return min(max(a, b), c); }"
),
(
r'__device__\s+__inline__\s+S32\s+min_max\s*\(\s*S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?vmin\.s32\.s32\.s32\.max[^{}]*?\}',
"__device__ __inline__ S32   min_max                                     (S32 a, S32 b, S32 c)   { return max(min(a, b), c); }"
),
]
for pat, repl in extra_repls:
    s, n = re.subn(pat, repl, s)
    if n:
        print(f"OK Util video-asm fallback patched: {n}")

# Comprehensive CUDA/PTX asm helper replacement for Util.inl.
# The original file uses many PTX "video" instructions. ROCm/gfx1201 cannot
# compile them, so we replace them with portable C++ equivalents.
def rep_func(name, ret, match_args, decl_args, body):
    global s
    pat = (
        r'__device__\s+__inline__\s+' + re.escape(ret) +
        r'\s+' + re.escape(name) +
        r'\s*\(\s*' + match_args + r'\s*\)\s*\{[^{}]*?asm\s*\([^{};]*?;[^{}]*?\}'
    )
    repl = "__device__ __inline__ " + ret + "   " + name + "(" + decl_args + ")   { " + body + " }"
    s, n = re.subn(pat, repl, s, flags=re.S)
    if n:
        print(f"OK Util asm helper patched: {name} x{n}")

func_repls = [
    ("findLeadingOne", "int", r'U32\s+v', "U32 v",
     'return (v == 0u) ? -1 : (31 - __clz(v));'),

    ("add_s16lo_s16lo", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)(a & 0xffff)) + (S32)((short)(b & 0xffff));'),
    ("add_s16hi_s16lo", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)((a >> 16) & 0xffff)) + (S32)((short)(b & 0xffff));'),
    ("add_s16lo_s16hi", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)(a & 0xffff)) + (S32)((short)((b >> 16) & 0xffff));'),
    ("add_s16hi_s16hi", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)((a >> 16) & 0xffff)) + (S32)((short)((b >> 16) & 0xffff));'),

    ("sub_s16lo_s16lo", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)(a & 0xffff)) - (S32)((short)(b & 0xffff));'),
    ("sub_s16hi_s16lo", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)((a >> 16) & 0xffff)) - (S32)((short)(b & 0xffff));'),
    ("sub_s16lo_s16hi", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)(a & 0xffff)) - (S32)((short)((b >> 16) & 0xffff));'),
    ("sub_s16hi_s16hi", "S32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b",
     'return (S32)((short)((a >> 16) & 0xffff)) - (S32)((short)((b >> 16) & 0xffff));'),

    ("sub_u16lo_u16lo", "S32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'return (S32)(a & 0xffffu) - (S32)(b & 0xffffu);'),
    ("sub_u16hi_u16lo", "S32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'return (S32)((a >> 16) & 0xffffu) - (S32)(b & 0xffffu);'),
    ("sub_u16lo_u16hi", "S32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'return (S32)(a & 0xffffu) - (S32)((b >> 16) & 0xffffu);'),
    ("sub_u16hi_u16hi", "S32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'return (S32)((a >> 16) & 0xffffu) - (S32)((b >> 16) & 0xffffu);'),

    ("add_b0", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b", 'return (a & 0xffu) + b;'),
    ("add_b1", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b", 'return ((a >> 8) & 0xffu) + b;'),
    ("add_b2", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b", 'return ((a >> 16) & 0xffu) + b;'),
    ("add_b3", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b", 'return ((a >> 24) & 0xffu) + b;'),

    ("vmad_b0", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return (a & 0xffu) * b + c;'),
    ("vmad_b1", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 8) & 0xffu) * b + c;'),
    ("vmad_b2", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 16) & 0xffu) * b + c;'),
    ("vmad_b3", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 24) & 0xffu) * b + c;'),

    ("vmad_b0_b3", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return (a & 0xffu) * ((b >> 24) & 0xffu) + c;'),
    ("vmad_b1_b3", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 8) & 0xffu) * ((b >> 24) & 0xffu) + c;'),
    ("vmad_b2_b3", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 16) & 0xffu) * ((b >> 24) & 0xffu) + c;'),
    ("vmad_b3_b3", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return ((a >> 24) & 0xffu) * ((b >> 24) & 0xffu) + c;'),

    ("add_mask8", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'U32 r=0u; r |= ((a & 0xffu) + (b & 0xffu)) & 0xffu; r |= ((((a >> 8) & 0xffu) + ((b >> 8) & 0xffu)) & 0xffu) << 8; r |= ((((a >> 16) & 0xffu) + ((b >> 16) & 0xffu)) & 0xffu) << 16; r |= ((((a >> 24) & 0xffu) + ((b >> 24) & 0xffu)) & 0xffu) << 24; return r;'),
    ("sub_mask8", "U32", r'U32\s+a\s*,\s*U32\s+b', "U32 a, U32 b",
     'U32 r=0u; r |= ((a & 0xffu) - (b & 0xffu)) & 0xffu; r |= ((((a >> 8) & 0xffu) - ((b >> 8) & 0xffu)) & 0xffu) << 8; r |= ((((a >> 16) & 0xffu) - ((b >> 16) & 0xffu)) & 0xffu) << 16; r |= ((((a >> 24) & 0xffu) - ((b >> 24) & 0xffu)) & 0xffu) << 24; return r;'),

    ("max_add", "S32", r'S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c', "S32 a, S32 b, S32 c", 'return ((a > b) ? a : b) + c;'),
    ("min_add", "S32", r'S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c', "S32 a, S32 b, S32 c", 'return ((a < b) ? a : b) + c;'),
    ("sub_add", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c", 'return a - b + c;'),

    ("add_clamp_0_x", "S32", r'S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c', "S32 a, S32 b, S32 c",
     'S32 v = a + b; return (v < 0) ? 0 : ((v > c) ? c : v);'),
    ("add_clamp_b0", "S32", r'S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c', "S32 a, S32 b, S32 c",
     'S32 v = a + b + c; return (v < 0) ? 0 : ((v > 255) ? 255 : v);'),
    ("add_clamp_b2", "S32", r'S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c', "S32 a, S32 b, S32 c",
     'S32 v = a + b + c; v = (v < 0) ? 0 : ((v > 255) ? 255 : v); return v << 16;'),

    ("prmt", "U32", r'U32\s+a\s*,\s*U32\s+b\s*,\s*U32\s+c', "U32 a, U32 b, U32 c",
     'U32 r = 0u; U64 src = ((U64)b << 32) | (U64)a; for (int i = 0; i < 4; ++i) { U32 sel = (c >> (i * 4)) & 0xfu; U32 bytev = (sel < 8u) ? (U32)((src >> (sel * 8)) & 0xffu) : 0u; r |= bytev << (i * 8); } return r;'),
    ("u32lo_sext", "S32", r'U32\s+a', "U32 a", 'return (S32)((short)(a & 0xffffu));'),
    ("isetge", "U32", r'S32\s+a\s*,\s*S32\s+b', "S32 a, S32 b", 'return (a >= b) ? 1u : 0u;'),
]
for name, ret, match_args, decl_args, body in func_repls:
    rep_func(name, ret, match_args, decl_args, body)

# Overloaded slct variants need specific handling.
s, n = re.subn(
    r'__device__\s+__inline__\s+U32\s+slct\s*\(\s*U32\s+a\s*,\s*U32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?asm\s*\([^{};]*?;[^{}]*?\}',
    '__device__ __inline__ U32   slct                    (U32 a, U32 b, S32 c)   { return (c >= 0) ? a : b; }',
    s, flags=re.S)
if n: print(f"OK Util asm helper patched: slct U32 x{n}")

s, n = re.subn(
    r'__device__\s+__inline__\s+S32\s+slct\s*\(\s*S32\s+a\s*,\s*S32\s+b\s*,\s*S32\s+c\s*\)\s*\{[^{}]*?asm\s*\([^{};]*?;[^{}]*?\}',
    '__device__ __inline__ S32   slct                    (S32 a, S32 b, S32 c)   { return (c >= 0) ? a : b; }',
    s, flags=re.S)
if n: print(f"OK Util asm helper patched: slct S32 x{n}")

p.write_text(s)

bad_substrings = ["asm(", "asm volatile", "lanemask_", "cvt.", "slct.", "rcp.approx", "fma.rm", "vmax.", "vmin.", "vadd.", "vsub.", "vmad.", "prmt.", "set.ge."]
bad_lines = []
for i, line in enumerate(s.splitlines(), 1):
    if any(tok in line for tok in bad_substrings):
        bad_lines.append((i, line))
if bad_lines:
    print("FEHLER: Util.inl still contains CUDA/PTX-only asm tokens:")
    for i, line in bad_lines[:100]:
        print(f"{i}: {line}")
    if len(bad_lines) > 100:
        print(f"... {len(bad_lines)-100} more line(s)")
    raise SystemExit(1)
print(f"OK Util.inl ROCm asm cleanup changed={s != orig}")

# -------------------------------------------------------------------------
# 3) CoarseRaster precedence fixes.
# -------------------------------------------------------------------------
p = impl / "CoarseRaster.inl"
s = p.read_text()
orig = s
s = s.replace(
    "bool force = (p.deferredClear & tileX <= maxTileXInBin & tileY <= maxTileYInBin);",
    "bool force = ((p.deferredClear != 0) && (tileX <= maxTileXInBin) && (tileY <= maxTileYInBin));"
)
s = s.replace(
    "U32 res = __ballot_sync((U64)actMask, ofs >= 0 | force);",
    "U32 res = (U32)__ballot_sync((U64)actMask, ((ofs >= 0) || force));"
)
s = s.replace("? ~0ull : binSegData", "? -1 : binSegData")
p.write_text(s)
print(f"OK CoarseRaster warning fixes changed={s != orig}")

# -------------------------------------------------------------------------
# 4) TriangleSetup prepareTriangle return workaround.
# -------------------------------------------------------------------------
p = impl / "TriangleSetup.inl"
s = p.read_text()
if "NVDR_ROCM_RECONSTRUCT_RES_BUNDLE" not in s:
    pat = re.compile(r'(\n\s*bool\s+res\s*=\s*prepareTriangle\s*\([^;]*?\)\s*;)', re.S)
    matches = list(pat.finditer(s))
    if not matches:
        print("FEHLER: prepareTriangle call-site not found in TriangleSetup.inl")
        raise SystemExit(1)
    patch = r'''
    // NVDR_ROCM_RECONSTRUCT_RES_BUNDLE
    // ROCm workaround: prepareTriangle() can return false to caller although
    // area/lo/hi describe a valid front-facing triangle.
    bool nvdr_rocm_reconstructed_res_bundle =
        (area > 0) &&
        (lo.x <= hi.x && lo.y <= hi.y) &&
        (hi.x >= 0 && hi.y >= 0) &&
        (lo.x < p.widthPixelsVp && lo.y < p.heightPixelsVp);

    if (nvdr_rocm_reconstructed_res_bundle)
        res = true;
'''.rstrip()
    out = []
    last = 0
    for m in matches:
        out.append(s[last:m.end()])
        out.append("\n" + patch)
        last = m.end()
    out.append(s[last:])
    s = ''.join(out)
    p.write_text(s)
    print(f"OK TriangleSetup res reconstruction inserted after {len(matches)} call-site(s)")
else:
    print("OK TriangleSetup res reconstruction already present")

# -------------------------------------------------------------------------
# 5) FineRaster mode.
# -------------------------------------------------------------------------
p = impl / "FineRaster.inl"
s = p.read_text()

def replace_fineRasterImpl(s: str, body: str) -> str:
    sig = re.search(
        r'__device__\s+__inline__\s+void\s+fineRasterImpl\s*\(\s*const\s+CRParams\s+p\s*\)',
        s,
        flags=re.S,
    )
    if not sig:
        raise SystemExit("FEHLER: fineRasterImpl signature not found")
    open_brace = s.find("{", sig.end())
    if open_brace < 0:
        raise SystemExit("FEHLER: fineRasterImpl opening brace not found")
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
        raise SystemExit("FEHLER: fineRasterImpl closing brace not found")
    return s[:sig.start()] + s[sig.start():open_brace] + body.rstrip() + s[close_brace+1:]

if mode == "coverage_probe":
    body = r'''
{
    // NVDR_ROCM_FINERASTER_BUNDLE_WAVE64_SAFE_COVERAGE_PROBE
    // Wave64-safe rebaseline: getTriangle + coverage + separate active-mask ballots only.

    __shared__ volatile U64 s_cover8x8_lut[CR_COVER8X8_LUT_SIZE];

    cover8x8_setupLUT(s_cover8x8_lut);
    __syncthreads();

    // AMD Wave64 safety:
    // CUDA assumed each threadIdx.y row is a separate warp32. On AMD, one wave64
    // can contain y=0 and y=1 lanes together. Therefore all lanes first compute
    // the participating y=0 mask, then non-participating lanes may return.
    U64 waveActiveMask = __ballot_sync(~0ull, (threadIdx.y == 0));

    if (threadIdx.y != 0)
        return;

    CRAtomics& atomics = p.atomics[blockIdx.z];

    const S32* activeTiles  = (const S32*)p.activeTiles  + CR_MAXTILES_SQR * blockIdx.z;
    const S32* tileFirstSeg = (const S32*)p.tileFirstSeg + CR_MAXTILES_SQR * blockIdx.z;
    const S32* tileSegCount = (const S32*)p.tileSegCount + p.maxTileSegs * blockIdx.z;

    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0)
        return;

    U32* pColorProbe = (U32*)p.colorBuffer + p.strideX * p.strideY * blockIdx.z;
    U32* pDepthProbe = (U32*)p.depthBuffer + p.strideX * p.strideY * blockIdx.z;

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 10 && i < p.strideX; ++i)
        {
            pColorProbe[i] = 0u;
            pDepthProbe[i] = 0u;
        }
        pColorProbe[0] = 1u;
        pDepthProbe[0] = 0u;
    }

    __syncwarp(waveActiveMask);

    if (atomics.numActiveTiles <= 0)
        return;

    int firstTile = activeTiles[0];
    int numTiles = p.widthTiles * p.heightTiles;
    bool tileOK = (firstTile >= 0 && firstTile < numTiles);

    if (threadIdx.x == 0 && tileOK)
    {
        pColorProbe[1] = 1u;
        pDepthProbe[1] = 0u;
    }

    if (!tileOK)
        return;

    S32 segment = tileFirstSeg[firstTile];
    bool segOK = (segment >= 0 && segment < p.maxTileSegs);

    if (threadIdx.x == 0 && segOK)
    {
        pColorProbe[2] = 1u;
        pDepthProbe[2] = 0u;
    }

    if (!segOK)
        return;

    int segCount = tileSegCount[segment];

    if (threadIdx.x == 0 && segCount > 0)
    {
        pColorProbe[3] = 1u;
        pDepthProbe[3] = 0u;
    }

    int tileY = firstTile / p.widthTiles;
    int tileX = firstTile - tileY * p.widthTiles;

    S32 triIdx = -999;
    S32 dataIdx = -999;
    uint4 triHeader = make_uint4(0, 0, 0, 0);

    getTriangle(p, triIdx, dataIdx, triHeader, segment);

    bool validTri = (triIdx >= 0);

    U64 coverage = trianglePixelCoverage(p, triHeader, tileX, tileY, s_cover8x8_lut);
    bool covNZ = (coverage != 0ull);

    U64 validMask    = __ballot_sync(waveActiveMask, validTri);
    U64 coverageMask = __ballot_sync(waveActiveMask, covNZ);

    if (threadIdx.x == 0)
    {
        if (validMask != 0ull)
        {
            pColorProbe[4] = 1u;
            pDepthProbe[4] = 0u;
        }

        if (coverageMask != 0ull)
        {
            pColorProbe[5] = 1u;
            pDepthProbe[5] = 0u;
        }

        pColorProbe[6] = 1u;
        pDepthProbe[6] = 0u;

        if (segment == -1 || (segment >= 0 && segment < p.maxTileSegs))
        {
            pColorProbe[7] = 1u;
            pDepthProbe[7] = 0u;
        }

        if (segCount <= CR_TILE_SEG_SIZE)
        {
            pColorProbe[8] = 1u;
            pDepthProbe[8] = 0u;
        }

        pColorProbe[9] = 1u;
        pDepthProbe[9] = 0u;
    }
}
'''
    s = replace_fineRasterImpl(s, body)
    p.write_text(s)
    print("OK FineRaster mode=coverage_probe installed")

elif mode == "runtime":
    s, n = re.subn(
        r'\buint4\s+triHeader\s*;',
        'uint4 triHeader = make_uint4(0, 0, 0, 0); // NVDR_ROCM_TRIHEADER_ZEROINIT_BUNDLE',
        s,
    )
    p.write_text(s)
    print(f"OK FineRaster mode=runtime; triHeader zero-init replacements={n}")

else:
    print(f"FEHLER: unknown MODE={mode!r}; use coverage_probe or runtime")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# 6) Final validation.
# -------------------------------------------------------------------------
util = (impl / "Util.inl").read_text()
bad = [tok for tok in ["asm(", "asm volatile", "lanemask_", "cvt.rni", "cvt.rmi", "slct.f32", "rcp.approx", "fma.rm.f32", "vmax.", "vmin.", "vadd.", "vsub."] if tok in util]
if bad:
    print("FEHLER: Util.inl still contains bad PTX tokens:", bad)
    raise SystemExit(1)

print("=== NVDR markers ===")
for path in sorted(impl.glob("*.inl")):
    txt = path.read_text()
    for i, line in enumerate(txt.splitlines(), 1):
        if "NVDR_ROCM_" in line:
            print(f"{path}:{i}:{line}")

print("OK source patch phase complete.")
PY

echo
echo "=== Ensure CUDA compat header dir ==="
COMPAT="$ROOT/nvdiffrast_rocm_cuda_compat"
mkdir -p "$COMPAT"

if [[ ! -f "$COMPAT/cuda_runtime.h" ]]; then
  cat > "$COMPAT/cuda_runtime.h" <<'EOF'
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStream_t hipStream_t
EOF
fi

if [[ ! -f "$COMPAT/cuda_fp16.h" ]]; then
  cat > "$COMPAT/cuda_fp16.h" <<'EOF'
#pragma once
#include <hip/hip_fp16.h>
EOF
fi

# NVDR_ROCM_C10_NO_CMAKE_CONFIGURE
# nvdiffrast's csrc/common/framework.h unconditionally pulls in
# <ATen/cuda/CUDAContext.h> -> <c10/cuda/CUDAGuard.h> -> <c10/cuda/CUDAMacros.h>.
# PyTorch's ROCm wheels ship the c10/cuda headers, but NOT the CMake-generated
# c10/cuda/impl/cuda_cmake_macros.h -- it only exists in CUDA builds. Result:
#   fatal error: 'c10/cuda/impl/cuda_cmake_macros.h' file not found
#
# PyTorch provides the sanctioned escape hatch for exactly this case. From
# c10/cuda/CUDAMacros.h:
#   "We have not yet modified the AMD HIP build to generate this file so
#    we add an extra option to specifically ignore it."
#   #ifndef C10_CUDA_NO_CMAKE_CONFIGURE_FILE
#   #include <c10/cuda/impl/cuda_cmake_macros.h>
#   #endif
# So we define that macro for the host C++ objects instead of faking the header.
# CPPFLAGS is appended to the compiler flags by distutils' customize_compiler(),
# which is what builds the torch_*.cpp wrappers.
echo "=== Enable PyTorch ROCm header workaround (NVDR_ROCM_C10_NO_CMAKE_CONFIGURE) ==="
# Idempotent: amd_nvdiffrast_setup.py already exports CPPFLAGS and passes it into
# this script, so appending unconditionally would define the macro twice.
case " ${CPPFLAGS:-} " in
  *" -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE "*)
    echo "CPPFLAGS already carries the guard; leaving it unchanged."
    ;;
  *)
    export CPPFLAGS="-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE ${CPPFLAGS:-}"
    ;;
esac
echo "CPPFLAGS=$CPPFLAGS"

# Belt-and-braces fallback: should the define fail to reach a translation unit
# (e.g. a setuptools version that does not honour CPPFLAGS for C++ objects), an
# EMPTY stub in the COMPAT dir still satisfies the #include. It deliberately
# defines nothing: the generated file's only macro, C10_CUDA_BUILD_SHARED_LIBS,
# is read exclusively inside "#ifdef _WIN32" in CUDAMacros.h and is therefore
# irrelevant on Linux. Defining it here would assert a build property we cannot
# know; an empty file asserts nothing.
mkdir -p "$COMPAT/c10/cuda/impl"
if [[ ! -f "$COMPAT/c10/cuda/impl/cuda_cmake_macros.h" ]]; then
  cat > "$COMPAT/c10/cuda/impl/cuda_cmake_macros.h" <<'EOF'
#pragma once
// NVDR_ROCM_C10_NO_CMAKE_CONFIGURE (fallback)
// Intentionally empty. See CUDAMacros.h: the real, CMake-generated header only
// ever defines C10_CUDA_BUILD_SHARED_LIBS, which is consumed solely under _WIN32.
EOF
fi

echo "COMPAT=$COMPAT"

echo
echo "=== Clean and build/install ==="
rm -rf build/ dist/ ./*.egg-info
python -m pip uninstall -y nvdiffrast || true

export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1201}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CPATH="$COMPAT:$ROCM_PATH/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v

echo
echo "=== Import smoke test ==="
python - <<'PY'
import torch
import nvdiffrast.torch as dr
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("nvdiffrast import: OK")
PY

echo
echo "OK: reinstall completed. Now run ./test_min_triangle.sh from the bundle."
EOS

chmod +x "$BUNDLE_DIR/reinstall_nvdiffrast_rocm72_gfx1201.sh"

cat > "$BUNDLE_DIR/test_min_triangle.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/therock_test}"
VENV="${VENV:-$ROOT/venv}"

source "$VENV/bin/activate"

cd "$ROOT"

AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}" python - <<'PY'
import torch
import nvdiffrast.torch as dr

device = "cuda"
ctx = dr.RasterizeCudaContext(device=device)

pos = torch.tensor([[
    [-1.0, -1.0, 0.5, 1.0],
    [ 1.0, -1.0, 0.5, 1.0],
    [ 0.0,  1.0, 0.5, 1.0],
]], device=device, dtype=torch.float32)

tri = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)

rast, _ = dr.rasterize(ctx, pos, tri, resolution=[16, 16])
torch.cuda.synchronize()

ch3 = rast[0, :, :, 3].detach().cpu()

print("finite:", torch.isfinite(rast).all().item())
print("sum:", rast.sum().item())
print("covered:", (rast[..., 3] > 0).sum().item())
print("tri id max:", rast[..., 3].max().item())
print("unique tri ids:", torch.unique(rast[..., 3]).detach().cpu().tolist())
print("row0:")
print(ch3[0, :10])
print("channel3:")
print(ch3)
PY
EOS

chmod +x "$BUNDLE_DIR/test_min_triangle.sh"

cat > "$BUNDLE_DIR/inspect_patch_state.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/therock_test}"
REPO="${REPO:-$ROOT/nvdiffrast}"
cd "$REPO"

echo "=== NVDR markers ==="
grep -R -n "NVDR_ROCM_" csrc/common/hipraster/impl/*.inl || true

echo
echo "=== Util bad PTX tokens check ==="
grep -R -n "asm(\|asm volatile\|lanemask_\|cvt\.rni\|cvt\.rmi\|slct\.f32\|rcp\.approx\|fma\.rm\.f32\|vmax\.\|vmin\.\|vadd\.\|vsub\." csrc/common/hipraster/impl/Util.inl || echo "OK: no bad PTX/ASM tokens"

echo
echo "=== FineRaster key lines ==="
grep -n "trianglePixelCoverage\|executeROP\|__ballot_sync\|NVDR_ROCM" csrc/common/hipraster/impl/FineRaster.inl | head -240

echo
echo "=== TriangleSetup key lines ==="
grep -n "prepareTriangle\|RECONSTRUCT_RES\|triSubtris\\[taskIdx\\]" csrc/common/hipraster/impl/TriangleSetup.inl | head -180
EOS

chmod +x "$BUNDLE_DIR/inspect_patch_state.sh"

cat > "$BUNDLE_DIR/README_ROCM72_GFX1201.md" <<'EOS'
# nvdiffrast ROCm 7.2 / gfx1201 runtime-baseline bundle for final v52

This bundle creates/rebuilds the ROCm runtime-baseline used before applying the final v52 patch stack. It can also create `csrc/common/hipraster` from `csrc/common/cudaraster` when starting from a clean NVlabs/nvdiffrast tree.

## Diagnostic reinstall

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_final_v52_bundle
MODE=coverage_probe ./reinstall_nvdiffrast_rocm72_gfx1201.sh
./test_min_triangle.sh
```

`MODE=coverage_probe` replaces `fineRasterImpl()` with a coverage-only diagnostic probe.

Expected row0 for the probe, if it reaches FineRaster cleanly:

```text
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

## Runtime reinstall

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_final_v52_bundle
MODE=runtime ./reinstall_nvdiffrast_rocm72_gfx1201.sh
./test_min_triangle.sh
```

Runtime mode keeps normal FineRaster but applies:
- Util.inl CUDA/PTX asm cleanup for ROCm
- HIP sync-mask literal fixes
- CoarseRaster precedence fixes
- TriangleSetup prepareTriangle result reconstruction
- FineRaster triHeader zero initialization

Runtime mode may still expose FineRaster runtime traps. Use coverage_probe first.\n\nThe coverage probe in v4 is Wave64-safe: it computes a `waveActiveMask` before `threadIdx.y != 0` lanes return, then uses that mask for `__syncwarp()` and `__ballot_sync()`.

## Important

After any HSA hardware exception / illegal address / launch failure:
- start a fresh shell or at least a fresh Python process before the next test
- do not trust the current Python process after a GPU trap
EOS

echo "OK: runtime-baseline bundle for final v52 created at $BUNDLE_DIR"
echo
echo "Run:"
echo "  cd $BUNDLE_DIR"
echo "  MODE=coverage_probe ./reinstall_nvdiffrast_rocm72_gfx1201.sh"
echo "  ./test_min_triangle.sh"
