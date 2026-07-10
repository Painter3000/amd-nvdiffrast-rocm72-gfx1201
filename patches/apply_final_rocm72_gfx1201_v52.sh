#!/usr/bin/env bash
set -euo pipefail

# apply_final_rocm72_gfx1201_v52.sh
#
# FINAL ROCm 7.2 / RDNA4 gfx1201 nvdiffrast patch stack.
#
# Intended basis:
#   - clean nvdiffrast / GitHub-v36-capable tree
#   - patch_rocm_v36_wave32.sh available either in:
#       $PATCH_DIR/patch_rocm_v36_wave32.sh
#     or:
#       $REPO/patches/patch_rocm_v36_wave32.sh
#
# Applies in this order:
#   v36    FineRaster RDNA Wave32 runtime fix
#   v38    AntialiasGrad ROCm Wave32
#   v38b   GradPos guard
#   v39b   Persistent-loop guard
#   v40f   AA forward bounds guards
#   v41c   Clang evHash narrowing fix
#   v41m2  Interpolate empty-warp allsync fix
#   v45    Remove numCTA override
#   v47    Forward workBuffer native zero fix
#   v51    Backward/Grad workBuffer y-counter native zero fix
#
# Not included:
#   v48, v49, v50, v50a, v50b, v51_dual_file_device_sync_diag
# These were diagnostic / failed-control / non-canonical patches.
#
# Usage:
#   cd ~/therock_test
#   REPO=/home/oem/therock_test/nvdiffrast \
#   PATCH_DIR=/home/oem/therock_test/Patches \
#   ./apply_final_rocm72_gfx1201_v52.sh
#
# After applying:
#   clean rebuild
#   marker check
#   path probe + AA forward/backward probes

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
PATCH_DIR="${PATCH_DIR:-$HOME/therock_test/Patches}"

SRC_AA="$REPO/csrc/torch/torch_antialias.cpp"
HIP_AA="$REPO/csrc/torch/torch_antialias_hip.cpp"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_patch() {
  local patch="$1"
  [[ -f "$patch" ]] || die "missing patch: $patch"
  echo
  echo "================================================================"
  echo "APPLY: $patch"
  echo "================================================================"
  REPO="$REPO" bash "$patch"
}

[[ -d "$REPO" ]] || die "REPO does not exist: $REPO"
[[ -d "$PATCH_DIR" ]] || die "PATCH_DIR does not exist: $PATCH_DIR"

# Find v36.
V36=""
if [[ -f "$PATCH_DIR/patch_rocm_v36_wave32.sh" ]]; then
  V36="$PATCH_DIR/patch_rocm_v36_wave32.sh"
elif [[ -f "$REPO/patches/patch_rocm_v36_wave32.sh" ]]; then
  V36="$REPO/patches/patch_rocm_v36_wave32.sh"
else
  die "patch_rocm_v36_wave32.sh not found in PATCH_DIR or REPO/patches"
fi

# Canonical post-v36 patches.
PATCHES=(
  "$V36"
  "$PATCH_DIR/patch_antialias_grad_rocm_wave32_v38.sh"
  "$PATCH_DIR/patch_antialias_grad_rocm_wave32_v38b_gradpos_guard.sh"
  "$PATCH_DIR/patch_antialias_persistent_loop_v39b.sh"
  "$PATCH_DIR/patch_active_hip_antialias_fwd_bounds_v40f_patch_cu_and_hip.sh"
  "$PATCH_DIR/patch_clang_antialias_evhash_narrowing_v41c.sh"
  "$PATCH_DIR/patch_rocm_interpolate_emptywarp_allsync_fix_v41m2.sh"
  "$PATCH_DIR/patch_remove_numcta_override_v45.sh"
  "$PATCH_DIR/patch_v47_native_workbuffer_zero.sh"
  "$PATCH_DIR/patch_v51_antialias_grad_workbuffer_y_zero.sh"
)

echo "REPO      = $REPO"
echo "PATCH_DIR = $PATCH_DIR"
echo "v36 patch = $V36"

echo
echo "Checking patch files..."
for p in "${PATCHES[@]}"; do
  [[ -f "$p" ]] || die "missing patch: $p"
  echo "OK: $p"
done

echo
echo "Checking that diagnostic patches are NOT in the final patch list..."
for bad in \
  "$PATCH_DIR/patch_v48_native_workbuffer_y_zero.sh" \
  "$PATCH_DIR/patch_v49_hard_sync_antialias_forward_diag.sh" \
  "$PATCH_DIR/patch_v50_device_wide_sync_diag.sh" \
  "$PATCH_DIR/patch_v50a_sync_after_discontinuity_diag.sh" \
  "$PATCH_DIR/patch_v50b_sync_after_analysis_diag.sh" \
  "$PATCH_DIR/patch_v51_dual_file_device_sync_diag.sh"
do
  if [[ -f "$bad" ]]; then
    echo "NOTE: diagnostic patch exists but will NOT be applied: $bad"
  fi
done

echo
echo "Applying final stack..."
for p in "${PATCHES[@]}"; do
  run_patch "$p"
done

echo
echo "================================================================"
echo "POST-APPLY MARKER CHECK"
echo "================================================================"

if [[ -f "$SRC_AA" && -f "$HIP_AA" ]]; then
  echo
  echo "v47 counts:"
  grep -c "v47 FIX" "$SRC_AA" "$HIP_AA" || true

  echo
  echo "v51 counts:"
  grep -c "v51 FIX" "$SRC_AA" "$HIP_AA" || true

  v47_src=$(grep -c "v47 FIX" "$SRC_AA" || true)
  v47_hip=$(grep -c "v47 FIX" "$HIP_AA" || true)
  v51_src=$(grep -c "v51 FIX" "$SRC_AA" || true)
  v51_hip=$(grep -c "v51 FIX" "$HIP_AA" || true)

  [[ "$v47_src" == "1" && "$v47_hip" == "1" ]] || die "v47 marker must be exactly 1/1"
  [[ "$v51_src" == "1" && "$v51_hip" == "1" ]] || die "v51 marker must be exactly 1/1"

  echo
  echo "Diagnostic marker/sync scan:"
  if grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" "$SRC_AA" "$HIP_AA"; then
    die "diagnostic marker/sync found in active antialias files"
  else
    echo "OK: no active diagnostic markers/syncs"
  fi
else
  echo "WARNING: antialias source files not found for marker check:"
  echo "  $SRC_AA"
  echo "  $HIP_AA"
fi

echo
echo "FINAL PATCH STACK APPLIED."
echo
echo "Next recommended commands:"
cat <<'EOF'

# Clean rebuild:
source ~/therock_test/venv/bin/activate
cd ~/therock_test/nvdiffrast

pip uninstall -y nvdiffrast

SITE="$HOME/therock_test/venv/lib/python3.10/site-packages"
rm -rf "$SITE"/nvdiffrast
rm -rf "$SITE"/nvdiffrast-*.dist-info
rm -rf "$SITE"/__editable__*nvdiffrast*
rm -f  "$SITE"/_nvdiffrast_c*.so

rm -rf build/ dist/ ./*.egg-info
find . -name "*.o" -delete
find . -name "*.so" -delete
find . -name "*.d" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf {} +

export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++
export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation --no-cache-dir -v 2>&1 | tee /tmp/build_final_v52_clean.log

# Marker check after build:
grep -c "v47 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
grep -c "v51 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" \
  csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp || true

# Validation:
cd ~/therock_test/continuous-remeshing
AMD_SERIALIZE_KERNEL=3 TORCH_DISABLE_ADDR2LINE=1 python ./nvdiffrast_path_probe_v1.py --timeout 30

python ./aa_matrix_stat_probe.py --runs 20 \
  --shapes single,grid1,grid4,grid16 \
  --res-list 160,180,182,192,224,256 \
  --colors interp --hashes explicit \
  --label "final v52 forward AA"

AMD_SERIALIZE_KERNEL=3 TORCH_DISABLE_ADDR2LINE=1 \
python ./nvdiffrast_antialias_grid_bisect_v44b.py \
  --timeout 30 \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages call,sync,finite,diff \
  --verbose

EOF
