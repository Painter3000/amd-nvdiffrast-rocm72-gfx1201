# AMD nvdiffrast ROCm 7.2 / RDNA4 gfx1201 final v52 Patch Stack

Community ROCm 7.2 / RDNA4 gfx1201 patch stack for `NVlabs/nvdiffrast`.

This repository is a patch and installer bundle for building `nvdiffrast` on
AMD RDNA4 / gfx1201 with ROCm 7.2. It is **not** a standalone replacement for
the upstream `NVlabs/nvdiffrast` source tree.

The setup flow is:

```text
clone NVlabs/nvdiffrast
→ create the ROCm/HIP runtime baseline
→ apply the final v52 patch stack
→ clean rebuild
→ verify v47/v51 markers after hipify
→ run validation tests
```

---

## Status

Current validated stack: **final v52**

Validated environment:

```text
GPU:        AMD Radeon AI PRO R9700 / gfx1201 / RDNA4
PyTorch:    2.12.0+rocm7.2
HIP:        7.2.53211
nvdiffrast: 0.4.0
Python:     3.10.x
ROCm arch:  gfx1201
```

Current validation summary:

```text
nvdiffrast_path_probe_v1.py:
  passed=14 failed_or_timeout=0 total=14

AA forward statistical matrix:
  aa_matrix_stat_probe.py --runs 20
  480/480 passed
  24/24 cases with 100% success

AA backward / gradient matrix:
  cells=1,4,16
  res=160,180,182,192,224,256
  stages=call,sync,finite,diff
  passed=72 failed=0 total=72

AA backward_pos statistical probe:
  aa_backward_pos_stat_probe_v52.py --runs 20
  cells=1,4,16, res=160,180,182,192,224,256
  passed=360 failed=0 total=360
  18/18 cases with 100% success over 20 runs
```

Validated paths include:

```text
- import and extension loading
- minimal rasterize forward
- grid rasterize forward
- rasterize/interpolate backward position gradients
- interpolate forward and backward color gradients
- smooth interior finite-difference rasterize/interpolate gradient check
- topology hash construction
- antialias forward single-triangle and grid cases
- antialias backward color direct
- antialias backward position-grid cases
- antialias forward matrix stress over critical resolutions
- antialias backward/grad matrix over cells=1,4,16 and res=160..256
- antialias backward_pos statistical probe (real .backward(), 20 runs, 360/360 passed) over cells=1,4,16 and res=160..256
- basic texture forward and backward
```

---

## Canonical final v52 patch stack

The final patch stack is applied in this order:

```text
v36    patch_rocm_v36_wave32.sh
v38    patch_antialias_grad_rocm_wave32_v38.sh
v38b   patch_antialias_grad_rocm_wave32_v38b_gradpos_guard.sh
v39b   patch_antialias_persistent_loop_v39b.sh
v40f   patch_active_hip_antialias_fwd_bounds_v40f_patch_cu_and_hip.sh
v41c   patch_clang_antialias_evhash_narrowing_v41c.sh
v41m2  patch_rocm_interpolate_emptywarp_allsync_fix_v41m2.sh
v45    patch_remove_numcta_override_v45.sh
v47    patch_v47_native_workbuffer_zero.sh
v51    patch_v51_antialias_grad_workbuffer_y_zero.sh
```

Use the wrapper:

```bash
bash patches/apply_final_rocm72_gfx1201_v52.sh
```

Diagnostic or failed-control patches are intentionally **not** part of the final
stack:

```text
v48
v49
v50
v50a / v50b
v51 device-sync diagnostics
```

---

## Core technical findings

### v36: FineRaster RDNA Wave32 fix

The original `nvdiffrast` FineRaster kernel assumes CUDA-style Warp32 behavior.
For RDNA4 / gfx1201, the correct runtime model is not a Wave64 upper/lower-half
split. Each `threadIdx.y` row must be treated as an independent native Wave32
execution group.

The final v36 FineRaster fix uses:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

Do **not** use this incorrect Wave64 half-split model for RDNA/gfx1201:

```cpp
const U32 nvdr_rowShift = (U32)((threadIdx.y & 1) << 5);
const U64 nvdr_rowMask  = (threadIdx.y & 1)
                         ? 0xffffffff00000000ull
                         : 0x00000000ffffffffull;
```

The critical rule is:

```text
A mask passed to __syncwarp() or __ballot_sync() must describe the lanes that
actually participate.
```

### v41m2: Interpolate empty-warp fix

The interpolate path previously had a crash pattern at certain resolutions,
especially around cases where `width % 8 == 4`.

The final v41m2 fix disables the problematic ROCm/HIP empty-warp
`__all_sync` shortcut while leaving the CUDA path unchanged:

```cpp
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // v41m2 ROCm/RDNA fix:
    // Do not use the full-mask empty-warp shortcut after edge lanes may have returned.
    // The normal triValid=false path below still writes zero output for live lanes.
    if (false)
#else
    if (__all_sync(0xffffffffffffffffull, !triValid))
#endif
```

### v47: Antialias forward workBuffer fix

The antialias forward crash was traced to raw HIP memset usage on a PyTorch-owned
`work_buffer` allocation:

```cpp
hipMemsetAsync(p.workBuffer, 0, sizeof(int4), stream)
```

The final v47 fix replaces this in the ROCm path with a torch-native operation:

```cpp
work_buffer.narrow(0, 0, 4).zero_();
```

### v51: Antialias backward / grad workBuffer fix

The remaining antialias backward/grad raw memset was:

```cpp
hipMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)
```

The final v51 fix applies the same torch-native principle:

```cpp
work_buffer.narrow(0, 1, 1).zero_();
```

### Important hipify / rebuild caveat

A clean rebuild can regenerate:

```text
csrc/torch/torch_antialias_hip.cpp
```

from:

```text
csrc/torch/torch_antialias.cpp
```

through hipify.

Therefore v47 and v51 must patch both files:

```text
csrc/torch/torch_antialias.cpp
csrc/torch/torch_antialias_hip.cpp
```

Always verify after a clean rebuild:

```bash
grep -c "v47 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
grep -c "v51 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
```

Expected:

```text
v47: 1 / 1
v51: 1 / 1
```

No diagnostic markers or hard-sync experiments should remain active:

```bash
grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" \
  csrc/torch/torch_antialias.cpp \
  csrc/torch/torch_antialias_hip.cpp || true
```

Expected: no output.

---

## Installation

### Recommended setup command

Do **not** install this repository directly with `pip install`.

This repository is a patch/installer bundle. It clones and patches upstream
`NVlabs/nvdiffrast`.

```bash
git clone https://github.com/Painter3000/amd-nvdiffrast-rocm72-gfx1201.git
cd amd-nvdiffrast-rocm72-gfx1201

python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick
```

For a stronger release check:

```bash
python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation full
```

Validation levels:

```text
none    build only
quick   marker check + v52 validation wrapper
full    quick + forward AA statistical probe
stress  full + additional stress wrappers
```

### Manual patch-stack flow

```bash
git clone https://github.com/NVlabs/nvdiffrast.git ~/therock_test/nvdiffrast

cd ~/therock_test/amd-nvdiffrast-rocm72-gfx1201

REPO=~/therock_test/nvdiffrast \
PATCH_DIR=~/therock_test/amd-nvdiffrast-rocm72-gfx1201/patches \
bash patches/apply_final_rocm72_gfx1201_v52.sh
```

Then clean rebuild:

```bash
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

python -m pip install . --no-build-isolation --no-cache-dir -v
```

Note about CPATH when following the manual flow:

The `nvdiffrast_rocm_cuda_compat` directory referenced in `CPATH` is created by
the automated generator (`scripts/nvdiffrast_rocm72_bundle_v4_generator.sh`) when
you use the recommended setup flow. If you apply the patches and rebuild
manually without running the generator, you will likely see missing-header
errors unless you create or populate that compatibility directory yourself.

Options when following the manual path:
- Run the generator instead (recommended) to produce `nvdiffrast_rocm_cuda_compat`.
- Manually create `~/therock_test/nvdiffrast_rocm_cuda_compat` and copy the
  required compatibility headers into it (these are small helper headers used
  to compile against ROCm/HIP).
- Adjust `CPATH` to point at whatever headers you have prepared.

---

## Validation

Quick validation:

```bash
cd tests
./run_v52_validation.sh
```

Marker-only validation:

```bash
REPO=~/therock_test/nvdiffrast ./tests/test_v52_marker_integrity.sh
```

Forward AA statistical validation:

```bash
cd tests

python ./aa_matrix_stat_probe.py --runs 20 \
  --shapes single,grid1,grid4,grid16 \
  --res-list 160,180,182,192,224,256 \
  --colors interp \
  --hashes explicit \
  --label "final v52 forward AA"
```

Backward / gradient AA validation:

```bash
cd tests

AMD_SERIALIZE_KERNEL=3 TORCH_DISABLE_ADDR2LINE=1 \
python ./test_antialias_backward_matrix_v52.py \
  --timeout 30 \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages call,sync,finite,diff \
  --verbose
```

Backward position-gradient statistical validation:

```bash
cd tests

AMD_SERIALIZE_KERNEL=3 TORCH_DISABLE_ADDR2LINE=1 \
python ./aa_backward_pos_stat_probe_v52.py --runs 20 \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages backward_pos \
  --probe-script ./test_antialias_backward_matrix_v52.py \
  --label "final v52 AA backward_pos statistical probe"
```

**Note:** The preceding backward/grad block tests only setup/data paths
(call, sync, finite, diff) without triggering a real `.backward()` call.
This backward_pos block is the only test that actually calls `loss.backward()`
and reads back `pos.grad`, repeated over 20 runs to exclude run-to-run
non-determinism.

---

## HIP crash hygiene

After any hardware trap, unspecified launch failure, illegal address, or
`HSA_STATUS_ERROR_EXCEPTION`, start a fresh shell or at least a fresh Python
process before testing again. HIP context state can be poisoned after a
device-side failure.

Useful debugging environment:

```bash
export AMD_SERIALIZE_KERNEL=3
export TORCH_DISABLE_ADDR2LINE=1
```

---

## Remaining open questions

See [`docs/KNOWN_OPEN_QUESTIONS_v52.md`](docs/KNOWN_OPEN_QUESTIONS_v52.md).

Current short version:

```text
- very large production meshes
- long PSHuman / continuous-remeshing optimization loops
- clipping / offscreen geometry stress cases
- depth-ordering / z-sensitive finite-difference tests
- texture stress beyond basic forward/backward
- performance benchmarking versus CUDA and/or native PyTorch paths
- multi-batch / multi-view real-world workloads
- very high resolutions beyond the current 160..256 AA validation matrix
```

These are not known failures in the current final v52 stack. They are outside
the current validation envelope.

---

## License and attribution

This repository contains setup scripts, patches, and documentation around
`NVlabs/nvdiffrast`.

- `NVlabs/nvdiffrast` is copyright NVIDIA Corporation and distributed under the
  NVIDIA Source Code License. See `third_party/nvdiffrast/LICENSE.txt`.
- Non-commercial use restrictions may apply; read the upstream NVIDIA Source
  Code License before redistribution or commercial use.
- Earlier ROCm patch work by `tashibi/nvdiffrast-rocm-patch` is acknowledged.
  See `third_party/tashibi-nvdiffrast-rocm-patch/LICENSE`.
- Original helper scripts and documentation in this repository are MIT-licensed
  unless marked otherwise.

---

## Short summary for local AI agents

```text
Project:
  Community ROCm 7.2 / RDNA4 gfx1201 patch stack for NVlabs/nvdiffrast.

Do not:
  Treat this repository as the nvdiffrast source tree.
  pip-install this repository directly.
  Apply v48/v49/v50 diagnostic patches as final fixes.

Correct flow:
  Clone upstream NVlabs/nvdiffrast.
  Generate ROCm/HIP runtime baseline.
  Apply final v52 patch stack.
  Clean rebuild.
  Verify v47/v51 markers after hipify.
  Run v52 validation tests.

Key fixes:
  v36: FineRaster RDNA Wave32 row-mask fix.
  v41m2: Interpolate empty-warp allsync ROCm fix.
  v47: Antialias forward workBuffer torch-native zero_().
  v51: Antialias backward/grad workBuffer y-counter torch-native zero_().

Validated:
  Path probe 14/14.
  AA forward statistical matrix 480/480.
  AA backward/grad matrix 72/72.
  AA backward_pos statistical probe 360/360 (real .backward(), 20 runs).
```
