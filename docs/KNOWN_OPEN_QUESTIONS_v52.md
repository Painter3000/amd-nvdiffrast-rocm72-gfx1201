# Known Validation Status and Open Questions

This document tracks the current validation status and remaining open questions
for the ROCm 7.2 / RDNA4 gfx1201 nvdiffrast patch stack.

Current canonical patch stack: **final v52**.

The final v52 stack builds on the earlier v36 RDNA Wave32 FineRaster fix and
adds the later antialias, interpolate, build-hygiene, and work-buffer fixes:

```text
v36    FineRaster RDNA Wave32 runtime fix
v38    AntialiasGrad ROCm Wave32 fix
v38b   GradPos guard
v39b   Persistent-loop guard
v40f   Antialias forward bounds guards
v41c   Clang evHash narrowing fix
v41m2  Interpolate empty-warp allsync fix
v45    Remove stale numCTA override
v47    Antialias forward workBuffer torch-native zero_() fix
v51    Antialias backward/grad workBuffer y-counter torch-native zero_() fix
```

Diagnostic or non-canonical patches such as v48, v49, v50, v50a/v50b, and
v51 device-sync diagnostics are intentionally not part of the final stack.

---

## 1. v36 FineRaster RDNA Wave32 validation

v36 restored the full FineRaster runtime path and corrected the RDNA/gfx1201
row mask behavior:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

The earlier Wave64 half-row split is not used:

```cpp
const U32 nvdr_rowShift = (U32)((threadIdx.y & 1) << 5);
const U64 nvdr_rowMask  = (threadIdx.y & 1)
                         ? 0xffffffff00000000ull
                         : 0x00000000ffffffffull;
```

Validated results included:

```text
minimal 16x16 triangle:
finite: True
covered: 128
tri id max: 1.0
unique tri ids: [0.0, 1.0]
```

and the advanced 256x256 / 512x512 multi-triangle forward+backward pipeline.

This validates the FineRaster path substantially beyond the original minimal
probe, but it still does not prove every possible production row/tile layout.

---

## 2. Rasterize / interpolate gradient validation

The earlier advanced autograd test verified finite and nonzero `pos.grad`.
A later finite-difference test added stronger numerical validation for the
smooth interior rasterize/interpolate path.

Validated interior finite-difference result:

```text
test_rasterizer_gradients_v4.py --test interior
resolution: 64x64
crop: 8x8 central interior crop

crop covered: 64/64
crop tri IDs: [1.0]

max |analytic - numeric| = 0.00218090
max relative error       = 0.02177946
PASS
```

This validates that analytical gradients from the rasterize/interpolate
interior path match numerical finite differences in a smooth, fully covered
region with no silhouette, no triangle-ID changes, and no coverage-edge jumps.

Observed z gradients were zero in this specific test while w gradients were
nonzero and matched finite differences. This is not automatically a failure:
the test uses constant depth and an interior color interpolation loss, so it
does not create a strong depth-ordering objective. A separate depth-ordering or
z-sensitive finite-difference test is still needed before making broad claims
about z-gradient behavior.

---

## 3. Interpolate validation after v41m2

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

Validation status:

```text
108/108 interpolate resolution/stage cases passed
shapes: single, grid1, grid16
resolutions: 160,164,168,172,176,180,184,192,256
stages: call, sync, scalar, all
```

The interpolate path is therefore no longer considered broadly unvalidated,
although production-scale scene tests are still recommended.

---

## 4. Antialias forward validation after v47

The original antialias forward crash was traced to raw HIP memset usage on a
PyTorch-owned `work_buffer` allocation:

```cpp
hipMemsetAsync(p.workBuffer, 0, sizeof(int4), stream)
```

The final v47 fix replaces this in the HIP/ROCm path with a torch-native
operation:

```cpp
work_buffer.narrow(0, 0, 4).zero_();
```

Important build note: v47 must patch both files:

```text
csrc/torch/torch_antialias.cpp
csrc/torch/torch_antialias_hip.cpp
```

This is required because a clean rebuild can regenerate
`torch_antialias_hip.cpp` from `torch_antialias.cpp` through hipify.

Validation status:

```text
aa_matrix_stat_probe.py --runs 20
shapes: single, grid1, grid4, grid16
resolutions: 160,180,182,192,224,256
colors: interp
hashes: explicit

Result:
480/480 passed
24/24 cases with 100% success
```

This validates the final antialias forward path across the previously unstable
resolution class, including 182 and 256.

---

## 5. Antialias backward / gradient validation after v51

The remaining antialias backward/grad raw memset was:

```cpp
hipMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)
```

The final v51 fix applies the same torch-native principle as v47:

```cpp
work_buffer.narrow(0, 1, 1).zero_();
```

As with v47, v51 must patch both:

```text
csrc/torch/torch_antialias.cpp
csrc/torch/torch_antialias_hip.cpp
```

Validation status:

```text
nvdiffrast_path_probe_v1.py:
passed=14 failed_or_timeout=0 total=14
```

The successful path probe covered:

```text
import
rasterize_forward_min
interpolate_forward_min
interpolate_backward_color
rasterize_interpolate_backward_pos
grid_forward
grid_backward_pos
topology_hash
antialias_forward_single_tri
texture_forward
antialias_forward_grid
antialias_backward_color_direct
antialias_backward_pos_grid
texture_backward
```

Additional targeted antialias backward/grad matrix validation:

```text
nvdiffrast_antialias_grid_bisect_v44b.py
cells: 1,4,16
resolutions: 160,180,182,192,224,256
topology: explicit
colors: interp
pos-grads: 1
stages: call,sync,finite,diff

Result:
passed=72 failed=0 total=72
```

This targeted matrix exercises `antialias_fwd`/`antialias_grad` setup and data
paths (call, sync, finite, diff) but does not itself trigger an actual
`loss.backward()` call. A separate statistical probe closes that gap by
running the `backward_pos` stage, which does call `.backward()` and reads
back `pos.grad`, repeated over multiple runs to rule out the run-to-run
non-determinism observed elsewhere in this validation effort:

```text
final v52 AA backward_pos statistical probe
cells: 1,4,16
resolutions: 160,180,182,192,224,256
topology: explicit
colors: interp
pos-grads: 1
stage: backward_pos
runs: 20

Result:
passed=360 failed=0 total=360
18/18 cases with 100% success over 20 runs
```

This makes antialias forward and backward/grad validated for the current v52
test matrix, including a real, repeated `.backward()` exercise of the v51
workBuffer y-counter fix.

---

## 6. Texture validation status

Texture forward and backward are included in the v52 path probe:

```text
texture_forward   OK
texture_backward  OK
```

This confirms basic texture operation in the current environment. However,
texture sampling has not yet received the same broad stress coverage as
rasterize/interpolate/antialias.

Recommended future texture stress coverage:

```text
- multiple texture resolutions
- mipmapping / filtering modes where applicable
- larger batches
- nontrivial UV layouts
- gradients through texture coordinates and texture values
```

---

## 7. Current validated scope

The current final v52 validation covers:

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
- antialias backward_pos statistical probe (real `.backward()`, 20 runs,
  360/360 passed) over the same cells/resolution range
- basic texture forward and backward
```

---

## 8. Remaining open questions / recommended future stress tests

The following paths still need broader stress tests before claiming production
coverage:

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

These are not known failures in the current final v52 stack. They are simply
outside the current validation envelope.

---

## 9. Known stress-case note: offscreen covering triangle

An earlier finite-difference test variant used a very large triangle extending
outside clipspace to remove visible silhouettes. On ROCm/gfx1201 this triggered
an illegal memory access in the test environment.

That test design was replaced by `test_rasterizer_gradients_v4.py`, which uses
an in-clip triangle and computes the loss only on a safe central interior crop.

The offscreen/clipping case should therefore remain listed as an open stress
case rather than as a validated path.

---

## 10. Build and hipify caveat

A clean rebuild may regenerate:

```text
csrc/torch/torch_antialias_hip.cpp
```

from:

```text
csrc/torch/torch_antialias.cpp
```

through hipify.

For this reason, source-level fixes that must survive rebuilds should patch
both files or at least patch the CUDA-source file before hipify regeneration.

The v52 marker check must pass after every clean rebuild:

```bash
grep -c "v47 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
grep -c "v51 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp
```

Expected:

```text
v47: 1 / 1
v51: 1 / 1
```

No diagnostic markers or hard syncs should remain active:

```bash
grep -n "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" \
  csrc/torch/torch_antialias.cpp \
  csrc/torch/torch_antialias_hip.cpp || true
```

Expected: no output.

---

## 11. Licensing / redistribution

This repository includes setup scripts and patches for a project that depends on
NVlabs/nvdiffrast. Upstream nvdiffrast is under the NVIDIA Source Code License
(1-Way Commercial). Read `NOTICE.md` and `third_party/nvdiffrast/LICENSE.txt`
before public redistribution or commercial use.

Non-commercial use restrictions may apply.
