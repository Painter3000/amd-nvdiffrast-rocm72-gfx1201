# nvdiffrast ROCm 7.2 / RDNA4 gfx1201 v36 Stable Patch

## Status

This bundle documents the first validated `nvdiffrast` FineRaster runtime fix for AMD RDNA4 / gfx1201 under ROCm 7.2.

Validated environment:

```text
GPU:        AMD Radeon AI PRO R9700 / gfx1201 / RDNA4
PyTorch:    2.12.0+rocm7.2
HIP:        7.2.53211
nvdiffrast: 0.4.0
Python:     3.10.20
```

Validated results:

```text
test_min_triangle.sh:
  finite:       True
  sum:          277.5
  covered:      128
  tri id max:   1.0
  unique IDs:   [0.0, 1.0]

test_advanced_pipeline.py:
  256x256:
    finite:       True
    covered:      20962
    tri IDs:      [0.0, 1.0, 2.0, 3.0, 4.0]
    grad finite:  True
    grad nonzero: True
    grad abs max: 1333.958

  512x512:
    finite:       True
    covered:      83641
    tri IDs:      [0.0, 1.0, 2.0, 3.0, 4.0]
    grad finite:  True
    grad nonzero: True
    grad abs max: 5289.046

Final:
  ADVANCED NVDIFFRAST ROCm/RDNA PIPELINE TEST PASSED
```

## Core technical finding

The original nvdiffrast FineRaster kernel assumes CUDA-style Warp32 behavior. For RDNA/gfx1201, the correct execution model is not a Wave64 upper/lower-half split. Each `threadIdx.y` row must be treated as an independent native Wave32 execution group.

The final v36 fix uses:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

This means:

```text
threadIdx.y row 0 -> own Wave32, lanes 0..31
threadIdx.y row 1 -> own Wave32, lanes 0..31
threadIdx.y row 2 -> own Wave32, lanes 0..31
...
```

Do **not** use this incorrect Wave64 half-split model for RDNA/gfx1201:

```cpp
const U32 nvdr_rowShift = (U32)((threadIdx.y & 1) << 5);
const U64 nvdr_rowMask  = (threadIdx.y & 1)
                         ? 0xffffffff00000000ull
                         : 0x00000000ffffffffull;
```

That model was useful as an intermediate diagnostic idea, but it is not the correct RDNA/gfx1201 runtime fix.

## What v36 changes

The v36 patch adapts FineRaster synchronization and ballot logic to RDNA Wave32 semantics:

```text
scan32_value(..., nvdr_rowMask)
updateTileZMax(..., nvdr_rowMask)
executeROP(..., U64 ropMask)
__syncwarp() / __syncwarp(~0ull) -> __syncwarp(nvdr_rowMask)
__ballot_sync(~0u/~0ull, ...)   -> Wave32 row-mask ballot
ropMask                         -> exact participant mask
```

The critical rule is:

```text
A mask passed to __syncwarp() or __ballot_sync() must describe the lanes that actually participate.
```

For `executeROP()` specifically:

```cpp
bool doROP = ...;
U64 ropMask = __ballot_sync(nvdr_rowMask, doROP);

if (doROP)
    executeROP(color, depth, pColor, pDepth, ropMask);
```

## What has been validated

The following parts are validated by the current tests:

```text
- nvdiffrast import
- minimal 16x16 triangle rasterization
- FineRaster forward path
- multiple triangle IDs
- 256x256 and 512x512 tile/bin scaling
- rasterize backward/autograd sanity
- finite, nonzero gradients
```

## What has NOT yet been fully validated

Avoid overclaiming. The following still needs production testing:

```text
- huge meshes with thousands or millions of triangles
- multi-view PSHuman / continuous-remeshing workloads
- long optimization loops
- numerical finite-difference gradient comparison
- antialias / interpolate / texture paths, if used
- performance benchmarking against CPU fallback or previous ROCm builds
- stress tests at 1024x1024 or higher
```

The current result proves that the core rasterize forward and backward pipeline is functional and stable for the tested cases.

## Recommended install / rebuild commands

From a clean v4 runtime source basis:

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4
MODE=runtime ./reinstall_nvdiffrast_rocm72_gfx1201.sh
```

Apply the v36 patch:

```bash
cd ~/therock_test/nvdiffrast
bash ~/therock_test/nvdiffrast_rocm72_gfx1201_v36_stable/patch_rocm_v36_wave32.sh
```

Rebuild:

```bash
cd ~/therock_test/nvdiffrast

rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast

export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v
```

Smoke test:

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_v36_stable
./test_min_triangle.sh
```

Advanced test:

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_v36_stable
./run_test_advanced_pipeline.sh
```

## HIP crash hygiene

After any hardware trap, unspecified launch failure, illegal address, or `HSA_STATUS_ERROR_EXCEPTION`, start a fresh shell before testing again. HIP context state can be poisoned after a device-side failure.

Useful debugging environment:

```bash
export AMD_SERIALIZE_KERNEL=3
export TORCH_DISABLE_ADDR2LINE=1
```

## Suggested next validation: real project

After the v36 bundle passes both included tests, validate inside the production project:

```text
1. Activate the main continuous-remeshing / PSHuman environment.
2. Install the patched nvdiffrast build into that environment.
3. Run a short 1-view or 6-view optimization smoke test.
4. Confirm:
   - finite output
   - finite gradients
   - no hardware trap
   - stable loss decrease
   - expected masks/silhouettes
5. Only then run long optimization loops.
```

## Short summary for local AI agents

```text
Problem:
  nvdiffrast FineRaster crashed on ROCm 7.2 / gfx1201 with 0x1016 hardware exceptions.

Root cause:
  CUDA Warp32 synchronization and ballot assumptions were not correctly represented for RDNA/gfx1201.

Final fix:
  Treat every threadIdx.y row as an independent RDNA Wave32:
    nvdr_rowShift = 0u
    nvdr_rowMask  = 0x00000000ffffffffull

Do not:
  Use Wave64 upper/lower half splitting on RDNA/gfx1201.

Validated:
  16x16 minimal rasterize, 256x256 and 512x512 multi-triangle forward, and rasterize backward/autograd.
```


## License and attribution

This documentation belongs to a patch/helper package around `NVlabs/nvdiffrast`.

- `NVlabs/nvdiffrast` is copyright NVIDIA Corporation and distributed under the
  NVIDIA Source Code License (1-Way Commercial). See
  `../third_party/nvdiffrast/LICENSE.txt` in the release package.
- Earlier ROCm patch work by `tashibi/nvdiffrast-rocm-patch` is acknowledged.
  See `../third_party/tashibi-nvdiffrast-rocm-patch/LICENSE`.
- Original helper scripts and documentation in this package are MIT-licensed
  unless marked otherwise.

## Known open questions

See `KNOWN_OPEN_QUESTIONS.md`.

Short version:

```text
- v34/v36 output relationship needs more complex scene coverage.
- z/w gradient behavior has not been finite-difference checked.
- antialias / interpolate / texture paths are not yet separately validated.
- large PSHuman / continuous-remeshing workloads remain the next production test.
```
