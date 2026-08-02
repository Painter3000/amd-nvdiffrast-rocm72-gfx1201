# AMD nvdiffrast ROCm 7.2 / RDNA4 gfx1201 — final v52 patch stack

<!-- PAINTER3000_STATUS_BLOCK_START -->
## Repository snapshot

- **Repository type:** Community ROCm/HIP patch, build, and validation bundle for upstream `NVlabs/nvdiffrast`.
- **Target GPU:** AMD Radeon AI PRO R9700.
- **Target architecture:** RDNA4 / `gfx1201`.
- **Target stack:** Ubuntu 24.04, ROCm 7.2, Python 3.12, PyTorch `2.13.0+rocm7.2`, HIP `7.2.53211`.
- **Validation status:** Fresh-host quick validation **110/110 PASS**; downstream `Profactor/continuous-remeshing` **100/100 steps PASS**.
- **Upstream base:** `NVlabs/nvdiffrast` `0.4.0`; final v52 patch stack.
<!-- PAINTER3000_STATUS_BLOCK_END -->


Community patch, build and validation bundle for running [`NVlabs/nvdiffrast`](https://github.com/NVlabs/nvdiffrast) on AMD RDNA4 / `gfx1201` with ROCm 7.2.

> [!IMPORTANT]
> This repository is **not** a standalone fork or replacement for the upstream nvdiffrast source tree.  
> It clones upstream nvdiffrast, creates a ROCm/HIP-compatible runtime baseline, applies the final v52 patch stack, performs a clean rebuild and runs validation.

The supported setup flow is:

```text
clone upstream nvdiffrast
→ generate ROCm/HIP runtime baseline
→ initial baseline build
→ apply final v52 patch stack
→ clean final rebuild
→ verify patch markers after hipify
→ run validation
```

---

## Status

**Current validated stack:** final v52  
**Fresh-host validation:** passed on 2026-07-11  
**Fresh-host quick-validation result:** **110 / 110 checks passed**  
**Downstream end-to-end validation:** passed — `Profactor/continuous-remeshing`, **100 / 100 steps**

### Fresh-host validated environment

| Component | Validated value |
|---|---|
| Operating system | Ubuntu 24.04 |
| GPU | AMD Radeon AI PRO R9700 |
| Architecture | RDNA4 / `gfx1201` |
| Python | 3.12.3 |
| PyTorch | `2.13.0+rocm7.2` |
| HIP reported by PyTorch | `7.2.53211` |
| Triton | `triton-rocm 3.7.1` |
| NumPy | `2.5.1` |
| Ninja | `1.13.0` |
| nvdiffrast | `0.4.0` |
| ROCm path | `/opt/rocm` |

PyTorch reported the generic device name `AMD Radeon Graphics`; the build still targeted the correct GPU explicitly through:

```text
PYTORCH_ROCM_ARCH=gfx1201
```

### Fresh-host evidence

- [Fresh-host validation report](docs/validation/fresh-host-ubuntu24.04-rocm72-gfx1201-v52.md)
- [Complete raw installation and validation log](docs/validation/logs/fresh-host-ubuntu24.04-rocm72-gfx1201-v52-full.log)

### Fresh-host quick-validation summary

| Validation group | Passed | Failed |
|---|---:|---:|
| Path probe | 14 | 0 |
| Antialias forward matrix | 24 | 0 |
| Antialias backward/gradient matrix | 72 | 0 |
| **Total** | **110** | **0** |

### Downstream end-to-end validation

The quick suite covers fixed resolutions and fixed topologies. A second,
independent validation runs a real differentiable-rendering workload in which the
mesh topology changes on every step:

| Metric | Value |
|---|---|
| Project | `Profactor/continuous-remeshing`, unmodified `example.py` |
| Steps completed | **100 / 100** |
| Final loss | `0.00829143` |
| Final vertices / faces | `5814` / `11624` |
| Throughput | ≈ 24.27 it/s |
| GPU architecture | `gfx1201`, 32 CUs, 29.86 GB |

- [Downstream end-to-end validation report](docs/validation/downstream-continuous-remeshing-gfx1201.md)

These figures act as a **regression anchor**: future changes to the patch stack
should keep the final loss, vertex count, face count and visual result in the
same range. They also make it possible to *exclude* nvdiffrast as a cause when a
downstream project misbehaves.

### Extended development validation

These extended tests were performed separately from the fresh-host quick run:

```text
AA forward statistical matrix:
  aa_matrix_stat_probe.py --runs 20
  480 / 480 passed
  24 / 24 cases with 100% success

AA backward_pos statistical probe:
  aa_backward_pos_stat_probe_v52.py --runs 20
  cells=1,4,16
  resolutions=160,180,182,192,224,256
  360 / 360 passed
  18 / 18 cases with 100% success
```

Validated functionality includes:

- extension import and loading,
- minimal and grid rasterization forward,
- rasterize/interpolate position gradients,
- interpolate color gradients,
- finite-difference gradient checks,
- topology-hash construction,
- antialias forward for single-triangle and grid cases,
- antialias backward for color and position,
- texture forward and backward,
- repeated statistical antialias validation over critical resolutions.

---

## What the automated installer does

`amd_nvdiffrast_setup.py` performs the complete supported process:

1. Uses or clones a clean upstream `NVlabs/nvdiffrast` source tree.
2. Verifies that the selected virtual environment contains a working ROCm-enabled PyTorch.
3. Generates the v52 ROCm runtime-baseline bundle.
4. Applies baseline compatibility fixes required before the first hipify/build.
5. Performs an initial baseline build and import smoke test.
6. Applies the canonical final v52 patch stack.
7. Removes old build artifacts and performs a second, clean final build.
8. Verifies v47 and v51 markers in both source and hipified files.
9. Rejects active diagnostic patches or hard synchronization experiments.
10. Runs the selected validation level.

### Why there are two builds

The two builds have different purposes:

```text
Build 1:
  creates and verifies the ROCm/HIP runtime baseline,
  generates hipified files,
  and proves that the upstream-derived baseline can compile and import.

Build 2:
  recompiles the fully patched final v52 source tree from scratch,
  prevents stale object reuse,
  verifies hipify regeneration,
  and produces the final installed extension.
```

Without the second clean build, the installed shared library could still represent the baseline rather than the complete final patch stack.

---

## Prerequisites

The setup script does **not** install ROCm, create the virtual environment or install PyTorch.

Required before running it:

- Ubuntu 24.04 or a compatible Linux distribution,
- ROCm 7.2 installed under `/opt/rocm`, or a custom path passed with `--rocm-path`,
- an AMD RDNA4 / `gfx1201` GPU with a working amdgpu/KFD stack,
- the current user in the appropriate `video` and `render` groups,
- Python 3.12,
- Git and standard native build tools,
- a working ROCm-enabled PyTorch installation.

Example Ubuntu packages:

```bash
sudo apt update
sudo apt install -y \
  git \
  build-essential \
  python3.12 \
  python3.12-venv \
  python3-dev \
  ninja-build
```

ROCm itself must already be installed and functional.

---

## Quick start

### 1. Create the virtual environment

```bash
mkdir -p ~/therock_test
python3.12 -m venv ~/therock_test/venv
source ~/therock_test/venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja numpy
```

### 2. Install ROCm PyTorch

Exact validated major/minor release:

```bash
python -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

An unpinned installation from the ROCm 7.2 index may install a newer release later; that newer combination is not covered by the validation report above.

Verify PyTorch before continuing:

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA API available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is not working")

print("Device:", torch.cuda.get_device_name(0))
PY
```

Expected for the validated setup:

```text
Torch: 2.13.0+rocm7.2
HIP: 7.2.53211
CUDA API available: True
```

PyTorch uses the `torch.cuda` API for both CUDA and ROCm backends; `True` is therefore expected on a working ROCm installation.

### 3. Clone this installer repository

```bash
cd ~/therock_test
git clone https://github.com/Painter3000/amd-nvdiffrast-rocm72-gfx1201.git
cd amd-nvdiffrast-rocm72-gfx1201
```

Do **not** run `pip install .` in this repository. It is a patch and installer bundle, not the upstream nvdiffrast package.

### 4. Run the complete setup

Recommended fresh-host command:

```bash
python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick
```

The default nvdiffrast checkout will be created at:

```text
~/therock_test/nvdiffrast
```

The generated runtime bundle will be created at:

```text
~/therock_test/nvdiffrast_rocm72_gfx1201_final_v52_bundle
```

### Stronger validation

```bash
python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation full
```

### Validation levels

| Level | Behavior |
|---|---|
| `none` | Build only |
| `quick` | Marker integrity plus the v52 validation wrapper |
| `full` | Quick validation plus the 20-run forward-AA statistical matrix |
| `stress` | Full validation plus the additional stress wrappers |

`--skip-tests` is an alias for `--validation none`.

### Other useful options

```text
--repo PATH          existing or target nvdiffrast checkout
--bundle-dir PATH    generated runtime bundle location
--repo-url URL       custom upstream nvdiffrast repository
--branch NAME        optional upstream branch or tag
--max-jobs N         native build parallelism; default 1
--skip-clone         require an already existing nvdiffrast checkout
```

---

## Fresh-host compatibility fixes discovered during validation

The clean Ubuntu 24.04 / Python 3.12 installation exposed several prerequisites that had previously existed only implicitly on the development machine.

The generator now applies these before the initial baseline build:

| Fix | Purpose |
|---|---|
| `NVDR_ROCM_AA_BALLOT64` | Uses a 64-bit HIP ballot mask in `antialias.cu` |
| `NVDR_ROCM_INTERP_ALLSYNC64` | Uses a 64-bit HIP `all_sync` mask in `interpolate.cu` |
| `NVDR_ROCM_FRCP_RZ` | Provides ROCm-compatible round-toward-zero reciprocal behavior |
| `evHashElements` narrowing fix | Avoids Clang/C++20 unsigned-to-signed braced-initializer narrowing |
| `NVDR_ROCM_TEXTURE_NO_TRANSITIVE_FRAMEWORK` | Prevents CUDA and HIP PyTorch headers from entering the same texture wrapper translation unit |
| `C10_CUDA_NO_CMAKE_CONFIGURE_FILE` | Avoids a missing PyTorch CMake-generated CUDA header in ROCm wheels |
| ROCm-safe rasterizer helpers | Replaces CUDA/PTX-only helper paths |
| TriangleSetup reconstruction | Restores required result values after HIP conversion |
| FineRaster initialization | Ensures deterministic baseline initialization |

The final patch wrapper also executes every patch from the nvdiffrast repository directory, because several patch scripts use paths relative to the upstream source tree.

---

## Canonical final v52 patch stack

The final stack is applied in this order:

| Version | Script | Purpose |
|---|---|---|
| v36 | `patch_rocm_v36_wave32.sh` | FineRaster RDNA Wave32 runtime fix |
| v38 | `patch_antialias_grad_rocm_wave32_v38.sh` | AntialiasGrad ROCm Wave32 fix |
| v38b | `patch_antialias_grad_rocm_wave32_v38b_gradpos_guard.sh` | GradPos guard |
| v39b | `patch_antialias_persistent_loop_v39b.sh` | Persistent-loop guard |
| v40f | `patch_active_hip_antialias_fwd_bounds_v40f_patch_cu_and_hip.sh` | Antialias forward bounds guards |
| v41c | `patch_clang_antialias_evhash_narrowing_v41c.sh` | Clang `evHash` narrowing fix |
| v41m2 | `patch_rocm_interpolate_emptywarp_allsync_fix_v41m2.sh` | Interpolate empty-warp `all_sync` fix |
| v45 | `patch_remove_numcta_override_v45.sh` | Remove or verify absence of the debug `numCTA = 1` override |
| v47 | `patch_v47_native_workbuffer_zero.sh` | Forward work-buffer native zero initialization |
| v51 | `patch_v51_antialias_grad_workbuffer_y_zero.sh` | Backward/Grad work-buffer Y-counter native zero initialization |

Diagnostic and failed-control patches are intentionally excluded:

```text
v48
v49
v50
v50a
v50b
v51 device-wide synchronization diagnostics
```

Do not reintroduce them as final fixes.

### v45 on a clean upstream tree

Current upstream nvdiffrast does not contain the old manual debug line:

```cpp
numCTA = 1;
```

Therefore v45 normally reports the source as **already clean** and verifies that the analysis kernel uses the full:

```text
numCTA * numSM
```

launch grid. On an older manually modified development tree, it removes the override.

---

## Core technical findings

### v36: FineRaster RDNA Wave32 execution model

The original FineRaster code assumes CUDA Warp32 behavior.

For RDNA4 / `gfx1201`, each `threadIdx.y` row must be handled as its own native Wave32 execution group:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

Do not use a synthetic Wave64 upper/lower-half split:

```cpp
const U32 nvdr_rowShift = (U32)((threadIdx.y & 1) << 5);
const U64 nvdr_rowMask  = (threadIdx.y & 1)
                       ? 0xffffffff00000000ull
                       : 0x00000000ffffffffull;
```

The critical rule is:

```text
A mask passed to __syncwarp() or __ballot_sync()
must describe the lanes that actually participate.
```

### v41m2: Interpolate empty-warp fix

The interpolate path showed resolution-dependent failures, especially around edge-lane patterns such as `width % 8 == 4`.

The final ROCm path avoids the unsafe full-mask empty-warp shortcut after edge lanes may already have returned, while the CUDA path remains unchanged.

### v47: Antialias forward work-buffer initialization

The forward failure was traced to raw asynchronous HIP memset on a PyTorch-owned allocation:

```cpp
hipMemsetAsync(p.workBuffer, 0, sizeof(int4), stream)
```

The final ROCm path uses a PyTorch-native operation:

```cpp
work_buffer.narrow(0, 0, 4).zero_();
```

### v51: Antialias backward/gradient work-buffer initialization

The remaining raw backward/gradient memset was:

```cpp
hipMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)
```

The final fix applies the same PyTorch-native principle:

```cpp
work_buffer.narrow(0, 1, 1).zero_();
```

---

## Hipify and rebuild integrity

A clean build can regenerate:

```text
csrc/torch/torch_antialias_hip.cpp
```

from:

```text
csrc/torch/torch_antialias.cpp
```

Therefore fixes that must survive hipify need to be present in the source file and verified in the generated HIP file.

After the final clean rebuild, the expected marker counts are:

```bash
cd ~/therock_test/nvdiffrast

grep -c "v47 FIX" \
  csrc/torch/torch_antialias.cpp \
  csrc/torch/torch_antialias_hip.cpp

grep -c "v51 FIX" \
  csrc/torch/torch_antialias.cpp \
  csrc/torch/torch_antialias_hip.cpp
```

Expected:

```text
v47: 1 / 1
v51: 1 / 1
```

No diagnostic markers or hard synchronization experiments should remain active:

```bash
grep -n \
  "v46\|v48\|v49\|v50\|hipStreamSynchronize\|hipDeviceSynchronize\|cudaStreamSynchronize\|cudaDeviceSynchronize" \
  csrc/torch/torch_antialias.cpp \
  csrc/torch/torch_antialias_hip.cpp || true
```

Expected: no output.

---

## Validation

Activate the validated environment first:

```bash
source ~/therock_test/venv/bin/activate
cd ~/therock_test/amd-nvdiffrast-rocm72-gfx1201
```

### Complete quick suite

```bash
REPO=~/therock_test/nvdiffrast \
./tests/run_v52_validation.sh
```

### Marker-only validation

```bash
REPO=~/therock_test/nvdiffrast \
./tests/test_v52_marker_integrity.sh
```

### Forward-AA statistical validation

```bash
cd tests

python ./aa_matrix_stat_probe.py \
  --runs 20 \
  --shapes single,grid1,grid4,grid16 \
  --res-list 160,180,182,192,224,256 \
  --colors interp \
  --hashes explicit \
  --label "final v52 forward AA"
```

### Backward/gradient matrix

```bash
cd tests

AMD_SERIALIZE_KERNEL=3 \
TORCH_DISABLE_ADDR2LINE=1 \
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

### Real backward position-gradient statistical validation

```bash
cd tests

AMD_SERIALIZE_KERNEL=3 \
TORCH_DISABLE_ADDR2LINE=1 \
python ./aa_backward_pos_stat_probe_v52.py \
  --runs 20 \
  --cells-list 1,4,16 \
  --res-list 160,180,182,192,224,256 \
  --topos explicit \
  --colors interp \
  --pos-grads 1 \
  --stages backward_pos \
  --probe-script ./test_antialias_backward_matrix_v52.py \
  --label "final v52 AA backward_pos statistical probe"
```

The setup/data matrix stages `call`, `sync`, `finite` and `diff` do not by themselves represent a real `.backward()` call. The `backward_pos` statistical probe explicitly calls `loss.backward()` and reads `pos.grad`.

---

## Advanced manual flow

> [!WARNING]
> Do not apply only the final patch stack directly to a fresh upstream clone.  
> A fresh tree first requires the generated ROCm runtime baseline and its initial build.

The automated installer is the supported route. The following is mainly for debugging or development.

### 1. Define paths

```bash
export ROOT="$HOME/therock_test"
export INSTALLER="$ROOT/amd-nvdiffrast-rocm72-gfx1201"
export REPO="$ROOT/nvdiffrast"
export VENV="$ROOT/venv"
export BUNDLE_DIR="$ROOT/nvdiffrast_rocm72_gfx1201_final_v52_bundle"
export ROCM_PATH="/opt/rocm"
```

### 2. Clone upstream if needed

```bash
git clone https://github.com/NVlabs/nvdiffrast.git "$REPO"
```

### 3. Generate the runtime-baseline bundle

```bash
BUNDLE_DIR="$BUNDLE_DIR" \
bash "$INSTALLER/scripts/nvdiffrast_rocm72_bundle_v52_generator.sh"
```

### 4. Apply the baseline and perform the initial build

```bash
ROOT="$ROOT" \
REPO="$REPO" \
VENV="$VENV" \
BUNDLE_DIR="$BUNDLE_DIR" \
ROCM_PATH="$ROCM_PATH" \
MODE=runtime \
PYTORCH_ROCM_ARCH=gfx1201 \
FORCE_CUDA=1 \
MAX_JOBS=1 \
bash "$BUNDLE_DIR/reinstall_nvdiffrast_rocm72_gfx1201.sh"
```

### 5. Apply the canonical final stack

```bash
REPO="$REPO" \
PATCH_DIR="$INSTALLER/patches" \
bash "$INSTALLER/patches/apply_final_rocm72_gfx1201_v52.sh"
```

The wrapper changes into `$REPO` for each patch invocation, because most patch scripts use paths relative to the nvdiffrast source tree.

### 6. Perform the clean final rebuild

```bash
source "$VENV/bin/activate"
cd "$REPO"

python -m pip uninstall -y nvdiffrast || true

SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

rm -rf "$SITE"/nvdiffrast
rm -rf "$SITE"/nvdiffrast-*.dist-info
rm -rf "$SITE"/__editable__*nvdiffrast*
rm -f  "$SITE"/_nvdiffrast_c*.so

rm -rf build/ dist/ ./*.egg-info
find . -name "*.o" -delete
find . -name "*.so" -delete
find . -name "*.d" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf {} +

export CC="$ROCM_PATH/llvm/bin/clang"
export CXX="$ROCM_PATH/llvm/bin/clang++"
export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$ROOT/nvdiffrast_rocm_cuda_compat:$ROCM_PATH/include/hipsparse:${CPATH:-}"
export CPPFLAGS="-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE ${CPPFLAGS:-}"

python -m pip install . \
  --no-build-isolation \
  --no-cache-dir \
  -v
```

---

## Troubleshooting

### `cuda_cmake_macros.h` not found

Symptom:

```text
fatal error: 'c10/cuda/impl/cuda_cmake_macros.h' file not found
```

Required host-wrapper flag:

```bash
export CPPFLAGS="-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE ${CPPFLAGS:-}"
```

The automated installer sets this for both builds.

### CUDA and HIP PyTorch header redefinitions

Symptoms include redefinitions from both:

```text
c10/cuda/...
c10/hip/...
ATen/cuda/...
ATen/hip/...
```

The v52 baseline removes the unnecessary transitive `framework.h` include from `texture.h`. A stale or manually regenerated tree may need a clean restart from upstream.

### `FineRaster.inl` not found while applying v36

The final stack must execute patches from the nvdiffrast repository directory.

Use the current wrapper:

```bash
REPO="$REPO" \
PATCH_DIR="$INSTALLER/patches" \
bash "$INSTALLER/patches/apply_final_rocm72_gfx1201_v52.sh"
```

Older copies of the wrapper did not change into `$REPO`.

### PyTorch reports `AMD Radeon Graphics`

A generic device string is not by itself a failure. `torch.cuda.get_device_name(0)`
may report the card generically; the architecture can be confirmed independently:

```python
import torch
p = torch.cuda.get_device_properties(0)
print(p.gcnArchName)            # gfx1201
print(p.multi_processor_count)  # 32 on the AI PRO R9700
```

Also confirm:

```text
torch.cuda.is_available() == True
PYTORCH_ROCM_ARCH=gfx1201
```

The compiler command should contain:

```text
--offload-arch=gfx1201
```

### HIP context after a hardware fault

After a hardware trap, illegal address, unspecified launch failure or `HSA_STATUS_ERROR_EXCEPTION`, use a fresh Python process. Preferably start a fresh shell before repeating validation.

Useful diagnostic environment:

```bash
export AMD_SERIALIZE_KERNEL=3
export TORCH_DISABLE_ADDR2LINE=1
```

---

## Known non-fatal build warnings

The validated build may emit warnings such as:

- `-lineinfo` unused by host Clang,
- ignored cleanup return values from `hipFree` or `hipHostFree`,
- unused variables in upstream-derived or generated wrappers,
- setuptools manifest exclusion warnings,
- legacy `cp -n` portability warnings in some backup paths.

These warnings did not prevent compilation, linking, installation or validation in the documented fresh-host run.

---

## Repository layout

```text
amd_nvdiffrast_setup.py
    complete automated clone, baseline, patch, rebuild and validation flow

scripts/
    runtime-baseline bundle generator

patches/
    canonical final v52 patches and patch-stack wrapper

tests/
    marker, path, forward, backward and stress validation tools

docs/
    technical notes and open questions

docs/validation/
    fresh-host installation report and raw log
    downstream end-to-end report (continuous-remeshing)

docs/validation/downstream/
    captured environment, dependency freeze, run log and result preview
    for the downstream end-to-end test

third_party/
    upstream licenses and attribution material
```

---

## Remaining validation scope

See [docs/KNOWN_OPEN_QUESTIONS_v52.md](docs/KNOWN_OPEN_QUESTIONS_v52.md).

Important areas outside the current validation envelope include:

- **a cross-vendor numerical reference.** The downstream test anchors the result
  against itself, not against CUDA. A run of the same example with the same seed
  on an NVIDIA GPU would confirm that gradient scaling matches. A reference
  `FINAL_LOSS` from a CUDA machine would be a welcome contribution;
- very large production meshes,
- longer optimization loops than the 100-step downstream run, and PSHuman-scale
  pipelines,
- clipping and offscreen geometry stress cases,
- depth-ordering and Z-sensitive finite-difference tests,
- texture stress beyond the current basic forward/backward coverage,
- performance benchmarking against CUDA or native PyTorch paths,
- multi-batch and multi-view production workloads,
- resolutions substantially above the current 160–256 antialias matrix,
- GPUs other than `gfx1201`. RDNA3 (`gfx1100`) uses the same native Wave32
  execution model and is expected to work, but has not been tested here.

These are not documented failures of final v52. They remain additional validation targets.

---

## Reproducibility notes

For release-quality validation artifacts, record the exact revisions:

```bash
git -C "$INSTALLER" rev-parse HEAD
git -C "$REPO" rev-parse HEAD
```

The published fresh-host log was created from fresh clones, but the original run did not print both exact `HEAD` SHAs into the console output.

---

## License and attribution

This repository contains setup scripts, patches and documentation around `NVlabs/nvdiffrast`.

- `NVlabs/nvdiffrast` is copyright NVIDIA Corporation and is distributed under the NVIDIA Source Code License. See [`third_party/nvdiffrast/LICENSE.txt`](third_party/nvdiffrast/LICENSE.txt).
- Non-commercial-use restrictions may apply. Read the upstream NVIDIA license before redistribution or commercial use.
- Earlier ROCm patch work by `tashibi/nvdiffrast-rocm-patch` is acknowledged. See [`third_party/tashibi-nvdiffrast-rocm-patch/LICENSE`](third_party/tashibi-nvdiffrast-rocm-patch/LICENSE).
- Original helper scripts and documentation in this repository are MIT-licensed unless marked otherwise.

---

## Short summary for local AI agents

```text
Project:
  Community ROCm 7.2 / RDNA4 gfx1201 build and patch stack
  for upstream NVlabs/nvdiffrast.

Validated fresh-host environment:
  Ubuntu 24.04
  Python 3.12.3
  PyTorch 2.13.0+rocm7.2
  HIP 7.2.53211
  AMD Radeon AI PRO R9700 / gfx1201
  110/110 quick-validation checks passed

Do not:
  Treat this repository as the upstream nvdiffrast source tree.
  pip-install this repository directly.
  Apply only the final patch stack to a raw fresh upstream clone.
  Reintroduce v48/v49/v50 diagnostic patches as final fixes.

Correct flow:
  Clone upstream nvdiffrast.
  Generate the ROCm/HIP runtime baseline.
  Perform the initial baseline build.
  Apply final v52.
  Perform a clean final rebuild.
  Verify v47/v51 markers after hipify.
  Run validation.

Key runtime fixes:
  v36: FineRaster RDNA Wave32 row-mask fix.
  v41m2: Interpolate empty-warp all_sync ROCm fix.
  v47: Antialias forward torch-native work-buffer zeroing.
  v51: Antialias backward/grad torch-native Y-counter zeroing.

Fresh-host evidence:
  docs/validation/fresh-host-ubuntu24.04-rocm72-gfx1201-v52.md
  docs/validation/logs/fresh-host-ubuntu24.04-rocm72-gfx1201-v52-full.log
```
