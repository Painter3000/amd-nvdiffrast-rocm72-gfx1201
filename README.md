# AMD nvdiffrast ROCm 7.2 / RDNA gfx1201 v36

This package builds and validates a patched `nvdiffrast` 0.4.0 runtime for AMD RDNA / gfx1201 under ROCm 7.2.

Validated target system:

```text
GPU:        AMD Radeon AI PRO R9700 / gfx1201 / RDNA4
PyTorch:    2.12.0+rocm7.2
HIP:        7.2.53211
Python:     3.10.20
nvdiffrast: 0.4.0
```

## What this fixes

The upstream FineRaster code is written around CUDA Warp32 synchronization and ballot assumptions. On RDNA/gfx1201, every `threadIdx.y` row must be handled as its own native Wave32 execution group.

Final v36 rule:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

Do **not** use Wave64 upper/lower half splitting on RDNA/gfx1201.

## Requirements

This package does not install AMD drivers, ROCm, or PyTorch. You need a working ROCm PyTorch environment first.

Verify before running:

```bash
source ~/therock_test/venv/bin/activate

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## Recommended one-command workflow

```bash
source ~/therock_test/venv/bin/activate

python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --advanced-test
```

What it does:

```text
1. Clone NVlabs/nvdiffrast if needed.
2. Generate the v4 ROCm/gfx1201 reinstall bundle.
3. Apply the v4 runtime baseline patches.
4. Apply the final v36 RDNA Wave32 FineRaster fix.
5. Rebuild and install nvdiffrast.
6. Run minimal and advanced validation tests.
```

## Manual workflow

Clone nvdiffrast:

```bash
mkdir -p ~/therock_test
cd ~/therock_test
git clone https://github.com/NVlabs/nvdiffrast.git
```

Generate the reinstall bundle:

```bash
cd <this-repo>
BUNDLE_DIR=~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4 \
bash scripts/nvdiffrast_rocm72_bundle_v4_generator.sh
```

Apply the v4 runtime baseline:

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4

ROOT=~/therock_test \
REPO=~/therock_test/nvdiffrast \
VENV=~/therock_test/venv \
ROCM_PATH=/opt/rocm \
MODE=runtime \
./reinstall_nvdiffrast_rocm72_gfx1201.sh
```

Apply the v36 RDNA Wave32 fix:

```bash
cd ~/therock_test/nvdiffrast
bash <this-repo>/patches/patch_rocm_v36_wave32.sh
```

Rebuild:

```bash
source ~/therock_test/venv/bin/activate
cd ~/therock_test/nvdiffrast

rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast

export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v
```

Run tests:

```bash
cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4
./test_min_triangle.sh
./run_test_advanced_pipeline.sh
```

## Known-good validation output

Minimal:

```text
finite: True
sum: 277.5
covered: 128
tri id max: 1.0
unique tri ids: [0.0, 1.0]
```

Advanced:

```text
256x256:
  finite: True
  covered px: 20962
  tri IDs: [0.0, 1.0, 2.0, 3.0, 4.0]
  grad finite: True
  grad nonzero: True

512x512:
  finite: True
  covered px: 83641
  tri IDs: [0.0, 1.0, 2.0, 3.0, 4.0]
  grad finite: True
  grad nonzero: True

Final:
  ADVANCED NVDIFFRAST ROCm/RDNA PIPELINE TEST PASSED
```

## Repository layout

```text
.
├── README.md
├── amd_nvdiffrast_setup.py
├── patches/
│   └── patch_rocm_v36_wave32.sh
├── scripts/
│   └── nvdiffrast_rocm72_bundle_v4_generator.sh
├── tests/
│   ├── test_min_triangle.sh
│   ├── test_advanced_pipeline.py
│   └── run_test_advanced_pipeline.sh
└── docs/
    ├── README_ROCM72_GFX1201_v36.md
    └── KNOWN_OPEN_QUESTIONS.md
```

## License and attribution

This repository contains original helper scripts plus patch content for
`NVlabs/nvdiffrast`.

- Original helper scripts and documentation in this repository are provided under
  the MIT License. See `LICENSE`.
- Upstream `NVlabs/nvdiffrast` is copyright NVIDIA Corporation and is made
  available under the NVIDIA Source Code License (1-Way Commercial). A complete
  copy is included at `third_party/nvdiffrast/LICENSE.txt`.
- Earlier ROCm patch work by `tashibi/nvdiffrast-rocm-patch` is acknowledged.
  Its MIT license is included at
  `third_party/tashibi-nvdiffrast-rocm-patch/LICENSE`.
- See `NOTICE.md` for attribution and license-boundary details.

This is not legal advice. Read the upstream licenses before redistributing or
using this package in a project.

## Known open questions

The current v36 patch is validated for the included rasterize forward/backward
tests, but the following are still open:

```text
- v34/v36 output relationship should be tested on more complex row/tile cases.
- z/w gradient behavior has not been checked with finite differences.
- antialias, interpolate, and texture paths need separate stress tests.
- huge meshes and long PSHuman / continuous-remeshing loops remain production tests.
```

See `docs/KNOWN_OPEN_QUESTIONS.md` for details.


## HIP crash hygiene

After any hardware trap, `HSA_STATUS_ERROR_EXCEPTION`, illegal address, or unspecified launch failure, start a fresh shell before testing again. HIP context state can be poisoned after a device-side failure.

```bash
export AMD_SERIALIZE_KERNEL=3
export TORCH_DISABLE_ADDR2LINE=1
```

## Current validation scope

Validated:

```text
- nvdiffrast import
- 16x16 minimal triangle
- 256x256 and 512x512 multi-triangle rasterize
- rasterize backward/autograd sanity
- finite and nonzero gradients
```

Not yet fully validated:

```text
- huge production meshes
- long PSHuman / continuous-remeshing optimization loops
- finite-difference gradient comparison
- antialias / interpolate / texture paths as separate stress tests
- performance benchmarks
```
