# NOTICE / Attribution

This repository is a patch and setup helper package for building `nvdiffrast`
on AMD ROCm/RDNA systems. It is not an official NVIDIA, AMD, or PyTorch project.

## Upstream nvdiffrast

- Upstream project: `NVlabs/nvdiffrast`
- Copyright: Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
- License: NVIDIA Source Code License (1-Way Commercial)
- Included license copy: `third_party/nvdiffrast/LICENSE.txt`

Important licensing note: the NVIDIA Source Code License requires redistributions
of the Work to include a complete copy of the license and retain copyright,
patent, trademark, and attribution notices. It also contains a non-commercial
use limitation for the Work and derivative works. Read the full license before
publishing, redistributing, or using this in a project.

## tashibi/nvdiffrast-rocm-patch

This project also acknowledges the earlier ROCm patch work by `tashibi`:

- Project: `tashibi/nvdiffrast-rocm-patch`
- License: MIT
- Included license copy: `third_party/tashibi-nvdiffrast-rocm-patch/LICENSE`

The v36 patch in this repository differs from the earlier Wave64-oriented
approach: for RDNA/gfx1201 the validated runtime fix treats each `threadIdx.y`
row as an independent native Wave32 group.

## This repository's original helper code

Original helper scripts and documentation in this repository are provided under
the MIT License in `LICENSE`, except where a file explicitly states otherwise or
contains third-party / derivative code.

## No endorsement

Mention of NVIDIA, AMD, ROCm, PyTorch, or tashibi is for attribution and
compatibility documentation only. No endorsement is implied.
