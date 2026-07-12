# Downstream end-to-end validation: Profactor/continuous-remeshing

**Date:** 2026-07-11
**Result:** passed — 100 / 100 optimization steps
**Purpose:** verify the patched nvdiffrast under a real differentiable-rendering
workload with continuously changing mesh topology, not just under fixed
synthetic test cases.

---

## Why this test exists

The quick-validation suite (110 / 110) exercises fixed resolutions and fixed
topologies. It cannot cover one property that matters in practice: during
remeshing, the vertex and triangle counts change on every step, so
`allocTriangles` is reallocated continuously. That is exactly the code path the
v47 / v51 work-buffer fixes and the topology hash operate on.

This test also confirms that gradients are not merely produced, but are
*usable*: a corrupted mask or a broken synchronization does not yield a slightly
imperfect mesh — it yields noise, spikes, holes at tile boundaries, or a
collapsed mesh. A run that converges over 100 iterations to a clean, closed
surface rules out those failure classes.

Finally, the analysis kernel is launched with a grid of `numCTA * numSM`. On this
GPU `numSM` (`multi_processor_count`) is 32. A successful run at full occupancy
is therefore also an end-to-end confirmation of the v45 finding (no
`numCTA = 1;` debug override in the tree).

---

## Environment

| Component | Value |
|---|---|
| Operating system | Ubuntu 24.04 |
| GPU (reported name) | `AMD Radeon Graphics` |
| GPU (architecture) | `gfx1201` (RDNA4) |
| Compute units (`multi_processor_count`) | 32 |
| VRAM | 29.86 GB |
| Python | 3.12 |
| PyTorch | `2.13.0+rocm7.2` |
| HIP reported by PyTorch | `7.2.53211` |
| Triton | `triton-rocm 3.7.1` |
| nvdiffrast | installed from the locally patched source tree |
| torch_scatter | `2.1.2`, built from source |

PyTorch reports the card generically as `AMD Radeon Graphics`. The architecture
is confirmed independently:

```python
import torch
p = torch.cuda.get_device_properties(0)
print(p.gcnArchName)            # gfx1201
print(p.multi_processor_count)  # 32
```

Captured environment files:

- `env_report.txt` — interpreter, PyTorch, HIP, device properties
- `pip_freeze.txt` — full dependency set, including the exact
  `pytorch_scatter` commit

---

## Additional dependencies

`continuous-remeshing` needs a few packages beyond the nvdiffrast stack:

```bash
pip install trimesh imageio tqdm matplotlib
pip install --no-build-isolation -v git+https://github.com/rusty1s/pytorch_scatter.git
```

`torch_scatter` has no prebuilt ROCm wheel and must be compiled from source.
`--no-build-isolation` is required so that it builds against the installed
ROCm PyTorch rather than pulling a fresh CUDA one.

---

## Test procedure

A fresh clone of `Profactor/continuous-remeshing` was run with its unmodified
`example.py` for 100 optimization / remeshing steps, using the AMD-patched
nvdiffrast as the rendering backend.

One helper adjustment was needed for current `imageio` / `Pillow`: single-channel
alpha previews are written as RGB-compatible preview images. This affects only
how preview PNGs are saved. It does not change rendering, loss, gradients,
optimization or remeshing.

---

## Result

```text
100/100 steps completed
FINAL_LOSS      = 0.00829143
FINAL_VERTICES  = 5814
FINAL_FACES     = 11624
throughput      ≈ 24.27 it/s
```

Generated outputs:

```text
out/result.obj
out/images/          16 rendered views
out/alpha/           16 rendered alpha views
out/target_images/   16 target views
out/target_alpha/    16 target alpha views
```

The reconstructed mesh is a clean, closed surface with an intact silhouette,
separated wings and a coherent base. There are no tile-boundary artifacts.

This validates the complete differentiable loop:

```text
nvdiffrast rasterize / interpolate / antialias
→ normal + alpha loss
→ backward
→ optimizer step
→ remeshing with changing topology
→ final mesh export
```

---

## Use as a regression anchor

The recorded values are a fixed reference point. Any future change to the patch
stack should keep the final loss, vertex count, face count and visual result in
the same range. If they move, the change is suspect.

Just as importantly, this run gives a way to *exclude* nvdiffrast as a cause: if
a downstream project misbehaves, re-running this test answers whether the
renderer is still healthy before anything else is investigated.

---

## Open point: cross-vendor numerical reference

This test anchors the result against *itself*. It does not yet anchor it against
CUDA.

A run of the same example with the same seed on an NVIDIA GPU would confirm that
gradient scaling matches, closing the last quiet failure class that a visual
check cannot catch. Contributions of a reference `FINAL_LOSS` from a CUDA
machine are welcome.

---

## Artifacts

```text
docs/validation/downstream/
    env_report.txt
    pip_freeze.txt
    example_run.log
    example_alpha_fix.py
    result_preview.jpg
```
