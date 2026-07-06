# Known Validation Notes and Open Questions

The v36 ROCm/RDNA gfx1201 patch has passed the current forward, backward,
and interior finite-difference validation suite. A few limitations should still
remain explicit before broader publication or production use.

## 1. v34/v36 output relationship

v34 forced the original FineRaster path to run only `threadIdx.y == 0`.
v36 restored the full runtime path but used the corrected RDNA Wave32 mask for
all `threadIdx.y` rows:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

Both v34 and v36 produced a valid 16x16 triangle result with `covered: 128`
in the minimal test. v36 also passed the advanced 256x256 and 512x512
multi-triangle forward/backward tests, which significantly increases confidence
that the full runtime path is working correctly.

This still does not prove every possible row/tile distribution or every
production mesh layout. Large real-world scenes should continue to be tested.

## 2. Finite-difference gradient validation

The earlier advanced autograd test verified that `pos.grad` was finite and
nonzero. A later finite-difference test added stronger numerical validation for
the smooth interior rasterize/interpolate path.

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

This validates that the analytical gradients from the rasterize/interpolate
interior path match numerical finite differences in a smooth, fully covered
region with no silhouette, no triangle-ID changes, and no coverage-edge jumps.

Observed z gradients were zero in this test while w gradients were nonzero and
matched finite differences. This is not automatically a failure: the current
test uses constant depth and an interior color interpolation loss, so it does
not create a strong depth-ordering objective. A separate depth-ordering or
z-sensitive finite-difference test would be needed before making broader claims
about z-gradient behavior.

## 3. Remaining unvalidated nvdiffrast paths

The following paths still need separate stress tests:

```text
- antialias / silhouette gradients
- texture sampling
- very large production meshes
- long PSHuman / continuous-remeshing optimization loops
- clipping / offscreen geometry stress cases
- depth-ordering / z-sensitive finite-difference tests
- performance benchmarking
```

The `interpolate` path is no longer fully unvalidated: the smooth interior
rasterize/interpolate gradient path has passed the v4 finite-difference test.
However, interpolation under more complex production conditions should still be
covered by real project tests.

## 4. Known stress-case note: offscreen covering triangle

An earlier finite-difference test variant used a very large triangle extending
outside the clipspace to remove visible silhouettes. On ROCm/gfx1201 this
triggered an illegal memory access in the test environment.

That test design was replaced by `test_rasterizer_gradients_v4.py`, which uses
an in-clip triangle and computes the loss only on a safe central interior crop.
The offscreen/clipping case should therefore remain listed as an open stress
case rather than as a validated path.

## 5. Licensing / redistribution

This repository includes setup scripts and patches for a project that depends on
NVlabs/nvdiffrast. Upstream nvdiffrast is under the NVIDIA Source Code License
(1-Way Commercial). Read `NOTICE.md` and `third_party/nvdiffrast/LICENSE.txt`
before public redistribution or commercial use.
