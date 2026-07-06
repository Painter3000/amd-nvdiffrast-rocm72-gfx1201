# Known Open Questions

The v36 ROCm/RDNA gfx1201 patch has passed the current forward and backward
validation suite, but a few points should remain explicit before broader
publication.

## 1. v34/v36 output relationship

v34 forced the original FineRaster path to run only `threadIdx.y == 0`.
v36 restored the full runtime path but used the corrected RDNA Wave32 mask for
all `threadIdx.y` rows:

```cpp
const U32 nvdr_rowShift = 0u;
const U64 nvdr_rowMask  = 0x00000000ffffffffull;
```

Both produced a valid 16x16 triangle result with `covered: 128` in the minimal
test. This is good, but it does not by itself prove that every possible row/tile
distribution behaves identically in all scenes. The advanced 256x256 and 512x512
multi-triangle tests significantly increase confidence, but production meshes
should still be tested.

## 2. z / w gradient interpretation

The advanced autograd test verifies that `pos.grad` is finite and nonzero.
In the observed results, z gradients were zero while w gradients were nonzero.
That is not automatically a failure: the current test loss and scene layout do
not create a strong depth-ordering objective. A separate finite-difference test
should be added before claiming numerical gradient accuracy.

## 3. Unvalidated nvdiffrast paths

The following paths have not yet been separately stress-tested in this package:

```text
- antialias
- interpolate
- texture sampling
- very large production meshes
- long PSHuman / continuous-remeshing optimization loops
- finite-difference gradient comparison
- performance benchmarking
```

## 4. Licensing / redistribution

This repository includes setup scripts and patches for a project that depends on
NVlabs/nvdiffrast. Upstream nvdiffrast is under the NVIDIA Source Code License
(1-Way Commercial). Read `NOTICE.md` and `third_party/nvdiffrast/LICENSE.txt`
before public redistribution or commercial use.
