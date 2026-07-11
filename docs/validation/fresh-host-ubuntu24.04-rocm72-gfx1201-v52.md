# Fresh-host validation: Ubuntu 24.04, ROCm 7.2, gfx1201, v52

**Status:** ✅ Passed  
**Validation date:** 2026-07-11  
**Result:** **110 / 110 checks passed**

This document summarizes a complete fresh-host installation and validation run of the community ROCm 7.2 / RDNA4 `gfx1201` patch stack for NVIDIA nvdiffrast.

The run started from:

- a fresh clone of this installer repository,
- a fresh clone of `NVlabs/nvdiffrast`,
- a clean Python virtual environment,
- and no pre-existing local nvdiffrast build artifacts.

The full raw console log is available here:

[Full fresh-host validation log](logs/fresh-host-ubuntu24.04-rocm72-gfx1201-v52-full.log)

---

## Validation environment

| Component | Validated value |
|---|---|
| Operating system | Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | `2.13.0+rocm7.2` |
| HIP runtime reported by PyTorch | `7.2.53211` |
| Triton | `triton-rocm 3.7.1` |
| NumPy | `2.5.1` |
| Ninja | `1.13.0` |
| nvdiffrast | `0.4.0` |
| GPU | AMD Radeon AI PRO R9700 |
| PyTorch device string | `AMD Radeon Graphics` |
| Explicit compile target | `gfx1201` |
| ROCm path | `/opt/rocm` |
| Validation mode | `quick` |

PyTorch reported:

```text
Torch: 2.13.0+rocm7.2
HIP: 7.2.53211
cuda available: True
device: AMD Radeon Graphics
Target arch requested: gfx1201
```

The generic PyTorch device string did not affect compilation because the installer explicitly targeted `gfx1201`.

---

## Command used

After preparing the virtual environment and installing the ROCm PyTorch stack, the installer was started with:

```bash
python ./amd_nvdiffrast_setup.py \
  --workdir ~/therock_test \
  --venv ~/therock_test/venv \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick
```

---

## Installation stages

The automated process completed all major stages successfully.

### 1. Fresh upstream clone

The installer cloned a new copy of:

```text
https://github.com/NVlabs/nvdiffrast.git
```

No existing nvdiffrast source tree was reused.

### 2. Runtime-baseline bundle generation

The v52 generator created the ROCm runtime-baseline bundle and applied the required pre-build compatibility changes before the first compile.

The baseline phase covered, among other changes:

- HIP-compatible 64-bit warp masks,
- antialias `__ballot_sync` mask widening,
- interpolate `__all_sync` mask widening,
- the ROCm `__frcp_rz` compatibility implementation,
- the Clang/C++20 `evHashElements` narrowing fix,
- removal of the unnecessary transitive `framework.h` include from `texture.h`,
- ROCm-safe replacements for CUDA/PTX-only rasterizer helpers,
- TriangleSetup result reconstruction,
- and FineRaster initialization fixes.

### 3. Initial baseline build

All 15 native compilation units built successfully.

The wheel was created and installed:

```text
Successfully built nvdiffrast
Successfully installed nvdiffrast-0.4.0
```

The initial import smoke test passed:

```text
torch: 2.13.0+rocm7.2
hip: 7.2.53211
cuda available: True
nvdiffrast import: OK
```

### 4. Final v52 patch stack

The final patch stack applied the canonical runtime fixes in this order:

| Patch | Purpose |
|---|---|
| v36 | FineRaster RDNA Wave32 runtime fix |
| v38 | AntialiasGrad ROCm Wave32 fix |
| v38b | GradPos guard |
| v39b | Persistent-loop guard |
| v40f | Antialias forward bounds guards |
| v41c | Clang `evHash` narrowing fix |
| v41m2 | Interpolate empty-warp `all_sync` fix |
| v45 | Verify removal of the debug `numCTA = 1` override |
| v47 | Forward work-buffer native zero initialization |
| v51 | Backward/Grad work-buffer Y-counter native zero initialization |

Diagnostic-only patches were not included in the final stack.

### 5. Clean final rebuild

The baseline installation and all generated build products were removed.

nvdiffrast was then rebuilt from the fully patched source tree. This second build verifies that:

- the final patch stack survives a clean rebuild,
- generated HIP files are recreated correctly,
- no stale object files are reused,
- and the installed extension contains the complete final patch set.

The final wheel build, link and installation all succeeded.

---

## Marker-integrity result

The dedicated marker-integrity test confirmed the expected final source state.

| Check | Result |
|---|---:|
| v47 marker in `torch_antialias.cpp` | 1 |
| v47 marker in `torch_antialias_hip.cpp` | 1 |
| v51 marker in `torch_antialias.cpp` | 1 |
| v51 marker in `torch_antialias_hip.cpp` | 1 |
| Active diagnostic markers | None |
| Hard device/stream synchronization diagnostics | None |

Final marker result:

```text
OK: v52 markers are clean.
```

---

## Validation results

### Path probe

The path probe exercised 14 separate forward, backward and utility paths.

| Path | Result |
|---|---:|
| Import | OK |
| Minimal rasterize forward | OK |
| Minimal interpolate forward | OK |
| Interpolate backward: color | OK |
| Rasterize/interpolate backward: position | OK |
| Grid forward | OK |
| Grid backward: position | OK |
| Topology hash | OK |
| Antialias forward: single triangle | OK |
| Texture forward | OK |
| Antialias forward: grid | OK |
| Antialias backward: direct color | OK |
| Antialias backward: grid position | OK |
| Texture backward | OK |

```text
passed=14 failed_or_timeout=0 total=14
```

### Antialias forward matrix

The forward matrix covered:

- shapes: `single`, `grid1`, `grid4`, `grid16`,
- resolutions: `160`, `180`, `182`, `192`, `224`, `256`,
- interpolated colors,
- explicit topology hashes.

```text
passed=24 failed_or_timeout=0 total=24
```

### Antialias backward and gradient matrix

The backward/gradient matrix covered:

- cells: `1`, `4`, `16`,
- resolutions: `160`, `180`, `182`, `192`, `224`, `256`,
- explicit topology,
- interpolated colors,
- position gradients enabled,
- stages: `call`, `sync`, `finite`, `diff`.

```text
passed=72 failed=0 total=72
```

The cases verified that calls returned, synchronization completed, results remained finite and output differences were observable where expected.

---

## Overall result

| Validation group | Passed | Failed |
|---|---:|---:|
| Path probe | 14 | 0 |
| Antialias forward matrix | 24 | 0 |
| Antialias backward/gradient matrix | 72 | 0 |
| **Total** | **110** | **0** |

> **Fresh-host installation, both native builds, marker integrity and all 110 quick-validation checks completed successfully.**

---

## Non-fatal warnings observed

The build emitted several warnings that did not affect compilation, linking, installation or validation:

- `-lineinfo` reported as unused by the host Clang invocation,
- ignored `hipFree` and `hipHostFree` return values in cleanup paths,
- unused local variables in generated or upstream-derived wrapper code,
- normal setuptools manifest exclusion warnings,
- and portability notices from legacy `cp -n` backup commands.

No warning corresponded to a failed test or invalid runtime result.

---

## Scope and limitations

This validation proves the tested installation path on the environment documented above.

It does not by itself prove:

- compatibility with every ROCm or PyTorch release,
- compatibility with other AMD architectures,
- long-duration stress stability,
- performance parity with CUDA,
- or correctness for every possible scene, tensor shape and texture configuration.

The repository's extended stress and diagnostic suites should be used for broader coverage.

---

## Reproducibility note

Both repositories were freshly cloned during the recorded run, but the installer did not print the exact Git commit SHA of either clone into the log.

Future validation runs should record at least:

```bash
git -C "$INSTALLER_REPO" rev-parse HEAD
git -C "$NVDR_REPO" rev-parse HEAD
```

This will bind each validation artifact to the exact installer and upstream source revisions.

The raw log itself was added to this repository in commit `19bc59d`.

---

## Evidence

- [Full raw validation log](logs/fresh-host-ubuntu24.04-rocm72-gfx1201-v52-full.log)
- Installer entry point: `amd_nvdiffrast_setup.py`
- Final stack: `patches/apply_final_rocm72_gfx1201_v52.sh`
- Marker test: `tests/test_v52_marker_integrity.sh`
- Validation suite: `tests/run_v52_validation.sh`
