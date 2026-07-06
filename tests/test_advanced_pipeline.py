#!/usr/bin/env python3
# test_advanced_pipeline.py
# Advanced nvdiffrast ROCm/RDNA validation:
#   1) multiple triangles
#   2) larger resolution / tile path
#   3) backward/autograd sanity
#
# Run in a fresh shell after any previous HIP crash:
#   cd ~/therock_test/nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4
#   python test_advanced_pipeline.py

import os
import sys
import traceback

import torch
import nvdiffrast.torch as dr


def sync(label: str) -> None:
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"\n❌ CUDA/HIP synchronize failed after: {label}")
        print(type(exc).__name__, exc)
        raise


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def summarize_raster(rast: torch.Tensor, name: str) -> None:
    ch3 = rast[..., 3]
    covered = (ch3 > 0).sum().item()
    unique_ids = torch.unique(ch3.detach()).cpu().tolist()
    finite = torch.isfinite(rast).all().item()

    print(f"{name} shape:        {tuple(rast.shape)}")
    print(f"{name} finite:       {finite}")
    print(f"{name} covered px:   {covered}")
    print(f"{name} ch3 sum:      {ch3.sum().item()}")
    print(f"{name} tri IDs:      {unique_ids}")
    print(f"{name} ch3 min/max:  {ch3.min().item()} / {ch3.max().item()}")

    if not finite:
        raise RuntimeError(f"{name}: raster output contains NaN/Inf")
    if covered <= 0:
        raise RuntimeError(f"{name}: no covered pixels")
    if ch3.max().item() < 1.0:
        raise RuntimeError(f"{name}: no triangle IDs found")


def run_forward_backward(ctx: dr.RasterizeCudaContext, resolution: int) -> None:
    print_header(f"FORWARD + BACKWARD TEST @ {resolution}x{resolution}")

    # Four triangles in different screen regions.
    # z differs slightly so depth path is exercised too.
    pos = torch.tensor([[
        # tri 0: lower left
        [-0.95, -0.95, 0.40, 1.0],
        [-0.15, -0.95, 0.40, 1.0],
        [-0.55, -0.15, 0.40, 1.0],

        # tri 1: upper right
        [ 0.15,  0.15, 0.55, 1.0],
        [ 0.95,  0.15, 0.55, 1.0],
        [ 0.55,  0.95, 0.55, 1.0],

        # tri 2: lower right
        [ 0.10, -0.90, 0.65, 1.0],
        [ 0.90, -0.90, 0.65, 1.0],
        [ 0.50, -0.10, 0.65, 1.0],

        # tri 3: upper left
        [-0.90,  0.10, 0.75, 1.0],
        [-0.10,  0.10, 0.75, 1.0],
        [-0.50,  0.90, 0.75, 1.0],
    ]], device="cuda", dtype=torch.float32, requires_grad=True)

    tri = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ], device="cuda", dtype=torch.int32)

    rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=[resolution, resolution])
    sync(f"rasterize {resolution}")

    summarize_raster(rast, f"rast{resolution}")

    # Channel 3 is triangle ID / coverage marker and is not a meaningful
    # differentiable loss. Use it only as a detached mask.
    mask = (rast[..., 3:4] > 0).detach().to(rast.dtype)

    # Differentiable scalar loss from barycentric/depth channels only.
    # Weighted terms avoid perfect symmetry cancelling all gradients.
    uvz = rast[..., :3]
    weights = torch.tensor([0.37, -0.23, 0.11], device="cuda", dtype=torch.float32)
    loss = ((uvz * weights) * mask).sum()

    print(f"loss:              {loss.item()}")

    if pos.grad is not None:
        pos.grad.zero_()

    loss.backward()
    sync(f"backward {resolution}")

    grad = pos.grad
    print("pos.grad:")
    print(grad.detach().cpu())

    grad_finite = torch.isfinite(grad).all().item()
    grad_absmax = grad.abs().max().item()
    grad_nonzero = (grad.abs() > 0).any().item()

    print(f"grad finite:       {grad_finite}")
    print(f"grad abs max:      {grad_absmax}")
    print(f"grad nonzero:      {grad_nonzero}")

    if not grad_finite:
        raise RuntimeError(f"{resolution}: gradient contains NaN/Inf")
    if not grad_nonzero or grad_absmax == 0.0:
        raise RuntimeError(f"{resolution}: gradient is all zero")

    print(f"✅ {resolution}x{resolution}: forward + backward OK")


def main() -> int:
    print_header("ENVIRONMENT")
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("cuda available:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    print("AMD_SERIALIZE_KERNEL:", os.environ.get("AMD_SERIALIZE_KERNEL"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available through torch.cuda")

    ctx = dr.RasterizeCudaContext(device="cuda")

    # Stage 1 + 2 + 3: medium-large path and autograd.
    run_forward_backward(ctx, 256)

    # Extra stress pass. If this fails while 256 passes, tile/bin scaling is suspect.
    run_forward_backward(ctx, 512)

    print_header("FINAL RESULT")
    print("✅ ADVANCED NVDIFFRAST ROCm/RDNA PIPELINE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\n❌ TEST FAILED")
        traceback.print_exc()
        raise
