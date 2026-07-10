#!/usr/bin/env python3
"""
test_many_triangles_stress_v1.py

nvdiffrast ROCm/RDNA gfx1201 stress test for CoarseRaster/BinRaster/FineRaster.

Purpose:
  The small tests validate the FineRaster path and smooth interior gradients,
  but they may not sufficiently stress binning/coarse rasterization. This test
  renders thousands of triangles spread across the viewport to force many
  tiles/bins and more complex queue/ballot paths.

Modes:
  forward:
    rasterize -> interpolate -> finite checks

  backward_color:
    col.requires_grad=True
    rasterize -> interpolate -> loss -> backward
    This stresses rasterize/interpolate forward plus interpolate/color backward.

  backward_pos:
    pos.requires_grad=True
    rasterize -> interpolate -> loss -> backward
    This stresses rasterize backward wrt positions on many triangles.

  antialias:
    pos.requires_grad=True
    rasterize -> interpolate -> antialias -> loss -> backward
    This additionally stresses antialias backward on a larger mesh.

Notes:
  - This is a stability/smoke test, not a finite-difference accuracy test.
  - After any HIP crash, start a fresh shell before continuing.
"""

import argparse
import math
import traceback

import torch
import nvdiffrast.torch as dr


def sync(label):
    print(f"[sync] {label}", flush=True)
    torch.cuda.synchronize()


def summarize(name, x):
    finite = torch.isfinite(x).all().item()
    print(
        f"{name}: shape={tuple(x.shape)} finite={finite} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"sum={x.sum().item():.6f} absmax={x.abs().max().item():.6f}",
        flush=True,
    )


def make_grid_mesh(device, cells, jitter=0.0, z_wave=False):
    """Regular grid in NDC [-0.95, 0.95]^2, triangulated into 2 tris/cell."""
    n = cells
    xs = torch.linspace(-0.95, 0.95, n + 1, device=device, dtype=torch.float32)
    ys = torch.linspace(-0.95, 0.95, n + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    if jitter > 0:
        # Deterministic tiny jitter for interior vertices only. Keep boundary stable.
        torch.manual_seed(1234)
        jx = (torch.rand_like(xx) - 0.5) * (1.9 / n) * jitter
        jy = (torch.rand_like(yy) - 0.5) * (1.9 / n) * jitter
        jx[0, :] = jx[-1, :] = jx[:, 0] = jx[:, -1] = 0
        jy[0, :] = jy[-1, :] = jy[:, 0] = jy[:, -1] = 0
        xx = xx + jx
        yy = yy + jy

    if z_wave:
        zz = 0.1 * torch.sin(xx * 12.0) * torch.cos(yy * 10.0)
    else:
        zz = torch.zeros_like(xx)

    ww = torch.ones_like(xx)
    pos = torch.stack([xx, yy, zz, ww], dim=-1).reshape(-1, 4).contiguous()

    # Vertex colors from coordinates; deliberately non-uniform.
    col = torch.stack([
        (xx + 1.0) * 0.5,
        (yy + 1.0) * 0.5,
        0.25 + 0.25 * torch.sin(xx * 8.0),
    ], dim=-1).reshape(-1, 3).contiguous()

    tris = []
    stride = n + 1
    for y in range(n):
        base = y * stride
        next_base = (y + 1) * stride
        for x in range(n):
            v00 = base + x
            v10 = base + x + 1
            v01 = next_base + x
            v11 = next_base + x + 1
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))

    tri = torch.tensor(tris, dtype=torch.int32, device=device)
    return pos, tri, col


def make_random_triangles(device, tris_count, scale=0.02):
    """Many small independent triangles across the viewport."""
    torch.manual_seed(4321)
    centers = torch.rand((tris_count, 2), device=device, dtype=torch.float32) * 1.8 - 0.9
    angles = torch.rand((tris_count,), device=device, dtype=torch.float32) * (2.0 * math.pi)

    base = torch.tensor([
        [0.0, 1.0],
        [-0.8660254, -0.5],
        [0.8660254, -0.5],
    ], device=device, dtype=torch.float32) * scale

    verts = []
    for i in range(tris_count):
        c = torch.cos(angles[i])
        s = torch.sin(angles[i])
        rot = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ])
        xy = base @ rot.T + centers[i]
        z = torch.zeros((3, 1), device=device, dtype=torch.float32)
        w = torch.ones((3, 1), device=device, dtype=torch.float32)
        verts.append(torch.cat([xy, z, w], dim=-1))

    pos = torch.cat(verts, dim=0).contiguous()
    tri = torch.arange(tris_count * 3, device=device, dtype=torch.int32).reshape(tris_count, 3)
    col = torch.rand((tris_count * 3, 3), device=device, dtype=torch.float32)
    return pos, tri, col


def weighted_loss(color):
    weights = color.new_tensor([0.37, -0.23, 0.61])
    return (color * weights).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["forward", "backward_color", "backward_pos", "antialias"], default="forward")
    parser.add_argument("--mesh", choices=["grid", "random"], default="grid")
    parser.add_argument("--cells", type=int, default=128, help="grid cells per axis; tris=2*cells*cells")
    parser.add_argument("--random-tris", type=int, default=20000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--z-wave", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Keine CUDA/HIP-GPU sichtbar.")

    device = "cuda"
    resolution = [args.resolution, args.resolution]

    print(f"mode={args.mode} mesh={args.mesh} resolution={resolution}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    if args.mesh == "grid":
        pos, tri, col = make_grid_mesh(device, args.cells, jitter=args.jitter, z_wave=args.z_wave)
    else:
        pos, tri, col = make_random_triangles(device, args.random_tris)

    if args.mode in ("backward_pos", "antialias"):
        pos = pos.detach().clone().requires_grad_(True)
    else:
        pos = pos.detach().clone()

    if args.mode == "backward_color":
        col = col.detach().clone().requires_grad_(True)
    else:
        col = col.detach().clone()

    print(f"vertices={pos.shape[0]} triangles={tri.shape[0]}", flush=True)
    summarize("pos", pos)
    summarize("col", col)

    glctx = dr.RasterizeCudaContext(device=device)

    print("[step] rasterize", flush=True)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=resolution)
    sync("after rasterize")
    summarize("rast", rast)

    tri_ids = rast[..., 3]
    covered = (tri_ids > 0).sum().item()
    unique_count = torch.unique(tri_ids).numel()
    print(f"covered_px={covered} unique_tri_id_count={unique_count}", flush=True)

    print("[step] interpolate", flush=True)
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    sync("after interpolate")
    summarize("color", color)

    if args.mode == "antialias":
        print("[step] construct topology hash", flush=True)
        topology_hash = dr.antialias_construct_topology_hash(tri)
        sync("after topology hash")

        print("[step] antialias", flush=True)
        color = dr.antialias(color, rast, pos[None, ...], tri, topology_hash=topology_hash)
        sync("after antialias")

    if args.mode == "forward":
        print("PASS: forward stress completed", flush=True)
        return

    print("[step] backward", flush=True)
    loss = weighted_loss(color)
    print(f"loss={loss.item():.6f}", flush=True)
    loss.backward()
    sync("after backward")

    if args.mode == "backward_color":
        if col.grad is None:
            raise RuntimeError("col.grad is None")
        summarize("col.grad", col.grad)
        if not torch.isfinite(col.grad).all():
            raise RuntimeError("col.grad has NaN/Inf")
        if col.grad.abs().max().item() == 0.0:
            raise RuntimeError("col.grad is all zero")

    if args.mode in ("backward_pos", "antialias"):
        if pos.grad is None:
            raise RuntimeError("pos.grad is None")
        summarize("pos.grad", pos.grad)
        if not torch.isfinite(pos.grad).all():
            raise RuntimeError("pos.grad has NaN/Inf")
        if pos.grad.abs().max().item() == 0.0:
            raise RuntimeError("pos.grad is all zero")

    print("PASS: stress backward completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ TEST FAILED", flush=True)
        traceback.print_exc()
        raise
