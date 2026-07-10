#!/usr/bin/env python3
"""
nvdiffrast_interpolate_resolution_probe_v6.py

Focused probe for nvdiffrast.interpolate() resolution-dependent crashes.

Why this exists:
  The AA stage probe showed crash already at:
    [step] interpolate
  with stack frame:
    interpolate_fwd(...)
  before topology_hash and before dr.antialias().

Each case runs in a fresh subprocess.

Examples:
  python ./nvdiffrast_interpolate_resolution_probe_v6.py --timeout 30

  python ./nvdiffrast_interpolate_resolution_probe_v6.py \
    --shapes single,grid1,grid16 \
    --res-list 160,164,168,172,176,180,184,192,256 \
    --stages call,sync,read

  python ./nvdiffrast_interpolate_resolution_probe_v6.py \
    --shape single --res 164 --stage call --verbose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


def make_single_triangle(torch, device="cuda"):
    pos = torch.tensor(
        [[-0.75, -0.75, 0.0, 1.0],
         [ 0.75, -0.75, 0.0, 1.0],
         [ 0.00,  0.75, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    tri = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32).contiguous()
    col = torch.tensor(
        [[1.0, 0.1, 0.2],
         [0.1, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    return pos, tri, col


def make_grid(torch, cells: int, device="cuda"):
    n = cells
    xs = torch.linspace(-0.95, 0.95, n + 1, device=device, dtype=torch.float32)
    ys = torch.linspace(-0.95, 0.95, n + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack(
        [xx, yy, torch.zeros_like(xx), torch.ones_like(xx)], dim=-1
    ).reshape(-1, 4).contiguous()

    col = torch.stack(
        [
            (xx + 1.0) * 0.5,
            (yy + 1.0) * 0.5,
            0.25 + 0.25 * torch.sin(xx * 8.0),
        ],
        dim=-1,
    ).reshape(-1, 3).contiguous()

    tris = []
    stride = n + 1
    for y in range(n):
        for x in range(n):
            v00 = y * stride + x
            v10 = v00 + 1
            v01 = (y + 1) * stride + x
            v11 = v01 + 1
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))

    tri = torch.tensor(tris, device=device, dtype=torch.int32).contiguous()
    return pos, tri, col


def ptr(x):
    return hex(x.data_ptr())


def get_shape(torch, shape: str):
    if shape == "single":
        return make_single_triangle(torch)
    if shape.startswith("grid"):
        return make_grid(torch, int(shape.replace("grid", "")))
    raise RuntimeError(f"unknown shape: {shape}")


def child_main(args) -> int:
    import torch
    import nvdiffrast.torch as dr

    shape = args.child_shape
    res = args.child_res
    stage = args.child_stage

    print(f"[child] shape={shape} res={res} stage={stage}", flush=True)
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} device={torch.cuda.get_device_name(0)}", flush=True)

    pos, tri, col = get_shape(torch, shape)
    pos_b = pos[None, ...].contiguous()
    col_b = col[None, ...].contiguous()

    print(f"[meta] pos={tuple(pos.shape)} tri={tuple(tri.shape)} col={tuple(col.shape)}", flush=True)

    ctx = dr.RasterizeCudaContext(device="cuda")
    torch.cuda.synchronize()
    print("[ok] context", flush=True)

    print("[step] rasterize call", flush=True)
    rast, _ = dr.rasterize(ctx, pos_b, tri, resolution=[res, res])
    print(f"[meta] rast shape={tuple(rast.shape)} stride={tuple(rast.stride())} ptr={ptr(rast)}", flush=True)

    if stage == "rast_call":
        print("PASS rast_call", flush=True)
        return 0

    print("[step] rasterize sync", flush=True)
    torch.cuda.synchronize()
    print("[ok] rasterize sync", flush=True)

    if stage == "rast_sync":
        print("PASS rast_sync", flush=True)
        return 0

    print("[step] rasterize read coverage", flush=True)
    covered = (rast[..., 3] > 0).sum().item()
    unique = torch.unique(rast[..., 3]).numel()
    print(f"[ok] covered={covered} unique_tri_id_count={unique}", flush=True)

    if stage == "rast_read":
        print("PASS rast_read", flush=True)
        return 0

    print("[step] interpolate call", flush=True)
    color, _ = dr.interpolate(col_b, rast, tri)
    print(f"[meta] color shape={tuple(color.shape)} stride={tuple(color.stride())} ptr={ptr(color)} contiguous={color.is_contiguous()}", flush=True)

    if stage == "call":
        print("PASS call", flush=True)
        return 0

    print("[step] interpolate sync", flush=True)
    torch.cuda.synchronize()
    print("[ok] interpolate sync", flush=True)

    if stage == "sync":
        print("PASS sync", flush=True)
        return 0

    print("[step] interpolate scalar read", flush=True)
    v = color[0, 0, 0, 0].item()
    print(f"[ok] scalar={v:.6f}", flush=True)

    if stage == "scalar":
        print("PASS scalar", flush=True)
        return 0

    print("[step] interpolate finite/min/max/sum", flush=True)
    finite = torch.isfinite(color).all().item()
    mn = color.min().item()
    mx = color.max().item()
    sm = color.sum().item()
    print(f"[ok] finite={finite} min={mn:.6f} max={mx:.6f} sum={sm:.6f}", flush=True)
    if not finite:
        raise RuntimeError("color non-finite")

    print("PASS all", flush=True)
    return 0


@dataclass
class Result:
    shape: str
    res: int
    stage: str
    status: str
    rc: int | None
    seconds: float
    out: str


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_child(script: str, shape: str, res: int, stage: str, timeout: int, verbose: bool) -> Result:
    env = os.environ.copy()
    env.setdefault("AMD_SERIALIZE_KERNEL", "3")
    env.setdefault("TORCH_DISABLE_ADDR2LINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        sys.executable,
        script,
        "--child",
        "--child-shape", shape,
        "--child-res", str(res),
        "--child-stage", stage,
    ]

    start = time.monotonic()
    try:
        cp = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        seconds = time.monotonic() - start
        out = cp.stdout or ""
        if cp.returncode == 0:
            status = "OK"
        elif cp.returncode < 0:
            status = f"CRASH(signal {-cp.returncode})"
        else:
            status = f"FAIL(rc={cp.returncode})"
        if verbose:
            print(out, end="" if out.endswith("\n") else "\n")
        return Result(shape, res, stage, status, cp.returncode, seconds, out)
    except subprocess.TimeoutExpired as e:
        seconds = time.monotonic() - start
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if verbose and out:
            print(out, end="" if out.endswith("\n") else "\n")
        return Result(shape, res, stage, "TIMEOUT", 124, seconds, out)


def parent_main(args) -> int:
    shapes = parse_csv(args.shapes if not args.shape else args.shape)
    res_list = parse_ints(args.res_list if not args.res else args.res)
    stages = parse_csv(args.stages if not args.stage else args.stage)

    print(f"Interpolate resolution probe: shapes={shapes} res={res_list} stages={stages} timeout={args.timeout}s", flush=True)
    print("", flush=True)

    results: List[Result] = []
    script = os.path.abspath(__file__)

    for shape in shapes:
        for stage in stages:
            for res in res_list:
                print(f"===== shape={shape} res={res} stage={stage} =====", flush=True)
                r = run_child(script, shape, res, stage, args.timeout, args.verbose)
                results.append(r)
                print(f"-> {r.status} rc={r.rc} time={r.seconds:.2f}s", flush=True)
                if r.status != "OK" and not args.verbose:
                    print("---- tail ----", flush=True)
                    print("\n".join(r.out.splitlines()[-24:]), flush=True)
                    print("-------------", flush=True)
                print("", flush=True)
                if args.stop_on_fail and r.status != "OK":
                    break

    print("===== SUMMARY =====", flush=True)
    failed = 0
    for r in results:
        print(f"{r.shape:<8} res={r.res:<4} stage={r.stage:<10} {r.status:<16} rc={r.rc} time={r.seconds:.2f}s", flush=True)
        if r.status != "OK":
            failed += 1

    print("\n===== COMPACT MATRIX =====", flush=True)
    keys = []
    for r in results:
        key = (r.shape, r.stage)
        if key not in keys:
            keys.append(key)
    width = max(len(str(x)) for x in res_list) if res_list else 3
    header = "shape/stage".ljust(22) + " " + " ".join(str(r).rjust(width) for r in res_list)
    print(header)
    print("-" * len(header))
    by_key = {(r.shape, r.stage, r.res): r for r in results}
    for key in keys:
        label = f"{key[0]}/{key[1]}".ljust(22)
        vals = []
        for res in res_list:
            r = by_key.get((key[0], key[1], res))
            if r is None:
                vals.append("?".rjust(width))
            elif r.status == "OK":
                vals.append("OK".rjust(width))
            elif r.status == "TIMEOUT":
                vals.append("TO".rjust(width))
            else:
                vals.append("XX".rjust(width))
        print(label + " " + " ".join(vals))

    print("", flush=True)
    print(f"passed={len(results)-failed} failed_or_timeout={failed} total={len(results)}", flush=True)
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--shapes", default="single")
    p.add_argument("--shape", default="")
    p.add_argument("--res-list", default="160,164,168,172,176,180,184,192,256")
    p.add_argument("--res", default="")
    p.add_argument("--stages", default="rast_sync,rast_read,call,sync,scalar,all")
    p.add_argument("--stage", default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--child-shape", default="", help=argparse.SUPPRESS)
    p.add_argument("--child-res", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--child-stage", default="", help=argparse.SUPPRESS)

    args = p.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
