#!/usr/bin/env python3
"""
test_antialias_backward_matrix_v52.py

Bisects the remaining validated failure after clean rebuild:
  - antialias_forward_grid
  - antialias_backward_pos_grid

Each case runs in its own subprocess so HIP crashes do not poison the runner.

Note: this file was formerly named nvdiffrast_antialias_grid_bisect_v44b.py.
Any wrapper that references it by the old name (e.g. --probe-script defaults
in statistical probe scripts) must be updated to this filename.
"""

import argparse
import os
import subprocess
import sys
import time


def parse_csv(s, typ=str):
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def child(args):
    import torch
    import nvdiffrast.torch as dr

    device = "cuda"
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(
        f"case cells={args.cells} res={args.res} topo={args.topo} "
        f"color={args.color} pos_grad={args.pos_grad} stage={args.stage}",
        flush=True,
    )

    ctx = dr.RasterizeCudaContext(device=device)

    n = args.cells
    xs = torch.linspace(-0.9, 0.9, n + 1, device=device, dtype=torch.float32)
    ys = torch.linspace(-0.9, 0.9, n + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    pos = torch.stack(
        [xx, yy, torch.zeros_like(xx), torch.ones_like(xx)], dim=-1
    ).reshape(-1, 4).contiguous()
    pos.requires_grad_(args.pos_grad)

    if args.color == "interp":
        col = torch.stack(
            [(xx + 1) * 0.5, (yy + 1) * 0.5, torch.full_like(xx, 0.5)], dim=-1
        ).reshape(-1, 3).contiguous()
        attr = col[None, ...]
    elif args.color == "ones_attr":
        col = torch.ones((pos.shape[0], 3), device=device, dtype=torch.float32).contiguous()
        attr = col[None, ...]
    elif args.color == "direct_image":
        attr = None
    else:
        raise ValueError(args.color)

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
    tri = torch.tensor(tris, dtype=torch.int32, device=device)

    print("vertices:", pos.shape[0], "triangles:", tri.shape[0], flush=True)

    rast, _ = dr.rasterize(ctx, pos[None, ...], tri, resolution=[args.res, args.res])
    torch.cuda.synchronize()
    print("rasterize sync OK", flush=True)

    if args.color == "direct_image":
        color = torch.full((1, args.res, args.res, 3), 0.5, device=device, dtype=torch.float32)
    else:
        color, _ = dr.interpolate(attr, rast, tri)
        torch.cuda.synchronize()
        print("interpolate sync OK", flush=True)

    topo = None
    if args.topo == "explicit":
        topo = dr.antialias_construct_topology_hash(tri)
        torch.cuda.synchronize()
        print("topology hash sync OK", flush=True)
    elif args.topo == "implicit":
        topo = None
    else:
        raise ValueError(args.topo)

    if args.stage == "pre_aa":
        return

    if topo is None:
        aa = dr.antialias(color, rast, pos[None, ...], tri)
    else:
        aa = dr.antialias(color, rast, pos[None, ...], tri, topology_hash=topo)
    print("antialias call returned", flush=True)

    if args.stage == "call":
        return

    torch.cuda.synchronize()
    print("antialias sync OK", flush=True)

    if args.stage == "sync":
        return

    if args.stage == "diff":
        diff = (aa - color).detach().abs()
        torch.cuda.synchronize()
        print("diff max:", float(diff.max()), "diff sum:", float(diff.sum()), flush=True)
        print("changed:", int((diff.max(dim=-1).values > 0).sum().item()), flush=True)
        return

    if args.stage == "finite":
        v = torch.isfinite(aa).all()
        torch.cuda.synchronize()
        print("finite:", bool(v.item()), flush=True)
        return

    if args.stage == "max":
        v = aa.detach().abs().max()
        torch.cuda.synchronize()
        print("max:", float(v), flush=True)
        return

    if args.stage == "backward_pos":
        loss = aa.sum()
        loss.backward()
        torch.cuda.synchronize()
        print("pos.grad none:", pos.grad is None, flush=True)
        if pos.grad is not None:
            print("pos.grad absmax:", float(pos.grad.abs().max()), flush=True)
            print("pos.grad finite:", bool(torch.isfinite(pos.grad).all().item()), flush=True)
        return

    raise ValueError(args.stage)


def parent(args):
    cells_list = parse_csv(args.cells_list, int)
    res_list = parse_csv(args.res_list, int)
    topo_list = parse_csv(args.topos, str)
    color_list = parse_csv(args.colors, str)
    pos_grad_list = parse_csv(args.pos_grads, str)
    stage_list = parse_csv(args.stages, str)

    cases = []
    for cells in cells_list:
        for res in res_list:
            for topo in topo_list:
                for color in color_list:
                    for pg in pos_grad_list:
                        for stage in stage_list:
                            cases.append((cells, res, topo, color, pg, stage))

    print(f"AA grid bisect cases={len(cases)} timeout={args.timeout}s")
    print()

    rows = []
    for i, (cells, res, topo, color, pg, stage) in enumerate(cases, 1):
        label = f"cells={cells} res={res} topo={topo} color={color} pg={pg} stage={stage}"
        print(f"[{i:03d}/{len(cases):03d}] {label}", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child",
            "--cells", str(cells),
            "--res", str(res),
            "--topo", topo,
            "--color", color,
            "--pos-grad", pg,
            "--stage", stage,
        ]
        t0 = time.time()
        try:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=args.timeout)
            dt = time.time() - t0
            ok = cp.returncode == 0
            print(f"    -> {'OK' if ok else 'XX'} rc={cp.returncode} time={dt:.2f}s", flush=True)
            if args.verbose or not ok:
                if cp.stdout.strip():
                    print("    stdout:")
                    for line in cp.stdout.rstrip().splitlines():
                        print("      " + line)
                if cp.stderr.strip():
                    print("    stderr tail:")
                    for line in cp.stderr.rstrip().splitlines()[-40:]:
                        print("      " + line)
            rows.append((label, ok, cp.returncode, dt))
        except subprocess.TimeoutExpired:
            print("    -> TIMEOUT", flush=True)
            rows.append((label, False, "TIMEOUT", args.timeout))

    print("\n===== SUMMARY =====")
    for label, ok, rc, dt in rows:
        print(f"{label:<95s} {'OK' if ok else 'XX'} rc={rc} time={dt:.2f}s")
    passed = sum(1 for _, ok, _, _ in rows if ok)
    print(f"\npassed={passed} failed={len(rows)-passed} total={len(rows)}")
    return 0 if passed == len(rows) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)

    # Parent args.
    ap.add_argument("--cells-list", default="1,2,4,8,16")
    ap.add_argument("--res-list", default="64,128,256")
    ap.add_argument("--topos", default="implicit,explicit")
    ap.add_argument("--colors", default="interp")
    ap.add_argument("--pos-grads", default="1")
    ap.add_argument("--stages", default="sync")
    ap.add_argument("--verbose", action="store_true")

    # Child args.
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--topo", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--color", choices=["interp", "ones_attr", "direct_image"], default="interp")
    ap.add_argument("--pos-grad", choices=["0", "1"], default="1")
    ap.add_argument("--stage", choices=["pre_aa", "call", "sync", "diff", "finite", "max", "backward_pos"], default="sync")

    args = ap.parse_args()
    if args.child:
        args.pos_grad = args.pos_grad == "1"
        child(args)
        return 0
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
