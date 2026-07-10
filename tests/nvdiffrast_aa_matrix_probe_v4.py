#!/usr/bin/env python3
"""
nvdiffrast_aa_matrix_probe_v4.py
Focused antialias-forward matrix probe for ROCm/HIP nvdiffrast.
Every case runs in a fresh subprocess with its own timeout.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import dataclass
from typing import List, Tuple


def make_single_triangle(torch, device="cuda"):
    pos = torch.tensor([[-0.75,-0.75,0.0,1.0],[0.75,-0.75,0.0,1.0],[0.0,0.75,0.0,1.0]], device=device, dtype=torch.float32).contiguous()
    tri = torch.tensor([[0,1,2]], device=device, dtype=torch.int32).contiguous()
    col = torch.tensor([[1.0,0.1,0.2],[0.1,1.0,0.3],[0.2,0.3,1.0]], device=device, dtype=torch.float32).contiguous()
    return pos, tri, col


def make_grid(torch, cells:int, device="cuda"):
    n = cells
    xs = torch.linspace(-0.95,0.95,n+1,device=device,dtype=torch.float32)
    ys = torch.linspace(-0.95,0.95,n+1,device=device,dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([xx, yy, torch.zeros_like(xx), torch.ones_like(xx)], dim=-1).reshape(-1,4).contiguous()
    col = torch.stack([(xx+1.0)*0.5, (yy+1.0)*0.5, 0.25+0.25*torch.sin(xx*8.0)], dim=-1).reshape(-1,3).contiguous()
    tris=[]; stride=n+1
    for y in range(n):
        for x in range(n):
            v00=y*stride+x; v10=v00+1; v01=(y+1)*stride+x; v11=v01+1
            tris.append((v00,v10,v11)); tris.append((v00,v11,v01))
    tri = torch.tensor(tris, device=device, dtype=torch.int32).contiguous()
    return pos, tri, col


def make_direct_color(torch, res:int, device="cuda"):
    yy, xx = torch.meshgrid(torch.linspace(0.0,1.0,res,device=device,dtype=torch.float32), torch.linspace(0.0,1.0,res,device=device,dtype=torch.float32), indexing="ij")
    return torch.stack([xx, yy, 0.25+0.5*xx], dim=-1)[None,...].contiguous()


def finite_summary(torch, name, x):
    ok = torch.isfinite(x).all().item()
    print(f"{name}: shape={tuple(x.shape)} finite={ok} min={x.min().item():.6f} max={x.max().item():.6f} sum={x.sum().item():.6f} absmax={x.abs().max().item():.6f}", flush=True)
    if not ok: raise RuntimeError(f"{name} non-finite")


def child_main(args)->int:
    import torch
    import nvdiffrast.torch as dr
    if not torch.cuda.is_available(): raise RuntimeError("torch.cuda.is_available() is False")
    shape=args.child_shape; res=args.child_res; color_mode=args.child_color; hash_mode=args.child_hash
    if shape == "single": pos, tri, col = make_single_triangle(torch)
    elif shape.startswith("grid"): pos, tri, col = make_grid(torch, int(shape.replace("grid","")))
    else: raise RuntimeError(f"unknown shape {shape}")
    pos_b=pos[None,...].contiguous()
    print(f"case shape={shape} res={res} color={color_mode} hash={hash_mode} verts={pos.shape[0]} tris={tri.shape[0]}", flush=True)
    print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} device={torch.cuda.get_device_name(0)}", flush=True)
    ctx=dr.RasterizeCudaContext(device="cuda")
    rast,_=dr.rasterize(ctx,pos_b,tri,resolution=[res,res]); torch.cuda.synchronize(); finite_summary(torch,"rast",rast)
    print(f"covered={(rast[...,3]>0).sum().item()} unique_tri_id_count={torch.unique(rast[...,3]).numel()}", flush=True)
    if color_mode=="interp": color,_=dr.interpolate(col[None,...].contiguous(),rast,tri)
    elif color_mode=="direct": color=make_direct_color(torch,res)
    else: raise RuntimeError(f"unknown color {color_mode}")
    torch.cuda.synchronize(); finite_summary(torch,"color",color)
    topology_hash=None
    if hash_mode=="explicit":
        topology_hash=dr.antialias_construct_topology_hash(tri); torch.cuda.synchronize(); print("topology_hash OK", flush=True)
    elif hash_mode!="auto": raise RuntimeError(f"unknown hash {hash_mode}")
    if topology_hash is None: aa=dr.antialias(color,rast,pos_b,tri)
    else: aa=dr.antialias(color,rast,pos_b,tri,topology_hash=topology_hash)
    torch.cuda.synchronize(); finite_summary(torch,"aa",aa); finite_summary(torch,"abs(aa-color)",(aa-color).abs())
    print("PASS", flush=True); return 0

@dataclass
class CaseResult:
    shape:str; res:int; color:str; hash_mode:str; status:str; rc:int|None; seconds:float; tail:str


def csv(s): return [x.strip() for x in s.split(',') if x.strip()]
def rescsv(s): return [int(x.strip()) for x in s.split(',') if x.strip()]


def run_case(script,shape,res,color,hash_mode,timeout,verbose):
    env=os.environ.copy(); env.setdefault("AMD_SERIALIZE_KERNEL","3"); env.setdefault("TORCH_DISABLE_ADDR2LINE","1"); env.setdefault("PYTHONUNBUFFERED","1")
    cmd=[sys.executable,script,"--child","--child-shape",shape,"--child-res",str(res),"--child-color",color,"--child-hash",hash_mode]
    start=time.monotonic()
    try:
        cp=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        sec=time.monotonic()-start; out=cp.stdout or ""
        if verbose: print(out,end="" if out.endswith("\n") else "\n")
        if cp.returncode==0: status="OK"
        elif cp.returncode<0: status=f"CRASH(signal {-cp.returncode})"
        else: status=f"FAIL(rc={cp.returncode})"
        return CaseResult(shape,res,color,hash_mode,status,cp.returncode,sec,"\n".join(out.splitlines()[-14:]))
    except subprocess.TimeoutExpired as e:
        sec=time.monotonic()-start; out=e.stdout or ""
        if isinstance(out,bytes): out=out.decode("utf-8",errors="replace")
        return CaseResult(shape,res,color,hash_mode,"TIMEOUT",124,sec,"\n".join(str(out).splitlines()[-14:]))


def parent_main(args)->int:
    shapes=csv(args.shapes); colors=csv(args.colors); hashes=csv(args.hashes); res_list=rescsv(args.res_list)
    cases=[(sh,r,c,h) for sh in shapes for c in colors for h in hashes for r in res_list]
    print(f"AA matrix probe: cases={len(cases)} timeout={args.timeout}s", flush=True)
    print(f"shapes={shapes}\ncolors={colors}\nhashes={hashes}\nres={res_list}\n", flush=True)
    results=[]; script=os.path.abspath(__file__)
    for i,(sh,r,c,h) in enumerate(cases,1):
        print(f"[{i:03d}/{len(cases):03d}] {sh:>6} res={r:<3} color={c:<6} hash={h:<8}", flush=True)
        cr=run_case(script,sh,r,c,h,args.timeout,args.verbose); results.append(cr)
        print(f"    -> {cr.status:<16} rc={cr.rc} time={cr.seconds:.2f}s", flush=True)
        if cr.status!="OK" and not args.verbose:
            print("    tail:")
            for line in cr.tail.splitlines()[-8:]: print(f"      {line}")
        if args.stop_on_fail and cr.status!="OK": break
    print("\n===== SUMMARY BY CASE =====", flush=True); failed=0
    for r in results:
        print(f"{r.shape:>6}  res={r.res:<3}  color={r.color:<6}  hash={r.hash_mode:<8}  {r.status}", flush=True)
        failed += (r.status!="OK")
    print("\n===== COMPACT MATRIX =====", flush=True)
    keys=[]
    for r in results:
        k=(r.shape,r.color,r.hash_mode)
        if k not in keys: keys.append(k)
    width=max([len(str(x)) for x in res_list]+[3])
    header="shape/color/hash".ljust(24)+" "+" ".join(str(r).rjust(width) for r in res_list)
    print(header); print("-"*len(header))
    by={(r.shape,r.color,r.hash_mode,r.res):r for r in results}
    for k in keys:
        label=f"{k[0]}/{k[1]}/{k[2]}".ljust(24); vals=[]
        for r in res_list:
            x=by.get((k[0],k[1],k[2],r))
            vals.append(("?" if x is None else ("OK" if x.status=="OK" else ("TO" if x.status=="TIMEOUT" else "XX"))).rjust(width))
        print(label+" "+" ".join(vals))
    print(f"\npassed={len(results)-failed} failed_or_timeout={failed} total={len(results)}", flush=True)
    return 0 if failed==0 else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--timeout",type=int,default=30)
    p.add_argument("--shapes",default="single,grid1,grid4,grid16")
    p.add_argument("--res-list",default="64,128,160,164,168,172,176,180,184,188,192,224,256")
    p.add_argument("--colors",default="interp")
    p.add_argument("--hashes",default="explicit")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--stop-on-fail",action="store_true")
    p.add_argument("--child",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--child-shape",default="",help=argparse.SUPPRESS)
    p.add_argument("--child-res",type=int,default=0,help=argparse.SUPPRESS)
    p.add_argument("--child-color",default="",help=argparse.SUPPRESS)
    p.add_argument("--child-hash",default="",help=argparse.SUPPRESS)
    args=p.parse_args()
    return child_main(args) if args.child else parent_main(args)

if __name__=="__main__": raise SystemExit(main())
