#!/usr/bin/env bash
set -euo pipefail

AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" \
TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}" \
python - <<'PY'
import torch
import nvdiffrast.torch as dr

device = "cuda"
ctx = dr.RasterizeCudaContext(device=device)

pos = torch.tensor([[
    [-1.0, -1.0, 0.5, 1.0],
    [ 1.0, -1.0, 0.5, 1.0],
    [ 0.0,  1.0, 0.5, 1.0],
]], device=device, dtype=torch.float32)

tri = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)

rast, _ = dr.rasterize(ctx, pos, tri, resolution=[16, 16])
torch.cuda.synchronize()

ch3 = rast[0, :, :, 3].detach().cpu()

print("finite:", torch.isfinite(rast).all().item())
print("sum:", rast.sum().item())
print("covered:", (rast[..., 3] > 0).sum().item())
print("tri id max:", rast[..., 3].max().item())
print("unique tri ids:", torch.unique(rast[..., 3]).detach().cpu().tolist())
print("row0:")
print(ch3[0, :10])
print("channel3:")
print(ch3)

assert torch.isfinite(rast).all().item()
assert (rast[..., 3] > 0).sum().item() > 0
assert rast[..., 3].max().item() >= 1.0
PY
