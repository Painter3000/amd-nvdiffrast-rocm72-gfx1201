#!/usr/bin/env bash
set -euo pipefail

# v39b diagnostic patch: avoid early return before later __syncthreads()
# in nvdiffrast antialias persistent-thread loops on HIP/ROCm.
#
# Problem pattern in AntialiasFwdAnalysisKernel and AntialiasGradKernel:
#
#   for(;;) {
#       __syncthreads();
#       if (threadIdx.x == 0)
#           s_base = atomicAdd(..., THREADS_PER_BLOCK);
#       __syncthreads();
#       int thread_idx = s_base + threadIdx.x;
#       if (thread_idx >= workCount)
#           return;
#       ...
#   }
#
# On CUDA this pattern is commonly tolerated. On HIP/RDNA it is suspicious:
# some threads may return while other threads in the same block continue and
# reach the next loop iteration's __syncthreads().
#
# Patch logic:
#
#   if (s_base >= workCount)
#       break;       // entire CTA has no work left
#   if (thread_idx >= workCount)
#       continue;    // only this lane has no item in the final partial chunk
#
# This keeps all threads in the CTA participating in later barriers until the
# entire block is done.

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
AA="$REPO/csrc/common/antialias.cu"

if [[ ! -f "$AA" ]]; then
  echo "ERROR: antialias.cu not found: $AA" >&2
  exit 1
fi

cp -n "$AA" "$AA.before_rocm_antialias_persistent_loop_v39b" || true

python - "$AA" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text()

if "ROCm/RDNA persistent-loop guard" in s:
    print("v39b persistent-loop guard already present")
    raise SystemExit(0)

old = """        int thread_idx = s_base + threadIdx.x;
        if (thread_idx >= workCount)
            return;
"""

new = """        int thread_idx = s_base + threadIdx.x;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // ROCm/RDNA persistent-loop guard:
        // Do not return from only part of a CTA before later __syncthreads().
        // If the whole CTA fetched beyond the end, all lanes leave together.
        if (s_base >= workCount)
            break;
        // In the final partial chunk, inactive lanes skip work but remain in
        // the loop so they can participate in the next barrier.
        if (thread_idx >= workCount)
            continue;
#else
        if (thread_idx >= workCount)
            return;
#endif
"""

count = s.count(old)
if count != 2:
    print(f"ERROR: expected exactly 2 persistent-loop return sites, found {count}")
    print("The source may already be patched or formatted differently.")
    raise SystemExit(1)

s = s.replace(old, new, 2)
path.write_text(s)
print(f"patched {path}")
print("persistent-loop guard count:", path.read_text().count("ROCm/RDNA persistent-loop guard"))
PY

echo
echo "Quick source check:"
grep -n "ROCm/RDNA persistent-loop guard" "$AA" || true

echo
echo "Rebuild:"
cat <<'EOF'
source ~/therock_test/venv/bin/activate
cd ~/therock_test/nvdiffrast

rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast

export PYTORCH_ROCM_ARCH=gfx1201
export FORCE_CUDA=1
export MAX_JOBS=1
export CPATH="$HOME/therock_test/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse:${CPATH:-}"

python -m pip install . --no-build-isolation -v
EOF
