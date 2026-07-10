#!/usr/bin/env bash
set -euo pipefail

# v38b: additional guard for antialias backward color-only mode on ROCm/RDNA.
#
# Observed after v38:
#   constant_color_only      PASS
#   ramp_color_posgrad       PASS
#   ramp_color_only          can hang/stall
#
# Interpretation:
#   When dd != 0, AntialiasGradKernel enters the position-gradient branch.
#   In color-only autograd, p.gradPos may be null/invalid because pos does not
#   require gradients. Guard this branch on HIP/ROCm.

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
AA="$REPO/csrc/common/antialias.cu"

if [[ ! -f "$AA" ]]; then
  echo "ERROR: antialias.cu not found: $AA" >&2
  exit 1
fi

cp -n "$AA" "$AA.before_rocm_antialias_grad_v38b" || true

python - "$AA" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text()

guard = """#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // In color-only backward, gradPos may be null/invalid. If dd != 0 but
        // positions do not require gradients, skip the position-gradient branch.
        if (p.gradPos == NULL)
            continue;
#endif
"""

if "if (p.gradPos == NULL)" in s:
    print("gradPos guard already present")
    raise SystemExit(0)

needle = """        if (noGrad)
            continue;

        // Fetch vertex indices of the active edge and their positions.
"""

replacement = """        if (noGrad)
            continue;

""" + guard + """
        // Fetch vertex indices of the active edge and their positions.
"""

if needle not in s:
    raise SystemExit("ERROR: insertion point not found near noGrad branch")

s = s.replace(needle, replacement)
path.write_text(s)
print(f"patched {path}")
print("gradPos guard count:", path.read_text().count("if (p.gradPos == NULL)"))
PY

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
