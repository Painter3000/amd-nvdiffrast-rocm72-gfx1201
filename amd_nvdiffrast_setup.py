#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    if cwd:
        print("  cwd:", cwd)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def path_expand(s: str) -> Path:
    return Path(s).expanduser().resolve()


def detect_venv(args_venv: str | None, root: Path) -> Path:
    if args_venv:
        return path_expand(args_venv)
    if os.environ.get("VIRTUAL_ENV"):
        return path_expand(os.environ["VIRTUAL_ENV"])
    return root / "venv"


def py_in_venv(venv: Path) -> Path:
    py = venv / "bin" / "python"
    if not py.exists():
        die(f"venv python not found: {py}")
    return py


def check_torch(venv: Path, require_arch: str) -> None:
    py = py_in_venv(venv)
    code = '''
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("HIP:", getattr(torch.version, "hip", None))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("torch.cuda is not available; ROCm PyTorch is not working")
'''
    run([py, "-c", code])
    print(f"Target arch requested: {require_arch}")


def ensure_repo(repo: Path, repo_url: str, branch: str | None) -> None:
    if (repo / ".git").exists():
        print(f"Using existing nvdiffrast repo: {repo}")
        return
    if repo.exists() and any(repo.iterdir()):
        die(f"repo path exists but is not an empty git repo: {repo}")
    repo.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", repo_url, str(repo)]
    if branch:
        cmd = ["git", "clone", "--branch", branch, repo_url, str(repo)]
    run(cmd)


def make_env(args: argparse.Namespace, root: Path, repo: Path, venv: Path, bundle_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "ROOT": str(root),
        "REPO": str(repo),
        "VENV": str(venv),
        "BUNDLE_DIR": str(bundle_dir),
        "ROCM_PATH": args.rocm_path,
        "PYTORCH_ROCM_ARCH": args.arch,
        "FORCE_CUDA": "1",
        "MAX_JOBS": str(args.max_jobs),
        "TORCH_DISABLE_ADDR2LINE": env.get("TORCH_DISABLE_ADDR2LINE", "1"),
    })

    # ROCm clang avoids GCC ICEs that were observed in common_hip.cpp.
    env.setdefault("CC", str(Path(args.rocm_path) / "llvm" / "bin" / "clang"))
    env.setdefault("CXX", str(Path(args.rocm_path) / "llvm" / "bin" / "clang++"))

    compat = root / "nvdiffrast_rocm_cuda_compat"
    cpath_parts = [str(compat), str(Path(args.rocm_path) / "include" / "hipsparse")]
    if env.get("CPATH"):
        cpath_parts.append(env["CPATH"])
    env["CPATH"] = ":".join(cpath_parts)
    return env


def copy_tests(bundle_dir: Path) -> None:
    tests_src = HERE / "tests"
    if not tests_src.exists():
        print("tests directory not found; skipping copy:", tests_src)
        return

    names = [
        "test_min_triangle.sh",
        "test_advanced_pipeline.py",
        "run_test_advanced_pipeline.sh",
        "test_rasterizer_gradients_v4.py",
        "test_v52_marker_integrity.sh",
        "run_v52_validation.sh",
        "run_v52_stress.sh",
        "nvdiffrast_path_probe_v1.py",
        "nvdiffrast_aa_matrix_probe_v4.py",
        "aa_matrix_stat_probe.py",
        "test_antialias_backward_matrix_v52.py",
        "nvdiffrast_interpolate_resolution_probe_v6.py",
        "test_many_triangles_stress_v1.py",
    ]

    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = tests_src / name
        if src.exists():
            dst = bundle_dir / name
            shutil.copy2(src, dst)
            if dst.suffix in {".sh", ".py"}:
                dst.chmod(0o755)
            print("copied:", dst)


def clean_rebuild(repo: Path, venv: Path, env: dict[str, str]) -> None:
    py = py_in_venv(venv)
    site_code = "import site; print(site.getsitepackages()[0])"
    site = Path(subprocess.check_output([str(py), "-c", site_code], text=True).strip())

    script = f'''
set -euo pipefail
source "{venv}/bin/activate"
cd "{repo}"

pip uninstall -y nvdiffrast || true

SITE="{site}"
rm -rf "$SITE"/nvdiffrast
rm -rf "$SITE"/nvdiffrast-*.dist-info
rm -rf "$SITE"/__editable__*nvdiffrast*
rm -f  "$SITE"/_nvdiffrast_c*.so

rm -rf build/ dist/ ./*.egg-info
find . -name "*.o" -delete
find . -name "*.so" -delete
find . -name "*.d" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf {{}} +

python -m pip install . --no-build-isolation --no-cache-dir -v
'''
    run(["bash", "-lc", script], env=env)


def verify_v52_markers(repo: Path) -> None:
    src = repo / "csrc" / "torch" / "torch_antialias.cpp"
    hip = repo / "csrc" / "torch" / "torch_antialias_hip.cpp"
    if not src.exists() or not hip.exists():
        die(f"missing antialias source files:\n  {src}\n  {hip}")

    def count(marker: str, path: Path) -> int:
        return path.read_text(errors="replace").count(marker)

    v47_src = count("v47 FIX", src)
    v47_hip = count("v47 FIX", hip)
    v51_src = count("v51 FIX", src)
    v51_hip = count("v51 FIX", hip)

    print("\n=== v52 marker check ===")
    print(f"{src}: v47={v47_src} v51={v51_src}")
    print(f"{hip}: v47={v47_hip} v51={v51_hip}")

    if (v47_src, v47_hip) != (1, 1):
        die("v47 FIX must be present exactly once in both torch_antialias.cpp and torch_antialias_hip.cpp")
    if (v51_src, v51_hip) != (1, 1):
        die("v51 FIX must be present exactly once in both torch_antialias.cpp and torch_antialias_hip.cpp")

    bad_patterns = [
        "v46", "v48", "v49", "v50",
        "hipStreamSynchronize", "hipDeviceSynchronize",
        "cudaStreamSynchronize", "cudaDeviceSynchronize",
    ]
    for path in (src, hip):
        text = path.read_text(errors="replace")
        for pat in bad_patterns:
            if pat in text:
                die(f"diagnostic marker/sync found in active source: {pat} in {path}")

    print("OK: v47/v51 markers clean and no diagnostic sync markers found.")


def run_v52_tests(repo: Path, tests_dir: Path, env: dict[str, str], validation: str) -> None:
    if validation == "none":
        return

    env = env.copy()
    env["REPO"] = str(repo)
    env.setdefault("AMD_SERIALIZE_KERNEL", "3")
    env.setdefault("TORCH_DISABLE_ADDR2LINE", "1")

    marker = tests_dir / "test_v52_marker_integrity.sh"
    if marker.exists():
        print("\n=== Run v52 marker integrity test ===")
        run(["bash", str(marker)], cwd=tests_dir, env=env)
    else:
        print("marker test not found; using internal marker check")
        verify_v52_markers(repo)

    if validation in {"quick", "full", "stress"}:
        val = tests_dir / "run_v52_validation.sh"
        if val.exists():
            print("\n=== Run v52 validation suite ===")
            run(["bash", str(val)], cwd=tests_dir, env=env)
        else:
            die(f"missing validation wrapper: {val}")

    if validation in {"full", "stress"}:
        stat = tests_dir / "aa_matrix_stat_probe.py"
        if stat.exists():
            print("\n=== Run v52 forward AA statistical probe ===")
            run([
                py_in_venv(Path(env["VENV"])), str(stat),
                "--runs", "20",
                "--shapes", "single,grid1,grid4,grid16",
                "--res-list", "160,180,182,192,224,256",
                "--colors", "interp",
                "--hashes", "explicit",
                "--label", "final v52 forward AA",
            ], cwd=tests_dir, env=env)

    if validation == "stress":
        stress = tests_dir / "run_v52_stress.sh"
        if stress.exists():
            print("\n=== Run v52 stress suite ===")
            run(["bash", str(stress)], cwd=tests_dir, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build and validate nvdiffrast ROCm 7.2 / RDNA4 gfx1201 final v52 patch stack."
    )
    ap.add_argument("--workdir", default="~/therock_test", help="Root workdir. Default: ~/therock_test")
    ap.add_argument("--repo", default=None, help="nvdiffrast repo path. Default: <workdir>/nvdiffrast")
    ap.add_argument("--venv", default=None, help="Existing ROCm PyTorch venv. Default: $VIRTUAL_ENV or <workdir>/venv")
    ap.add_argument("--bundle-dir", default=None, help="Generated reinstall/test bundle path")
    ap.add_argument("--repo-url", default="https://github.com/NVlabs/nvdiffrast.git", help="nvdiffrast upstream URL")
    ap.add_argument("--branch", default=None, help="Optional nvdiffrast branch/tag")
    ap.add_argument("--rocm-path", default="/opt/rocm", help="ROCm path. Default: /opt/rocm")
    ap.add_argument("--arch", default="gfx1201", help="PYTORCH_ROCM_ARCH. Default: gfx1201")
    ap.add_argument("--max-jobs", type=int, default=1, help="Build parallelism. Default: 1")
    ap.add_argument("--skip-clone", action="store_true", help="Do not clone; require repo to exist")
    ap.add_argument("--skip-tests", action="store_true", help="Alias for --validation none")
    ap.add_argument("--validation", choices=["none", "quick", "full", "stress"], default="quick",
                    help="Validation level after build. Default: quick")
    args = ap.parse_args(argv)

    if args.skip_tests:
        args.validation = "none"

    root = path_expand(args.workdir)
    repo = path_expand(args.repo) if args.repo else root / "nvdiffrast"
    venv = detect_venv(args.venv, root)
    bundle_dir = path_expand(args.bundle_dir) if args.bundle_dir else root / "nvdiffrast_rocm72_gfx1201_final_v52_bundle"

    print("=== AMD nvdiffrast ROCm72/gfx1201 final v52 setup ===")
    print("workdir:   ", root)
    print("repo:      ", repo)
    print("venv:      ", venv)
    print("bundle:    ", bundle_dir)
    print("rocm path: ", args.rocm_path)
    print("arch:      ", args.arch)
    print("validation:", args.validation)

    root.mkdir(parents=True, exist_ok=True)

    if args.skip_clone:
        if not repo.exists():
            die(f"--skip-clone was set but repo does not exist: {repo}")
    else:
        ensure_repo(repo, args.repo_url, args.branch)

    check_torch(venv, args.arch)

    generator = HERE / "scripts" / "nvdiffrast_rocm72_bundle_v52_generator.sh"
    final_stack = HERE / "patches" / "apply_final_rocm72_gfx1201_v52.sh"

    if not generator.exists():
        die(f"missing generator: {generator}")
    if not final_stack.exists():
        die(f"missing final v52 patch stack script: {final_stack}")

    env = make_env(args, root, repo, venv, bundle_dir)

    print("\n=== Generate v52 ROCm runtime baseline bundle ===")
    run(["bash", str(generator)], env=env)

    reinstall = bundle_dir / "reinstall_nvdiffrast_rocm72_gfx1201.sh"
    if not reinstall.exists():
        die(f"reinstall script was not generated: {reinstall}")

    print("\n=== Apply v52 runtime baseline and initial build ===")
    env_runtime = env.copy()
    env_runtime["MODE"] = "runtime"
    run(["bash", str(reinstall)], env=env_runtime)

    print("\n=== Apply final v52 patch stack ===")
    env_stack = env.copy()
    env_stack["PATCH_DIR"] = str(HERE / "patches")
    run(["bash", str(final_stack)], cwd=HERE, env=env_stack)

    print("\n=== Clean rebuild after final v52 stack ===")
    clean_rebuild(repo, venv, env)

    print("\n=== Verify v52 markers after clean rebuild / hipify ===")
    verify_v52_markers(repo)

    print("\n=== Copy validation tests into bundle ===")
    copy_tests(bundle_dir)

    print("\n=== Run validation ===")
    run_v52_tests(repo, HERE / "tests", env, args.validation)

    print("\n=== DONE ===")
    print("Patched nvdiffrast is installed in:", venv)
    print("Patched nvdiffrast repo:", repo)
    print("Validation bundle:", bundle_dir)
    print("\nRecommended commands:")
    print(f'  cd "{repo}" && grep -c "v47 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp')
    print(f'  cd "{repo}" && grep -c "v51 FIX" csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp')
    print(f'  cd "{HERE / "tests"}" && ./run_v52_validation.sh')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
