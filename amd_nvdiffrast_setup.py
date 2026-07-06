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
    print("\n$", " ".join(cmd))
    if cwd:
        print("  cwd:", cwd)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


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
    run([str(py), "-c", code])
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


def copy_tests(bundle_dir: Path) -> None:
    tests_src = HERE / "tests"
    for name in ["test_advanced_pipeline.py", "run_test_advanced_pipeline.sh"]:
        src = tests_src / name
        if src.exists():
            dst = bundle_dir / name
            shutil.copy2(src, dst)
            dst.chmod(0o755)
            print("copied:", dst)


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
    })

    compat = root / "nvdiffrast_rocm_cuda_compat"
    cpath_parts = [
        str(compat),
        str(Path(args.rocm_path) / "include" / "hipsparse"),
    ]
    if env.get("CPATH"):
        cpath_parts.append(env["CPATH"])
    env["CPATH"] = ":".join(cpath_parts)
    return env


def rebuild_after_v36(repo: Path, venv: Path, env: dict[str, str]) -> None:
    script = f'''
set -euo pipefail
source "{venv}/bin/activate"
cd "{repo}"
rm -rf build/ dist/ ./*.egg-info
pip uninstall -y nvdiffrast || true
python -m pip install . --no-build-isolation -v
'''
    run(["bash", "-lc", script], env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build and validate nvdiffrast ROCm 7.2 / RDNA gfx1201 v36 patch."
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
    ap.add_argument("--skip-tests", action="store_true", help="Build only, do not run tests")
    ap.add_argument("--advanced-test", action="store_true", help="Also run 256/512 forward+backward test")
    args = ap.parse_args(argv)

    root = path_expand(args.workdir)
    repo = path_expand(args.repo) if args.repo else root / "nvdiffrast"
    venv = detect_venv(args.venv, root)
    bundle_dir = path_expand(args.bundle_dir) if args.bundle_dir else root / "nvdiffrast_rocm72_gfx1201_reinstall_bundle_v4"

    print("=== AMD nvdiffrast ROCm72/gfx1201 v36 setup ===")
    print("workdir:   ", root)
    print("repo:      ", repo)
    print("venv:      ", venv)
    print("bundle:    ", bundle_dir)
    print("rocm path: ", args.rocm_path)
    print("arch:      ", args.arch)

    root.mkdir(parents=True, exist_ok=True)

    if args.skip_clone:
        if not repo.exists():
            die(f"--skip-clone was set but repo does not exist: {repo}")
    else:
        ensure_repo(repo, args.repo_url, args.branch)

    check_torch(venv, args.arch)

    generator = HERE / "scripts" / "nvdiffrast_rocm72_bundle_v4_generator.sh"
    patch_v36 = HERE / "patches" / "patch_rocm_v36_wave32.sh"

    if not generator.exists():
        die(f"missing generator: {generator}")
    if not patch_v36.exists():
        die(f"missing v36 patch: {patch_v36}")

    env = make_env(args, root, repo, venv, bundle_dir)

    print("\n=== Generate v4 reinstall bundle ===")
    run(["bash", str(generator)], env=env)

    reinstall = bundle_dir / "reinstall_nvdiffrast_rocm72_gfx1201.sh"
    if not reinstall.exists():
        die(f"reinstall script was not generated: {reinstall}")

    print("\n=== Apply v4 runtime baseline and build once ===")
    env_runtime = env.copy()
    env_runtime["MODE"] = "runtime"
    run(["bash", str(reinstall)], env=env_runtime)

    print("\n=== Apply v36 RDNA Wave32 FineRaster fix ===")
    run(["bash", str(patch_v36)], cwd=repo, env=env)

    print("\n=== Rebuild nvdiffrast after v36 ===")
    rebuild_after_v36(repo, venv, env)

    print("\n=== Copy validation tests into bundle ===")
    copy_tests(bundle_dir)

    if not args.skip_tests:
        print("\n=== Run minimal triangle test ===")
        run(["bash", str(bundle_dir / "test_min_triangle.sh")], cwd=bundle_dir, env=env)

        if args.advanced_test:
            print("\n=== Run advanced forward+backward test ===")
            run(["bash", str(bundle_dir / "run_test_advanced_pipeline.sh")], cwd=bundle_dir, env=env)

    print("\n=== DONE ===")
    print("Patched nvdiffrast is installed in:", venv)
    print("Validation bundle:", bundle_dir)
    print("\nRecommended next command:")
    print(f'  cd "{bundle_dir}" && ./run_test_advanced_pipeline.sh')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
