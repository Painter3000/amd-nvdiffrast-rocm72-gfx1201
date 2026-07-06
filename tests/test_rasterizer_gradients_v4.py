#!/usr/bin/env python3
"""
Finite-Differenzen-Gradiententest v4 - nvdiffrast ROCm / gfx1201

Warum v4?
---------
v3 nutzte ein sehr grosses Dreieck ausserhalb des Clipspace. Das ist als
"Interior ohne Silhouette"-Idee zwar logisch, kann aber genau die Out-of-bounds /
Clipping-/BBox-Pfade stressen, die fuer einen sauberen Gradienten-FD-Test nicht
noetig sind.

v4 ist konservativer:
  - Interior-Test nutzt ein normales Dreieck komplett innerhalb des Clipspace.
  - Der Loss wird nur auf einem kleinen zentralen Crop berechnet.
  - Dieser Crop liegt weit weg von Silhouette und Dreieckskanten.
  - Dadurch testen wir Interpolate/Rasterize-Gradienten im glatten Inneren,
    nicht diskrete Kanten-/Coverage-Aenderungen.

Nutzung:
  python test_rasterizer_gradients_v4.py --test interior
  python test_rasterizer_gradients_v4.py --test silhouette

Bei HIP-Crash:
  frische Shell starten, dann erneut testen.
"""

import argparse
import traceback

import torch
import nvdiffrast.torch as dr


def sync(label: str) -> None:
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"\n❌ HIP/CUDA synchronize failed after: {label}")
        print(type(exc).__name__, exc)
        raise


def make_interior_triangle(device):
    """Ein normales In-Clip-Dreieck.

    Der zentrale Crop liegt sicher im Dreiecksinneren. Keine riesigen Offscreen-
    Koordinaten, keine interne Quad-Diagonale.
    """
    pos = torch.tensor(
        [
            [-0.85, -0.85, 0.2, 1.0],
            [ 0.85, -0.85, 0.2, 1.0],
            [ 0.00,  0.85, 0.2, 1.0],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)

    # Nicht-symmetrische Farben.
    col = torch.tensor(
        [
            [1.00, 0.10, 0.20],
            [0.25, 1.20, 0.15],
            [0.05, 0.35, 1.40],
        ],
        dtype=torch.float32,
        device=device,
    )
    return pos, tri, col


def make_small_quad(device):
    """Kleines Quad mit echter Silhouette gegen Hintergrund fuer antialias()."""
    pos = torch.tensor(
        [
            [-0.55, -0.55, 0.0, 1.0],
            [ 0.55, -0.45, 0.0, 1.0],
            [ 0.50,  0.55, 0.0, 1.0],
            [-0.45,  0.50, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)
    col = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.1],
            [0.1, 0.2, 1.0],
            [0.8, 0.8, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return pos, tri, col


def render_plain(glctx, pos, tri, col, resolution):
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=resolution)
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    return color, rast


def render_silhouette(glctx, pos, tri, col, resolution):
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=resolution)
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    color = dr.antialias(color, rast, pos[None, ...], tri)
    return color, rast


def center_crop_slices(resolution, crop_size):
    h, w = resolution
    cy, cx = h // 2, w // 2
    r = crop_size // 2
    return slice(cy - r, cy + r), slice(cx - r, cx + r)


def weighted_color_loss(color, crop_y=None, crop_x=None):
    # Gewichtung bricht Symmetrien. Kein neues CUDA-Tensor-Allocation nach
    # einem potenziell fehlerhaften Kernel: weights wird per color.new_tensor
    # erzeugt, was aber immer noch eine GPU-Operation ist; deshalb synchronisieren
    # wir vor dem Loss-Aufruf.
    weights = color.new_tensor([0.37, -0.23, 0.61])
    if crop_y is not None and crop_x is not None:
        color = color[:, crop_y, crop_x, :]
    return (color * weights).sum()


def check_crop_coverage(rast, crop_y, crop_x, label):
    ch3 = rast[..., 3]
    crop = ch3[:, crop_y, crop_x]
    covered = (crop > 0).sum().item()
    total = crop.numel()
    unique = torch.unique(crop.detach()).cpu().tolist()

    print(f"{label} crop covered: {covered}/{total}")
    print(f"{label} crop tri IDs:  {unique}")

    if covered != total:
        raise RuntimeError(
            f"{label}: central crop is not fully covered. "
            f"Use smaller --crop-size or different geometry."
        )


def analytic_grad_interior(glctx, pos, tri, col, resolution, crop_size):
    pos = pos.detach().clone().requires_grad_(True)
    crop_y, crop_x = center_crop_slices(resolution, crop_size)

    color, rast = render_plain(glctx, pos, tri, col, resolution)
    sync("analytic interior forward")
    check_crop_coverage(rast, crop_y, crop_x, "analytic")

    loss = weighted_color_loss(color, crop_y, crop_x)
    loss.backward()
    sync("analytic interior backward")

    if pos.grad is None:
        raise RuntimeError("pos.grad is None after backward()")

    return loss.item(), pos.grad.detach().clone()


def numeric_grad_interior(glctx, pos, tri, col, resolution, eps, crop_size):
    base = pos.detach().clone()
    grad = torch.zeros_like(base)
    crop_y, crop_x = center_crop_slices(resolution, crop_size)

    with torch.no_grad():
        flat = base.view(-1)
        grad_flat = grad.view(-1)

        for i in range(base.numel()):
            orig = flat[i].item()

            flat[i] = orig + eps
            color_plus, rast_plus = render_plain(glctx, base, tri, col, resolution)
            sync(f"numeric +eps index {i}")
            check_crop_coverage(rast_plus, crop_y, crop_x, f"numeric +eps index {i}")
            loss_plus = weighted_color_loss(color_plus, crop_y, crop_x).item()

            flat[i] = orig - eps
            color_minus, rast_minus = render_plain(glctx, base, tri, col, resolution)
            sync(f"numeric -eps index {i}")
            check_crop_coverage(rast_minus, crop_y, crop_x, f"numeric -eps index {i}")
            loss_minus = weighted_color_loss(color_minus, crop_y, crop_x).item()

            flat[i] = orig
            grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad


def analytic_grad_silhouette(glctx, pos, tri, col, resolution):
    pos = pos.detach().clone().requires_grad_(True)
    color, rast = render_silhouette(glctx, pos, tri, col, resolution)
    sync("analytic silhouette forward")
    loss = weighted_color_loss(color)
    loss.backward()
    sync("analytic silhouette backward")

    if pos.grad is None:
        raise RuntimeError("pos.grad is None after backward()")

    return loss.item(), pos.grad.detach().clone()


def numeric_grad_silhouette(glctx, pos, tri, col, resolution, eps):
    base = pos.detach().clone()
    grad = torch.zeros_like(base)

    with torch.no_grad():
        flat = base.view(-1)
        grad_flat = grad.view(-1)

        for i in range(base.numel()):
            orig = flat[i].item()

            flat[i] = orig + eps
            color_plus, _ = render_silhouette(glctx, base, tri, col, resolution)
            sync(f"silhouette numeric +eps index {i}")
            loss_plus = weighted_color_loss(color_plus).item()

            flat[i] = orig - eps
            color_minus, _ = render_silhouette(glctx, base, tri, col, resolution)
            sync(f"silhouette numeric -eps index {i}")
            loss_minus = weighted_color_loss(color_minus).item()

            flat[i] = orig
            grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad


def compare(name, analytic, numeric, rel_tol, abs_tol, show_top=12):
    if not torch.isfinite(analytic).all():
        print("❌ analytic gradient contains NaN/Inf")
        return False
    if not torch.isfinite(numeric).all():
        print("❌ numeric gradient contains NaN/Inf")
        return False

    diff = (analytic - numeric).abs()
    denom = torch.maximum(analytic.abs(), numeric.abs()).clamp_min(1e-6)
    rel = diff / denom

    elementwise_ok = (diff <= abs_tol) | (rel <= rel_tol)
    passed = bool(elementwise_ok.all().item())

    max_abs = diff.max().item()
    max_rel = rel.max().item()

    print(f"\n--- {name} ---")
    print(f"analytic:\n{analytic}")
    print(f"numeric (FD):\n{numeric}")
    print(f"diff:\n{diff}")
    print(f"rel:\n{rel}")
    print(f"max |analytic - numeric| = {max_abs:.8f}")
    print(f"max relative error       = {max_rel:.8f}")

    if not passed:
        flat_diff = diff.view(-1)
        k = min(show_top, flat_diff.numel())
        vals, idxs = torch.topk(flat_diff, k)
        print("\nTop Abweichungen:")
        a_flat = analytic.view(-1)
        n_flat = numeric.view(-1)
        r_flat = rel.view(-1)
        for val, idx in zip(vals.tolist(), idxs.tolist()):
            print(
                f"  flat[{idx:02d}] "
                f"analytic={a_flat[idx].item(): .8f} "
                f"numeric={n_flat[idx].item(): .8f} "
                f"abs={val: .8f} "
                f"rel={r_flat[idx].item(): .8f}"
            )

    print("PASS" if passed else "FAIL")
    return passed


def run_interior(glctx, device, resolution, eps, rel_tol, abs_tol, crop_size):
    pos, tri, col = make_interior_triangle(device)
    loss_a, grad_a = analytic_grad_interior(glctx, pos, tri, col, resolution, crop_size)
    grad_n = numeric_grad_interior(glctx, pos, tri, col, resolution, eps, crop_size)
    print(f"analytic loss: {loss_a:.8f}")
    return compare("Interior gradients: central crop inside one in-clip triangle",
                   grad_a, grad_n, rel_tol, abs_tol)


def run_silhouette(glctx, device, resolution, eps, rel_tol, abs_tol):
    pos, tri, col = make_small_quad(device)
    loss_a, grad_a = analytic_grad_silhouette(glctx, pos, tri, col, resolution)
    grad_n = numeric_grad_silhouette(glctx, pos, tri, col, resolution, eps)
    print(f"analytic loss: {loss_a:.8f}")
    return compare("Silhouette gradients: antialias() path",
                   grad_a, grad_n, rel_tol, abs_tol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["interior", "silhouette"], required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--rel-tol", type=float, default=5e-2)
    parser.add_argument("--abs-tol", type=float, default=2e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Keine CUDA/HIP-GPU sichtbar.")

    if args.crop_size <= 0 or args.crop_size % 2 != 0:
        raise ValueError("--crop-size must be a positive even integer")

    device = "cuda"
    resolution = [args.resolution, args.resolution]
    glctx = dr.RasterizeCudaContext(device=device)

    print(f"Test: {args.test}, resolution={resolution}, eps={args.eps}, crop={args.crop_size}")
    print(f"torch: {torch.__version__}")
    print(f"hip: {getattr(torch.version, 'hip', None)}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    if args.test == "interior":
        ok = run_interior(glctx, device, resolution, args.eps,
                          args.rel_tol, args.abs_tol, args.crop_size)
    else:
        print("\nHinweis: antialias() kann bei einem echten ROCm-Kernelproblem hart abstuerzen.")
        print("Bei Crash danach frische Shell nutzen.\n")
        ok = run_silhouette(glctx, device, resolution, args.eps,
                            args.rel_tol, args.abs_tol)

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ TEST FAILED WITH EXCEPTION")
        traceback.print_exc()
        raise
