#!/usr/bin/env python3
"""
nvdiffrast_path_probe_v1.py

Isolierter Pfad-Test fuer nvdiffrast (ROCm/gfx1201).

Jeder einzelne Test ("case") laeuft in einem EIGENEN Subprozess mit Timeout.
Wenn ein HIP-Kernel crasht, haengt oder den GPU-Kontext vergiftet, betrifft
das nur diesen einen Subprozess - der Runner selbst laeuft weiter und
sammelt am Ende eine Uebersicht ueber alle Pfade.

Nutzung
-------
  python nvdiffrast_path_probe_v1.py --timeout 30 --quick
  python nvdiffrast_path_probe_v1.py --timeout 30
  python nvdiffrast_path_probe_v1.py --timeout 30 --only antialias
  python nvdiffrast_path_probe_v1.py --timeout 30 --skip antialias,texture

Funktionsweise
--------------
Ohne --run-case: Runner-Modus. Startet fuer jeden Case einen Subprozess
  `python nvdiffrast_path_probe_v1.py --run-case <name>` und klassifiziert
  das Ergebnis anhand von Returncode/Timeout/Signal.

Mit --run-case <name>: Fuehrt NUR diesen einen Test direkt aus (kein Runner-
  Verhalten). Das ist der Teil, der im Subprozess laeuft.
"""

import argparse
import subprocess
import sys
import time
import traceback


# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer die einzelnen Testfaelle (laufen jeweils im Subprozess)
# ---------------------------------------------------------------------------

def _sync(label):
    import torch
    print(f"[sync] {label}", flush=True)
    torch.cuda.synchronize()


def _make_single_triangle(device):
    import torch
    pos = torch.tensor(
        [[-0.6, -0.6, 0.0, 1.0],
         [ 0.6, -0.6, 0.0, 1.0],
         [ 0.0,  0.6, 0.0, 1.0]],
        dtype=torch.float32, device=device, requires_grad=True,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    col = torch.tensor(
        [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]],
        dtype=torch.float32, device=device,
    )
    return pos, tri, col


def _make_grid(device, cells):
    import torch
    n = cells
    xs = torch.linspace(-0.9, 0.9, n + 1, device=device, dtype=torch.float32)
    ys = torch.linspace(-0.9, 0.9, n + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack(
        [xx, yy, torch.zeros_like(xx), torch.ones_like(xx)], dim=-1
    ).reshape(-1, 4).contiguous()
    pos.requires_grad_(True)
    col = torch.stack(
        [(xx + 1) * 0.5, (yy + 1) * 0.5, torch.full_like(xx, 0.5)], dim=-1
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
    tri = torch.tensor(tris, dtype=torch.int32, device=device)
    return pos, tri, col


# ---------------------------------------------------------------------------
# Die eigentlichen Testfaelle. Jede Funktion: nimmt keine Argumente, wirft
# bei Fehlschlag eine Exception (-> FAIL), gibt bei Erfolg formlos zurueck.
# Ein Absturz/Hang des Prozesses wird vom Runner ueber Timeout/Signal erkannt,
# nicht ueber eine Python-Exception.
# ---------------------------------------------------------------------------

def case_import():
    import torch
    import nvdiffrast.torch as dr  # noqa: F401
    assert torch.cuda.is_available(), "keine CUDA/HIP-GPU sichtbar"
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)


def case_rasterize_forward_min():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_single_triangle(device)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    _sync("rasterize forward")
    assert torch.isfinite(rast).all()


def case_interpolate_forward_min():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_single_triangle(device)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    _sync("interpolate forward")
    assert torch.isfinite(color).all()


def case_interpolate_backward_color():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_single_triangle(device)
    with torch.no_grad():
        rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    col = col.detach().clone().requires_grad_(True)
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    _sync("interpolate backward color forward-pass")
    loss = color.sum()
    loss.backward()
    _sync("interpolate backward color")
    assert col.grad is not None and torch.isfinite(col.grad).all()


def case_rasterize_interpolate_backward_pos():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_single_triangle(device)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    _sync("rast/interp backward pos forward-pass")
    loss = color.sum()
    loss.backward()
    _sync("rast/interp backward pos")
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def case_grid_forward():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_grid(device, cells=8)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[128, 128])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    _sync("grid forward")
    assert torch.isfinite(color).all()


def case_grid_backward_pos():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_grid(device, cells=8)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[128, 128])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    _sync("grid backward pos forward-pass")
    loss = color.sum()
    loss.backward()
    _sync("grid backward pos")
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def case_topology_hash():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    _, tri, _ = _make_grid(device, cells=16)
    th = dr.antialias_construct_topology_hash(tri)
    _sync("topology hash")
    assert th is not None


def case_antialias_forward_single_tri():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_single_triangle(device)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    aa = dr.antialias(color, rast, pos[None, ...], tri)
    _sync("antialias forward single-tri")
    assert torch.isfinite(aa).all()


def case_antialias_forward_grid():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_grid(device, cells=16)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[256, 256])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    aa = dr.antialias(color, rast, pos[None, ...], tri)
    _sync("antialias forward grid")
    assert torch.isfinite(aa).all()


def case_antialias_backward_color_direct():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, _ = _make_single_triangle(device)
    with torch.no_grad():
        rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    color = torch.full((1, 64, 64, 3), 0.5, device=device, requires_grad=True)
    aa = dr.antialias(color, rast, pos.detach()[None, ...], tri)
    _sync("antialias backward color-direct forward-pass")
    loss = aa.sum()
    loss.backward()
    _sync("antialias backward color-direct")
    assert color.grad is not None and torch.isfinite(color.grad).all()


def case_antialias_backward_pos_grid():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_grid(device, cells=16)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[256, 256])
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    aa = dr.antialias(color, rast, pos[None, ...], tri)
    _sync("antialias backward pos (grid) forward-pass")
    loss = aa.sum()
    loss.backward()
    _sync("antialias backward pos (grid)")
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def case_texture_forward():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, _ = _make_single_triangle(device)
    uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=torch.float32, device=device,
    )
    tex = torch.rand(1, 16, 16, 3, device=device, dtype=torch.float32)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    texc, _ = dr.interpolate(uv[None, ...], rast, tri)
    out = dr.texture(tex, texc, filter_mode="linear")
    _sync("texture forward")
    assert torch.isfinite(out).all()


def case_texture_backward():
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, _ = _make_single_triangle(device)
    uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=torch.float32, device=device,
    )
    tex = torch.rand(1, 16, 16, 3, device=device, dtype=torch.float32, requires_grad=True)
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=[64, 64])
    texc, _ = dr.interpolate(uv[None, ...], rast, tri)
    out = dr.texture(tex, texc, filter_mode="linear")
    _sync("texture backward forward-pass")
    loss = out.sum()
    loss.backward()
    _sync("texture backward")
    assert tex.grad is not None and torch.isfinite(tex.grad).all()


def case_antialias_grid_param(cells, resolution, stage):
    """Parametrisierte Variante von antialias_forward_grid/antialias_backward_pos_grid.
    cells und resolution getrennt steuerbar, um den Schwellenwert zu isolieren,
    an dem der Crash einsetzt. stage='forward' testet nur den Forward-Pfad,
    stage='backward' haengt zusaetzlich ein backward() an."""
    import torch
    import nvdiffrast.torch as dr
    device = "cuda"
    glctx = dr.RasterizeCudaContext(device=device)
    pos, tri, col = _make_grid(device, cells=cells)
    res = [resolution, resolution]
    rast, _ = dr.rasterize(glctx, pos[None, ...], tri, resolution=res)
    color, _ = dr.interpolate(col[None, ...], rast, tri)
    aa = dr.antialias(color, rast, pos[None, ...], tri)
    _sync(f"antialias forward cells={cells} res={resolution}")
    assert torch.isfinite(aa).all()

    if stage == "backward":
        loss = aa.sum()
        loss.backward()
        _sync(f"antialias backward cells={cells} res={resolution}")
        assert pos.grad is not None and torch.isfinite(pos.grad).all()


# ---------------------------------------------------------------------------
# Case-Registry: Name -> (Funktion, Tags, quick?)
# ---------------------------------------------------------------------------

CASES = {
    "import":                              (case_import,                          {"core"},              True),
    "rasterize_forward_min":               (case_rasterize_forward_min,           {"core", "rasterize"}, True),
    "interpolate_forward_min":             (case_interpolate_forward_min,         {"core", "interpolate"}, True),
    "interpolate_backward_color":          (case_interpolate_backward_color,      {"core", "interpolate", "backward"}, True),
    "rasterize_interpolate_backward_pos":  (case_rasterize_interpolate_backward_pos, {"core", "rasterize", "backward"}, True),
    "grid_forward":                        (case_grid_forward,                    {"grid", "core"},      True),
    "grid_backward_pos":                   (case_grid_backward_pos,               {"grid", "backward"},  True),
    "topology_hash":                       (case_topology_hash,                   {"topology", "antialias"}, True),
    "antialias_forward_single_tri":        (case_antialias_forward_single_tri,    {"antialias"},         True),
    "texture_forward":                     (case_texture_forward,                 {"texture"},           True),
    # Risikoreiche Pfade - nur im vollen Lauf (--quick laesst sie aus):
    "antialias_forward_grid":              (case_antialias_forward_grid,          {"antialias", "grid"}, False),
    "antialias_backward_color_direct":     (case_antialias_backward_color_direct, {"antialias", "backward"}, False),
    "antialias_backward_pos_grid":         (case_antialias_backward_pos_grid,     {"antialias", "grid", "backward"}, False),
    "texture_backward":                    (case_texture_backward,               {"texture", "backward"}, False),
}


def _match_tags(name, tags, spec):
    """spec: kommagetrennte Liste von Namen ODER Tags."""
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    if name in wanted:
        return True
    return bool(wanted & tags)


def select_cases(args):
    names = []
    for name, (_fn, tags, is_quick) in CASES.items():
        if args.quick and not is_quick:
            continue
        if args.only and not _match_tags(name, tags, args.only):
            continue
        if args.skip and _match_tags(name, tags, args.skip):
            continue
        names.append(name)
    return names


def _as_text(x):
    """stdout/stderr koennen je nach Python-Version/Pfad bytes oder str sein
    (besonders bei TimeoutExpired). Immer sicher in str umwandeln."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x


def run_case_in_subprocess(name, timeout, extra_args=None):
    cmd = [sys.executable, __file__, "--run-case", name]
    if extra_args:
        cmd += extra_args
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        tail = (_as_text(e.stdout) + _as_text(e.stderr))[-800:]
        return {
            "status": "TIMEOUT",
            "returncode": None,
            "elapsed": elapsed,
            "tail": tail,
        }

    elapsed = time.time() - start
    rc = result.returncode
    if rc == 0:
        status = "OK"
    elif rc is not None and rc < 0:
        status = f"CRASH(signal {-rc})"
    else:
        status = "FAIL"

    tail = (_as_text(result.stdout) + _as_text(result.stderr))[-800:]
    return {"status": status, "returncode": rc, "elapsed": elapsed, "tail": tail}


def run_single_case(name, args=None):
    if name == "antialias_grid_param":
        case_antialias_grid_param(args.cells, args.resolution, args.stage)
        print(f"PASS: {name} (cells={args.cells}, resolution={args.resolution}, stage={args.stage})", flush=True)
        return
    fn, _tags, _quick = CASES[name]
    print(f"=== running case: {name} ===", flush=True)
    fn()
    print(f"PASS: {name}", flush=True)


def run_bisection(args):
    """Variiert Dreieckszahl (cells) und Aufloesung GETRENNT, um den
    Schwellenwert zu finden, an dem antialias_forward_grid crasht.
    Jeder Punkt laeuft in einem eigenen Subprozess."""
    stage = args.stage

    # Achse 1: Dreieckszahl steigt, Aufloesung fest.
    cells_axis = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    fixed_res = args.resolution

    # Achse 2: Aufloesung steigt, Dreieckszahl fest.
    res_axis = [32, 64, 96, 128, 160, 192, 224, 256]
    fixed_cells = args.cells

    def run_point(cells, resolution):
        extra = ["--cells", str(cells), "--resolution", str(resolution), "--stage", stage]
        res = run_case_in_subprocess("antialias_grid_param", args.timeout, extra_args=extra)
        return res

    print(f"===== Achse 1: Dreieckszahl (cells), Aufloesung fest={fixed_res}, stage={stage} =====\n")
    axis1_results = []
    for cells in cells_axis:
        label = f"cells={cells:<3d} res={fixed_res}"
        res = run_point(cells, fixed_res)
        axis1_results.append((cells, res["status"]))
        print(f"{label:<24s} -> {res['status']:<16s} ({res['elapsed']:.1f}s)", flush=True)

    print(f"\n===== Achse 2: Aufloesung, Dreieckszahl fest=cells={fixed_cells}, stage={stage} =====\n")
    axis2_results = []
    for resolution in res_axis:
        label = f"cells={fixed_cells:<3d} res={resolution}"
        res = run_point(fixed_cells, resolution)
        axis2_results.append((resolution, res["status"]))
        print(f"{label:<24s} -> {res['status']:<16s} ({res['elapsed']:.1f}s)", flush=True)

    def first_failure(results):
        for key, status in results:
            if status != "OK":
                return key, status
        return None, None

    fail_cells, status1 = first_failure(axis1_results)
    fail_res, status2 = first_failure(axis2_results)

    print("\n===== BISEKTIONS-ZUSAMMENFASSUNG =====")
    if fail_cells is not None:
        print(f"Achse 1 (cells, res={fixed_res}): erster Fehlschlag bei cells={fail_cells} ({status1})")
    else:
        print(f"Achse 1 (cells, res={fixed_res}): kein Fehlschlag im getesteten Bereich {cells_axis}")

    if fail_res is not None:
        print(f"Achse 2 (res, cells={fixed_cells}): erster Fehlschlag bei res={fail_res} ({status2})")
    else:
        print(f"Achse 2 (res, cells={fixed_cells}): kein Fehlschlag im getesteten Bereich {res_axis}")

    print("\nHinweis: Wenn BEIDE Achsen erst bei hohen Werten failen, aber die")
    print("Kombination (hohe cells + hohe res) frueher failt als jede Achse")
    print("einzeln, deutet das auf ein Produkt-/Speicherlimit hin, nicht auf")
    print("einen reinen Dreieckszahl- oder Aufloesungs-Schwellenwert.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-case", type=str, default=None,
                         help="Intern: fuehrt genau einen Case direkt aus (kein Runner-Modus).")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--quick", action="store_true",
                         help="Nur die als sicher markierten Pfade testen.")
    parser.add_argument("--only", type=str, default=None,
                         help="Kommagetrennte Namen/Tags, nur diese testen (z.B. 'antialias').")
    parser.add_argument("--skip", type=str, default=None,
                         help="Kommagetrennte Namen/Tags, diese auslassen.")
    parser.add_argument("--bisect", action="store_true",
                         help="Sweep ueber cells/resolution fuer antialias_grid_param, "
                              "um den Crash-Schwellenwert getrennt pro Achse zu finden.")
    parser.add_argument("--cells", type=int, default=16,
                         help="Dreieckszahl-Parameter (Grid-Kanten) fuer antialias_grid_param/--bisect.")
    parser.add_argument("--resolution", type=int, default=256,
                         help="Aufloesungs-Parameter fuer antialias_grid_param/--bisect.")
    parser.add_argument("--stage", type=str, default="forward", choices=["forward", "backward"],
                         help="Nur Forward oder Forward+Backward testen (antialias_grid_param/--bisect).")
    args = parser.parse_args()

    if args.run_case is not None:
        # Subprozess-Modus: genau einen Test ausfuehren, harte Fehler
        # (HIP-Trap/Crash) duerfen den Prozess ruhig beenden - das ist
        # gewollt und wird vom Runner ueber Signal/Returncode erkannt.
        try:
            run_single_case(args.run_case, args)
        except Exception:
            print(f"\n❌ FAIL: {args.run_case}", flush=True)
            traceback.print_exc()
            sys.exit(2)
        sys.exit(0)

    if args.bisect:
        run_bisection(args)
        return

    # Runner-Modus
    names = select_cases(args)
    if not names:
        print("Keine passenden Testfaelle gefunden (Filter zu eng?).")
        return

    print(f"Teste {len(names)} Pfad(e), Timeout={args.timeout}s pro Pfad:\n")
    results = {}
    for name in names:
        print(f"--- {name} ---", flush=True)
        res = run_case_in_subprocess(name, args.timeout)
        results[name] = res
        print(f"  -> {res['status']} (rc={res['returncode']}, {res['elapsed']:.1f}s)", flush=True)
        if res["status"] != "OK" and res["tail"]:
            print("  --- letzte Ausgabe ---")
            print("  " + res["tail"].replace("\n", "\n  ")[-1200:])
        print(flush=True)

    print("\n===== SUMMARY =====")
    passed = 0
    failed = 0
    for name in names:
        status = results[name]["status"]
        marker = "OK" if status == "OK" else status
        print(f"{name:<40s} {marker}")
        if status == "OK":
            passed += 1
        else:
            failed += 1
    print(f"\npassed={passed} failed_or_timeout={failed} total={len(names)}")


if __name__ == "__main__":
    main()
