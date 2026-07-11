#!/usr/bin/env python3
"""
aa_backward_pos_stat_probe_v52.py

Statistical validation probe for the final v52 antialias backward/grad
workBuffer y-counter fix (v51 in this repo's patch numbering).

Wraps test_antialias_backward_matrix_v52.py (--stages backward_pos), which
is the only stage that actually calls loss.backward() and reads back
pos.grad - unlike --stages call,sync,finite,diff, which exercises the
antialias_fwd/antialias_grad setup and data paths without ever triggering a
real backward pass.

Runs the bisect probe N times and aggregates the success rate PER CASE
(cells/res/topo/color/pg/stage), instead of relying on a single run, which
was shown to be insufficient given the run-to-run non-determinism observed
elsewhere in this validation effort.

Dependency: test_antialias_backward_matrix_v52.py must be present in the
same directory (or pointed to via --probe-script). This file was formerly
named nvdiffrast_antialias_grid_bisect_v44b.py before the final v52 rename.

Usage:
  python aa_backward_pos_stat_probe_v52.py --runs 20 \\
    --cells-list 1,4,16 --res-list 160,180,182,192,224,256 \\
    --topos explicit --colors interp --pos-grads 1 --stages backward_pos \\
    --probe-script ./test_antialias_backward_matrix_v52.py \\
    --label "final v52 AA backward_pos statistical probe"
"""

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict

# Format: "cells=1 res=160 topo=explicit color=interp pg=1 stage=backward_pos    OK rc=0 time=1.72s"
CASE_LINE_RE = re.compile(
    r"^\s*cells=(\d+)\s+res=(\d+)\s+topo=(\S+)\s+color=(\S+)\s+pg=(\S+)\s+stage=(\S+)\s+(OK|XX|TIMEOUT)\b"
)


def run_once(probe_script, cells_list, res_list, topos, colors, pos_grads, stages, timeout):
    cmd = [
        sys.executable, probe_script,
        "--timeout", str(timeout),
        "--cells-list", cells_list,
        "--res-list", res_list,
        "--topos", topos,
        "--colors", colors,
        "--pos-grads", pos_grads,
        "--stages", stages,
    ]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=timeout * 60)
        return cp.stdout
    except subprocess.TimeoutExpired as e:
        return e.stdout or ""


def parse_summary(stdout_text):
    """Extrahiert die Zeilen aus dem '===== SUMMARY ====='-Block."""
    results = {}
    in_summary = False
    for line in stdout_text.splitlines():
        if "===== SUMMARY" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = CASE_LINE_RE.match(line)
        if m:
            cells, res, topo, color, pg, stage, status = m.groups()
            key = (int(cells), int(res), topo, color, pg, stage)
            ok = status.strip() == "OK"
            results[key] = ok
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--cells-list", default="1,4,16")
    ap.add_argument("--res-list", default="160,180,182,192,224,256")
    ap.add_argument("--topos", default="explicit")
    ap.add_argument("--colors", default="interp")
    ap.add_argument("--pos-grads", default="1")
    ap.add_argument("--stages", default="backward_pos")
    ap.add_argument("--probe-script", default="./test_antialias_backward_matrix_v52.py")
    ap.add_argument("--label", default="final v52 AA backward_pos statistical probe")
    args = ap.parse_args()

    print(f"Aggregiere {args.runs} Wiederholungen"
          + (f" [{args.label}]" if args.label else "") + " ...\n")

    counts = defaultdict(lambda: [0, 0])
    t0 = time.time()

    for run_idx in range(1, args.runs + 1):
        print(f"[{run_idx:03d}/{args.runs:03d}] laeuft...", flush=True)
        stdout_text = run_once(
            args.probe_script, args.cells_list, args.res_list,
            args.topos, args.colors, args.pos_grads, args.stages, args.timeout,
        )
        parsed = parse_summary(stdout_text)
        if not parsed:
            print(f"  [WARN] Lauf {run_idx}: keine auswertbare SUMMARY gefunden "
                  f"(Timeout/Crash des gesamten Probe-Prozesses?)")
            continue
        for key, ok in parsed.items():
            counts[key][1] += 1
            if ok:
                counts[key][0] += 1

    dt = time.time() - t0
    print(f"\nFertig nach {dt:.1f}s.\n")

    print("===== FEHLERQUOTE PRO FALL (sortiert nach schlechtester Quote) =====")
    rows = []
    for key, (ok_count, total) in counts.items():
        cells, res, topo, color, pg, stage = key
        rate = ok_count / total if total else float("nan")
        rows.append((rate, cells, res, topo, color, pg, stage, ok_count, total))
    rows.sort(key=lambda r: (r[0], r[2]))

    for rate, cells, res, topo, color, pg, stage, ok_count, total in rows:
        marker = "  <-- AUFFAELLIG" if rate < 1.0 else ""
        print(f"cells={cells:<3d} res={res:<5d} topo={topo:<9s} color={color:<8s} "
              f"pg={pg} stage={stage:<13s} {ok_count:>3d}/{total:<3d} ({rate*100:5.1f}%){marker}")

    total_ok = sum(r[7] for r in rows)
    total_all = sum(r[8] for r in rows)
    print(f"\nGesamt: {total_ok}/{total_all} "
          f"({(total_ok/total_all*100 if total_all else float('nan')):.1f}%)")

    n_perfect = sum(1 for r in rows if r[0] == 1.0)
    print(f"Faelle mit 100%-Erfolgsquote: {n_perfect}/{len(rows)}")

    if args.label:
        print(f"\n(Label: {args.label})")


if __name__ == "__main__":
    main()
