#!/usr/bin/env python3
"""
aa_matrix_stat_probe.py

Fuehrt nvdiffrast_aa_matrix_probe_v4.py N-mal hintereinander aus und
aggregiert die Ergebnisse PRO EINZELFALL (shape/res-Kombination), statt
nur Gesamtsummen zu vergleichen. Zweck: belastbare Fehlerquote pro Fall
statt manuellem Vergleich einzelner 24er-Zusammenfassungen.

WICHTIG: Das ist ein Diagnose-Werkzeug, kein Weg, fehlerhafte Laeufe zu
verstecken. Ziel ist eine PRAEZISE Fehlerquote pro Fall, um Patches
objektiv zu vergleichen (vorher/nachher) - nicht, einen "guten" Lauf
auszuwaehlen.

Nutzung:
  python aa_matrix_stat_probe.py --runs 20 \\
    --shapes single,grid1,grid4,grid16 \\
    --res-list 160,180,182,192,224,256 \\
    --colors interp --hashes explicit \\
    --probe-script ./nvdiffrast_aa_matrix_probe_v4.py
"""

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict

CASE_LINE_RE = re.compile(
    r"^\s*(\S+)\s+res=(\d+)\s+color=(\S+)\s+hash=(\S+)\s+(OK|CRASH.*|TIMEOUT.*)\s*$"
)


def run_once(probe_script, shapes, res_list, colors, hashes, timeout):
    cmd = [
        sys.executable, probe_script,
        "--timeout", str(timeout),
        "--shapes", shapes,
        "--res-list", res_list,
        "--colors", colors,
        "--hashes", hashes,
    ]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=timeout * 60)
        return cp.stdout
    except subprocess.TimeoutExpired as e:
        return e.stdout or ""


def parse_summary(stdout_text):
    """Extrahiert die 'SUMMARY BY CASE'-Zeilen (shape res=.. color=.. hash=.. STATUS)."""
    results = {}
    in_summary = False
    for line in stdout_text.splitlines():
        if "SUMMARY BY CASE" in line:
            in_summary = True
            continue
        if in_summary and "COMPACT MATRIX" in line:
            break
        if not in_summary:
            continue
        m = CASE_LINE_RE.match(line)
        if m:
            shape, res, color, hashmode, status = m.groups()
            key = (shape, int(res), color, hashmode)
            ok = status.strip() == "OK"
            results[key] = ok
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=30, help="Timeout pro Einzelfall-Prozess im Probe-Skript (Sekunden)")
    ap.add_argument("--shapes", default="single,grid1,grid4,grid16")
    ap.add_argument("--res-list", default="160,180,182,192,224,256")
    ap.add_argument("--colors", default="interp")
    ap.add_argument("--hashes", default="explicit")
    ap.add_argument("--probe-script", default="./nvdiffrast_aa_matrix_probe_v4.py")
    ap.add_argument("--label", default="", help="Optionale Bezeichnung fuer diesen Lauf (z.B. Patch-Version) in der Ausgabe")
    args = ap.parse_args()

    print(f"Aggregiere {args.runs} Wiederholungen"
          + (f" [{args.label}]" if args.label else "") + " ...\n")

    counts = defaultdict(lambda: [0, 0])  # key -> [ok_count, total_count]
    t0 = time.time()

    for run_idx in range(1, args.runs + 1):
        print(f"[{run_idx:03d}/{args.runs:03d}] laeuft...", flush=True)
        stdout_text = run_once(
            args.probe_script, args.shapes, args.res_list,
            args.colors, args.hashes, args.timeout,
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
        shape, res, color, hashmode = key
        rate = ok_count / total if total else float("nan")
        rows.append((rate, shape, res, color, hashmode, ok_count, total))
    rows.sort(key=lambda r: (r[0], r[2]))  # schlechteste Quote zuerst, dann nach Aufloesung

    for rate, shape, res, color, hashmode, ok_count, total in rows:
        marker = "  <-- AUFFAELLIG" if rate < 1.0 else ""
        print(f"{shape:<10s} res={res:<5d} color={color:<8s} hash={hashmode:<10s} "
              f"{ok_count:>3d}/{total:<3d} ({rate*100:5.1f}%){marker}")

    total_ok = sum(r[5] for r in rows)
    total_all = sum(r[6] for r in rows)
    print(f"\nGesamt: {total_ok}/{total_all} "
          f"({(total_ok/total_all*100 if total_all else float('nan')):.1f}%)")

    n_perfect = sum(1 for r in rows if r[0] == 1.0)
    n_total_cases = len(rows)
    print(f"Faelle mit 100%-Erfolgsquote: {n_perfect}/{n_total_cases}")

    if args.label:
        print(f"\n(Label: {args.label})")


if __name__ == "__main__":
    main()
