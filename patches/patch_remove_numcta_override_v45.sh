#!/usr/bin/env bash
# patch_remove_numcta_override_v45.sh
#
# Entfernt eine liegen gebliebene Debug-Zeile in torch_antialias(_hip).cpp:
#
#   NVDR_CHECK_CUDA_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, ...));
#   numCTA = 1;                              <-- diese Zeile wird entfernt
#   NVDR_CHECK_CUDA_ERROR(hipDeviceGetAttribute(&numSM, ...));
#
# Diese Zeile hat den echten, per Occupancy-API ermittelten Wert sofort
# wieder verworfen und den persistenten Thread-Pool von
# AntialiasFwdAnalysisKernel kuenstlich auf numSM*256 Threads gedrosselt,
# statt des eigentlich verfuegbaren numCTA*numSM*256. Das erklaert den
# reinen Gesamtpixelzahl-Kipppunkt: sobald workCount den kuenstlich
# verkleinerten Pool uebersteigt, braucht die Fetch-Schleife mehrere
# Runden statt einer einzigen.
#
# Nutzung:
#   REPO=/home/oem/therock_test/nvdiffrast ./patch_remove_numcta_override_v45.sh

set -euo pipefail

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
cd "$REPO"

SRC_FILES=(csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp)
TS=$(date +%Y%m%d_%H%M%S)

for f in "${SRC_FILES[@]}"; do
  if [ -f "$f" ]; then
    target="${f}.before_v45_remove_numcta_override"
    if [ -f "$target" ]; then
      target="${f}.before_v45_remove_numcta_override_${TS}"
    fi
    cp "$f" "$target"
    echo "[backup] $f -> $target"
  else
    echo "[WARN] Datei nicht gefunden: $f"
  fi
done

python3 - "${SRC_FILES[@]}" <<'PYEOF'
import re, sys

# Drei Zustaende sauber unterscheiden:
#   1) Override vorhanden      -> entfernen                      -> [ok]
#   2) Override nicht vorhanden, Occupancy-Block aber intakt
#      und numCTA wird verwendet -> already clean                -> [ok]
#      (Das ist der Normalfall bei einem frischen Upstream-Klon: nvdiffrast
#       enthaelt die Debug-Zeile gar nicht. Sie stammte aus einer aelteren,
#       manuell veraenderten Entwicklungskopie.)
#   3) Occupancy-Block nicht auffindbar -> echte Strukturabweichung -> FEHLER
occ_re = re.compile(
    r"NVDR_CHECK_CUDA_ERROR\(\w*OccupancyMaxActiveBlocksPerMultiprocessor\("
    r"&numCTA,\s*\(void\*\)AntialiasFwdAnalysisKernel,[^\n]*\)\);\s*\n"
)
override_re = re.compile(
    r"(NVDR_CHECK_CUDA_ERROR\(\w*OccupancyMaxActiveBlocksPerMultiprocessor\("
    r"&numCTA,\s*\(void\*\)AntialiasFwdAnalysisKernel,[^\n]*\)\);\s*\n)"
    r"\s*numCTA\s*=\s*1;\s*\n"
)
# Zielzustand: der Analysis-Kernel wird mit dem vollen Grid gestartet.
launch_re = re.compile(
    r"\w*LaunchKernel\(\(void\*\)AntialiasFwdAnalysisKernel,\s*numCTA\s*\*\s*numSM"
)

failed = False
for path in sys.argv[1:]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            t = f.read()
    except FileNotFoundError:
        # Wird oben bereits gemeldet; torch_antialias_hip.cpp existiert erst
        # nach dem ersten Hipify-Lauf.
        continue

    if not occ_re.search(t):
        print(f"[FEHLER] {path}: OccupancyMaxActiveBlocksPerMultiprocessor-Block fuer "
              f"AntialiasFwdAnalysisKernel nicht gefunden. Quellstruktur unerwartet - "
              f"v45 kann nicht verifizieren, ob der Debug-Override fehlt.")
        failed = True
        continue

    new_t, n = override_re.subn(r"\1", t, count=1)
    if n:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_t)
        print(f"[ok] {path}: 'numCTA = 1;'-Debug-Override entfernt")
        t = new_t
    else:
        print(f"[ok] {path}: kein 'numCTA = 1;'-Override vorhanden - already clean "
              f"(Upstream enthaelt die Debug-Zeile nicht)")

    # Zielzustand in beiden Faellen verifizieren.
    if re.search(r"^\s*numCTA\s*=\s*1;\s*$", t, re.M):
        print(f"[FEHLER] {path}: es existiert weiterhin eine 'numCTA = 1;'-Zeile - bitte pruefen!")
        failed = True
    elif not launch_re.search(t):
        print(f"[FEHLER] {path}: AntialiasFwdAnalysisKernel-Launch nutzt nicht das erwartete "
              f"'numCTA * numSM'-Grid - Zielzustand NICHT bestaetigt.")
        failed = True
    else:
        print(f"[ok] {path}: AntialiasFwdAnalysisKernel laeuft mit vollem Grid (numCTA * numSM)")

sys.exit(1 if failed else 0)
PYEOF

echo
echo "v45 abgeschlossen: AntialiasFwdAnalysisKernel nutzt das volle numCTA*numSM-Grid."
