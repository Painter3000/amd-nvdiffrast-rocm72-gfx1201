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

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()

    # Nur die Ueberschreibung fuer AntialiasFwdAnalysisKernel betreffen -
    # NICHT den AntialiasGradKernel-Block (der hat laut Grep keine
    # "numCTA = 1;"-Zeile und soll unangetastet bleiben).
    pattern = re.compile(
        r"(NVDR_CHECK_CUDA_ERROR\(\w*OccupancyMaxActiveBlocksPerMultiprocessor\("
        r"&numCTA,\s*\(void\*\)AntialiasFwdAnalysisKernel,[^\n]*\)\);\s*\n)"
        r"\s*numCTA\s*=\s*1;\s*\n"
    )

    new_t, n = pattern.subn(r"\1", t, count=1)
    if n == 0:
        print(f"[WARN] {path}: 'numCTA = 1;'-Anker nicht exakt gematcht - manuell pruefen!")
        continue

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_t)
    print(f"[ok] {path}: 'numCTA = 1;'-Debug-Override entfernt")
PYEOF

echo
echo "Fertig. Falls [WARN] erschien: bitte Zeile manuell direkt nach der"
echo "hipOccupancyMaxActiveBlocksPerMultiprocessor(...)-Abfrage fuer"
echo "AntialiasFwdAnalysisKernel loeschen (Grep-Zeile ~160 in torch_antialias_hip.cpp)."
echo
echo "Naechster Schritt: rebuild, dann exakt dieselben Faelle erneut testen:"
echo "  cells=4 wh-list 181,182,256x128,256x140,182x64,64x182"
