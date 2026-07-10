#!/usr/bin/env bash
# patch_v47_native_workbuffer_zero.sh
#
# ECHTER FIX (kein Diagnose-Skip): ersetzt
#
#   NVDR_CHECK_CUDA_ERROR(hipMemsetAsync(p.workBuffer, 0, sizeof(int4), stream));
#
# durch eine PyTorch-native Zero-Init auf dem bereits vorhandenen
# work_buffer-Tensor:
#
#   work_buffer.narrow(0, 0, 4).zero_();
#
# Das nullt exakt dieselben 16 Bytes (4 floats = 1 int4), aber ueber
# PyTorchs eigene ATen-Dispatch-/Stream-Buchhaltung statt eines rohen
# hipMemsetAsync-Aufrufs, der am Caching-Allokator vorbei direkt auf dem
# rohen Pointer operiert. Behebt vermutlich einen Stream-Ordering-Hazard
# bei frischen (nicht aus dem Pool bedienten), groesseren Allokationen.
#
# WICHTIG: Dieses Skript ersetzt den Aufruf FUNKTIONAL (nicht nur zu
# Diagnosezwecken skippend wie v46e) - work_buffer[0..3] wird weiterhin
# korrekt auf 0 gesetzt.
#
# Voraussetzung: v46d/v46e sollten VORHER zurueckgesetzt sein (auf den
# Stand VOR diesen Diagnose-Patches), damit hier der echte Fix auf einer
# sauberen Basis mit echten Kernel-Launches getestet wird:
#
#   cp csrc/torch/torch_antialias.cpp.before_v46d_skip_disc_and_analysis csrc/torch/torch_antialias.cpp
#   cp csrc/torch/torch_antialias_hip.cpp.before_v46d_skip_disc_and_analysis csrc/torch/torch_antialias_hip.cpp
#
# Nutzung:
#   REPO=/home/oem/therock_test/nvdiffrast ./patch_v47_native_workbuffer_zero.sh

set -euo pipefail

REPO="${REPO:-$HOME/therock_test/nvdiffrast}"
cd "$REPO"

SRC_FILES=(csrc/torch/torch_antialias.cpp csrc/torch/torch_antialias_hip.cpp)
TS=$(date +%Y%m%d_%H%M%S)

for f in "${SRC_FILES[@]}"; do
  if [ -f "$f" ]; then
    target="${f}.before_v47_native_workbuffer_zero"
    [ -f "$target" ] && target="${f}.before_v47_native_workbuffer_zero_${TS}"
    cp "$f" "$target"
    echo "[backup] $f -> $target"
  else
    echo "[WARN] Datei nicht gefunden: $f"
  fi
done

python3 - "${SRC_FILES[@]}" <<'PYEOF'
import re, sys

MARKER = "v47 FIX"

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()

    if MARKER in t:
        print(f"[skip] {path}: bereits gepatcht")
        continue

    pattern = re.compile(
        r"([ \t]*NVDR_CHECK_CUDA_ERROR\(\w*MemsetAsync\(p\.workBuffer,"
        r"[^\n]*\)\);\n)"
    )

    def repl(m):
        original = m.group(1)
        indent = re.match(r"[ \t]*", original).group(0)
        return (
            f"{indent}#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)\n"
            f"{indent}// {MARKER}: replace raw hipMemsetAsync (bypasses PyTorch's\n"
            f"{indent}// caching-allocator stream tracking) with a torch-native\n"
            f"{indent}// zero_() on the first int4 (4 floats) of work_buffer. Fixes\n"
            f"{indent}// a suspected stream-ordering hazard on fresh, large (non-\n"
            f"{indent}// pooled) allocations at width*height>=~32768.\n"
            f"{indent}work_buffer.narrow(0, 0, 4).zero_();\n"
            f"{indent}#else\n"
            f"{original}"
            f"{indent}#endif\n"
        )

    new_t, n = pattern.subn(repl, t)
    if n == 0:
        print(f"[WARN] {path}: hipMemsetAsync(p.workBuffer,...)-Anker nicht gefunden - manuell pruefen!")
        continue

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_t)
    print(f"[ok] {path}: {n} Aufruf(e) durch native zero_() ersetzt (v47)")
PYEOF

echo
echo "Fertig. Rebuild, dann testen:"
echo "  1) Stabilitaet: 182x182, 256x128, 181x181 (sollten jetzt ALLE OK sein,"
echo "     inklusive ECHTER Kernel-Launches - kein Diagnose-Skip mehr aktiv)"
echo "  2) Korrektheit: nvdiffrast_aa_matrix_probe_v4.py / den Silhouette-FD-"
echo "     Sweep erneut laufen lassen, um zu bestaetigen dass echte Workitems"
echo "     wieder korrekt erzeugt/verarbeitet werden (workCount > 0, nicht 0"
echo "     wie in den v46-Diagnoseversionen mit uebersprungenen Kerneln)."
