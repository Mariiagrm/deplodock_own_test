#!/bin/bash
# Compila un caso con varios juegos de knobs fijados (EMMY_KNOBS) y vuelca el CUDA de cada uno,
# para comparar p. ej. la elección de emmy sin autotuning frente a la elección tras --tune.
#   bash tfm_pruebas/porque/generar_knobs.sh <familia/caso> <etiqueta>=<KNOBS> [<etiqueta>=<KNOBS> ...]
set -uo pipefail
ID=$1; shift
C=${ID/\//__}
RUN=tfm_pruebas/runs/20261111_102641/casos/$C
OUT=tfm_pruebas/porque/$C
mkdir -p "$OUT"
for kv in "$@"; do
  TAG=${kv%%=*}; KNOBS=${kv#*=}
  rm -rf "$OUT/dump_$TAG"
  EMMY_NVCC_FLAGS="" EMMY_KNOBS="$KNOBS" ./venv/bin/python -m emmy.emmy run --ir "$RUN/dump/00_input.json" --bench \
    --bench-backends emmy --no-record-evidence --dump-dir "$OUT/dump_$TAG" --json "$OUT/bench_$TAG.json" \
    > "$OUT/stdout_$TAG.txt" 2>&1 || echo "rc=$? ($TAG)"
  echo "== $C $TAG ($KNOBS)"; grep -E '^k_' "$OUT/stdout_$TAG.txt" | head -2 | cut -c1-170
done
