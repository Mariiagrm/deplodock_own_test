#!/bin/bash
# Reconstruye, para un caso de la ejecución --tune, las dos versiones que se comparan:
#  1) torch.compile (Inductor, max-autotune-no-cudagraphs, como en la suite): código generado en inductor/
#     (TORCH_COMPILE_DEBUG + caché propia) y tiempos en bench_golden.json
#  2) el kernel que emmy eligió tras --tune, fijado con EMMY_KNOBS -> dump_emmy/05_lowering_cuda.kernels/
#   bash tfm_pruebas/porque/generar.sh <familia/caso> "<KNOBS del greedy autotuneado>"
set -euo pipefail
ID=$1; KNOBS=$2
C=${ID/\//__}
RUN=tfm_pruebas/runs/20261109_005347/casos/$C
CASE=tests/compiler/realization/cases/$ID.json
OUT=$PWD/tfm_pruebas/porque/$C
rm -rf "$OUT"; mkdir -p "$OUT/inductor"
export EMMY_TUNE_DB=$PWD/$RUN/autotune.db EMMY_NVCC_FLAGS=""
ROW=$(./venv/bin/python -c "import json;print(json.load(open('$CASE'))['rows'][0]['name'])")
( cd "$OUT/inductor" && TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR="$OUT/inductor/cache" \
  /workspace/deplodock/venv/bin/python -m emmy.emmy run --golden "/workspace/deplodock/$CASE" --realization "$ROW" --bench \
  --bench-backends eager,tcompile,emmy --no-record-evidence --json "$OUT/bench_golden.json" \
  > "$OUT/golden_stdout.txt" 2> "$OUT/golden_stderr.txt" ) || echo "golden rc=$?"
EMMY_KNOBS="$KNOBS" ./venv/bin/python -m emmy.emmy run --ir "$RUN/dump/00_input.json" --bench --bench-backends emmy \
  --no-record-evidence --dump-dir "$OUT/dump_emmy" --json "$OUT/bench_emmy.json" > "$OUT/emmy_stdout.txt" 2>&1 || echo "emmy rc=$?"
grep -E 'Eager|torch.compile|Emmy|^k_' "$OUT/golden_stdout.txt" "$OUT/emmy_stdout.txt" | cut -c1-200
find "$OUT/inductor" -name 'output_code.py' | head
