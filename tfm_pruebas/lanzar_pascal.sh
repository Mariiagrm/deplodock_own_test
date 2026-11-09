#!/bin/bash
# Lanza el subconjunto TFM de la suite de emmy dentro del contenedor cutlass-gemm de pascal.
#   sbatch -p pascal-benchmark --exclusive -J emmy-tfm -o ~/deplodock/tfm_pruebas/slurm_%j.log \
#          --wrap 'bash ~/deplodock/tfm_pruebas/lanzar_pascal.sh [opciones de correr_suite_tfm.py]'
# Ejemplos de opciones: --cat matmul   --solo qwen3emb/rmsnorm   --max 5   --ncu   --tune 24   --workers 4
set -euo pipefail
cd "$HOME"
docker compose up -d >/dev/null 2>&1 || docker start cutlass-gemm >/dev/null
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="--out tfm_pruebas/runs/$STAMP"
RUN_AS="$(id -u):$(id -g)"
for a in "$@"; do
  [ "$a" = "--reanalizar" ] && OUT=""
  # ncu necesita root dentro del contenedor (contadores de GPU en modo solo-admin + cap SYS_ADMIN del compose)
  [ "$a" = "--ncu" ] && RUN_AS="0:0"
done
# HOME nuevo por ejecución: base de datos de autotune vacía, como el "stock lane" de make bench-kernels.
set +e
docker exec -u "$RUN_AS" -e HOME=/tmp/emmy_tfm_$STAMP -e PYTHONPATH=/workspace/deplodock \
  -w /workspace/deplodock cutlass-gemm \
  bash -c 'mkdir -p "$HOME" && ./venv/bin/python -u tfm_pruebas/correr_suite_tfm.py '"$OUT"' "$@"' _ "$@"
rc=$?
if [ "$RUN_AS" = "0:0" ]; then  # devolver a tu usuario lo que haya escrito root
  docker exec -u 0:0 cutlass-gemm chown -R "$(id -u):$(id -g)" /workspace/deplodock/tfm_pruebas /workspace/deplodock/venv
fi
exit $rc
