#!/bin/bash
# Verificaciones P1–P5 en pascal (RTX 5090). Lanzar en exclusiva:
#   sbatch -p pascal-benchmark --exclusive -o ~/deplodock/tfm_pruebas/verif/slurm_%j.log \
#          --wrap 'bash ~/deplodock/tfm_pruebas/verif/verificar_pascal.sh'
# Cada paso escribe en tfm_pruebas/verif/<paso>/ (dentro del contenedor: /workspace/deplodock/...).
set -u
cd "$HOME"
docker compose up -d >/dev/null 2>&1 || docker start cutlass-gemm >/dev/null
U="$(id -u):$(id -g)"
V=tfm_pruebas/verif
CASOS=tfm_pruebas/runs/20261111_102641/casos
ex() {  # ex <usuario> <HOME> <comando bash>   (dentro del contenedor, en /workspace/deplodock)
  docker exec -u "$1" -e HOME="$2" -e PYTHONPATH=/workspace/deplodock -e EMMY_NVCC_FLAGS= \
    -e TORCHINDUCTOR_CACHE_DIR="$2/inductor" -w /workspace/deplodock cutlass-gemm bash -c "mkdir -p \$HOME; $3"
}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 0. Entorno: TF32
log "== 0. entorno TF32"
mkdir -p "$HOME/deplodock/$V/0_entorno"
ex "$U" /tmp/v0 "env | grep -i -E 'tf32|cublas|nvidia_tf32' ; ./venv/bin/python -c \"import torch,os;print('torch',torch.__version__,'allow_tf32',torch.backends.cuda.matmul.allow_tf32,'prec',torch.get_float32_matmul_precision(),'override',os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))\"" \
  > "$HOME/deplodock/$V/0_entorno/salida.txt" 2>&1
docker inspect cutlass-gemm --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i -E 'tf32|cublas' >> "$HOME/deplodock/$V/0_entorno/salida.txt"
cat "$HOME/deplodock/$V/0_entorno/salida.txt"

# 1. P5 en la RTX 5090 (con y sin la variable de NGC)
log "== 1. P5 TF32 en RTX 5090"
mkdir -p "$HOME/deplodock/$V/1_tf32"
ex "$U" /tmp/v1 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 ./venv/bin/python tfm_pruebas/tf32/tf32_bench.py 1024 2048 4096 8192" > "$HOME/deplodock/$V/1_tf32/rtx5090.txt" 2>&1
cat "$HOME/deplodock/$V/1_tf32/rtx5090.txt"

# 2. P1: Tensor Core frente a paralelismo (f16 -> f32, 128x128x128)
log "== 2. P1 factorial TC x bloques"
D=$V/2_factorial; mkdir -p "$HOME/deplodock/$D"
for kv in "tc_64bloques=WORK=w1x2,TILE=mma_m16n8k16_f16_f32/f1x1/k8,STAGE=d2/smem-async" \
          "tc_8bloques=WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f2x2/k8,STAGE=d2/smem-async" \
          "escalar_64bloques=WORK=t8x8,TILE=f2x2,STAGE=d2/smem-async" \
          "escalar_8bloques=WORK=t16x16,TILE=f2x4,STAGE=d2/smem-async" \
          "escalar_64bloques_sinsmem=WORK=t8x8,TILE=f2x2" \
          "escalar_8bloques_sinsmem=WORK=t16x16,TILE=f2x4"; do
  T=${kv%%=*}; K=${kv#*=}
  ex "$U" /tmp/v2 "EMMY_KNOBS='$K' ./venv/bin/python -m emmy.emmy run --ir $CASOS/matmul__f16-mma-m128n128k128-f32/dump/00_input.json --bench --bench-backends emmy --no-record-evidence --dump-dir $D/dump_$T --json $D/$T.json" \
    > "$HOME/deplodock/$D/$T.txt" 2>&1
  echo "$T ($K): $(grep -E '^k_' "$HOME/deplodock/$D/$T.txt" | head -1 | cut -c1-150)"
done

# 3. P3: eager con TF32 apagado, y ncu emmy (autotuneado) frente a cuBLAS
log "== 3. P3 eager fp32 real y ncu"
D=$V/3_fp32; mkdir -p "$HOME/deplodock/$D"
for c in qwen3emb/matmul-qproj-s512 qwen3emb/matmul-kvproj-s512 qwen3emb/gated-mlp-s32 qwen3emb/gated-mlp-s512; do
  C=${c/\//__}; ROW=$(python3 -c "import json;print(json.load(open('$HOME/deplodock/tests/compiler/realization/cases/$c.json'))['rows'][0]['name'])")
  for ov in 1 0; do
    ex "$U" /tmp/v3 "EMMY_TUNE_DB=\$PWD/$CASOS/$C/autotune.db TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=$ov ./venv/bin/python -m emmy.emmy run --golden tests/compiler/realization/cases/$c.json --realization $ROW --bench --bench-backends eager,emmy --no-record-evidence --json $D/${C}_tf32override$ov.json" \
      > "$HOME/deplodock/$D/${C}_tf32override$ov.txt" 2>&1
    echo "$c override=$ov: $(grep -E '^Eager|^Emmy' "$HOME/deplodock/$D/${C}_tf32override$ov.txt" | tr -s ' ' | tr '\n' ' ')"
  done
  ex "0:0" /tmp/v3r "EMMY_TUNE_DB=\$PWD/$CASOS/$C/autotune.db TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 ./venv/bin/python -m emmy.emmy run --golden tests/compiler/realization/cases/$c.json --realization $ROW --bench --bench-backends eager,emmy --no-record-evidence --profile --dump-dir $D/ncu_$C --json $D/${C}_ncu.json" \
    > "$HOME/deplodock/$D/${C}_ncu.txt" 2>&1
done
docker exec -u 0:0 cutlass-gemm chown -R "$U" /workspace/deplodock/$V

# 4. P4: elección sin mediciones (prior) según disposición y M; y emmy eval prior
log "== 4. P4 prior"
D=$V/4_prior; mkdir -p "$HOME/deplodock/$D"
i=0
for code in "torch.matmul(torch.randn(32,1024),torch.randn(1024,1024))" \
            "torch.nn.functional.linear(torch.randn(32,1024),torch.randn(1024,1024))" \
            "torch.matmul(torch.randn(128,1024),torch.randn(1024,1024))" \
            "torch.matmul(torch.randn(512,1024),torch.randn(1024,1024))" \
            "torch.nn.functional.linear(torch.randn(512,1024),torch.randn(1024,1024))" \
            "torch.matmul(torch.randn(32,1024).half(),torch.randn(1024,1024).half())"; do
  i=$((i+1))
  ex "$U" /tmp/v4_$i "./venv/bin/python -m emmy.emmy run --code '$code' --bench --bench-backends eager,emmy --no-record-evidence --json $D/code_$i.json" > "$HOME/deplodock/$D/code_$i.txt" 2>&1
  echo "[$i] $code -> $(grep -E '^k_' "$HOME/deplodock/$D/code_$i.txt" | head -1 | tr -s ' ' | cut -c1-160) | $(grep -E '^Eager|^Emmy' "$HOME/deplodock/$D/code_$i.txt" | tr -s ' ' | tr '\n' ' ')"
done
ex "$U" /tmp/v4e "./venv/bin/python -m emmy.emmy db import --db $D/dataset.db --fresh emmy/compiler/pipeline/search/golden/records/*.json && ./venv/bin/python -m emmy.emmy db export --db $D/dataset.db $D/dataset && ./venv/bin/python -m emmy.emmy eval prior $D/dataset" > "$HOME/deplodock/$D/eval_prior.txt" 2>&1
tail -40 "$HOME/deplodock/$D/eval_prior.txt"

# 5. P1–P2: concurrencia y autotuning de torch.compile (caché de Inductor vacía en cada ejecución)
log "== 5. P1-P2 concurrencia"
SOLO="qwen3emb/rmsnorm,fused/fp4-tma-w4a4-two-channel-p2,matmul/f16-mma-m128n128k128-f32,matmul/f16-mma-epilogue-residual,matmul/imap-reshape-a-gmem,matmul/bf16-mma-trans-b-masked-n,matmul/nvfp4-w4a4-packed-slab-loopify"
for rep in 1 2 3; do
  for w in 4 1; do
    log "   rep $rep workers $w"
    bash "$HOME/deplodock/tfm_pruebas/lanzar_pascal.sh" --solo "$SOLO" --workers $w 2>&1 | grep -E 'Salida|Hecho'
  done
done
log "== FIN"
