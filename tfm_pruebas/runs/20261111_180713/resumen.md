# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T17:07:14 · deplodock 57a05f0a 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 17 (17 medidos, 0 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 1.63 | 3.87 | 1.09 | — | 0.43x | 1.53x | 1.50x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.67 | 1.65 | 4.22 | 1.08 | — | 0.40x | 1.55x | 1.52x | — |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.69 | 3.60 | 6.61 | 1.27 | 11.33 | 0.86x | 4.49x | 2.84x | 0.58 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.74 | 2.31 | 4.82 | 1.25 | 6.07 | 0.98x | 3.78x | 1.84x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.39 | 1.72 | 4.71 | 1.09 | 4.79 | 0.51x | 2.19x | 1.58x | 0.98 |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.67 | 5.40 | 1.15 | — | 0.29x | 1.37x | 1.46x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.10 | 6.47 | 3.14 | 21.79 | — | 10.23x | 1.47x | 0.30x | — |

Media geométrica vs eager: emmy fijado 0.80x · greedy 2.10x · greedy vs torch.compile 1.35x (greedy gana a torch.compile en 6/7). Reproducimos la referencia CloudRift (±5%) en 3/3 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.34 | — | 1.27x | 1.27x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.02 | 1.24 | 1.24 | — | 1.32x | 1.32x | 0.82x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.71 | 2.71 | — | 1.21x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.34 | 1.46 | 1.46 | — | 1.46x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.17 | 1.17 | — | 1.60x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.87 | 1.88 | 1.90 | — | 1.35x | 1.34x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.37 | 1.43 | 1.79 | 1.79 | — | 1.33x | 1.32x | 0.80x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.08 | 1.25 | 1.25 | — | 1.32x | 1.32x | 0.86x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.22 | 2.34 | 4.63 | 4.62 | — | 1.13x | 1.13x | 0.51x | — |

Media geométrica vs eager: emmy fijado 1.33x · greedy 1.32x · greedy vs torch.compile 0.83x (greedy gana a torch.compile en 0/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.10 | 6.47 | 3.14 | 21.79 | — | 10.23x | 1.47x | 0.30x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.34 | 10.85 | 5.15 | 24.46 | — | 12.50x | 2.63x | 0.44x | — |

Media geométrica vs eager: emmy fijado 11.31x · greedy 1.97x · greedy vs torch.compile 0.36x (greedy gana a torch.compile en 0/2). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.34 | — | 1.27x | 1.27x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.02 | 1.24 | 1.24 | — | 1.32x | 1.32x | 0.82x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.71 | 2.71 | — | 1.21x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.34 | 1.46 | 1.46 | — | 1.46x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.17 | 1.17 | — | 1.60x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.87 | 1.88 | 1.90 | — | 1.35x | 1.34x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.37 | 1.43 | 1.79 | 1.79 | — | 1.33x | 1.32x | 0.80x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.08 | 1.25 | 1.25 | — | 1.32x | 1.32x | 0.86x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.22 | 2.34 | 4.63 | 4.62 | — | 1.13x | 1.13x | 0.51x | — |

Media geométrica vs eager: emmy fijado 1.33x · greedy 1.32x · greedy vs torch.compile 0.83x (greedy gana a torch.compile en 0/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## Matriz de técnicas (schedule fijado)

✓ = presente en algún kernel del caso · · = no · – = no aplica. Detalle y evidencia en `logs/<caso>.log`.

| caso | Fact | RegT | Vec | TC | Smem | Async | Pipe | WSpec | Bank | Rast | SplitK | Coop | PDL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul/bf16-mma-trans-b-masked-n-async | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/bf16-mma-trans-b-masked-n-tma | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-epilogue-residual | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k128-f32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k128-f32-trans | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/imap-reshape-a-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/nvfp4-w4a4-packed-slab-loopify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| qwen3emb/rmsnorm-knorm-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| fused/fp4-tma-w4a4-two-channel-p2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |

Leyenda: **Fact** Factorización de bucles · **RegT** Register tiling · **Vec** Cargas vectorizadas e intercalado · **TC** Tensor Cores (mma.sync + ldmatrix) · **Smem** Almacenamiento por fases en smem · **Async** Transferencias asíncronas (cp.async / TMA) · **Pipe** Software pipelining / doble búfer · **WSpec** Especialización de warps · **Bank** Padding y swizzling · **Rast** CTA swizzle (rasterizado agrupado) · **SplitK** Split-K · **Coop** Reducción cooperativa (shuffle / árbol) · **PDL** Programmatic Dependent Launch

