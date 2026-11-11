# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T17:03:19 · deplodock 57a05f0a 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 17 (17 medidos, 0 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 1.32 | 3.86 | 1.08 | — | 0.43x | 1.54x | 1.22x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.67 | 1.32 | 4.21 | 1.08 | — | 0.40x | 1.54x | 1.22x | — |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.70 | 3.66 | 6.61 | 1.28 | 11.33 | 0.86x | 4.45x | 2.86x | 0.58 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.75 | 3.45 | 4.81 | 1.26 | 6.07 | 0.99x | 3.77x | 2.74x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.38 | 1.35 | 4.72 | 1.08 | 4.79 | 0.51x | 2.20x | 1.25x | 0.98 |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.67 | 5.43 | 1.15 | — | 0.29x | 1.37x | 1.46x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.07 | 6.47 | 3.13 | 21.38 | — | 10.24x | 1.50x | 0.30x | — |

Media geométrica vs eager: emmy fijado 0.80x · greedy 2.10x · greedy vs torch.compile 1.30x (greedy gana a torch.compile en 6/7). Reproducimos la referencia CloudRift (±5%) en 3/3 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.35 | — | 1.28x | 1.27x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.06 | 1.22 | 1.23 | — | 1.34x | 1.34x | 0.86x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.70 | 2.70 | — | 1.22x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.30 | 1.46 | 1.47 | — | 1.46x | 1.45x | 0.88x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.23 | 1.17 | 1.18 | — | 1.60x | 1.58x | 1.04x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.85 | 1.89 | 1.88 | — | 1.34x | 1.35x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.36 | 1.78 | 1.80 | — | 1.33x | 1.32x | 0.75x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.16 | 1.24 | 1.25 | — | 1.33x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.33 | 4.62 | 4.62 | — | 1.13x | 1.13x | 0.50x | — |

Media geométrica vs eager: emmy fijado 1.33x · greedy 1.32x · greedy vs torch.compile 0.84x (greedy gana a torch.compile en 1/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.07 | 6.47 | 3.13 | 21.38 | — | 10.24x | 1.50x | 0.30x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.32 | 10.65 | 5.13 | 23.32 | — | 12.53x | 2.76x | 0.46x | — |

Media geométrica vs eager: emmy fijado 11.33x · greedy 2.03x · greedy vs torch.compile 0.37x (greedy gana a torch.compile en 0/2). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.35 | — | 1.28x | 1.27x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.06 | 1.22 | 1.23 | — | 1.34x | 1.34x | 0.86x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.70 | 2.70 | — | 1.22x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.30 | 1.46 | 1.47 | — | 1.46x | 1.45x | 0.88x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.23 | 1.17 | 1.18 | — | 1.60x | 1.58x | 1.04x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.85 | 1.89 | 1.88 | — | 1.34x | 1.35x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.36 | 1.78 | 1.80 | — | 1.33x | 1.32x | 0.75x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.16 | 1.24 | 1.25 | — | 1.33x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.33 | 4.62 | 4.62 | — | 1.13x | 1.13x | 0.50x | — |

Media geométrica vs eager: emmy fijado 1.33x · greedy 1.32x · greedy vs torch.compile 0.84x (greedy gana a torch.compile en 1/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

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

