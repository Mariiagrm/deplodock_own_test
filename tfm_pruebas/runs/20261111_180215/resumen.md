# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T17:02:16 · deplodock 57a05f0a 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 17 (17 medidos, 0 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.56 | 1.58 | 3.75 | 0.98 | — | 0.41x | 1.59x | 1.62x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.56 | 1.52 | 4.07 | 0.97 | — | 0.38x | 1.61x | 1.57x | — |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.26 | 2.13 | 6.49 | 1.17 | 11.33 | 0.81x | 4.50x | 1.82x | 0.57 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.45 | 3.32 | 4.68 | 1.13 | 6.07 | 0.95x | 3.93x | 2.93x | 0.77 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.15 | 1.23 | 4.60 | 0.97 | 4.79 | 0.47x | 2.22x | 1.27x | 0.96 |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.45 | 1.22 | 5.30 | 1.06 | — | 0.27x | 1.37x | 1.16x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.17 | 6.79 | 3.12 | 21.61 | — | 10.32x | 1.49x | 0.31x | — |

Media geométrica vs eager: emmy fijado 0.77x · greedy 2.14x · greedy vs torch.compile 1.30x (greedy gana a torch.compile en 6/7). Reproducimos la referencia CloudRift (±5%) en 3/3 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.60 | 1.11 | 1.23 | 1.24 | — | 1.31x | 1.29x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.53 | 0.87 | 1.12 | 1.11 | — | 1.37x | 1.38x | 0.78x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.19 | 2.31 | 2.61 | 2.61 | — | 1.22x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.33 | 1.34 | 1.46 | — | 1.58x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.75 | 1.12 | 1.05 | 1.06 | — | 1.66x | 1.65x | 1.06x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.42 | 1.85 | 1.77 | 1.78 | — | 1.37x | 1.36x | 1.04x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.42 | 1.67 | 1.80 | — | 1.43x | 1.32x | 0.79x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.66 | 1.16 | 1.24 | 1.26 | — | 1.33x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.09 | 2.38 | 4.53 | 4.53 | — | 1.12x | 1.13x | 0.53x | — |

Media geométrica vs eager: emmy fijado 1.37x · greedy 1.34x · greedy vs torch.compile 0.85x (greedy gana a torch.compile en 2/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.17 | 6.79 | 3.12 | 21.61 | — | 10.32x | 1.49x | 0.31x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.48 | 10.66 | 5.14 | 23.27 | — | 12.53x | 2.77x | 0.46x | — |

Media geométrica vs eager: emmy fijado 11.37x · greedy 2.03x · greedy vs torch.compile 0.38x (greedy gana a torch.compile en 0/2). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.60 | 1.11 | 1.23 | 1.24 | — | 1.31x | 1.29x | 0.89x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.53 | 0.87 | 1.12 | 1.11 | — | 1.37x | 1.38x | 0.78x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.19 | 2.31 | 2.61 | 2.61 | — | 1.22x | 1.22x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.33 | 1.34 | 1.46 | — | 1.58x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.75 | 1.12 | 1.05 | 1.06 | — | 1.66x | 1.65x | 1.06x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.42 | 1.85 | 1.77 | 1.78 | — | 1.37x | 1.36x | 1.04x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.42 | 1.67 | 1.80 | — | 1.43x | 1.32x | 0.79x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.66 | 1.16 | 1.24 | 1.26 | — | 1.33x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.09 | 2.38 | 4.53 | 4.53 | — | 1.12x | 1.13x | 0.53x | — |

Media geométrica vs eager: emmy fijado 1.37x · greedy 1.34x · greedy vs torch.compile 0.85x (greedy gana a torch.compile en 2/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

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

