# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T16:58:22 · deplodock 57a05f0a 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 17 (17 medidos, 0 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.56 | 1.52 | 3.79 | 0.96 | — | 0.41x | 1.63x | 1.58x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.56 | 1.53 | 4.12 | 0.98 | — | 0.38x | 1.60x | 1.57x | — |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.30 | 3.48 | 6.49 | 1.17 | 11.33 | 0.82x | 4.53x | 2.98x | 0.57 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.42 | 2.19 | 4.68 | 1.14 | 6.07 | 0.94x | 3.88x | 1.93x | 0.77 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.16 | 1.23 | 4.60 | 0.98 | 4.79 | 0.47x | 2.20x | 1.25x | 0.96 |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.46 | 1.22 | 5.41 | 1.05 | — | 0.27x | 1.39x | 1.16x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.22 | 6.52 | 3.14 | 21.31 | — | 10.27x | 1.51x | 0.31x | — |

Media geométrica vs eager: emmy fijado 0.77x · greedy 2.15x · greedy vs torch.compile 1.30x (greedy gana a torch.compile en 6/7). Reproducimos la referencia CloudRift (±5%) en 3/3 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.73 | 1.22 | 1.23 | 1.35 | — | 1.41x | 1.28x | 0.90x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 6.38 | 11.67 | 1.12 | 1.52 | — | 5.69x | 4.19x | 7.67x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.18 | 1.61 | 2.60 | 2.60 | — | 1.22x | 1.23x | 0.62x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.27 | 1.45 | 1.46 | — | 1.46x | 1.45x | 0.87x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.87 | 1.06 | 1.16 | 1.17 | — | 1.61x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.44 | 1.75 | 1.79 | 1.79 | — | 1.36x | 1.36x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.57 | 1.67 | 1.79 | — | 1.42x | 1.33x | 0.87x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.66 | 1.15 | 1.26 | 1.26 | — | 1.32x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.22 | 2.30 | 4.64 | 4.63 | — | 1.13x | 1.13x | 0.50x | — |

Media geométrica vs eager: emmy fijado 1.59x · greedy 1.51x · greedy vs torch.compile 1.03x (greedy gana a torch.compile en 1/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.22 | 6.52 | 3.14 | 21.31 | — | 10.27x | 1.51x | 0.31x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.56 | 10.68 | 5.14 | 24.62 | — | 12.56x | 2.62x | 0.43x | — |

Media geométrica vs eager: emmy fijado 11.36x · greedy 1.99x · greedy vs torch.compile 0.36x (greedy gana a torch.compile en 0/2). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.73 | 1.22 | 1.23 | 1.35 | — | 1.41x | 1.28x | 0.90x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 6.38 | 11.67 | 1.12 | 1.52 | — | 5.69x | 4.19x | 7.67x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.18 | 1.61 | 2.60 | 2.60 | — | 1.22x | 1.23x | 0.62x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.27 | 1.45 | 1.46 | — | 1.46x | 1.45x | 0.87x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.87 | 1.06 | 1.16 | 1.17 | — | 1.61x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.44 | 1.75 | 1.79 | 1.79 | — | 1.36x | 1.36x | 0.98x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.57 | 1.67 | 1.79 | — | 1.42x | 1.33x | 0.87x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.66 | 1.15 | 1.26 | 1.26 | — | 1.32x | 1.32x | 0.92x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.22 | 2.30 | 4.64 | 4.63 | — | 1.13x | 1.13x | 0.50x | — |

Media geométrica vs eager: emmy fijado 1.59x · greedy 1.51x · greedy vs torch.compile 1.03x (greedy gana a torch.compile en 1/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

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

