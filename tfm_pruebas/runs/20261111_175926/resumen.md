# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T16:59:27 · deplodock 57a05f0a 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 17 (17 medidos, 0 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 1.63 | 3.86 | 1.08 | — | 0.43x | 1.54x | 1.50x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.67 | 1.63 | 4.21 | 1.09 | — | 0.40x | 1.53x | 1.49x | — |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.70 | 2.24 | 6.62 | 1.27 | 11.33 | 0.86x | 4.50x | 1.77x | 0.58 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.75 | 3.45 | 4.80 | 1.25 | 6.07 | 0.99x | 3.78x | 2.75x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.39 | 1.35 | 4.70 | 1.10 | 4.79 | 0.51x | 2.16x | 1.22x | 0.98 |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.68 | 5.41 | 1.15 | — | 0.29x | 1.36x | 1.45x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.17 | 6.47 | 3.13 | 21.47 | — | 10.26x | 1.50x | 0.30x | — |

Media geométrica vs eager: emmy fijado 0.81x · greedy 2.10x · greedy vs torch.compile 1.29x (greedy gana a torch.compile en 6/7). Reproducimos la referencia CloudRift (±5%) en 3/3 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.20 | 1.34 | 1.34 | — | 1.28x | 1.28x | 0.90x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 6.93 | 11.80 | 1.24 | 1.50 | — | 5.60x | 4.63x | 7.88x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.28 | 1.70 | 2.71 | 2.71 | — | 1.21x | 1.21x | 0.63x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.32 | 1.46 | 1.46 | — | 1.47x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.16 | 1.18 | — | 1.61x | 1.57x | 0.88x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.55 | 1.87 | 1.89 | 1.88 | — | 1.35x | 1.35x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 6.64 | 12.42 | 1.79 | 2.16 | — | 3.71x | 3.08x | 5.76x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.07 | 1.24 | 1.26 | — | 1.33x | 1.31x | 0.85x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.24 | 4.63 | 4.63 | — | 1.12x | 1.12x | 0.48x | — |

Media geométrica vs eager: emmy fijado 1.75x · greedy 1.67x · greedy vs torch.compile 1.27x (greedy gana a torch.compile en 2/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.17 | 6.47 | 3.13 | 21.47 | — | 10.26x | 1.50x | 0.30x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.44 | 10.68 | 5.14 | 23.32 | — | 12.54x | 2.76x | 0.46x | — |

Media geométrica vs eager: emmy fijado 11.34x · greedy 2.03x · greedy vs torch.compile 0.37x (greedy gana a torch.compile en 0/2). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.20 | 1.34 | 1.34 | — | 1.28x | 1.28x | 0.90x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 6.93 | 11.80 | 1.24 | 1.50 | — | 5.60x | 4.63x | 7.88x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.28 | 1.70 | 2.71 | 2.71 | — | 1.21x | 1.21x | 0.63x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.32 | 1.46 | 1.46 | — | 1.47x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.16 | 1.18 | — | 1.61x | 1.57x | 0.88x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.55 | 1.87 | 1.89 | 1.88 | — | 1.35x | 1.35x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 6.64 | 12.42 | 1.79 | 2.16 | — | 3.71x | 3.08x | 5.76x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.07 | 1.24 | 1.26 | — | 1.33x | 1.31x | 0.85x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.24 | 4.63 | 4.63 | — | 1.12x | 1.12x | 0.48x | — |

Media geométrica vs eager: emmy fijado 1.75x · greedy 1.67x · greedy vs torch.compile 1.27x (greedy gana a torch.compile en 2/9). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

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

