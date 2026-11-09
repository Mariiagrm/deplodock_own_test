# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-09T01:10:58 · deplodock 5f0a3770 2026-11-09 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 130 (128 medidos, 2 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-sm120 | matmul | (128,256) x (256,128) | 2.15 | 1.89 | 1.78 | 1.67 | — | 1.21x | 1.29x | 1.13x | — |
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 1.64 | 3.87 | 1.08 | — | 0.43x | 1.55x | 1.51x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.68 | 1.63 | 4.22 | 1.09 | — | 0.40x | 1.54x | 1.50x | — |
| bf16-rowless-grouped-pair | matmul+elementwise | (256) x (4,256,16) x (4,256,16) | 4.80 | 2.55 | 1.63 | 1.48 | — | 2.95x | 3.25x | 1.72x | — |
| bf16-scalar-sync-d2-sm120 | linear | (32,384) x (64,384) | 2.71 | 1.84 | 4.44 | 1.66 | — | 0.61x | 1.64x | 1.11x | — |
| f16-matvec-reshaped-output-tma | linear+reshape+index_map | (1,1,1024) x (2048,1024) | 4.55 | 2.35 | 6.29 | 2.23 | — | 0.72x | 2.04x | 1.05x | — |
| f16-mma-batched-pv-b1h4kv128 | matmul | (1,4,128,128) x (1,4,128,128) | 4.55 | 4.76 | 5.40 | 1.17 | 5.80 | 0.84x | 3.89x | 4.07x | 0.93 |
| f16-mma-batched-pv-b2h3kv256 | matmul | (2,3,128,256) x (2,3,256,128) | 6.42 | 5.95 | 8.57 | 1.93 | 6.02 | 0.75x | 3.33x | 3.09x | **1.42** |
| f16-mma-batched-qk-b1h4d64 | matmul+transpose | (1,4,128,64) x (1,4,128,64) | 2.99 | 3.36 | 3.83 | 1.52 | 5.92 | 0.78x | 1.97x | 2.21x | 0.65 |
| f16-mma-batched-qk-b2h3d32 | matmul+transpose | (2,3,128,32) x (2,3,128,32) | 2.90 | 2.68 | 2.82 | 1.02 | 2.87 | 1.03x | 2.84x | 2.62x | 0.98 |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.72 | 3.60 | 6.62 | 1.27 | 11.33 | 0.86x | 4.51x | 2.84x | 0.58 |
| f16-mma-f16acc-cp | matmul | (128,128) x (128,128) | 1.54 | 1.34 | 3.67 | 1.19 | 4.91 | 0.42x | 1.29x | 1.13x | 0.75 |
| f16-mma-f16acc-gmem | matmul | (128,128) x (128,128) | 1.53 | 1.34 | 7.33 | 1.19 | 8.68 | 0.21x | 1.29x | 1.13x | 0.84 |
| f16-mma-f16acc-symbolic-k | matmul | (128,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-mma-f16acc-tma | matmul | (128,128) x (128,128) | 1.55 | 1.37 | 3.87 | 1.22 | 4.54 | 0.40x | 1.27x | 1.13x | 0.85 |
| f16-mma-m128n128k128-f16 | matmul | (128,128) x (128,128) | 1.53 | 1.34 | 4.70 | 1.15 | 6.21 | 0.33x | 1.33x | 1.17x | 0.76 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.78 | 2.30 | 4.80 | 1.26 | 6.07 | 0.99x | 3.80x | 1.83x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.39 | 1.35 | 4.72 | 1.09 | 4.79 | 0.51x | 2.20x | 1.24x | 0.99 |
| f16-mma-m128n128k132-f32-masked-k | matmul | (128,132) x (132,128) | 5.04 | 3.97 | 6.46 | 9.76 | 7.23 | 0.78x | 0.52x | 0.41x | 0.89 |
| f16-mma-m128n128k132-f32-trans-masked-k | linear | (128,132) x (128,132) | 2.90 | 1.80 | 6.39 | 8.21 | 6.54 | 0.45x | 0.35x | 0.22x | 0.98 |
| f16-mma-m128n128k136-f32-masked-k | matmul | (128,136) x (136,128) | 4.92 | 3.95 | 6.20 | 9.93 | 7.09 | 0.79x | 0.50x | 0.40x | 0.87 |
| f16-mma-parity-m256-cp | matmul | (256,512) x (512,1024) | 4.58 | 3.63 | 3.84 | 3.72 | 5.58 | 1.19x | 1.23x | 0.98x | 0.69 |
| f16-mma-parity-m256-tma | matmul | (256,512) x (512,1024) | 4.57 | 3.62 | 3.99 | 3.72 | 4.51 | 1.15x | 1.23x | 0.97x | 0.88 |
| f16-mma-raster-gm8 | matmul | (1280,256) x (256,512) | 3.96 | 3.46 | 3.41 | 5.29 | 4.98 | 1.16x | 0.75x | 0.65x | 0.68 |
| f16-mma-splitk-atomic | matmul | (128,512) x (512,128) | 4.91 | 4.88 | 8.61 | 1.74 | — | 0.57x | 2.82x | 2.80x | — |
| f16-mma-splitk-atomic-f16acc-tma-rolled | matmul | (32,1024) x (1024,256) | 5.60 | 5.47 | 4.84 | 2.30 | — | 1.16x | 2.43x | 2.38x | — |
| f16-mma-splitk-deferred | matmul | (128,512) x (512,128) | 2.76 | 3.28 | 26.40 | 2.12 | — | 0.10x | 1.30x | 1.55x | — |
| f16-mma-trans-b-staged-cp | linear | (128,128) x (128,128) | 1.48 | 1.29 | 4.18 | 1.10 | 6.07 | 0.35x | 1.35x | 1.17x | 0.69 |
| f16-mma-trans-b-staged-tma | linear | (128,128) x (128,128) | 1.48 | 1.29 | 4.64 | 1.09 | 5.79 | 0.32x | 1.36x | 1.18x | 0.80 |
| f16-scalar-register-tile | matmul | (512,512) x (512,512) | 4.64 | 3.50 | 107.84 | 8.12 | 97.24 | 0.04x | 0.57x | 0.43x | **1.11** |
| f16-symbolic-batched-mk-greedy | matmul | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 12.32 | 8.32 | 34.63 | 353.40 | 38.31 | 0.36x | 0.03x | 0.02x | 0.90 |
| f16-symbolic-computed-a-k-warp | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 22.53 | 13.79 | 31.55 | 4599.74 | 33.38 | 0.71x | 0.00x | 0.00x | 0.95 |
| f16-symbolic-demoted-n-greedy | linear+elementwise | ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) | 3.94 | 3.36 | 19.31 | 117.72 | 22.59 | 0.20x | 0.03x | 0.03x | 0.85 |
| f16-symbolic-demoted-pv-greedy | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-symbolic-k-gmem | matmul | (64,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | 2.73 | 2.82 | 10.69 | 3.53 | 12.74 | 0.26x | 0.77x | 0.80x | 0.84 |
| f16-symbolic-k-trans | linear | (64,{'sym': 'seq_len', 'hint': 512}) x (32,{'sym': 'seq_len', 'hint': 512}) | 2.83 | 2.08 | 16.17 | 46.78 | 13.86 | 0.18x | 0.06x | 0.04x | **1.17** |
| f16-symbolic-m-residual-cp | matmul+elementwise | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) x ({'sym': 'seq_len', 'hint': 512},1024) | 5.86 | 5.63 | 6.43 | 7.05 | 7.62 | 0.91x | 0.83x | 0.80x | 0.84 |
| f16-symbolic-m-tma | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.65 | 4.65 | 5.34 | 5.51 | 6.28 | 0.87x | 0.84x | 0.85x | 0.85 |
| f16-symbolic-mn-gmem | matmul | ({'sym': 'seq_len', 'hint': 512},128) x (128,{'sym': 'seq_len', 'hint': 512}) | 2.74 | 2.76 | 4.29 | 210.60 | 5.16 | 0.64x | 0.01x | 0.01x | 0.83 |
| f16-warp-band-p1-tma-d1-m128 | matmul | (128,256) x (256,128) | 2.02 | 1.88 | 4.03 | 1.45 | 4.47 | 0.50x | 1.39x | 1.30x | 0.90 |
| f16-warp-band-p1-tma-d2-m100 | matmul | (100,256) x (256,128) | 2.00 | 2.17 | 2.61 | 1.34 | 2.91 | 0.77x | 1.49x | 1.62x | 0.90 |
| f16-warp-band-p1-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.02 | 1.89 | 2.62 | 1.46 | 2.95 | 0.77x | 1.38x | 1.29x | 0.89 |
| f16-warp-band-p2-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.02 | 1.88 | 2.65 | 1.45 | 2.93 | 0.76x | 1.39x | 1.30x | 0.90 |
| f16-warp-gmem-direct | matmul | (128,128) x (128,128) | 1.53 | 1.34 | 2.18 | 1.14 | 3.42 | 0.70x | 1.34x | 1.17x | 0.64 |
| f16-warp-splitk | matmul | (64,4096) x (4096,512) | 5.85 | 5.84 | 53.10 | 10.85 | — | 0.11x | 0.54x | 0.54x | — |
| f16-warp-symbolic-m-cp | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.65 | 4.64 | 5.24 | 5.56 | 6.20 | 0.89x | 0.84x | 0.83x | 0.85 |
| f32-epilogue-bias | matmul+elementwise | (1,64,64) x (64,64) x (64) | 4.28 | 2.97 | 1.71 | 3.12 | 2.10 | 2.50x | 1.37x | 0.95x | 0.82 |
| f32-epilogue-relu | matmul+elementwise | (1,64,64) x (64,64) | 4.09 | 2.21 | 1.71 | 2.91 | 1.90 | 2.40x | 1.40x | 0.76x | 0.90 |
| f32-epilogue-residual | matmul+elementwise | (1,64,64) x (64,64) x (1,64,64) | 4.10 | 2.30 | 1.73 | 3.13 | 2.20 | 2.37x | 1.31x | 0.73x | 0.78 |
| f32-epilogue-scale | matmul+elementwise | (1,64,64) x (64,64) x (1) | 4.24 | 3.02 | 1.73 | 3.39 | 2.09 | 2.45x | 1.25x | 0.89x | 0.83 |
| f32-mixed-dtype-slabs | matmul | (64,128) x (128,96) | 3.92 | 2.45 | 2.18 | 1.34 | — | 1.80x | 2.93x | 1.83x | — |
| f32-reduce-coop | matmul | (1,64,64) x (64,64) | 3.16 | 2.18 | 3.23 | 2.91 | 4.58 | 0.98x | 1.08x | 0.75x | 0.71 |
| f32-reduce-coop-ilp | matmul | (1,64,64) x (64,64) | 3.16 | 2.18 | 3.23 | 2.92 | 3.66 | 0.98x | 1.08x | 0.75x | 0.88 |
| f32-reduce-ilp | matmul | (1,64,64) x (64,64) | 3.16 | 2.18 | 1.79 | 2.91 | 4.07 | 1.76x | 1.08x | 0.75x | 0.44 |
| f32-scalar-bm8-outside-hints | matmul | (2048,2048) x (2048,2048) | 184.28 | 175.77 | 1363.17 | 238.50 | 1596.48 | 0.14x | 0.77x | 0.74x | 0.85 |
| f32-scalar-composite-fm4-fn4 | matmul | (2048,2048) x (2048,2048) | 184.13 | 175.75 | 348.54 | 238.00 | 530.50 | 0.53x | 0.77x | 0.74x | 0.66 |
| f32-scalar-composite-fn4 | matmul | (2048,2048) x (2048,2048) | 184.10 | 176.28 | 777.28 | 238.50 | 1271.23 | 0.24x | 0.77x | 0.74x | 0.61 |
| f32-scalar-masked-m-fm26 | matmul | (2048,2048) x (2048,2048) | 184.32 | 175.76 | 410.26 | 237.99 | 659.81 | 0.45x | 0.77x | 0.74x | 0.62 |
| f32-scalar-masked-n-fn26 | matmul | (512,512) x (512,512) | 7.26 | 5.40 | 300.12 | 8.98 | 75.39 | 0.02x | 0.81x | 0.60x | **3.98** |
| f32-scalar-masked-n-gmem | matmul | (64,64) x (64,47) | 3.62 | 1.58 | 4.31 | 2.73 | 2.84 | 0.84x | 1.33x | 0.58x | **1.52** |
| f32-scalar-staging-d1-128x128x128 | matmul | (128,128) x (128,128) | 3.00 | 2.62 | 6.23 | 9.87 | 3.29 | 0.48x | 0.30x | 0.27x | **1.90** |
| f32-scalar-staging-d1-64x64x64 | matmul | (64,64) x (64,64) | 3.16 | 2.18 | 1.32 | 2.91 | 1.26 | 2.40x | 1.09x | 0.75x | 1.04 |
| f32-scalar-staging-d2-128x128x128 | matmul | (128,128) x (128,128) | 3.01 | 2.61 | 6.34 | 9.84 | 3.06 | 0.48x | 0.31x | 0.27x | **2.07** |
| f32-scalar-staging-d2-64x64x64 | matmul | (64,64) x (64,64) | 3.16 | 2.18 | 1.31 | 2.91 | 1.38 | 2.41x | 1.09x | 0.75x | 0.95 |
| f32-scalar-staging-gmem-128x128x128 | matmul | (128,128) x (128,128) | 3.02 | 2.61 | 13.46 | 9.82 | 3.24 | 0.22x | 0.31x | 0.27x | **4.15** |
| f32-scalar-staging-gmem-64x64x64 | matmul | (64,64) x (64,64) | 3.15 | 2.18 | 2.72 | 2.92 | 2.00 | 1.16x | 1.08x | 0.75x | **1.36** |
| f32-scalar-tma-staged | matmul | (2048,2048) x (2048,2048) | 184.66 | 175.75 | 348.63 | 238.00 | 426.18 | 0.53x | 0.78x | 0.74x | 0.82 |
| f32-tile-per-cell | matmul | (1,64,64) x (64,64) | 3.15 | 2.19 | 1.77 | 2.91 | 1.91 | 1.78x | 1.08x | 0.75x | 0.93 |
| f32-tile-reg-2d | matmul | (1,64,64) x (64,64) | 3.16 | 2.18 | 2.87 | 3.06 | 3.11 | 1.10x | 1.03x | 0.71x | 0.92 |
| f32-tile-reg-fn3 | matmul | (1,64,64) x (64,64) | 3.15 | 2.19 | 3.52 | 3.06 | 3.95 | 0.89x | 1.03x | 0.71x | 0.89 |
| f32-tile-reg-inner-fn4 | matmul | (1,64,64) x (64,64) | 3.17 | 2.19 | 2.87 | 2.91 | 3.14 | 1.10x | 1.09x | 0.75x | 0.91 |
| f32-tile-single-cta | matmul | (1,64,64) x (64,64) | 3.16 | 2.18 | 2.85 | 2.92 | 2.88 | 1.11x | 1.08x | 0.75x | 0.99 |
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 279.25 | 53.10 | 4.12 | 18.16 | — | 67.75x | 15.37x | 2.92x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.13 | 5.50 | 2.84 | 17.73 | — | 11.30x | 1.81x | 0.31x | — |
| imap-reshape-a-cp | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.34 | 5.57 | 1.16 | 1.75 | 0.28x | 1.35x | 1.15x | **3.17** |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.34 | 5.41 | 1.17 | — | 0.29x | 1.35x | 1.15x | — |
| imap-reshape-b-gmem | matmul+reshape | (64,128) x (256,64) | 1.69 | 1.36 | 5.01 | 5.76 | — | 0.34x | 0.29x | 0.24x | — |
| imap-slice-a-cp | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 2.61 | 1.15 | 1.85 | 0.64x | 1.45x | 1.14x | **1.41** |
| imap-slice-a-gmem | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 5.44 | 1.15 | 2.06 | 0.31x | 1.45x | 1.15x | **2.64** |
| imap-slice-a-tma | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 1.83 | 1.15 | 2.10 | 0.91x | 1.44x | 1.15x | 0.87 |
| imap-transpose-a-gmem | matmul+transpose | (256,128) x (256,64) | 2.26 | 2.21 | 11.67 | 11.90 | 7.78 | 0.19x | 0.19x | 0.19x | **1.50** |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.64 | 4.84 | 1.79 | 1.41 | — | 23.26x | 29.46x | 3.43x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 2.37 | 2.08 | 1.43 | — | 44.21x | 64.31x | 1.65x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.29 | 5.47 | 2.31 | 2.38 | — | 13.57x | 13.17x | 2.30x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.66 | 3.80 | 4.43 | 1.39 | — | 6.02x | 19.16x | 2.73x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 188.46 | 41.58 | 3.13 | 21.78 | — | 60.27x | 8.65x | 1.91x | — |

Media geométrica vs eager: emmy fijado 0.82x · greedy 1.11x · greedy vs torch.compile 0.76x (greedy gana a torch.compile en 39/84). Reproducimos la referencia CloudRift (±5%) en 51/64 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-qk-sdpa-composed-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.99 | 4.74 | 3.49 | 3.17 | — | 1.72x | 1.89x | 1.49x | — |
| rmsnorm-qk-sdpa-output-axis | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.73 | 1.92 | 3.24 | — | 3.10x | 1.85x | 1.46x | — |
| rmsnorm-qk-sdpa-stat-b-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.73 | 4.21 | 3.19 | — | 1.42x | 1.87x | 1.48x | — |
| rmsnorm-qk-sdpa-stat-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.96 | 4.72 | 3.51 | 3.17 | — | 1.70x | 1.88x | 1.49x | — |
| norm-linear-f16-scalar-reduce | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.96 | 2.21 | 106.84 | 2.24 | 80.04 | 0.03x | 1.32x | 0.99x | **1.33** |
| norm-linear-f16-warp-masked-m | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.96 | 2.22 | 6.43 | 2.21 | 11.58 | 0.46x | 1.34x | 1.00x | 0.56 |
| norm-weight-linear-f16-computed-b | linear+rms_norm | (64,128) x (64,128) x (128) | 3.33 | 2.28 | 78.82 | 90.82 | — | 0.04x | 0.04x | 0.03x | — |
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.72 | 1.21 | 1.35 | 1.33 | — | 1.27x | 1.29x | 0.91x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.05 | 1.23 | 1.24 | — | 1.34x | 1.32x | 0.84x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 1.59 | 2.71 | 2.72 | — | 1.21x | 1.21x | 0.59x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.12 | 1.53 | 1.45 | 1.46 | — | 1.46x | 1.45x | 1.05x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.16 | 1.17 | — | 1.60x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.87 | 1.88 | 1.88 | — | 1.35x | 1.35x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.36 | 1.79 | 1.79 | — | 1.33x | 1.33x | 0.76x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.07 | 1.24 | 1.24 | — | 1.33x | 1.33x | 0.86x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.46 | 4.63 | 4.62 | — | 1.12x | 1.13x | 0.53x | — |
| fused-add-rms-norm | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.34 | 0.91 | 1.05 | 1.07 | — | 2.22x | 2.19x | 0.85x | — |
| fused-add-rms-norm-escaping-residual | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.34 | 0.91 | 1.07 | 1.07 | — | 2.19x | 2.18x | 0.85x | — |
| rms-norm-cut-sweep-work | rms_norm | (512,4096) | 5.14 | 4.60 | 358.12 | 4.91 | — | 0.01x | 1.05x | 0.94x | — |
| rms-norm-packed-cells | rms_norm | (2048,128) | 2.32 | 1.26 | 1.43 | 1.96 | — | 1.62x | 1.19x | 0.64x | — |

Media geométrica vs eager: emmy fijado 0.79x · greedy 1.23x · greedy vs torch.compile 0.78x (greedy gana a torch.compile en 6/20). Reproducimos la referencia CloudRift (±5%) en 1/2 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 279.25 | 53.10 | 4.12 | 18.16 | — | 67.75x | 15.37x | 2.92x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.13 | 5.50 | 2.84 | 17.73 | — | 11.30x | 1.81x | 0.31x | — |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.64 | 4.84 | 1.79 | 1.41 | — | 23.26x | 29.46x | 3.43x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 2.37 | 2.08 | 1.43 | — | 44.21x | 64.31x | 1.65x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.29 | 5.47 | 2.31 | 2.38 | — | 13.57x | 13.17x | 2.30x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.66 | 3.80 | 4.43 | 1.39 | — | 6.02x | 19.16x | 2.73x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 188.46 | 41.58 | 3.13 | 21.78 | — | 60.27x | 8.65x | 1.91x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.61 | 10.58 | 5.15 | 24.41 | — | 12.56x | 2.65x | 0.43x | — |
| nvfp4-gate-up-requant-full-projection-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.98 | 12.39 | 497.58 | 61.13 | — | 0.35x | 2.81x | 0.20x | — |
| nvfp4-gate-up-requant-place-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.98 | 12.38 | 646.72 | 61.38 | — | 0.27x | 2.80x | 0.20x | — |
| nvfp4-quantize-cut-shared-normalizer | elementwise+index_map+mean+reduce+reshape | (32,128) | 127.12 | 3.10 | 38.48 | 36.84 | — | 3.30x | 3.45x | 0.08x | — |
| nvfp4-w4a16-sibling-cut-async | elementwise+index_map+linear+reshape+gather | (48,256) | 38.97 | 6.32 | 12.77 | 7.07 | — | 3.05x | 5.51x | 0.89x | — |
| nvfp4-w4a16-sibling-cut-tma-p2 | elementwise+index_map+linear+reshape+gather | (48,256) | 38.96 | 6.31 | 13.60 | 7.06 | — | 2.86x | 5.52x | 0.89x | — |

Media geométrica vs eager: emmy fijado 7.16x · greedy 7.53x · greedy vs torch.compile 0.82x (greedy gana a torch.compile en 6/13). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.72 | 1.21 | 1.35 | 1.33 | — | 1.27x | 1.29x | 0.91x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.05 | 1.23 | 1.24 | — | 1.34x | 1.32x | 0.84x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 1.59 | 2.71 | 2.72 | — | 1.21x | 1.21x | 0.59x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.12 | 1.53 | 1.45 | 1.46 | — | 1.46x | 1.45x | 1.05x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.05 | 1.16 | 1.17 | — | 1.60x | 1.59x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.54 | 1.87 | 1.88 | 1.88 | — | 1.35x | 1.35x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.36 | 1.79 | 1.79 | — | 1.33x | 1.33x | 0.76x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.07 | 1.24 | 1.24 | — | 1.33x | 1.33x | 0.86x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.46 | 4.63 | 4.62 | — | 1.12x | 1.13x | 0.53x | — |
| gated-mlp-s128 | matmul+elementwise | (1,128,1024) x (1024,3072) x (1024,3072) | 31.81 | 29.06 | 438.82 | 7323.33 | — | 0.07x | 0.00x | 0.00x | — |
| gated-mlp-s32 | matmul+elementwise | (1,32,1024) x (1024,3072) x (1024,3072) | 28.29 | 22.91 | 117.12 | 1842.82 | — | 0.24x | 0.02x | 0.01x | — |
| gated-mlp-s512 | matmul+elementwise | (1,512,1024) x (1024,3072) x (1024,3072) | 104.35 | 92.67 | 1736.13 | 28864.16 | — | 0.06x | 0.00x | 0.00x | — |
| matmul-add-downproj-s128 | matmul+elementwise | (1,128,3072) x (3072,1024) x (1,128,1024) | 13.25 | 13.21 | 38.12 | 19.72 | 44.56 | 0.35x | 0.67x | 0.67x | 0.86 |
| matmul-add-downproj-s32 | matmul+elementwise | (1,32,3072) x (3072,1024) x (1,32,1024) | 13.89 | 14.02 | 33.61 | 903.55 | 39.57 | 0.41x | 0.02x | 0.02x | 0.85 |
| matmul-add-downproj-s512 | matmul+elementwise | (1,512,3072) x (3072,1024) x (1,512,1024) | 38.45 | 38.27 | 133.70 | 82.23 | 153.72 | 0.29x | 0.47x | 0.47x | 0.87 |
| matmul-add-oproj-s128 | matmul+elementwise | (1,128,2048) x (2048,1024) x (1,128,1024) | 11.00 | 10.94 | 25.12 | 14.00 | 30.06 | 0.44x | 0.79x | 0.78x | 0.84 |
| matmul-add-oproj-s32 | matmul+elementwise | (1,32,2048) x (2048,1024) x (1,32,1024) | 10.37 | 9.58 | 22.73 | 598.19 | 26.94 | 0.46x | 0.02x | 0.02x | 0.84 |
| matmul-add-oproj-s512 | matmul+elementwise | (1,512,2048) x (2048,1024) x (1,512,1024) | 26.52 | 26.54 | 89.35 | 56.19 | 102.91 | 0.30x | 0.47x | 0.47x | 0.87 |
| matmul-kvproj-s128 | matmul | (1,128,1024) x (1024,1024) | 13.14 | 10.09 | 13.68 | 11.78 | 15.69 | 0.96x | 1.11x | 0.86x | 0.87 |
| matmul-kvproj-s32 | matmul | (1,32,1024) x (1024,1024) | 7.37 | 6.68 | 11.99 | 296.95 | 13.77 | 0.61x | 0.02x | 0.02x | 0.87 |
| matmul-kvproj-s512 | matmul | (1,512,1024) x (1024,1024) | 13.99 | 13.63 | 45.65 | 23.49 | 52.62 | 0.31x | 0.60x | 0.58x | 0.87 |
| matmul-qproj-s128 | matmul | (1,128,1024) x (1024,2048) | 13.06 | 13.06 | 27.59 | 15.06 | 31.25 | 0.47x | 0.87x | 0.87x | 0.88 |
| matmul-qproj-s32 | matmul | (1,32,1024) x (1024,2048) | 8.42 | 8.40 | 12.33 | 8.05 | 14.20 | 0.68x | 1.05x | 1.04x | 0.87 |
| matmul-qproj-s512 | matmul | (1,512,1024) x (1024,2048) | 25.74 | 25.54 | 55.64 | 38.90 | 62.24 | 0.46x | 0.66x | 0.66x | 0.89 |
| sdpa-s128 | sdpa | (1,16,128,128) x (1,8,128,128) x (1,8,128,128) | 27.58 | 25.06 | 1441.66 | 9176.61 | — | 0.02x | 0.00x | 0.00x | — |
| sdpa-s32 | sdpa | (1,16,32,128) x (1,8,32,128) x (1,8,32,128) | 18.78 | 18.24 | 95.95 | 581.44 | — | 0.20x | 0.03x | 0.03x | — |
| sdpa-s512 | sdpa | (1,16,512,128) x (1,8,512,128) x (1,8,512,128) | 89.38 | 86.37 | 18452.13 | 143489.70 | — | 0.00x | 0.00x | 0.00x | — |

Media geométrica vs eager: emmy fijado 0.40x · greedy 0.19x · greedy vs torch.compile 0.16x (greedy gana a torch.compile en 2/27). Reproducimos la referencia CloudRift (±5%) en 12/12 casos con referencia.

## Matriz de técnicas (schedule fijado)

✓ = presente en algún kernel del caso · · = no · – = no aplica. Detalle y evidencia en `logs/<caso>.log`.

| caso | Fact | RegT | Vec | TC | Smem | Async | Pipe | WSpec | Bank | Rast | SplitK | Coop | PDL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul/bf16-mma-sm120 | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/bf16-mma-trans-b-masked-n-async | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/bf16-mma-trans-b-masked-n-tma | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/bf16-rowless-grouped-pair | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | · | · | – | ✓ |
| matmul/bf16-scalar-sync-d2-sm120 | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | – | ✓ | · | · | – | ✓ |
| matmul/f16-matvec-reshaped-output-tma | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-batched-pv-b1h4kv128 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-batched-pv-b2h3kv256 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-batched-qk-b1h4d64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-batched-qk-b2h3d32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-epilogue-residual | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-f16acc-cp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-f16acc-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-mma-f16acc-tma | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k128-f16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k128-f32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k128-f32-trans | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-m128n128k132-f32-masked-k | ✓ | ✓ | ✓ | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-mma-m128n128k132-f32-trans-masked-k | ✓ | ✓ | ✓ | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-mma-m128n128k136-f32-masked-k | ✓ | ✓ | ✓ | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-mma-parity-m256-cp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-parity-m256-tma | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-raster-gm8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | ✓ |
| matmul/f16-mma-splitk-atomic | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | ✓ | – | ✓ |
| matmul/f16-mma-splitk-atomic-f16acc-tma-rolled | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | – | ✓ |
| matmul/f16-mma-splitk-deferred | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | ✓ | – | ✓ |
| matmul/f16-mma-trans-b-staged-cp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-mma-trans-b-staged-tma | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-scalar-register-tile | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f16-symbolic-batched-mk-greedy | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-symbolic-computed-a-k-warp | ✓ | ✓ | · | ✓ | ✓ | ✓ | · | · | ✓ | · | · | ✓ | ✓ |
| matmul/f16-symbolic-demoted-n-greedy | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | · | · | – | ✓ |
| matmul/f16-symbolic-k-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-symbolic-k-trans | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-symbolic-m-residual-cp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-symbolic-m-tma | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-symbolic-mn-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/f16-warp-band-p1-tma-d1-m128 | ✓ | ✓ | · | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | · | – | ✓ |
| matmul/f16-warp-band-p1-tma-d2-m100 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | – | ✓ |
| matmul/f16-warp-band-p1-tma-d2-m128 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | – | ✓ |
| matmul/f16-warp-band-p2-tma-d2-m128 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | – | ✓ |
| matmul/f16-warp-gmem-direct | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f16-warp-splitk | – | – | · | · | · | – | – | – | – | – | ✓ | – | ✓ |
| matmul/f16-warp-symbolic-m-cp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/f32-epilogue-bias | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-relu | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-residual | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-scale | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-mixed-dtype-slabs | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | ✓ | · | · | – | ✓ |
| matmul/f32-reduce-coop | – | – | · | · | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-reduce-coop-ilp | – | – | · | · | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-reduce-ilp | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-scalar-bm8-outside-hints | ✓ | · | · | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-composite-fm4-fn4 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-composite-fn4 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-masked-m-fm26 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-masked-n-fn26 | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-masked-n-gmem | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-staging-d1-128x128x128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| matmul/f32-scalar-staging-d1-64x64x64 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | ✓ | · | · | – | ✓ |
| matmul/f32-scalar-staging-d2-128x128x128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-staging-d2-64x64x64 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | ✓ | · | · | – | ✓ |
| matmul/f32-scalar-staging-gmem-128x128x128 | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-staging-gmem-64x64x64 | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-tma-staged | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-tile-per-cell | – | – | · | · | · | – | – | – | – | – | – | – | ✓ |
| matmul/f32-tile-reg-2d | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-reg-fn3 | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-reg-inner-fn4 | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-single-cta | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/fp4-tma-w4a4-linear-ring | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| matmul/fp4-tma-w4a4-split-k | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | ✓ | – | ✓ |
| matmul/imap-reshape-a-cp | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | – | ✓ | · | · | – | ✓ |
| matmul/imap-reshape-a-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/imap-reshape-b-gmem | ✓ | ✓ | · | ✓ | · | – | – | · | – | · | · | – | ✓ |
| matmul/imap-slice-a-cp | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | – | ✓ | · | · | – | ✓ |
| matmul/imap-slice-a-gmem | – | – | · | · | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/imap-slice-a-tma | ✓ | · | · | · | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/imap-transpose-a-gmem | – | – | · | · | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/mma-fp4-encode-epilogue | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| matmul/mma-fp4-packed-block-scales | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| matmul/nvfp4-runtime-rows | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| matmul/nvfp4-unit-n | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| matmul/nvfp4-w4a4-packed-slab-loopify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| attention/rmsnorm-qk-sdpa-composed-cut | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| attention/rmsnorm-qk-sdpa-output-axis | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| attention/rmsnorm-qk-sdpa-stat-b-cut | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| attention/rmsnorm-qk-sdpa-stat-cut | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| fused/norm-linear-f16-scalar-reduce | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| fused/norm-linear-f16-warp-masked-m | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | · | · | ✓ | ✓ |
| fused/norm-weight-linear-f16-computed-b | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | · | · | – | ✓ |
| qwen3emb/rmsnorm-knorm-s128 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s32 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s512 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s128 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s32 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s512 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s128 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s32 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s512 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/fused-add-rms-norm | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/fused-add-rms-norm-escaping-residual | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/rms-norm-cut-sweep-work | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| reduce/rms-norm-packed-cells | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| fused/fp4-tma-w4a4-two-channel-p2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| fused/nvfp4-gate-up-requant-full-projection-cut | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| fused/nvfp4-gate-up-requant-place-cut | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| fused/nvfp4-quantize-cut-shared-normalizer | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| fused/nvfp4-w4a16-sibling-cut-async | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| fused/nvfp4-w4a16-sibling-cut-tma-p2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| qwen3emb/gated-mlp-s128 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/gated-mlp-s32 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/gated-mlp-s512 | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/matmul-add-downproj-s128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-downproj-s32 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-downproj-s512 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s32 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s512 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s32 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s512 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s128 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s32 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s512 | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/sdpa-s128 | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| qwen3emb/sdpa-s32 | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |
| qwen3emb/sdpa-s512 | – | – | ✓ | · | · | – | – | – | – | – | – | – | ✓ |

Leyenda: **Fact** Factorización de bucles · **RegT** Register tiling · **Vec** Cargas vectorizadas e intercalado · **TC** Tensor Cores (mma.sync + ldmatrix) · **Smem** Almacenamiento por fases en smem · **Async** Transferencias asíncronas (cp.async / TMA) · **Pipe** Software pipelining / doble búfer · **WSpec** Especialización de warps · **Bank** Padding y swizzling · **Rast** CTA swizzle (rasterizado agrupado) · **SplitK** Split-K · **Coop** Reducción cooperativa (shuffle / árbol) · **PDL** Programmatic Dependent Launch

