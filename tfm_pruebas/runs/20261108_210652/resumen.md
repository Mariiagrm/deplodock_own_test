# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-08T20:17:08 · deplodock 6764d63a 2026-11-08 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 130 (128 medidos, 2 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-sm120 | matmul | (128,256) x (256,128) | 2.16 | 1.88 | 1.65 | 1.66 | — | 1.31x | 1.30x | 1.13x | — |
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.56 | 2.22 | 3.79 | 0.97 | — | 0.41x | 1.61x | 2.29x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.57 | 1.54 | 4.15 | 0.96 | — | 0.38x | 1.63x | 1.60x | — |
| bf16-rowless-grouped-pair | matmul+elementwise | (256) x (4,256,16) x (4,256,16) | 4.85 | 2.55 | 1.66 | 1.47 | — | 2.92x | 3.29x | 1.73x | — |
| bf16-scalar-sync-d2-sm120 | linear | (32,384) x (64,384) | 2.73 | 1.83 | 4.38 | 1.69 | — | 0.62x | 1.62x | 1.08x | — |
| f16-matvec-reshaped-output-tma | linear+reshape+index_map | (1,1,1024) x (2048,1024) | 4.37 | 2.24 | 6.19 | 2.14 | — | 0.71x | 2.04x | 1.05x | — |
| f16-mma-batched-pv-b1h4kv128 | matmul | (1,4,128,128) x (1,4,128,128) | 4.56 | 4.79 | 5.39 | 1.16 | 5.80 | 0.85x | 3.92x | 4.12x | 0.93 |
| f16-mma-batched-pv-b2h3kv256 | matmul | (2,3,128,256) x (2,3,256,128) | 6.46 | 6.46 | 8.58 | 1.93 | 6.02 | 0.75x | 3.35x | 3.35x | **1.42** |
| f16-mma-batched-qk-b1h4d64 | matmul+transpose | (1,4,128,64) x (1,4,128,64) | 3.03 | 2.99 | 3.83 | 1.53 | 5.92 | 0.79x | 1.98x | 1.95x | 0.65 |
| f16-mma-batched-qk-b2h3d32 | matmul+transpose | (2,3,128,32) x (2,3,128,32) | 2.92 | 2.89 | 2.82 | 1.03 | 2.87 | 1.04x | 2.84x | 2.81x | 0.98 |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.73 | 2.25 | 6.50 | 1.28 | 11.33 | 0.88x | 4.50x | 1.76x | 0.57 |
| f16-mma-f16acc-cp | matmul | (128,128) x (128,128) | 1.42 | 1.57 | 3.57 | 1.09 | 4.91 | 0.40x | 1.30x | 1.45x | 0.73 |
| f16-mma-f16acc-gmem | matmul | (128,128) x (128,128) | 1.54 | 1.54 | 7.20 | 1.19 | 8.68 | 0.21x | 1.30x | 1.29x | 0.83 |
| f16-mma-f16acc-symbolic-k | matmul | (128,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-mma-f16acc-tma | matmul | (128,128) x (128,128) | 1.42 | 1.56 | 3.75 | 1.09 | 4.54 | 0.38x | 1.31x | 1.43x | 0.83 |
| f16-mma-m128n128k128-f16 | matmul | (128,128) x (128,128) | 1.42 | 1.56 | 4.57 | 1.04 | 6.21 | 0.31x | 1.37x | 1.50x | 0.74 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.43 | 2.18 | 4.68 | 1.15 | 6.07 | 0.95x | 3.85x | 1.89x | 0.77 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.39 | 1.34 | 4.71 | 1.10 | 4.79 | 0.51x | 2.18x | 1.22x | 0.98 |
| f16-mma-m128n128k132-f32-masked-k | matmul | (128,132) x (132,128) | 5.07 | 3.83 | 6.36 | 9.77 | 7.23 | 0.80x | 0.52x | 0.39x | 0.88 |
| f16-mma-m128n128k132-f32-trans-masked-k | linear | (128,132) x (128,132) | 2.67 | 1.83 | 6.29 | 8.10 | 6.54 | 0.42x | 0.33x | 0.23x | 0.96 |
| f16-mma-m128n128k136-f32-masked-k | matmul | (128,136) x (136,128) | 4.94 | 4.03 | 6.11 | 9.94 | 7.09 | 0.81x | 0.50x | 0.41x | 0.86 |
| f16-mma-parity-m256-cp | matmul | (256,512) x (512,1024) | 4.51 | 3.52 | 3.72 | 3.62 | 5.58 | 1.21x | 1.25x | 0.97x | 0.67 |
| f16-mma-parity-m256-tma | matmul | (256,512) x (512,1024) | 4.48 | 3.51 | 3.87 | 3.62 | 4.51 | 1.16x | 1.24x | 0.97x | 0.86 |
| f16-mma-raster-gm8 | matmul | (1280,256) x (256,512) | 3.97 | 3.50 | 3.25 | 5.35 | 4.98 | 1.22x | 0.74x | 0.65x | 0.65 |
| f16-mma-splitk-atomic | matmul | (128,512) x (512,128) | 4.92 | 4.69 | 8.64 | 1.74 | — | 0.57x | 2.82x | 2.69x | — |
| f16-mma-splitk-atomic-f16acc-tma-rolled | matmul | (32,1024) x (1024,256) | 5.62 | 5.50 | 4.82 | 2.32 | — | 1.16x | 2.42x | 2.37x | — |
| f16-mma-splitk-deferred | matmul | (128,512) x (512,128) | 2.76 | 3.21 | 26.34 | 2.10 | — | 0.10x | 1.31x | 1.52x | — |
| f16-mma-trans-b-staged-cp | linear | (128,128) x (128,128) | 1.37 | 1.19 | 4.06 | 0.98 | 6.07 | 0.34x | 1.40x | 1.21x | 0.67 |
| f16-mma-trans-b-staged-tma | linear | (128,128) x (128,128) | 1.48 | 1.29 | 4.52 | 1.08 | 5.79 | 0.33x | 1.36x | 1.19x | 0.78 |
| f16-scalar-register-tile | matmul | (512,512) x (512,512) | 4.52 | 3.39 | 107.40 | 8.04 | 97.24 | 0.04x | 0.56x | 0.42x | **1.10** |
| f16-symbolic-batched-mk-greedy | matmul | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 12.30 | 8.10 | 34.63 | 353.40 | 38.31 | 0.36x | 0.03x | 0.02x | 0.90 |
| f16-symbolic-computed-a-k-warp | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 21.15 | 13.98 | 31.29 | 4563.39 | 33.38 | 0.68x | 0.00x | 0.00x | 0.94 |
| f16-symbolic-demoted-n-greedy | linear+elementwise | ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) | 4.24 | 3.51 | 20.46 | 118.05 | 22.59 | 0.21x | 0.04x | 0.03x | 0.91 |
| f16-symbolic-demoted-pv-greedy | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-symbolic-k-gmem | matmul | (64,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | 2.63 | 2.98 | 10.58 | 3.30 | 12.74 | 0.25x | 0.80x | 0.90x | 0.83 |
| f16-symbolic-k-trans | linear | (64,{'sym': 'seq_len', 'hint': 512}) x (32,{'sym': 'seq_len', 'hint': 512}) | 2.80 | 2.78 | 16.07 | 46.88 | 13.86 | 0.17x | 0.06x | 0.06x | **1.16** |
| f16-symbolic-m-residual-cp | matmul+elementwise | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) x ({'sym': 'seq_len', 'hint': 512},1024) | 5.71 | 5.52 | 6.31 | 7.01 | 7.62 | 0.91x | 0.82x | 0.79x | 0.83 |
| f16-symbolic-m-tma | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.60 | 4.57 | 5.20 | 5.44 | 6.28 | 0.88x | 0.84x | 0.84x | 0.83 |
| f16-symbolic-mn-gmem | matmul | ({'sym': 'seq_len', 'hint': 512},128) x (128,{'sym': 'seq_len', 'hint': 512}) | 2.81 | 2.37 | 4.19 | 209.43 | 5.16 | 0.67x | 0.01x | 0.01x | 0.81 |
| f16-warp-band-p1-tma-d1-m128 | matmul | (128,256) x (256,128) | 1.92 | 1.78 | 3.93 | 1.35 | 4.47 | 0.49x | 1.42x | 1.32x | 0.88 |
| f16-warp-band-p1-tma-d2-m100 | matmul | (100,256) x (256,128) | 1.88 | 2.10 | 2.49 | 1.23 | 2.91 | 0.76x | 1.53x | 1.71x | 0.86 |
| f16-warp-band-p1-tma-d2-m128 | matmul | (128,256) x (256,128) | 1.90 | 1.78 | 2.52 | 1.35 | 2.95 | 0.75x | 1.40x | 1.32x | 0.85 |
| f16-warp-band-p2-tma-d2-m128 | matmul | (128,256) x (256,128) | 1.90 | 1.77 | 2.54 | 1.34 | 2.93 | 0.75x | 1.42x | 1.32x | 0.87 |
| f16-warp-gmem-direct | matmul | (128,128) x (128,128) | 1.41 | 1.56 | 2.08 | 1.03 | 3.42 | 0.68x | 1.37x | 1.51x | 0.61 |
| f16-warp-splitk | matmul | (64,4096) x (4096,512) | 5.83 | 5.80 | 52.88 | 10.87 | — | 0.11x | 0.54x | 0.53x | — |
| f16-warp-symbolic-m-cp | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.66 | 4.66 | 5.10 | 5.52 | 6.20 | 0.91x | 0.84x | 0.84x | 0.82 |
| f32-epilogue-bias | matmul+elementwise | (1,64,64) x (64,64) x (64) | 3.95 | 1.56 | 1.59 | 3.39 | 2.10 | 2.48x | 1.17x | 0.46x | 0.76 |
| f32-epilogue-relu | matmul+elementwise | (1,64,64) x (64,64) | 3.73 | 1.49 | 1.60 | 2.95 | 1.90 | 2.33x | 1.26x | 0.50x | 0.84 |
| f32-epilogue-residual | matmul+elementwise | (1,64,64) x (64,64) x (1,64,64) | 3.76 | 1.58 | 1.61 | 3.01 | 2.20 | 2.33x | 1.25x | 0.53x | 0.73 |
| f32-epilogue-scale | matmul+elementwise | (1,64,64) x (64,64) x (1) | 3.90 | 1.57 | 1.62 | 3.28 | 2.09 | 2.40x | 1.19x | 0.48x | 0.78 |
| f32-mixed-dtype-slabs | matmul | (64,128) x (128,96) | 3.92 | 2.43 | 2.06 | 1.34 | — | 1.91x | 2.94x | 1.82x | — |
| f32-reduce-coop | matmul | (1,64,64) x (64,64) | 2.93 | 1.49 | 3.23 | 2.87 | 4.58 | 0.91x | 1.02x | 0.52x | 0.71 |
| f32-reduce-coop-ilp | matmul | (1,64,64) x (64,64) | 2.93 | 1.49 | 3.20 | 2.89 | 3.66 | 0.92x | 1.02x | 0.52x | 0.87 |
| f32-reduce-ilp | matmul | (1,64,64) x (64,64) | 2.92 | 1.49 | 1.69 | 2.87 | 4.07 | 1.73x | 1.02x | 0.52x | 0.42 |
| f32-scalar-bm8-outside-hints | matmul | (2048,2048) x (2048,2048) | 184.94 | 175.77 | 1366.05 | 237.49 | 1596.48 | 0.14x | 0.78x | 0.74x | 0.86 |
| f32-scalar-composite-fm4-fn4 | matmul | (2048,2048) x (2048,2048) | 184.85 | 176.31 | 347.87 | 237.99 | 530.50 | 0.53x | 0.78x | 0.74x | 0.66 |
| f32-scalar-composite-fn4 | matmul | (2048,2048) x (2048,2048) | 184.21 | 175.64 | 775.90 | 236.97 | 1271.23 | 0.24x | 0.78x | 0.74x | 0.61 |
| f32-scalar-masked-m-fm26 | matmul | (2048,2048) x (2048,2048) | 184.37 | 176.30 | 409.50 | 238.00 | 659.81 | 0.45x | 0.77x | 0.74x | 0.62 |
| f32-scalar-masked-n-fn26 | matmul | (512,512) x (512,512) | 7.26 | 5.40 | 299.52 | 8.64 | 75.39 | 0.02x | 0.84x | 0.63x | **3.97** |
| f32-scalar-masked-n-gmem | matmul | (64,64) x (64,47) | 3.41 | 1.47 | 4.21 | 2.77 | 2.84 | 0.81x | 1.23x | 0.53x | **1.48** |
| f32-scalar-staging-d1-128x128x128 | matmul | (128,128) x (128,128) | 2.90 | 2.90 | 6.13 | 9.75 | 3.29 | 0.47x | 0.30x | 0.30x | **1.86** |
| f32-scalar-staging-d1-64x64x64 | matmul | (64,64) x (64,64) | 2.93 | 1.49 | 1.21 | 2.80 | 1.26 | 2.42x | 1.04x | 0.53x | 0.95 |
| f32-scalar-staging-d2-128x128x128 | matmul | (128,128) x (128,128) | 2.89 | 2.90 | 6.26 | 9.78 | 3.06 | 0.46x | 0.30x | 0.30x | **2.05** |
| f32-scalar-staging-d2-64x64x64 | matmul | (64,64) x (64,64) | 3.16 | 1.59 | 1.31 | 3.06 | 1.38 | 2.41x | 1.03x | 0.52x | 0.95 |
| f32-scalar-staging-gmem-128x128x128 | matmul | (128,128) x (128,128) | 3.02 | 3.00 | 13.33 | 9.81 | 3.24 | 0.23x | 0.31x | 0.31x | **4.11** |
| f32-scalar-staging-gmem-64x64x64 | matmul | (64,64) x (64,64) | 3.14 | 1.60 | 2.59 | 2.91 | 2.00 | 1.22x | 1.08x | 0.55x | **1.29** |
| f32-scalar-tma-staged | matmul | (2048,2048) x (2048,2048) | 184.44 | 175.77 | 347.90 | 236.71 | 426.18 | 0.53x | 0.78x | 0.74x | 0.82 |
| f32-tile-per-cell | matmul | (1,64,64) x (64,64) | 3.15 | 1.60 | 1.64 | 2.92 | 1.91 | 1.92x | 1.08x | 0.55x | 0.86 |
| f32-tile-reg-2d | matmul | (1,64,64) x (64,64) | 3.15 | 1.60 | 2.77 | 2.91 | 3.11 | 1.14x | 1.08x | 0.55x | 0.89 |
| f32-tile-reg-fn3 | matmul | (1,64,64) x (64,64) | 3.14 | 1.60 | 3.42 | 2.99 | 3.95 | 0.92x | 1.05x | 0.53x | 0.87 |
| f32-tile-reg-inner-fn4 | matmul | (1,64,64) x (64,64) | 2.94 | 1.48 | 2.73 | 2.80 | 3.14 | 1.08x | 1.05x | 0.53x | 0.87 |
| f32-tile-single-cta | matmul | (1,64,64) x (64,64) | 3.16 | 1.60 | 2.74 | 2.90 | 2.88 | 1.16x | 1.09x | 0.55x | 0.95 |
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.26 | 5.49 | 4.12 | 17.63 | — | 7.83x | 1.83x | 0.31x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.16 | 5.48 | 2.81 | 16.60 | — | 11.46x | 1.94x | 0.33x | — |
| imap-reshape-a-cp | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.34 | 5.67 | 1.16 | 1.75 | 0.28x | 1.35x | 1.16x | **3.23** |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.46 | 1.22 | 5.31 | 1.04 | — | 0.28x | 1.41x | 1.18x | — |
| imap-reshape-b-gmem | matmul+reshape | (64,128) x (256,64) | 1.55 | 1.54 | 4.89 | 5.70 | — | 0.32x | 0.27x | 0.27x | — |
| imap-slice-a-cp | matmul+slice | (128,256) x (128,64) | 1.55 | 1.51 | 2.50 | 1.04 | 1.85 | 0.62x | 1.49x | 1.45x | **1.35** |
| imap-slice-a-gmem | matmul+slice | (128,256) x (128,64) | 1.67 | 1.64 | 5.34 | 1.16 | 2.06 | 0.31x | 1.44x | 1.41x | **2.59** |
| imap-slice-a-tma | matmul+slice | (128,256) x (128,64) | 1.67 | 1.64 | 1.72 | 1.15 | 2.10 | 0.97x | 1.45x | 1.42x | 0.82 |
| imap-transpose-a-gmem | matmul+transpose | (256,128) x (256,64) | 2.28 | 2.24 | 11.86 | 12.00 | 7.78 | 0.19x | 0.19x | 0.19x | **1.52** |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 37.79 | 4.27 | 1.68 | 1.30 | — | 22.43x | 29.02x | 3.28x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.17 | 2.39 | 2.12 | 1.43 | — | 43.55x | 64.49x | 1.67x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.24 | 5.54 | 2.31 | 2.32 | — | 13.50x | 13.45x | 2.39x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.66 | 3.39 | 4.42 | 1.28 | — | 6.03x | 20.76x | 2.64x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.21 | 6.48 | 3.12 | 21.25 | — | 10.32x | 1.52x | 0.30x | — |

Media geométrica vs eager: emmy fijado 0.78x · greedy 1.06x · greedy vs torch.compile 0.70x (greedy gana a torch.compile en 37/84). Reproducimos la referencia CloudRift (±5%) en 51/64 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-qk-sdpa-composed-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.98 | 4.71 | 3.44 | 3.10 | — | 1.74x | 1.93x | 1.52x | — |
| rmsnorm-qk-sdpa-output-axis | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.01 | 4.71 | 1.98 | 3.16 | — | 3.03x | 1.90x | 1.49x | — |
| rmsnorm-qk-sdpa-stat-b-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.00 | 4.75 | 4.03 | 3.09 | — | 1.49x | 1.94x | 1.54x | — |
| rmsnorm-qk-sdpa-stat-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.01 | 4.73 | 3.36 | 3.06 | — | 1.79x | 1.96x | 1.54x | — |
| norm-linear-f16-scalar-reduce | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.72 | 2.00 | 106.62 | 2.11 | 80.04 | 0.03x | 1.29x | 0.95x | **1.33** |
| norm-linear-f16-warp-masked-m | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.97 | 2.22 | 6.45 | 2.23 | 11.58 | 0.46x | 1.33x | 1.00x | 0.56 |
| norm-weight-linear-f16-computed-b | linear+rms_norm | (64,128) x (64,128) x (128) | 3.27 | 2.27 | 78.41 | 90.82 | — | 0.04x | 0.04x | 0.02x | — |
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.61 | 1.06 | 1.22 | 1.24 | — | 1.32x | 1.30x | 0.85x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.65 | 1.03 | 1.12 | 1.22 | — | 1.47x | 1.35x | 0.84x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.30 | 2.43 | 2.72 | 2.72 | — | 1.21x | 1.21x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.32 | 1.45 | 1.46 | — | 1.47x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.85 | 1.18 | 1.16 | 1.17 | — | 1.60x | 1.59x | 1.02x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.42 | 1.76 | 1.77 | 1.78 | — | 1.37x | 1.36x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.37 | 1.56 | 1.79 | 1.80 | — | 1.33x | 1.32x | 0.87x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.54 | 0.96 | 1.13 | 1.15 | — | 1.37x | 1.34x | 0.84x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 4.29 | 4.63 | 4.63 | — | 1.12x | 1.12x | 0.93x | — |
| fused-add-rms-norm | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.10 | 1.28 | 0.94 | 0.93 | — | 2.23x | 2.25x | 1.37x | — |
| fused-add-rms-norm-escaping-residual | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.98 | 0.95 | 1.07 | — | 2.45x | 2.17x | 0.91x | — |
| rms-norm-cut-sweep-work | rms_norm | (512,4096) | 5.13 | 4.55 | 357.46 | 4.81 | — | 0.01x | 1.07x | 0.95x | — |
| rms-norm-packed-cells | rms_norm | (2048,128) | 2.33 | 1.56 | 1.33 | 1.93 | — | 1.75x | 1.21x | 0.81x | — |

Media geométrica vs eager: emmy fijado 0.81x · greedy 1.24x · greedy vs torch.compile 0.86x (greedy gana a torch.compile en 6/20). Reproducimos la referencia CloudRift (±5%) en 1/2 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.26 | 5.49 | 4.12 | 17.63 | — | 7.83x | 1.83x | 0.31x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.16 | 5.48 | 2.81 | 16.60 | — | 11.46x | 1.94x | 0.33x | — |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 37.79 | 4.27 | 1.68 | 1.30 | — | 22.43x | 29.02x | 3.28x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.17 | 2.39 | 2.12 | 1.43 | — | 43.55x | 64.49x | 1.67x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.24 | 5.54 | 2.31 | 2.32 | — | 13.50x | 13.45x | 2.39x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.66 | 3.39 | 4.42 | 1.28 | — | 6.03x | 20.76x | 2.64x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.21 | 6.48 | 3.12 | 21.25 | — | 10.32x | 1.52x | 0.30x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.52 | 10.61 | 5.13 | 23.37 | — | 12.58x | 2.76x | 0.45x | — |
| nvfp4-gate-up-requant-full-projection-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.97 | 12.76 | 499.41 | 60.88 | — | 0.34x | 2.82x | 0.21x | — |
| nvfp4-gate-up-requant-place-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 172.01 | 12.69 | 649.12 | 60.90 | — | 0.26x | 2.82x | 0.21x | — |
| nvfp4-quantize-cut-shared-normalizer | elementwise+index_map+mean+reduce+reshape | (32,128) | 127.28 | 2.96 | 38.47 | 36.81 | — | 3.31x | 3.46x | 0.08x | — |
| nvfp4-w4a16-sibling-cut-async | elementwise+index_map+linear+reshape+gather | (48,256) | 253.48 | 40.51 | 12.80 | 7.79 | — | 19.80x | 32.54x | 5.20x | — |
| nvfp4-w4a16-sibling-cut-tma-p2 | elementwise+index_map+linear+reshape+gather | (48,256) | 39.08 | 6.59 | 13.57 | 7.10 | — | 2.88x | 5.50x | 0.93x | — |

Media geométrica vs eager: emmy fijado 6.09x · greedy 6.52x · greedy vs torch.compile 0.69x (greedy gana a torch.compile en 5/13). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.61 | 1.06 | 1.22 | 1.24 | — | 1.32x | 1.30x | 0.85x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.65 | 1.03 | 1.12 | 1.22 | — | 1.47x | 1.35x | 0.84x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.30 | 2.43 | 2.72 | 2.72 | — | 1.21x | 1.21x | 0.89x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.32 | 1.45 | 1.46 | — | 1.47x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.85 | 1.18 | 1.16 | 1.17 | — | 1.60x | 1.59x | 1.02x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.42 | 1.76 | 1.77 | 1.78 | — | 1.37x | 1.36x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.37 | 1.56 | 1.79 | 1.80 | — | 1.33x | 1.32x | 0.87x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.54 | 0.96 | 1.13 | 1.15 | — | 1.37x | 1.34x | 0.84x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 4.29 | 4.63 | 4.63 | — | 1.12x | 1.12x | 0.93x | — |
| gated-mlp-s128 | matmul+elementwise | (1,128,1024) x (1024,3072) x (1024,3072) | 30.53 | 29.02 | 439.33 | 7284.38 | — | 0.07x | 0.00x | 0.00x | — |
| gated-mlp-s32 | matmul+elementwise | (1,32,1024) x (1024,3072) x (1024,3072) | 28.61 | 22.18 | 116.99 | 1840.80 | — | 0.24x | 0.02x | 0.01x | — |
| gated-mlp-s512 | matmul+elementwise | (1,512,1024) x (1024,3072) x (1024,3072) | 104.13 | 92.74 | 1744.26 | 28989.09 | — | 0.06x | 0.00x | 0.00x | — |
| matmul-add-downproj-s128 | matmul+elementwise | (1,128,3072) x (3072,1024) x (1,128,1024) | 13.38 | 13.26 | 37.95 | 19.77 | 44.56 | 0.35x | 0.68x | 0.67x | 0.85 |
| matmul-add-downproj-s32 | matmul+elementwise | (1,32,3072) x (3072,1024) x (1,32,1024) | 13.66 | 13.95 | 33.68 | 905.73 | 39.57 | 0.41x | 0.02x | 0.02x | 0.85 |
| matmul-add-downproj-s512 | matmul+elementwise | (1,512,3072) x (3072,1024) x (1,512,1024) | 38.22 | 42.47 | 132.40 | 82.40 | 153.72 | 0.29x | 0.46x | 0.52x | 0.86 |
| matmul-add-oproj-s128 | matmul+elementwise | (1,128,2048) x (2048,1024) x (1,128,1024) | 10.99 | 10.99 | 25.51 | 14.01 | 30.06 | 0.43x | 0.78x | 0.78x | 0.85 |
| matmul-add-oproj-s32 | matmul+elementwise | (1,32,2048) x (2048,1024) x (1,32,1024) | 9.74 | 9.62 | 22.82 | 600.54 | 26.94 | 0.43x | 0.02x | 0.02x | 0.85 |
| matmul-add-oproj-s512 | matmul+elementwise | (1,512,2048) x (2048,1024) x (1,512,1024) | 26.69 | 26.54 | 88.72 | 56.24 | 102.91 | 0.30x | 0.47x | 0.47x | 0.86 |
| matmul-kvproj-s128 | matmul | (1,128,1024) x (1024,1024) | 13.08 | 10.06 | 13.62 | 11.66 | 15.69 | 0.96x | 1.12x | 0.86x | 0.87 |
| matmul-kvproj-s32 | matmul | (1,32,1024) x (1024,1024) | 6.86 | 6.65 | 11.98 | 297.14 | 13.77 | 0.57x | 0.02x | 0.02x | 0.87 |
| matmul-kvproj-s512 | matmul | (1,512,1024) x (1024,1024) | 13.85 | 13.48 | 45.63 | 23.39 | 52.62 | 0.30x | 0.59x | 0.58x | 0.87 |
| matmul-qproj-s128 | matmul | (1,128,1024) x (1024,2048) | 12.98 | 12.95 | 27.68 | 14.92 | 31.25 | 0.47x | 0.87x | 0.87x | 0.89 |
| matmul-qproj-s32 | matmul | (1,32,1024) x (1024,2048) | 8.22 | 6.27 | 12.26 | 7.98 | 14.20 | 0.67x | 1.03x | 0.78x | 0.86 |
| matmul-qproj-s512 | matmul | (1,512,1024) x (1024,2048) | 25.88 | 25.62 | 55.78 | 38.98 | 62.24 | 0.46x | 0.66x | 0.66x | 0.90 |
| sdpa-s128 | sdpa | (1,16,128,128) x (1,8,128,128) x (1,8,128,128) | 25.86 | 24.96 | 1450.91 | 9191.78 | — | 0.02x | 0.00x | 0.00x | — |
| sdpa-s32 | sdpa | (1,16,32,128) x (1,8,32,128) x (1,8,32,128) | 18.78 | 18.26 | 95.95 | 584.51 | — | 0.20x | 0.03x | 0.03x | — |
| sdpa-s512 | sdpa | (1,16,512,128) x (1,8,512,128) x (1,8,512,128) | 90.02 | 86.82 | 18514.18 | 144118.44 | — | 0.00x | 0.00x | 0.00x | — |

Media geométrica vs eager: emmy fijado 0.40x · greedy 0.19x · greedy vs torch.compile 0.16x (greedy gana a torch.compile en 1/27). Reproducimos la referencia CloudRift (±5%) en 12/12 casos con referencia.

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

