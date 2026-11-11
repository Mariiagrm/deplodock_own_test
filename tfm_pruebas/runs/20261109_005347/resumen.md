# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T09:25:18 · deplodock bf784a76 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 130 (128 medidos, 2 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-sm120 | matmul | (128,256) x (256,128) | 2.03 | 2.11 | 1.67 | 1.24 | — | 1.21x | 1.63x | 1.70x | — |
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.68 | 1.71 | 3.76 | 1.05 | — | 0.45x | 1.60x | 1.63x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.57 | 1.59 | 4.09 | 0.94 | — | 0.38x | 1.67x | 1.69x | — |
| bf16-rowless-grouped-pair | matmul+elementwise | (256) x (4,256,16) x (4,256,16) | 4.85 | 2.46 | 1.62 | 1.13 | — | 2.98x | 4.28x | 2.17x | — |
| bf16-scalar-sync-d2-sm120 | linear | (32,384) x (64,384) | 2.72 | 1.90 | 4.36 | 1.32 | — | 0.62x | 2.05x | 1.43x | — |
| f16-matvec-reshaped-output-tma | linear+reshape+index_map | (1,1,1024) x (2048,1024) | 4.65 | 1.97 | 6.21 | 2.25 | — | 0.75x | 2.07x | 0.88x | — |
| f16-mma-batched-pv-b1h4kv128 | matmul | (1,4,128,128) x (1,4,128,128) | 4.58 | 4.42 | 5.40 | 1.16 | 5.80 | 0.85x | 3.95x | 3.81x | 0.93 |
| f16-mma-batched-pv-b2h3kv256 | matmul | (2,3,128,256) x (2,3,256,128) | 6.46 | 6.13 | 8.59 | 1.66 | 6.02 | 0.75x | 3.88x | 3.69x | **1.43** |
| f16-mma-batched-qk-b1h4d64 | matmul+transpose | (1,4,128,64) x (1,4,128,64) | 3.04 | 3.03 | 3.84 | 0.96 | 5.92 | 0.79x | 3.15x | 3.14x | 0.65 |
| f16-mma-batched-qk-b2h3d32 | matmul+transpose | (2,3,128,32) x (2,3,128,32) | 2.93 | 2.64 | 2.83 | 0.98 | 2.87 | 1.04x | 2.99x | 2.70x | 0.98 |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.29 | 2.14 | 6.50 | 1.04 | 11.33 | 0.81x | 5.09x | 2.06x | 0.57 |
| f16-mma-f16acc-cp | matmul | (128,128) x (128,128) | 1.43 | 1.55 | 3.56 | 1.03 | 4.91 | 0.40x | 1.39x | 1.51x | 0.72 |
| f16-mma-f16acc-gmem | matmul | (128,128) x (128,128) | 1.42 | 1.56 | 7.28 | 1.03 | 8.68 | 0.20x | 1.38x | 1.52x | 0.84 |
| f16-mma-f16acc-symbolic-k | matmul | (128,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-mma-f16acc-tma | matmul | (128,128) x (128,128) | 1.43 | 1.55 | 3.75 | 1.03 | 4.54 | 0.38x | 1.38x | 1.50x | 0.83 |
| f16-mma-m128n128k128-f16 | matmul | (128,128) x (128,128) | 1.53 | 1.67 | 4.57 | 1.15 | 6.21 | 0.34x | 1.34x | 1.46x | 0.73 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.78 | 3.44 | 4.66 | 1.13 | 6.07 | 1.03x | 4.22x | 3.04x | 0.77 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.38 | 1.76 | 4.59 | 1.10 | 4.79 | 0.52x | 2.16x | 1.60x | 0.96 |
| f16-mma-m128n128k132-f32-masked-k | matmul | (128,132) x (132,128) | 4.69 | 3.71 | 6.35 | 1.41 | 7.23 | 0.74x | 3.31x | 2.62x | 0.88 |
| f16-mma-m128n128k132-f32-trans-masked-k | linear | (128,132) x (128,132) | 2.90 | 1.79 | 6.25 | 1.38 | 6.54 | 0.46x | 2.11x | 1.30x | 0.95 |
| f16-mma-m128n128k136-f32-masked-k | matmul | (128,136) x (136,128) | 4.88 | 3.82 | 6.12 | 1.52 | 7.09 | 0.80x | 3.21x | 2.52x | 0.86 |
| f16-mma-parity-m256-cp | matmul | (256,512) x (512,1024) | 4.47 | 3.51 | 3.76 | 3.24 | 5.58 | 1.19x | 1.38x | 1.09x | 0.67 |
| f16-mma-parity-m256-tma | matmul | (256,512) x (512,1024) | 4.65 | 3.62 | 4.00 | 3.37 | 4.51 | 1.16x | 1.38x | 1.07x | 0.89 |
| f16-mma-raster-gm8 | matmul | (1280,256) x (256,512) | 3.86 | 3.66 | 3.28 | 3.41 | 4.98 | 1.18x | 1.13x | 1.08x | 0.66 |
| f16-mma-splitk-atomic | matmul | (128,512) x (512,128) | 4.93 | 4.98 | 8.63 | 1.58 | — | 0.57x | 3.11x | 3.14x | — |
| f16-mma-splitk-atomic-f16acc-tma-rolled | matmul | (32,1024) x (1024,256) | 5.64 | 5.40 | 4.85 | 2.32 | — | 1.16x | 2.43x | 2.33x | — |
| f16-mma-splitk-deferred | matmul | (128,512) x (512,128) | 2.75 | 3.25 | 26.25 | 1.71 | — | 0.10x | 1.61x | 1.90x | — |
| f16-mma-trans-b-staged-cp | linear | (128,128) x (128,128) | 1.50 | 1.69 | 4.06 | 1.09 | 6.07 | 0.37x | 1.38x | 1.56x | 0.67 |
| f16-mma-trans-b-staged-tma | linear | (128,128) x (128,128) | 1.49 | 1.70 | 4.54 | 1.09 | 5.79 | 0.33x | 1.37x | 1.55x | 0.78 |
| f16-scalar-register-tile | matmul | (512,512) x (512,512) | 4.62 | 3.48 | 107.31 | 3.44 | 97.24 | 0.04x | 1.34x | 1.01x | **1.10** |
| f16-symbolic-batched-mk-greedy | matmul | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 12.19 | 8.00 | 34.66 | 354.82 | 38.31 | 0.35x | 0.03x | 0.02x | 0.90 |
| f16-symbolic-computed-a-k-warp | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 21.31 | 13.98 | 31.53 | 4626.66 | 33.38 | 0.68x | 0.00x | 0.00x | 0.94 |
| f16-symbolic-demoted-n-greedy | linear+elementwise | ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) | 3.68 | 3.34 | 19.33 | 20.42 | 22.59 | 0.19x | 0.18x | 0.16x | 0.86 |
| f16-symbolic-demoted-pv-greedy | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-symbolic-k-gmem | matmul | (64,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | 2.64 | 3.00 | 10.57 | 3.34 | 12.74 | 0.25x | 0.79x | 0.90x | 0.83 |
| f16-symbolic-k-trans | linear | (64,{'sym': 'seq_len', 'hint': 512}) x (32,{'sym': 'seq_len', 'hint': 512}) | 2.72 | 2.05 | 16.12 | 7.85 | 13.86 | 0.17x | 0.35x | 0.26x | **1.16** |
| f16-symbolic-m-residual-cp | matmul+elementwise | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) x ({'sym': 'seq_len', 'hint': 512},1024) | 5.91 | 5.65 | 6.50 | 5.62 | 7.62 | 0.91x | 1.05x | 1.01x | 0.85 |
| f16-symbolic-m-tma | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.71 | 4.69 | 5.30 | 4.88 | 6.28 | 0.89x | 0.97x | 0.96x | 0.84 |
| f16-symbolic-mn-gmem | matmul | ({'sym': 'seq_len', 'hint': 512},128) x (128,{'sym': 'seq_len', 'hint': 512}) | 2.12 | 1.80 | 4.21 | 4.18 | 5.16 | 0.50x | 0.51x | 0.43x | 0.82 |
| f16-warp-band-p1-tma-d1-m128 | matmul | (128,256) x (256,128) | 2.01 | 2.33 | 3.90 | 1.39 | 4.47 | 0.52x | 1.44x | 1.67x | 0.87 |
| f16-warp-band-p1-tma-d2-m100 | matmul | (100,256) x (256,128) | 2.00 | 1.89 | 2.60 | 1.38 | 2.91 | 0.77x | 1.45x | 1.37x | 0.89 |
| f16-warp-band-p1-tma-d2-m128 | matmul | (128,256) x (256,128) | 1.90 | 2.19 | 2.55 | 1.25 | 2.95 | 0.74x | 1.52x | 1.75x | 0.87 |
| f16-warp-band-p2-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.01 | 2.31 | 2.55 | 1.35 | 2.93 | 0.79x | 1.49x | 1.71x | 0.87 |
| f16-warp-gmem-direct | matmul | (128,128) x (128,128) | 1.53 | 1.67 | 2.20 | 1.15 | 3.42 | 0.69x | 1.33x | 1.45x | 0.64 |
| f16-warp-splitk | matmul | (64,4096) x (4096,512) | 5.83 | 5.83 | 53.14 | 8.00 | — | 0.11x | 0.73x | 0.73x | — |
| f16-warp-symbolic-m-cp | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.67 | 4.65 | 5.23 | 4.87 | 6.20 | 0.89x | 0.96x | 0.96x | 0.84 |
| f32-epilogue-bias | matmul+elementwise | (1,64,64) x (64,64) x (64) | 3.96 | 2.61 | 1.59 | 1.36 | 2.10 | 2.49x | 2.90x | 1.92x | 0.76 |
| f32-epilogue-relu | matmul+elementwise | (1,64,64) x (64,64) | 3.74 | 1.94 | 1.59 | 1.01 | 1.90 | 2.36x | 3.70x | 1.92x | 0.83 |
| f32-epilogue-residual | matmul+elementwise | (1,64,64) x (64,64) x (1,64,64) | 4.10 | 2.84 | 1.61 | 1.31 | 2.20 | 2.55x | 3.13x | 2.16x | 0.73 |
| f32-epilogue-scale | matmul+elementwise | (1,64,64) x (64,64) x (1) | 3.91 | 2.05 | 1.61 | 1.15 | 2.09 | 2.42x | 3.38x | 1.78x | 0.77 |
| f32-mixed-dtype-slabs | matmul | (64,128) x (128,96) | 3.58 | 3.24 | 2.06 | 1.10 | — | 1.74x | 3.25x | 2.94x | — |
| f32-reduce-coop | matmul | (1,64,64) x (64,64) | 3.15 | 2.03 | 3.35 | 1.30 | 4.58 | 0.94x | 2.42x | 1.56x | 0.73 |
| f32-reduce-coop-ilp | matmul | (1,64,64) x (64,64) | 3.16 | 2.04 | 3.21 | 1.31 | 3.66 | 0.98x | 2.42x | 1.56x | 0.88 |
| f32-reduce-ilp | matmul | (1,64,64) x (64,64) | 2.91 | 1.92 | 1.68 | 1.20 | 4.07 | 1.74x | 2.43x | 1.61x | 0.41 |
| f32-scalar-bm8-outside-hints | matmul | (2048,2048) x (2048,2048) | 186.47 | 177.67 | 1373.70 | 240.55 | 1596.48 | 0.14x | 0.78x | 0.74x | 0.86 |
| f32-scalar-composite-fm4-fn4 | matmul | (2048,2048) x (2048,2048) | 184.66 | 176.78 | 352.14 | 251.56 | 530.50 | 0.52x | 0.73x | 0.70x | 0.66 |
| f32-scalar-composite-fn4 | matmul | (2048,2048) x (2048,2048) | 185.89 | 176.78 | 782.05 | 240.55 | 1271.23 | 0.24x | 0.77x | 0.73x | 0.62 |
| f32-scalar-masked-m-fm26 | matmul | (2048,2048) x (2048,2048) | 185.06 | 177.30 | 413.68 | 252.33 | 659.81 | 0.45x | 0.73x | 0.70x | 0.63 |
| f32-scalar-masked-n-fn26 | matmul | (512,512) x (512,512) | 7.27 | 5.43 | 300.79 | 8.21 | 75.39 | 0.02x | 0.89x | 0.66x | **3.99** |
| f32-scalar-masked-n-gmem | matmul | (64,64) x (64,47) | 3.65 | 1.58 | 4.16 | 1.43 | 2.84 | 0.88x | 2.55x | 1.10x | **1.46** |
| f32-scalar-staging-d1-128x128x128 | matmul | (128,128) x (128,128) | 2.88 | 3.26 | 6.13 | 1.52 | 3.29 | 0.47x | 1.90x | 2.15x | **1.87** |
| f32-scalar-staging-d1-64x64x64 | matmul | (64,64) x (64,64) | 3.17 | 2.05 | 1.21 | 1.31 | 1.26 | 2.62x | 2.41x | 1.56x | 0.96 |
| f32-scalar-staging-d2-128x128x128 | matmul | (128,128) x (128,128) | 2.99 | 3.37 | 6.26 | 1.63 | 3.06 | 0.48x | 1.83x | 2.06x | **2.05** |
| f32-scalar-staging-d2-64x64x64 | matmul | (64,64) x (64,64) | 2.95 | 1.94 | 1.21 | 1.20 | 1.38 | 2.44x | 2.46x | 1.62x | 0.88 |
| f32-scalar-staging-gmem-128x128x128 | matmul | (128,128) x (128,128) | 2.87 | 3.24 | 13.36 | 1.51 | 3.24 | 0.21x | 1.90x | 2.15x | **4.12** |
| f32-scalar-staging-gmem-64x64x64 | matmul | (64,64) x (64,64) | 3.17 | 2.05 | 2.61 | 1.31 | 2.00 | 1.21x | 2.41x | 1.56x | **1.30** |
| f32-scalar-tma-staged | matmul | (2048,2048) x (2048,2048) | 185.17 | 176.78 | 350.86 | 239.02 | 426.18 | 0.53x | 0.77x | 0.74x | 0.82 |
| f32-tile-per-cell | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 1.64 | 1.32 | 1.91 | 1.93x | 2.39x | 1.55x | 0.86 |
| f32-tile-reg-2d | matmul | (1,64,64) x (64,64) | 3.17 | 2.06 | 2.79 | 1.31 | 3.11 | 1.14x | 2.42x | 1.57x | 0.90 |
| f32-tile-reg-fn3 | matmul | (1,64,64) x (64,64) | 2.95 | 1.94 | 3.46 | 1.21 | 3.95 | 0.85x | 2.45x | 1.61x | 0.88 |
| f32-tile-reg-inner-fn4 | matmul | (1,64,64) x (64,64) | 2.92 | 1.93 | 2.75 | 1.20 | 3.14 | 1.06x | 2.45x | 1.61x | 0.88 |
| f32-tile-single-cta | matmul | (1,64,64) x (64,64) | 2.93 | 1.93 | 2.74 | 1.20 | 2.88 | 1.07x | 2.45x | 1.62x | 0.95 |
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.17 | 5.50 | 4.13 | 2.29 | — | 7.78x | 14.06x | 2.40x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.17 | 5.50 | 2.84 | 2.29 | — | 11.32x | 14.03x | 2.40x | — |
| imap-reshape-a-cp | matmul+reshape | (128,256) x (128,64) | 1.56 | 1.35 | 5.64 | 1.15 | 1.75 | 0.28x | 1.35x | 1.17x | **3.21** |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.45 | 1.23 | 5.32 | 1.04 | — | 0.27x | 1.39x | 1.18x | — |
| imap-reshape-b-gmem | matmul+reshape | (64,128) x (256,64) | 1.66 | 1.68 | 4.96 | 1.15 | — | 0.34x | 1.45x | 1.46x | — |
| imap-slice-a-cp | matmul+slice | (128,256) x (128,64) | 1.68 | 1.34 | 2.50 | 1.17 | 1.85 | 0.67x | 1.44x | 1.15x | **1.36** |
| imap-slice-a-gmem | matmul+slice | (128,256) x (128,64) | 1.55 | 1.22 | 5.26 | 1.04 | 2.06 | 0.30x | 1.50x | 1.17x | **2.56** |
| imap-slice-a-tma | matmul+slice | (128,256) x (128,64) | 1.66 | 1.32 | 1.83 | 1.15 | 2.10 | 0.90x | 1.44x | 1.15x | 0.87 |
| imap-transpose-a-gmem | matmul+transpose | (256,128) x (256,64) | 2.14 | 1.81 | 3.43 | 3.96 | 7.78 | 0.62x | 0.54x | 0.46x | 0.44 |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.63 | 6.15 | 1.67 | 1.37 | — | 24.93x | 30.35x | 4.48x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.18 | 3.02 | 2.10 | 1.44 | — | 43.93x | 64.09x | 2.10x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.35 | 5.38 | 2.24 | 1.24 | — | 14.01x | 25.26x | 4.34x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.75 | 3.67 | 4.41 | 1.25 | — | 6.07x | 21.48x | 2.95x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 183.23 | 38.02 | 3.12 | 1.66 | — | 58.76x | 110.29x | 22.89x | — |

Media geométrica vs eager: emmy fijado 0.80x · greedy 1.88x · greedy vs torch.compile 1.34x (greedy gana a torch.compile en 67/84). Reproducimos la referencia CloudRift (±5%) en 52/64 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-qk-sdpa-composed-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.01 | 4.89 | 3.49 | 1.68 | — | 1.72x | 3.57x | 2.91x | — |
| rmsnorm-qk-sdpa-output-axis | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.05 | 4.93 | 1.89 | 1.60 | — | 3.20x | 3.77x | 3.07x | — |
| rmsnorm-qk-sdpa-stat-b-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.03 | 4.89 | 4.03 | 1.63 | — | 1.50x | 3.69x | 2.99x | — |
| rmsnorm-qk-sdpa-stat-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 6.02 | 4.89 | 3.38 | 1.59 | — | 1.78x | 3.80x | 3.08x | — |
| norm-linear-f16-scalar-reduce | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.96 | 2.65 | 106.84 | 2.20 | 80.04 | 0.03x | 1.34x | 1.20x | **1.33** |
| norm-linear-f16-warp-masked-m | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.74 | 2.41 | 6.34 | 2.10 | 11.58 | 0.43x | 1.31x | 1.15x | 0.55 |
| norm-weight-linear-f16-computed-b | linear+rms_norm | (64,128) x (64,128) x (128) | 2.86 | 1.99 | 78.69 | 10.93 | — | 0.04x | 0.26x | 0.18x | — |
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.72 | 1.22 | 1.23 | 1.10 | — | 1.40x | 1.57x | 1.11x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.65 | 1.05 | 1.22 | 0.92 | — | 1.35x | 1.81x | 1.15x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.17 | 2.31 | 2.61 | 1.48 | — | 1.22x | 2.14x | 1.56x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.32 | 1.34 | 1.45 | — | 1.59x | 1.47x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.88 | 1.05 | 1.06 | 1.17 | — | 1.78x | 1.61x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.53 | 1.98 | 1.88 | 1.88 | — | 1.35x | 1.35x | 1.05x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.28 | 1.32 | 1.68 | 1.18 | — | 1.36x | 1.92x | 1.11x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.67 | 1.16 | 1.15 | 1.01 | — | 1.45x | 1.65x | 1.15x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.24 | 4.29 | 4.54 | 2.46 | — | 1.15x | 2.13x | 1.75x | — |
| fused-add-rms-norm | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.92 | 0.95 | 0.91 | — | 2.45x | 2.57x | 1.01x | — |
| fused-add-rms-norm-escaping-residual | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.97 | 0.94 | 0.93 | — | 2.48x | 2.52x | 1.05x | — |
| rms-norm-cut-sweep-work | rms_norm | (512,4096) | 5.12 | 4.59 | 358.24 | 4.58 | — | 0.01x | 1.12x | 1.00x | — |
| rms-norm-packed-cells | rms_norm | (2048,128) | 2.35 | 1.56 | 1.25 | 1.41 | — | 1.88x | 1.66x | 1.10x | — |

Media geométrica vs eager: emmy fijado 0.82x · greedy 1.81x · greedy vs torch.compile 1.26x (greedy gana a torch.compile en 17/20). Reproducimos la referencia CloudRift (±5%) en 1/2 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.17 | 5.50 | 4.13 | 2.29 | — | 7.78x | 14.06x | 2.40x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.17 | 5.50 | 2.84 | 2.29 | — | 11.32x | 14.03x | 2.40x | — |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.63 | 6.15 | 1.67 | 1.37 | — | 24.93x | 30.35x | 4.48x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.18 | 3.02 | 2.10 | 1.44 | — | 43.93x | 64.09x | 2.10x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.35 | 5.38 | 2.24 | 1.24 | — | 14.01x | 25.26x | 4.34x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.75 | 3.67 | 4.41 | 1.25 | — | 6.07x | 21.48x | 2.95x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 183.23 | 38.02 | 3.12 | 1.66 | — | 58.76x | 110.29x | 22.89x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 64.47 | 10.70 | 5.16 | 2.72 | — | 12.49x | 23.73x | 3.94x | — |
| nvfp4-gate-up-requant-full-projection-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 172.05 | 11.38 | 500.21 | 61.32 | — | 0.34x | 2.81x | 0.19x | — |
| nvfp4-gate-up-requant-place-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 172.03 | 11.44 | 654.99 | 61.38 | — | 0.26x | 2.80x | 0.19x | — |
| nvfp4-quantize-cut-shared-normalizer | elementwise+index_map+mean+reduce+reshape | (32,128) | 128.44 | 2.96 | 38.48 | 36.98 | — | 3.34x | 3.47x | 0.08x | — |
| nvfp4-w4a16-sibling-cut-async | elementwise+index_map+linear+reshape+gather | (48,256) | 39.00 | 6.24 | 12.85 | 7.13 | — | 3.03x | 5.47x | 0.88x | — |
| nvfp4-w4a16-sibling-cut-tma-p2 | elementwise+index_map+linear+reshape+gather | (48,256) | 39.05 | 6.26 | 13.57 | 7.18 | — | 2.88x | 5.44x | 0.87x | — |

Media geométrica vs eager: emmy fijado 6.09x · greedy 13.39x · greedy vs torch.compile 1.46x (greedy gana a torch.compile en 8/13). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.72 | 1.22 | 1.23 | 1.10 | — | 1.40x | 1.57x | 1.11x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.65 | 1.05 | 1.22 | 0.92 | — | 1.35x | 1.81x | 1.15x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.17 | 2.31 | 2.61 | 1.48 | — | 1.22x | 2.14x | 1.56x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.13 | 1.32 | 1.34 | 1.45 | — | 1.59x | 1.47x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.88 | 1.05 | 1.06 | 1.17 | — | 1.78x | 1.61x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.53 | 1.98 | 1.88 | 1.88 | — | 1.35x | 1.35x | 1.05x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.28 | 1.32 | 1.68 | 1.18 | — | 1.36x | 1.92x | 1.11x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.67 | 1.16 | 1.15 | 1.01 | — | 1.45x | 1.65x | 1.15x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.24 | 4.29 | 4.54 | 2.46 | — | 1.15x | 2.13x | 1.75x | — |
| gated-mlp-s128 | matmul+elementwise | (1,128,1024) x (1024,3072) x (1024,3072) | 29.68 | 28.06 | 441.14 | 40.86 | — | 0.07x | 0.73x | 0.69x | — |
| gated-mlp-s32 | matmul+elementwise | (1,32,1024) x (1024,3072) x (1024,3072) | 27.47 | 20.75 | 117.53 | 113.28 | — | 0.23x | 0.24x | 0.18x | — |
| gated-mlp-s512 | matmul+elementwise | (1,512,1024) x (1024,3072) x (1024,3072) | 103.07 | 95.56 | 1750.34 | 122.58 | — | 0.06x | 0.84x | 0.78x | — |
| matmul-add-downproj-s128 | matmul+elementwise | (1,128,3072) x (3072,1024) x (1,128,1024) | 13.28 | 13.37 | 37.85 | 19.86 | 44.56 | 0.35x | 0.67x | 0.67x | 0.85 |
| matmul-add-downproj-s32 | matmul+elementwise | (1,32,3072) x (3072,1024) x (1,32,1024) | 11.49 | 11.33 | 33.79 | 14.97 | 39.57 | 0.34x | 0.77x | 0.76x | 0.85 |
| matmul-add-downproj-s512 | matmul+elementwise | (1,512,3072) x (3072,1024) x (1,512,1024) | 39.38 | 38.51 | 133.15 | 69.17 | 153.72 | 0.30x | 0.57x | 0.56x | 0.87 |
| matmul-add-oproj-s128 | matmul+elementwise | (1,128,2048) x (2048,1024) x (1,128,1024) | 11.01 | 11.11 | 24.93 | 14.04 | 30.06 | 0.44x | 0.78x | 0.79x | 0.83 |
| matmul-add-oproj-s32 | matmul+elementwise | (1,32,2048) x (2048,1024) x (1,32,1024) | 9.27 | 9.09 | 22.81 | 10.34 | 26.94 | 0.41x | 0.90x | 0.88x | 0.85 |
| matmul-add-oproj-s512 | matmul+elementwise | (1,512,2048) x (2048,1024) x (1,512,1024) | 26.96 | 26.65 | 88.42 | 46.60 | 102.91 | 0.30x | 0.58x | 0.57x | 0.86 |
| matmul-kvproj-s128 | matmul | (1,128,1024) x (1024,1024) | 13.07 | 10.05 | 13.81 | 11.74 | 15.69 | 0.95x | 1.11x | 0.86x | 0.88 |
| matmul-kvproj-s32 | matmul | (1,32,1024) x (1024,1024) | 6.57 | 6.55 | 11.94 | 5.99 | 13.77 | 0.55x | 1.10x | 1.09x | 0.87 |
| matmul-kvproj-s512 | matmul | (1,512,1024) x (1024,1024) | 13.98 | 13.63 | 45.69 | 23.54 | 52.62 | 0.31x | 0.59x | 0.58x | 0.87 |
| matmul-qproj-s128 | matmul | (1,128,1024) x (1024,2048) | 13.00 | 10.56 | 27.75 | 14.47 | 31.25 | 0.47x | 0.90x | 0.73x | 0.89 |
| matmul-qproj-s32 | matmul | (1,32,1024) x (1024,2048) | 8.47 | 8.45 | 12.27 | 8.10 | 14.20 | 0.69x | 1.05x | 1.04x | 0.86 |
| matmul-qproj-s512 | matmul | (1,512,1024) x (1024,2048) | 25.78 | 25.65 | 56.60 | 39.31 | 62.24 | 0.46x | 0.66x | 0.65x | 0.91 |
| sdpa-s128 | sdpa | (1,16,128,128) x (1,8,128,128) x (1,8,128,128) | 26.21 | 25.12 | 1453.60 | 1408.67 | — | 0.02x | 0.02x | 0.02x | — |
| sdpa-s32 | sdpa | (1,16,32,128) x (1,8,32,128) x (1,8,32,128) | 18.65 | 18.56 | 96.59 | 75.75 | — | 0.19x | 0.25x | 0.25x | — |
| sdpa-s512 | sdpa | (1,16,512,128) x (1,8,512,128) x (1,8,512,128) | 90.50 | 88.42 | 18520.61 | 6266.53 | — | 0.00x | 0.01x | 0.01x | — |

Media geométrica vs eager: emmy fijado 0.40x · greedy 0.70x · greedy vs torch.compile 0.59x (greedy gana a torch.compile en 9/27). Reproducimos la referencia CloudRift (±5%) en 12/12 casos con referencia.

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
| matmul/f32-epilogue-bias | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-relu | ✓ | ✓ | · | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-residual | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-epilogue-scale | ✓ | ✓ | · | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-mixed-dtype-slabs | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | – | ✓ | · | · | – | ✓ |
| matmul/f32-reduce-coop | – | – | · | – | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-reduce-coop-ilp | – | – | · | – | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-reduce-ilp | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| matmul/f32-scalar-bm8-outside-hints | ✓ | · | · | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-composite-fm4-fn4 | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-composite-fn4 | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-masked-m-fm26 | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-masked-n-fn26 | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-masked-n-gmem | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-staging-d1-128x128x128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| matmul/f32-scalar-staging-d1-64x64x64 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | ✓ | · | · | – | ✓ |
| matmul/f32-scalar-staging-d2-128x128x128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-scalar-staging-d2-64x64x64 | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | ✓ | · | · | – | ✓ |
| matmul/f32-scalar-staging-gmem-128x128x128 | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-staging-gmem-64x64x64 | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-scalar-tma-staged | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
| matmul/f32-tile-per-cell | – | – | · | – | · | – | – | – | – | – | – | – | ✓ |
| matmul/f32-tile-reg-2d | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-reg-fn3 | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-reg-inner-fn4 | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| matmul/f32-tile-single-cta | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | – | · | · | · | – | ✓ |
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
| qwen3emb/rmsnorm-knorm-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-knorm-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-layer-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/rmsnorm-qnorm-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/fused-add-rms-norm | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/fused-add-rms-norm-escaping-residual | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| reduce/rms-norm-cut-sweep-work | ✓ | ✓ | ✓ | – | · | – | – | – | – | · | · | – | ✓ |
| reduce/rms-norm-packed-cells | – | – | ✓ | · | · | – | – | – | – | – | – | ✓ | ✓ |
| fused/fp4-tma-w4a4-two-channel-p2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| fused/nvfp4-gate-up-requant-full-projection-cut | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · | – | ✓ |
| fused/nvfp4-gate-up-requant-place-cut | ✓ | ✓ | ✓ | · | · | – | – | – | – | · | · | – | ✓ |
| fused/nvfp4-quantize-cut-shared-normalizer | – | – | ✓ | – | · | – | – | – | – | – | – | – | ✓ |
| fused/nvfp4-w4a16-sibling-cut-async | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| fused/nvfp4-w4a16-sibling-cut-tma-p2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | – | ✓ |
| qwen3emb/gated-mlp-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/gated-mlp-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/gated-mlp-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | ✓ | ✓ |
| qwen3emb/matmul-add-downproj-s128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-downproj-s32 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-downproj-s512 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s32 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-add-oproj-s512 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s32 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-kvproj-s512 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s128 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s32 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/matmul-qproj-s512 | ✓ | ✓ | ✓ | – | ✓ | ✓ | · | – | · | · | · | – | ✓ |
| qwen3emb/sdpa-s128 | – | – | ✓ | – | · | – | – | – | – | – | – | – | ✓ |
| qwen3emb/sdpa-s32 | – | – | ✓ | – | · | – | – | – | – | – | – | – | ✓ |
| qwen3emb/sdpa-s512 | – | – | ✓ | – | · | – | – | – | – | – | – | – | ✓ |

Leyenda: **Fact** Factorización de bucles · **RegT** Register tiling · **Vec** Cargas vectorizadas e intercalado · **TC** Tensor Cores (mma.sync + ldmatrix) · **Smem** Almacenamiento por fases en smem · **Async** Transferencias asíncronas (cp.async / TMA) · **Pipe** Software pipelining / doble búfer · **WSpec** Especialización de warps · **Bank** Padding y swizzling · **Rast** CTA swizzle (rasterizado agrupado) · **SplitK** Split-K · **Coop** Reducción cooperativa (shuffle / árbol) · **PDL** Programmatic Dependent Launch

