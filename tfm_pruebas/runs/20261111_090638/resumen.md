# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T09:25:22 · deplodock bf784a76 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 130 (128 medidos, 2 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-sm120 | matmul | (128,256) x (256,128) | 2.16 | 2.24 | 1.78 | 1.36 | — | 1.21x | 1.58x | 1.64x | — |
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 1.70 | 3.86 | 1.06 | — | 0.43x | 1.58x | 1.61x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.67 | 1.70 | 4.24 | 1.06 | — | 0.39x | 1.57x | 1.60x | — |
| bf16-rowless-grouped-pair | matmul+elementwise | (256) x (4,256,16) x (4,256,16) | 4.85 | 2.41 | 1.66 | 1.07 | — | 2.93x | 4.55x | 2.26x | — |
| bf16-scalar-sync-d2-sm120 | linear | (32,384) x (64,384) | 2.73 | 1.83 | 4.46 | 1.39 | — | 0.61x | 1.96x | 1.32x | — |
| f16-matvec-reshaped-output-tma | linear+reshape+index_map | (1,1,1024) x (2048,1024) | 4.65 | 1.99 | 6.32 | 2.26 | — | 0.74x | 2.06x | 0.88x | — |
| f16-mma-batched-pv-b1h4kv128 | matmul | (1,4,128,128) x (1,4,128,128) | 4.57 | 4.62 | 5.41 | 1.16 | 5.80 | 0.85x | 3.94x | 3.98x | 0.93 |
| f16-mma-batched-pv-b2h3kv256 | matmul | (2,3,128,256) x (2,3,256,128) | 6.48 | 6.15 | 8.61 | 1.69 | 6.02 | 0.75x | 3.84x | 3.65x | **1.43** |
| f16-mma-batched-qk-b1h4d64 | matmul+transpose | (1,4,128,64) x (1,4,128,64) | 3.03 | 3.02 | 3.83 | 0.97 | 5.92 | 0.79x | 3.13x | 3.13x | 0.65 |
| f16-mma-batched-qk-b2h3d32 | matmul+transpose | (2,3,128,32) x (2,3,128,32) | 2.91 | 2.65 | 2.84 | 0.99 | 2.87 | 1.03x | 2.94x | 2.68x | 0.99 |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.70 | 2.25 | 6.59 | 1.15 | 11.33 | 0.86x | 4.93x | 1.95x | 0.58 |
| f16-mma-f16acc-cp | matmul | (128,128) x (128,128) | 1.53 | 1.67 | 3.66 | 1.13 | 4.91 | 0.42x | 1.35x | 1.47x | 0.75 |
| f16-mma-f16acc-gmem | matmul | (128,128) x (128,128) | 1.53 | 1.67 | 7.35 | 1.13 | 8.68 | 0.21x | 1.35x | 1.47x | 0.85 |
| f16-mma-f16acc-symbolic-k | matmul | (128,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-mma-f16acc-tma | matmul | (128,128) x (128,128) | 1.54 | 1.67 | 3.86 | 1.15 | 4.54 | 0.40x | 1.34x | 1.45x | 0.85 |
| f16-mma-m128n128k128-f16 | matmul | (128,128) x (128,128) | 1.54 | 1.67 | 4.73 | 1.16 | 6.21 | 0.33x | 1.33x | 1.44x | 0.76 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.75 | 3.46 | 4.81 | 1.15 | 6.07 | 0.99x | 4.15x | 3.02x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.40 | 1.78 | 4.73 | 1.09 | 4.79 | 0.51x | 2.20x | 1.64x | 0.99 |
| f16-mma-m128n128k132-f32-masked-k | matmul | (128,132) x (132,128) | 5.03 | 3.84 | 6.48 | 1.51 | 7.23 | 0.78x | 3.34x | 2.55x | 0.90 |
| f16-mma-m128n128k132-f32-trans-masked-k | linear | (128,132) x (128,132) | 2.91 | 1.80 | 6.43 | 1.35 | 6.54 | 0.45x | 2.16x | 1.34x | 0.98 |
| f16-mma-m128n128k136-f32-masked-k | matmul | (128,136) x (136,128) | 4.92 | 3.83 | 6.21 | 1.52 | 7.09 | 0.79x | 3.24x | 2.52x | 0.88 |
| f16-mma-parity-m256-cp | matmul | (256,512) x (512,1024) | 4.60 | 3.62 | 3.87 | 3.37 | 5.58 | 1.19x | 1.36x | 1.07x | 0.69 |
| f16-mma-parity-m256-tma | matmul | (256,512) x (512,1024) | 4.58 | 3.63 | 3.98 | 3.36 | 4.51 | 1.15x | 1.36x | 1.08x | 0.88 |
| f16-mma-raster-gm8 | matmul | (1280,256) x (256,512) | 3.94 | 3.76 | 3.39 | 3.42 | 4.98 | 1.16x | 1.15x | 1.10x | 0.68 |
| f16-mma-splitk-atomic | matmul | (128,512) x (512,128) | 4.92 | 4.96 | 8.60 | 1.59 | — | 0.57x | 3.10x | 3.13x | — |
| f16-mma-splitk-atomic-f16acc-tma-rolled | matmul | (32,1024) x (1024,256) | 5.62 | 5.46 | 4.82 | 2.32 | — | 1.16x | 2.42x | 2.35x | — |
| f16-mma-splitk-deferred | matmul | (128,512) x (512,128) | 2.76 | 3.24 | 26.37 | 1.69 | — | 0.10x | 1.63x | 1.92x | — |
| f16-mma-trans-b-staged-cp | linear | (128,128) x (128,128) | 1.49 | 1.70 | 4.17 | 1.11 | 6.07 | 0.36x | 1.35x | 1.53x | 0.69 |
| f16-mma-trans-b-staged-tma | linear | (128,128) x (128,128) | 1.49 | 1.70 | 4.64 | 1.09 | 5.79 | 0.32x | 1.37x | 1.56x | 0.80 |
| f16-scalar-register-tile | matmul | (512,512) x (512,512) | 4.60 | 3.50 | 108.02 | 3.45 | 97.24 | 0.04x | 1.33x | 1.01x | **1.11** |
| f16-symbolic-batched-mk-greedy | matmul | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 12.29 | 8.30 | 34.62 | 352.82 | 38.31 | 0.35x | 0.03x | 0.02x | 0.90 |
| f16-symbolic-computed-a-k-warp | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 21.82 | 14.37 | 31.40 | 4675.46 | 33.38 | 0.69x | 0.00x | 0.00x | 0.94 |
| f16-symbolic-demoted-n-greedy | linear+elementwise | ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) | 3.63 | 3.35 | 19.37 | 19.17 | 22.59 | 0.19x | 0.19x | 0.17x | 0.86 |
| f16-symbolic-demoted-pv-greedy | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run falló: el schedule fijado por el caso ya no se real ||||||||
| f16-symbolic-k-gmem | matmul | (64,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | 2.75 | 3.16 | 10.68 | 3.51 | 12.74 | 0.26x | 0.78x | 0.90x | 0.84 |
| f16-symbolic-k-trans | linear | (64,{'sym': 'seq_len', 'hint': 512}) x (32,{'sym': 'seq_len', 'hint': 512}) | 2.73 | 2.05 | 16.11 | 7.77 | 13.86 | 0.17x | 0.35x | 0.26x | **1.16** |
| f16-symbolic-m-residual-cp | matmul+elementwise | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) x ({'sym': 'seq_len', 'hint': 512},1024) | 5.92 | 5.61 | 6.46 | 5.59 | 7.62 | 0.92x | 1.06x | 1.00x | 0.85 |
| f16-symbolic-m-tma | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.72 | 4.69 | 5.32 | 4.88 | 6.28 | 0.89x | 0.97x | 0.96x | 0.85 |
| f16-symbolic-mn-gmem | matmul | ({'sym': 'seq_len', 'hint': 512},128) x (128,{'sym': 'seq_len', 'hint': 512}) | 2.22 | 1.91 | 4.28 | 4.29 | 5.16 | 0.52x | 0.52x | 0.45x | 0.83 |
| f16-warp-band-p1-tma-d1-m128 | matmul | (128,256) x (256,128) | 2.03 | 2.34 | 4.03 | 1.38 | 4.47 | 0.50x | 1.47x | 1.69x | 0.90 |
| f16-warp-band-p1-tma-d2-m100 | matmul | (100,256) x (256,128) | 2.00 | 1.88 | 2.59 | 1.38 | 2.91 | 0.77x | 1.45x | 1.36x | 0.89 |
| f16-warp-band-p1-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.03 | 2.33 | 2.65 | 1.40 | 2.95 | 0.76x | 1.45x | 1.67x | 0.90 |
| f16-warp-band-p2-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.02 | 2.33 | 2.66 | 1.38 | 2.93 | 0.76x | 1.47x | 1.69x | 0.91 |
| f16-warp-gmem-direct | matmul | (128,128) x (128,128) | 1.53 | 1.66 | 2.19 | 1.14 | 3.42 | 0.70x | 1.34x | 1.46x | 0.64 |
| f16-warp-splitk | matmul | (64,4096) x (4096,512) | 5.84 | 5.83 | 53.09 | 7.47 | — | 0.11x | 0.78x | 0.78x | — |
| f16-warp-symbolic-m-cp | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.71 | 4.69 | 5.19 | 4.88 | 6.20 | 0.91x | 0.96x | 0.96x | 0.84 |
| f32-epilogue-bias | matmul+elementwise | (1,64,64) x (64,64) x (64) | 4.28 | 2.84 | 1.71 | 1.30 | 2.10 | 2.50x | 3.29x | 2.18x | 0.82 |
| f32-epilogue-relu | matmul+elementwise | (1,64,64) x (64,64) | 4.07 | 2.05 | 1.71 | 1.13 | 1.90 | 2.38x | 3.60x | 1.81x | 0.90 |
| f32-epilogue-residual | matmul+elementwise | (1,64,64) x (64,64) x (1,64,64) | 4.11 | 2.85 | 1.74 | 1.27 | 2.20 | 2.37x | 3.24x | 2.24x | 0.79 |
| f32-epilogue-scale | matmul+elementwise | (1,64,64) x (64,64) x (1) | 4.24 | 2.18 | 1.73 | 1.26 | 2.09 | 2.45x | 3.36x | 1.73x | 0.83 |
| f32-mixed-dtype-slabs | matmul | (64,128) x (128,96) | 3.91 | 3.34 | 2.17 | 1.23 | — | 1.80x | 3.18x | 2.72x | — |
| f32-reduce-coop | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 3.34 | 1.31 | 4.58 | 0.94x | 2.40x | 1.56x | 0.73 |
| f32-reduce-coop-ilp | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 3.32 | 1.32 | 3.66 | 0.95x | 2.38x | 1.56x | 0.91 |
| f32-reduce-ilp | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 1.80 | 1.31 | 4.07 | 1.74x | 2.39x | 1.56x | 0.44 |
| f32-scalar-bm8-outside-hints | matmul | (2048,2048) x (2048,2048) | 184.69 | 175.76 | 1360.77 | 248.74 | 1596.48 | 0.14x | 0.74x | 0.71x | 0.85 |
| f32-scalar-composite-fm4-fn4 | matmul | (2048,2048) x (2048,2048) | 184.82 | 175.26 | 345.83 | 248.74 | 530.50 | 0.53x | 0.74x | 0.70x | 0.65 |
| f32-scalar-composite-fn4 | matmul | (2048,2048) x (2048,2048) | 184.83 | 175.79 | 773.38 | 248.74 | 1271.23 | 0.24x | 0.74x | 0.71x | 0.61 |
| f32-scalar-masked-m-fm26 | matmul | (2048,2048) x (2048,2048) | 184.70 | 175.26 | 409.17 | 236.72 | 659.81 | 0.45x | 0.78x | 0.74x | 0.62 |
| f32-scalar-masked-n-fn26 | matmul | (512,512) x (512,512) | 7.24 | 5.40 | 300.79 | 8.10 | 75.39 | 0.02x | 0.89x | 0.67x | **3.99** |
| f32-scalar-masked-n-gmem | matmul | (64,64) x (64,47) | 3.64 | 1.57 | 4.32 | 1.48 | 2.84 | 0.84x | 2.45x | 1.06x | **1.52** |
| f32-scalar-staging-d1-128x128x128 | matmul | (128,128) x (128,128) | 2.99 | 3.34 | 6.22 | 1.63 | 3.29 | 0.48x | 1.83x | 2.05x | **1.89** |
| f32-scalar-staging-d1-64x64x64 | matmul | (64,64) x (64,64) | 3.15 | 2.05 | 1.32 | 1.31 | 1.26 | 2.38x | 2.40x | 1.57x | 1.05 |
| f32-scalar-staging-d2-128x128x128 | matmul | (128,128) x (128,128) | 2.98 | 3.35 | 6.34 | 1.63 | 3.06 | 0.47x | 1.83x | 2.06x | **2.07** |
| f32-scalar-staging-d2-64x64x64 | matmul | (64,64) x (64,64) | 3.15 | 2.05 | 1.33 | 1.31 | 1.38 | 2.37x | 2.41x | 1.57x | 0.96 |
| f32-scalar-staging-gmem-128x128x128 | matmul | (128,128) x (128,128) | 2.98 | 3.35 | 13.45 | 1.66 | 3.24 | 0.22x | 1.80x | 2.02x | **4.15** |
| f32-scalar-staging-gmem-64x64x64 | matmul | (64,64) x (64,64) | 3.15 | 2.05 | 2.70 | 1.31 | 2.00 | 1.16x | 2.41x | 1.57x | **1.35** |
| f32-scalar-tma-staged | matmul | (2048,2048) x (2048,2048) | 184.10 | 174.75 | 345.96 | 235.94 | 426.18 | 0.53x | 0.78x | 0.74x | 0.81 |
| f32-tile-per-cell | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 1.73 | 1.31 | 1.91 | 1.82x | 2.41x | 1.57x | 0.91 |
| f32-tile-reg-2d | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 2.83 | 1.31 | 3.11 | 1.11x | 2.40x | 1.56x | 0.91 |
| f32-tile-reg-fn3 | matmul | (1,64,64) x (64,64) | 3.16 | 2.05 | 3.51 | 1.32 | 3.95 | 0.90x | 2.40x | 1.56x | 0.89 |
| f32-tile-reg-inner-fn4 | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 2.86 | 1.31 | 3.14 | 1.10x | 2.40x | 1.57x | 0.91 |
| f32-tile-single-cta | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 2.86 | 1.32 | 2.88 | 1.10x | 2.38x | 1.55x | 0.99 |
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.03 | 5.55 | 4.13 | 2.26 | — | 7.75x | 14.19x | 2.46x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.01 | 5.55 | 2.81 | 2.27 | — | 11.38x | 14.08x | 2.44x | — |
| imap-reshape-a-cp | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.34 | 5.86 | 1.15 | 1.75 | 0.27x | 1.36x | 1.16x | **3.34** |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.34 | 5.43 | 1.16 | — | 0.29x | 1.35x | 1.16x | — |
| imap-reshape-b-gmem | matmul+reshape | (64,128) x (256,64) | 1.68 | 1.66 | 4.95 | 1.16 | — | 0.34x | 1.46x | 1.43x | — |
| imap-slice-a-cp | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 2.72 | 1.15 | 1.85 | 0.61x | 1.45x | 1.15x | **1.48** |
| imap-slice-a-gmem | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 5.35 | 1.16 | 2.06 | 0.31x | 1.44x | 1.14x | **2.60** |
| imap-slice-a-tma | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 1.83 | 1.16 | 2.10 | 0.91x | 1.44x | 1.14x | 0.87 |
| imap-transpose-a-gmem | matmul+transpose | (256,128) x (256,64) | 2.26 | 1.93 | 3.50 | 3.67 | 7.78 | 0.65x | 0.61x | 0.52x | 0.45 |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.62 | 6.16 | 1.79 | 1.39 | — | 23.23x | 29.85x | 4.42x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 3.01 | 2.11 | 1.43 | — | 43.60x | 64.65x | 2.12x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.29 | 5.64 | 2.30 | 1.20 | — | 13.64x | 26.19x | 4.72x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.78 | 3.67 | 4.41 | 1.25 | — | 6.07x | 21.51x | 2.95x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 181.96 | 38.21 | 3.13 | 1.69 | — | 58.20x | 107.84x | 22.64x | — |

Media geométrica vs eager: emmy fijado 0.80x · greedy 1.88x · greedy vs torch.compile 1.34x (greedy gana a torch.compile en 67/84). Reproducimos la referencia CloudRift (±5%) en 52/64 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-qk-sdpa-composed-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.96 | 4.85 | 3.50 | 1.59 | — | 1.70x | 3.74x | 3.04x | — |
| rmsnorm-qk-sdpa-output-axis | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.98 | 4.85 | 1.92 | 1.60 | — | 3.11x | 3.73x | 3.03x | — |
| rmsnorm-qk-sdpa-stat-b-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.85 | 4.20 | 1.63 | — | 1.42x | 3.67x | 2.98x | — |
| rmsnorm-qk-sdpa-stat-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.85 | 3.41 | 1.58 | — | 1.75x | 3.77x | 3.06x | — |
| norm-linear-f16-scalar-reduce | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.95 | 2.64 | 106.87 | 2.20 | 80.04 | 0.03x | 1.34x | 1.20x | **1.34** |
| norm-linear-f16-warp-masked-m | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.95 | 2.64 | 6.40 | 2.21 | 11.58 | 0.46x | 1.34x | 1.19x | 0.55 |
| norm-weight-linear-f16-computed-b | linear+rms_norm | (64,128) x (64,128) x (128) | 3.08 | 2.18 | 78.42 | 11.10 | — | 0.04x | 0.28x | 0.20x | — |
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.11 | — | 1.28x | 1.55x | 1.08x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.05 | 1.23 | 0.94 | — | 1.33x | 1.75x | 1.12x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.28 | 2.42 | 2.70 | 1.60 | — | 1.21x | 2.05x | 1.51x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.33 | 1.46 | 1.46 | — | 1.46x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.04 | 1.16 | 1.16 | — | 1.60x | 1.60x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.58 | 1.96 | 1.88 | 1.89 | — | 1.37x | 1.36x | 1.03x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.43 | 1.79 | 1.30 | — | 1.33x | 1.84x | 1.10x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.14 | 1.24 | 1.00 | — | 1.33x | 1.65x | 1.15x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 4.29 | 4.63 | 2.38 | — | 1.12x | 2.18x | 1.80x | — |
| fused-add-rms-norm | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.92 | 1.06 | 0.92 | — | 2.20x | 2.53x | 1.00x | — |
| fused-add-rms-norm-escaping-residual | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.98 | 1.06 | 0.93 | — | 2.20x | 2.50x | 1.05x | — |
| rms-norm-cut-sweep-work | rms_norm | (512,4096) | 5.14 | 4.58 | 357.64 | 4.57 | — | 0.01x | 1.12x | 1.00x | — |
| rms-norm-packed-cells | rms_norm | (2048,128) | 2.34 | 1.55 | 1.42 | 1.36 | — | 1.65x | 1.72x | 1.14x | — |

Media geométrica vs eager: emmy fijado 0.79x · greedy 1.81x · greedy vs torch.compile 1.26x (greedy gana a torch.compile en 16/20). Reproducimos la referencia CloudRift (±5%) en 1/2 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.03 | 5.55 | 4.13 | 2.26 | — | 7.75x | 14.19x | 2.46x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 32.01 | 5.55 | 2.81 | 2.27 | — | 11.38x | 14.08x | 2.44x | — |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.62 | 6.16 | 1.79 | 1.39 | — | 23.23x | 29.85x | 4.42x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 3.01 | 2.11 | 1.43 | — | 43.60x | 64.65x | 2.12x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.29 | 5.64 | 2.30 | 1.20 | — | 13.64x | 26.19x | 4.72x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.78 | 3.67 | 4.41 | 1.25 | — | 6.07x | 21.51x | 2.95x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 181.96 | 38.21 | 3.13 | 1.69 | — | 58.20x | 107.84x | 22.64x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 367.16 | 62.11 | 5.15 | 2.85 | — | 71.25x | 129.04x | 21.83x | — |
| nvfp4-gate-up-requant-full-projection-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.97 | 11.36 | 500.29 | 60.74 | — | 0.34x | 2.83x | 0.19x | — |
| nvfp4-gate-up-requant-place-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.97 | 11.36 | 646.96 | 60.74 | — | 0.27x | 2.83x | 0.19x | — |
| nvfp4-quantize-cut-shared-normalizer | elementwise+index_map+mean+reduce+reshape | (32,128) | 127.01 | 2.96 | 38.48 | 36.82 | — | 3.30x | 3.45x | 0.08x | — |
| nvfp4-w4a16-sibling-cut-async | elementwise+index_map+linear+reshape+gather | (48,256) | 38.96 | 6.24 | 12.77 | 7.09 | — | 3.05x | 5.49x | 0.88x | — |
| nvfp4-w4a16-sibling-cut-tma-p2 | elementwise+index_map+linear+reshape+gather | (48,256) | 38.95 | 6.24 | 13.51 | 7.10 | — | 2.88x | 5.48x | 0.88x | — |

Media geométrica vs eager: emmy fijado 6.91x · greedy 15.30x · greedy vs torch.compile 1.68x (greedy gana a torch.compile en 8/13). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.19 | 1.34 | 1.11 | — | 1.28x | 1.55x | 1.08x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.05 | 1.23 | 0.94 | — | 1.33x | 1.75x | 1.12x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.28 | 2.42 | 2.70 | 1.60 | — | 1.21x | 2.05x | 1.51x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.33 | 1.46 | 1.46 | — | 1.46x | 1.46x | 0.91x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.04 | 1.16 | 1.16 | — | 1.60x | 1.60x | 0.90x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.58 | 1.96 | 1.88 | 1.89 | — | 1.37x | 1.36x | 1.03x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.38 | 1.43 | 1.79 | 1.30 | — | 1.33x | 1.84x | 1.10x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.14 | 1.24 | 1.00 | — | 1.33x | 1.65x | 1.15x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 4.29 | 4.63 | 2.38 | — | 1.12x | 2.18x | 1.80x | — |
| gated-mlp-s128 | matmul+elementwise | (1,128,1024) x (1024,3072) x (1024,3072) | 29.58 | 27.95 | 438.90 | 40.29 | — | 0.07x | 0.73x | 0.69x | — |
| gated-mlp-s32 | matmul+elementwise | (1,32,1024) x (1024,3072) x (1024,3072) | 28.01 | 20.77 | 117.12 | 113.05 | — | 0.24x | 0.25x | 0.18x | — |
| gated-mlp-s512 | matmul+elementwise | (1,512,1024) x (1024,3072) x (1024,3072) | 103.02 | 95.55 | 1748.16 | 122.45 | — | 0.06x | 0.84x | 0.78x | — |
| matmul-add-downproj-s128 | matmul+elementwise | (1,128,3072) x (3072,1024) x (1,128,1024) | 13.32 | 13.38 | 37.45 | 19.60 | 44.56 | 0.36x | 0.68x | 0.68x | 0.84 |
| matmul-add-downproj-s32 | matmul+elementwise | (1,32,3072) x (3072,1024) x (1,32,1024) | 11.40 | 11.22 | 33.73 | 14.83 | 39.57 | 0.34x | 0.77x | 0.76x | 0.85 |
| matmul-add-downproj-s512 | matmul+elementwise | (1,512,3072) x (3072,1024) x (1,512,1024) | 38.12 | 37.93 | 133.05 | 68.00 | 153.72 | 0.29x | 0.56x | 0.56x | 0.87 |
| matmul-add-oproj-s128 | matmul+elementwise | (1,128,2048) x (2048,1024) x (1,128,1024) | 10.97 | 11.02 | 25.47 | 13.92 | 30.06 | 0.43x | 0.79x | 0.79x | 0.85 |
| matmul-add-oproj-s32 | matmul+elementwise | (1,32,2048) x (2048,1024) x (1,32,1024) | 9.26 | 9.06 | 22.83 | 10.36 | 26.94 | 0.41x | 0.89x | 0.87x | 0.85 |
| matmul-add-oproj-s512 | matmul+elementwise | (1,512,2048) x (2048,1024) x (1,512,1024) | 26.95 | 26.56 | 88.37 | 46.11 | 102.91 | 0.30x | 0.58x | 0.58x | 0.86 |
| matmul-kvproj-s128 | matmul | (1,128,1024) x (1024,1024) | 13.15 | 10.11 | 13.86 | 11.75 | 15.69 | 0.95x | 1.12x | 0.86x | 0.88 |
| matmul-kvproj-s32 | matmul | (1,32,1024) x (1024,1024) | 6.54 | 6.53 | 12.02 | 5.98 | 13.77 | 0.54x | 1.09x | 1.09x | 0.87 |
| matmul-kvproj-s512 | matmul | (1,512,1024) x (1024,1024) | 13.89 | 13.59 | 46.14 | 23.90 | 52.62 | 0.30x | 0.58x | 0.57x | 0.88 |
| matmul-qproj-s128 | matmul | (1,128,1024) x (1024,2048) | 13.10 | 10.58 | 27.51 | 14.36 | 31.25 | 0.48x | 0.91x | 0.74x | 0.88 |
| matmul-qproj-s32 | matmul | (1,32,1024) x (1024,2048) | 8.39 | 8.37 | 12.36 | 8.03 | 14.20 | 0.68x | 1.05x | 1.04x | 0.87 |
| matmul-qproj-s512 | matmul | (1,512,1024) x (1024,2048) | 25.62 | 25.49 | 55.66 | 40.04 | 62.24 | 0.46x | 0.64x | 0.64x | 0.89 |
| sdpa-s128 | sdpa | (1,16,128,128) x (1,8,128,128) x (1,8,128,128) | 28.03 | 24.99 | 1445.18 | 1399.36 | — | 0.02x | 0.02x | 0.02x | — |
| sdpa-s32 | sdpa | (1,16,32,128) x (1,8,32,128) x (1,8,32,128) | 18.88 | 18.40 | 95.98 | 75.12 | — | 0.20x | 0.25x | 0.25x | — |
| sdpa-s512 | sdpa | (1,16,512,128) x (1,8,512,128) x (1,8,512,128) | 89.73 | 86.78 | 18516.80 | 6264.48 | — | 0.00x | 0.01x | 0.01x | — |

Media geométrica vs eager: emmy fijado 0.39x · greedy 0.70x · greedy vs torch.compile 0.59x (greedy gana a torch.compile en 9/27). Reproducimos la referencia CloudRift (±5%) en 12/12 casos con referencia.

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

