# Suite emmy · subconjunto TFM · NVIDIA GeForce RTX 5090

- Fecha: 2026-11-11T09:26:42 · deplodock bf784a76 2026-11-11 · driver 610.43.02 · torch 2.15.0+cu132
- Casos: 130 (128 medidos, 2 con error) · categorías: matmul, rmsnorm/fused, nvfp4, qwen3emb
- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.
- **Ejecución con `--tune 24`**: antes de medir, emmy autotunea cada caso (N variantes del schedule, base de datos propia del caso); `greedy` es lo que elige después. Autotuneados 118/130 (el resto tiene varios kernels o falló: ver logs).
- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).
- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido. Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.

## matmul

_multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16-mma-sm120 | matmul | (128,256) x (256,128) | 2.16 | 1.89 | 1.78 | 1.36 | — | 1.21x | 1.59x | 1.39x | — |
| bf16-mma-trans-b-masked-n-async | linear | (128,128) x (72,128) | 1.67 | 2.33 | 3.89 | 1.08 | — | 0.43x | 1.55x | 2.15x | — |
| bf16-mma-trans-b-masked-n-tma | linear | (128,128) x (72,128) | 1.67 | 2.32 | 4.23 | 1.05 | — | 0.40x | 1.59x | 2.20x | — |
| bf16-rowless-grouped-pair | matmul+elementwise | (256) x (4,256,16) x (4,256,16) | 4.85 | 2.55 | 1.63 | 1.08 | — | 2.98x | 4.51x | 2.36x | — |
| bf16-scalar-sync-d2-sm120 | linear | (32,384) x (64,384) | 2.73 | 1.82 | 4.47 | 1.35 | — | 0.61x | 2.02x | 1.35x | — |
| f16-matvec-reshaped-output-tma | linear+reshape+index_map | (1,1,1024) x (2048,1024) | 4.66 | 2.03 | 6.33 | 2.25 | — | 0.74x | 2.07x | 0.90x | — |
| f16-mma-batched-pv-b1h4kv128 | matmul | (1,4,128,128) x (1,4,128,128) | 4.57 | 4.45 | 5.41 | 1.17 | 5.80 | 0.85x | 3.89x | 3.79x | 0.93 |
| f16-mma-batched-pv-b2h3kv256 | matmul | (2,3,128,256) x (2,3,256,128) | 6.47 | 6.34 | 8.58 | 1.67 | 6.02 | 0.75x | 3.87x | 3.79x | **1.42** |
| f16-mma-batched-qk-b1h4d64 | matmul+transpose | (1,4,128,64) x (1,4,128,64) | 3.04 | 3.74 | 3.84 | 0.96 | 5.92 | 0.79x | 3.15x | 3.89x | 0.65 |
| f16-mma-batched-qk-b2h3d32 | matmul+transpose | (2,3,128,32) x (2,3,128,32) | 2.91 | 2.88 | 2.82 | 0.96 | 2.87 | 1.03x | 3.02x | 3.00x | 0.98 |
| f16-mma-epilogue-residual | matmul+elementwise | (128,128) x (128,128) x (128,128) | 5.70 | 3.59 | 6.62 | 1.15 | 11.33 | 0.86x | 4.96x | 3.13x | 0.58 |
| f16-mma-f16acc-cp | matmul | (128,128) x (128,128) | 1.53 | 1.66 | 3.66 | 1.14 | 4.91 | 0.42x | 1.35x | 1.46x | 0.74 |
| f16-mma-f16acc-gmem | matmul | (128,128) x (128,128) | 1.54 | 1.67 | 7.33 | 1.14 | 8.68 | 0.21x | 1.35x | 1.46x | 0.84 |
| f16-mma-f16acc-symbolic-k | matmul | (128,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run salió con código 1: el schedule fijado por el caso  ||||||||
| f16-mma-f16acc-tma | matmul | (128,128) x (128,128) | 1.54 | 1.67 | 3.86 | 1.14 | 4.54 | 0.40x | 1.34x | 1.46x | 0.85 |
| f16-mma-m128n128k128-f16 | matmul | (128,128) x (128,128) | 1.54 | 1.67 | 4.73 | 1.14 | 6.21 | 0.32x | 1.34x | 1.46x | 0.76 |
| f16-mma-m128n128k128-f32 | matmul | (128,128) x (128,128) | 4.74 | 3.46 | 4.79 | 1.14 | 6.07 | 0.99x | 4.16x | 3.04x | 0.79 |
| f16-mma-m128n128k128-f32-trans | linear | (128,128) x (128,128) | 2.39 | 1.71 | 4.72 | 1.08 | 4.79 | 0.51x | 2.21x | 1.58x | 0.98 |
| f16-mma-m128n128k132-f32-masked-k | matmul | (128,132) x (132,128) | 5.01 | 3.96 | 6.47 | 1.52 | 7.23 | 0.78x | 3.29x | 2.60x | 0.89 |
| f16-mma-m128n128k132-f32-trans-masked-k | linear | (128,132) x (128,132) | 2.89 | 1.79 | 6.38 | 1.37 | 6.54 | 0.45x | 2.10x | 1.30x | 0.98 |
| f16-mma-m128n128k136-f32-masked-k | matmul | (128,136) x (136,128) | 4.88 | 3.82 | 6.25 | 1.61 | 7.09 | 0.78x | 3.03x | 2.37x | 0.88 |
| f16-mma-parity-m256-cp | matmul | (256,512) x (512,1024) | 4.56 | 3.61 | 3.85 | 3.35 | 5.58 | 1.19x | 1.36x | 1.08x | 0.69 |
| f16-mma-parity-m256-tma | matmul | (256,512) x (512,1024) | 4.57 | 3.61 | 3.97 | 3.34 | 4.51 | 1.15x | 1.36x | 1.08x | 0.88 |
| f16-mma-raster-gm8 | matmul | (1280,256) x (256,512) | 3.95 | 3.75 | 3.37 | 3.42 | 4.98 | 1.17x | 1.16x | 1.10x | 0.68 |
| f16-mma-splitk-atomic | matmul | (128,512) x (512,128) | 4.91 | 4.80 | 8.61 | 1.58 | — | 0.57x | 3.11x | 3.03x | — |
| f16-mma-splitk-atomic-f16acc-tma-rolled | matmul | (32,1024) x (1024,256) | 5.61 | 5.47 | 4.82 | 2.29 | — | 1.16x | 2.45x | 2.38x | — |
| f16-mma-splitk-deferred | matmul | (128,512) x (512,128) | 2.75 | 3.15 | 26.38 | 1.70 | — | 0.10x | 1.62x | 1.86x | — |
| f16-mma-trans-b-staged-cp | linear | (128,128) x (128,128) | 1.49 | 1.65 | 4.18 | 1.09 | 6.07 | 0.36x | 1.36x | 1.51x | 0.69 |
| f16-mma-trans-b-staged-tma | linear | (128,128) x (128,128) | 1.49 | 1.65 | 4.66 | 1.09 | 5.79 | 0.32x | 1.36x | 1.52x | 0.80 |
| f16-scalar-register-tile | matmul | (512,512) x (512,512) | 4.59 | 3.50 | 108.05 | 3.43 | 97.24 | 0.04x | 1.34x | 1.02x | **1.11** |
| f16-symbolic-batched-mk-greedy | matmul | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 12.31 | 8.22 | 34.72 | 351.93 | 38.31 | 0.35x | 0.03x | 0.02x | 0.91 |
| f16-symbolic-computed-a-k-warp | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | 21.89 | 14.27 | 31.29 | 4685.57 | 33.38 | 0.70x | 0.00x | 0.00x | 0.94 |
| f16-symbolic-demoted-n-greedy | linear+elementwise | ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) x ({'sym': 'seq_len', 'hint': 512},128) | 3.66 | 3.24 | 19.31 | 20.42 | 22.59 | 0.19x | 0.18x | 0.16x | 0.85 |
| f16-symbolic-demoted-pv-greedy | matmul+softmax | (16,{'sym': 'seq_len', 'hint': 512},{'sym': 'seq_len', 'hint': 512}) x (16,{'sym': 'seq_len', 'hint': 512},128) | ERROR: emmy run salió con código 1: el schedule fijado por el caso  ||||||||
| f16-symbolic-k-gmem | matmul | (64,{'sym': 'seq_len', 'hint': 512}) x ({'sym': 'seq_len', 'hint': 512},128) | 2.75 | 3.22 | 10.84 | 3.52 | 12.74 | 0.25x | 0.78x | 0.91x | 0.85 |
| f16-symbolic-k-trans | linear | (64,{'sym': 'seq_len', 'hint': 512}) x (32,{'sym': 'seq_len', 'hint': 512}) | 2.73 | 2.05 | 16.15 | 7.76 | 13.86 | 0.17x | 0.35x | 0.26x | **1.17** |
| f16-symbolic-m-residual-cp | matmul+elementwise | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) x ({'sym': 'seq_len', 'hint': 512},1024) | 5.91 | 5.62 | 6.43 | 5.42 | 7.62 | 0.92x | 1.09x | 1.04x | 0.84 |
| f16-symbolic-m-tma | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.70 | 5.45 | 5.31 | 4.89 | 6.28 | 0.88x | 0.96x | 1.11x | 0.85 |
| f16-symbolic-mn-gmem | matmul | ({'sym': 'seq_len', 'hint': 512},128) x (128,{'sym': 'seq_len', 'hint': 512}) | 2.21 | 1.91 | 4.28 | 4.28 | 5.16 | 0.52x | 0.52x | 0.45x | 0.83 |
| f16-warp-band-p1-tma-d1-m128 | matmul | (128,256) x (256,128) | 2.02 | 1.88 | 4.02 | 1.36 | 4.47 | 0.50x | 1.49x | 1.39x | 0.90 |
| f16-warp-band-p1-tma-d2-m100 | matmul | (100,256) x (256,128) | 1.99 | 1.88 | 2.60 | 1.38 | 2.91 | 0.77x | 1.44x | 1.36x | 0.89 |
| f16-warp-band-p1-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.02 | 1.88 | 2.63 | 1.36 | 2.95 | 0.77x | 1.49x | 1.38x | 0.89 |
| f16-warp-band-p2-tma-d2-m128 | matmul | (128,256) x (256,128) | 2.03 | 1.89 | 2.66 | 1.38 | 2.93 | 0.76x | 1.47x | 1.37x | 0.91 |
| f16-warp-gmem-direct | matmul | (128,128) x (128,128) | 1.53 | 1.67 | 2.19 | 1.14 | 3.42 | 0.70x | 1.34x | 1.46x | 0.64 |
| f16-warp-splitk | matmul | (64,4096) x (4096,512) | 5.85 | 5.84 | 53.02 | 8.01 | — | 0.11x | 0.73x | 0.73x | — |
| f16-warp-symbolic-m-cp | matmul | ({'sym': 'seq_len', 'hint': 512},512) x (512,1024) | 4.70 | 5.45 | 5.18 | 4.87 | 6.20 | 0.91x | 0.97x | 1.12x | 0.84 |
| f32-epilogue-bias | matmul+elementwise | (1,64,64) x (64,64) x (64) | 4.29 | 2.79 | 1.72 | 1.21 | 2.10 | 2.50x | 3.55x | 2.31x | 0.82 |
| f32-epilogue-relu | matmul+elementwise | (1,64,64) x (64,64) | 4.07 | 2.05 | 1.71 | 1.13 | 1.90 | 2.39x | 3.60x | 1.81x | 0.90 |
| f32-epilogue-residual | matmul+elementwise | (1,64,64) x (64,64) x (1,64,64) | 4.09 | 2.82 | 1.73 | 1.32 | 2.20 | 2.36x | 3.10x | 2.14x | 0.79 |
| f32-epilogue-scale | matmul+elementwise | (1,64,64) x (64,64) x (1) | 4.24 | 2.18 | 1.73 | 1.27 | 2.09 | 2.46x | 3.35x | 1.72x | 0.83 |
| f32-mixed-dtype-slabs | matmul | (64,128) x (128,96) | 3.91 | 3.34 | 2.18 | 1.23 | — | 1.80x | 3.17x | 2.71x | — |
| f32-reduce-coop | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 3.24 | 1.31 | 4.58 | 0.97x | 2.40x | 1.56x | 0.71 |
| f32-reduce-coop-ilp | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 3.28 | 1.31 | 3.66 | 0.96x | 2.40x | 1.56x | 0.90 |
| f32-reduce-ilp | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 1.80 | 1.31 | 4.07 | 1.75x | 2.40x | 1.57x | 0.44 |
| f32-scalar-bm8-outside-hints | matmul | (2048,2048) x (2048,2048) | 184.70 | 175.75 | 1365.12 | 236.46 | 1596.48 | 0.14x | 0.78x | 0.74x | 0.86 |
| f32-scalar-composite-fm4-fn4 | matmul | (2048,2048) x (2048,2048) | 185.30 | 175.75 | 346.72 | 248.74 | 530.50 | 0.53x | 0.74x | 0.71x | 0.65 |
| f32-scalar-composite-fn4 | matmul | (2048,2048) x (2048,2048) | 185.25 | 175.75 | 773.22 | 248.74 | 1271.23 | 0.24x | 0.74x | 0.71x | 0.61 |
| f32-scalar-masked-m-fm26 | matmul | (2048,2048) x (2048,2048) | 184.73 | 176.29 | 408.06 | 249.26 | 659.81 | 0.45x | 0.74x | 0.71x | 0.62 |
| f32-scalar-masked-n-fn26 | matmul | (512,512) x (512,512) | 7.24 | 5.39 | 300.36 | 8.11 | 75.39 | 0.02x | 0.89x | 0.66x | **3.98** |
| f32-scalar-masked-n-gmem | matmul | (64,64) x (64,47) | 3.64 | 1.57 | 4.27 | 1.37 | 2.84 | 0.85x | 2.65x | 1.15x | **1.50** |
| f32-scalar-staging-d1-128x128x128 | matmul | (128,128) x (128,128) | 2.98 | 3.59 | 6.23 | 1.62 | 3.29 | 0.48x | 1.84x | 2.21x | **1.90** |
| f32-scalar-staging-d1-64x64x64 | matmul | (64,64) x (64,64) | 3.15 | 2.05 | 1.32 | 1.32 | 1.26 | 2.38x | 2.39x | 1.56x | 1.04 |
| f32-scalar-staging-d2-128x128x128 | matmul | (128,128) x (128,128) | 2.99 | 3.59 | 6.34 | 1.62 | 3.06 | 0.47x | 1.84x | 2.21x | **2.07** |
| f32-scalar-staging-d2-64x64x64 | matmul | (64,64) x (64,64) | 3.14 | 2.05 | 1.33 | 1.31 | 1.38 | 2.37x | 2.41x | 1.57x | 0.96 |
| f32-scalar-staging-gmem-128x128x128 | matmul | (128,128) x (128,128) | 2.98 | 3.58 | 13.45 | 1.63 | 3.24 | 0.22x | 1.83x | 2.20x | **4.15** |
| f32-scalar-staging-gmem-64x64x64 | matmul | (64,64) x (64,64) | 3.14 | 2.05 | 2.70 | 1.31 | 2.00 | 1.16x | 2.40x | 1.56x | **1.35** |
| f32-scalar-tma-staged | matmul | (2048,2048) x (2048,2048) | 183.87 | 174.78 | 345.89 | 248.23 | 426.18 | 0.53x | 0.74x | 0.70x | 0.81 |
| f32-tile-per-cell | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 1.75 | 1.31 | 1.91 | 1.80x | 2.40x | 1.57x | 0.92 |
| f32-tile-reg-2d | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 2.85 | 1.31 | 3.11 | 1.10x | 2.40x | 1.56x | 0.92 |
| f32-tile-reg-fn3 | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 3.52 | 1.32 | 3.95 | 0.89x | 2.39x | 1.56x | 0.89 |
| f32-tile-reg-inner-fn4 | matmul | (1,64,64) x (64,64) | 3.15 | 2.05 | 2.86 | 1.31 | 3.14 | 1.10x | 2.40x | 1.56x | 0.91 |
| f32-tile-single-cta | matmul | (1,64,64) x (64,64) | 3.14 | 2.05 | 2.86 | 1.32 | 2.88 | 1.10x | 2.38x | 1.55x | 0.99 |
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 31.97 | 5.60 | 4.13 | 2.30 | — | 7.74x | 13.88x | 2.43x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 31.96 | 5.60 | 2.83 | 2.30 | — | 11.31x | 13.92x | 2.44x | — |
| imap-reshape-a-cp | matmul+reshape | (128,256) x (128,64) | 1.56 | 1.66 | 5.85 | 1.15 | 1.75 | 0.27x | 1.35x | 1.44x | **3.34** |
| imap-reshape-a-gmem | matmul+reshape | (128,256) x (128,64) | 1.57 | 1.66 | 5.42 | 1.16 | — | 0.29x | 1.35x | 1.43x | — |
| imap-reshape-b-gmem | matmul+reshape | (64,128) x (256,64) | 1.68 | 1.32 | 5.00 | 1.15 | — | 0.34x | 1.46x | 1.15x | — |
| imap-slice-a-cp | matmul+slice | (128,256) x (128,64) | 1.67 | 1.33 | 2.61 | 1.15 | 1.85 | 0.64x | 1.44x | 1.15x | **1.41** |
| imap-slice-a-gmem | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 5.35 | 1.15 | 2.06 | 0.31x | 1.45x | 1.15x | **2.60** |
| imap-slice-a-tma | matmul+slice | (128,256) x (128,64) | 1.67 | 1.32 | 1.83 | 1.16 | 2.10 | 0.91x | 1.44x | 1.15x | 0.87 |
| imap-transpose-a-gmem | matmul+transpose | (256,128) x (256,64) | 2.26 | 2.22 | 3.96 | 3.74 | 7.78 | 0.57x | 0.60x | 0.59x | 0.51 |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.62 | 4.89 | 1.79 | 1.40 | — | 23.26x | 29.82x | 3.50x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 2.38 | 2.11 | 1.43 | — | 43.61x | 64.31x | 1.66x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.28 | 5.65 | 2.24 | 1.22 | — | 13.98x | 25.68x | 4.64x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.70 | 3.19 | 4.38 | 1.24 | — | 6.09x | 21.51x | 2.57x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.02 | 6.50 | 3.12 | 1.25 | — | 10.26x | 25.56x | 5.19x | — |

Media geométrica vs eager: emmy fijado 0.79x · greedy 1.85x · greedy vs torch.compile 1.32x (greedy gana a torch.compile en 69/84). Reproducimos la referencia CloudRift (±5%) en 52/64 casos con referencia.

## rmsnorm/fused

_casos con RMSNorm, sola o fusionada con la proyección lineal / atención_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-qk-sdpa-composed-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.98 | 4.86 | 3.48 | 1.59 | — | 1.72x | 3.76x | 3.06x | — |
| rmsnorm-qk-sdpa-output-axis | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.85 | 1.92 | 1.61 | — | 3.11x | 3.70x | 3.01x | — |
| rmsnorm-qk-sdpa-stat-b-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.98 | 4.85 | 4.18 | 1.59 | — | 1.43x | 3.75x | 3.04x | — |
| rmsnorm-qk-sdpa-stat-cut | rms_norm+sdpa | (1,2,8,16) x (1,2,8,16) x (1,2,8,16) x (16) x (16) | 5.97 | 4.84 | 3.51 | 1.61 | — | 1.70x | 3.70x | 3.01x | — |
| norm-linear-f16-scalar-reduce | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.95 | 2.16 | 106.81 | 2.20 | 80.04 | 0.03x | 1.34x | 0.98x | **1.33** |
| norm-linear-f16-warp-masked-m | rms_norm+linear | (1,32,128) x (128) x (512,128) | 2.96 | 2.16 | 6.43 | 2.21 | 11.58 | 0.46x | 1.34x | 0.98x | 0.55 |
| norm-weight-linear-f16-computed-b | linear+rms_norm | (64,128) x (64,128) x (128) | 3.09 | 2.19 | 78.94 | 11.11 | — | 0.04x | 0.28x | 0.20x | — |
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.41 | 1.35 | 1.10 | — | 1.27x | 1.55x | 1.28x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.01 | 1.23 | 0.94 | — | 1.33x | 1.73x | 1.07x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.70 | 1.59 | — | 1.21x | 2.06x | 1.52x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.32 | 1.46 | 1.44 | — | 1.47x | 1.48x | 0.92x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.50 | 1.16 | 1.17 | — | 1.60x | 1.59x | 1.29x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.58 | 1.87 | 1.88 | 1.89 | — | 1.37x | 1.36x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.39 | 1.43 | 1.78 | 1.30 | — | 1.34x | 1.84x | 1.10x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.06 | 1.25 | 1.01 | — | 1.31x | 1.64x | 1.06x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.36 | 4.63 | 2.38 | — | 1.12x | 2.19x | 0.99x | — |
| fused-add-rms-norm | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.92 | 1.05 | 0.92 | — | 2.23x | 2.54x | 1.00x | — |
| fused-add-rms-norm-escaping-residual | elementwise+rms_norm | (8,256) x (8,256) x (256) | 2.33 | 0.98 | 1.06 | 0.93 | — | 2.20x | 2.50x | 1.05x | — |
| rms-norm-cut-sweep-work | rms_norm | (512,4096) | 5.15 | 4.60 | 357.44 | 4.59 | — | 0.01x | 1.12x | 1.00x | — |
| rms-norm-packed-cells | rms_norm | (2048,128) | 2.33 | 1.24 | 1.34 | 1.43 | — | 1.74x | 1.63x | 0.87x | — |

Media geométrica vs eager: emmy fijado 0.80x · greedy 1.80x · greedy vs torch.compile 1.20x (greedy gana a torch.compile en 12/20). Reproducimos la referencia CloudRift (±5%) en 1/2 casos con referencia.

## nvfp4

_baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4-tma-w4a4-linear-ring | linear+index_map+reshape+elementwise+gather | (24,2048) | 31.97 | 5.60 | 4.13 | 2.30 | — | 7.74x | 13.88x | 2.43x | — |
| fp4-tma-w4a4-split-k | linear+index_map+reshape+elementwise+gather | (24,2048) | 31.96 | 5.60 | 2.83 | 2.30 | — | 11.31x | 13.92x | 2.44x | — |
| mma-fp4-encode-epilogue | elementwise+matmul | (128,256) x (256,128) | 41.62 | 4.89 | 1.79 | 1.40 | — | 23.26x | 29.82x | 3.50x | — |
| mma-fp4-packed-block-scales | elementwise+index_map+matmul | (16,32) x (32,128) x (16,8) | 92.15 | 2.38 | 2.11 | 1.43 | — | 43.61x | 64.31x | 1.66x | — |
| nvfp4-runtime-rows | elementwise+index_map+gather+reshape+linear | ({'sym': 'num_tokens', 'hint': 40},512) | 31.28 | 5.65 | 2.24 | 1.22 | — | 13.98x | 25.68x | 4.64x | — |
| nvfp4-unit-n | elementwise+index_map+gather+reshape+linear | (1,512) | 26.70 | 3.19 | 4.38 | 1.24 | — | 6.09x | 21.51x | 2.57x | — |
| nvfp4-w4a4-packed-slab-loopify | elementwise+index_map+gather+reshape+linear | (128,512) | 32.02 | 6.50 | 3.12 | 1.25 | — | 10.26x | 25.56x | 5.19x | — |
| fp4-tma-w4a4-two-channel-p2 | elementwise+index_map+gather+reshape+linear | (24,2048) | 366.98 | 61.07 | 5.14 | 2.90 | — | 71.35x | 126.66x | 21.08x | — |
| nvfp4-gate-up-requant-full-projection-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.97 | 12.38 | 496.21 | 60.67 | — | 0.35x | 2.83x | 0.20x | — |
| nvfp4-gate-up-requant-place-cut | linear+elementwise+reduce+reshape+index_map+gather | (128,512) | 171.98 | 12.38 | 655.98 | 60.73 | — | 0.26x | 2.83x | 0.20x | — |
| nvfp4-quantize-cut-shared-normalizer | elementwise+index_map+mean+reduce+reshape | (32,128) | 127.04 | 2.88 | 38.45 | 36.84 | — | 3.30x | 3.45x | 0.08x | — |
| nvfp4-w4a16-sibling-cut-async | elementwise+index_map+linear+reshape+gather | (48,256) | 38.94 | 6.67 | 12.76 | 7.09 | — | 3.05x | 5.49x | 0.94x | — |
| nvfp4-w4a16-sibling-cut-tma-p2 | elementwise+index_map+linear+reshape+gather | (48,256) | 38.94 | 6.65 | 13.51 | 7.08 | — | 2.88x | 5.50x | 0.94x | — |

Media geométrica vs eager: emmy fijado 6.05x · greedy 13.62x · greedy vs torch.compile 1.45x (greedy gana a torch.compile en 8/13). Reproducimos la referencia CloudRift (±5%) en 0/0 casos con referencia.

## qwen3emb

_las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512_

| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm-knorm-s128 | rms_norm | (1,8,128,128) x (128) | 1.71 | 1.41 | 1.35 | 1.10 | — | 1.27x | 1.55x | 1.28x | — |
| rmsnorm-knorm-s32 | rms_norm | (1,8,32,128) x (128) | 1.64 | 1.01 | 1.23 | 0.94 | — | 1.33x | 1.73x | 1.07x | — |
| rmsnorm-knorm-s512 | rms_norm | (1,8,512,128) x (128) | 3.29 | 2.42 | 2.70 | 1.59 | — | 1.21x | 2.06x | 1.52x | — |
| rmsnorm-layer-s128 | rms_norm | (1,128,1024) x (1024) | 2.14 | 1.32 | 1.46 | 1.44 | — | 1.47x | 1.48x | 0.92x | — |
| rmsnorm-layer-s32 | rms_norm | (1,32,1024) x (1024) | 1.86 | 1.50 | 1.16 | 1.17 | — | 1.60x | 1.59x | 1.29x | — |
| rmsnorm-layer-s512 | rms_norm | (1,512,1024) x (1024) | 2.58 | 1.87 | 1.88 | 1.89 | — | 1.37x | 1.36x | 0.99x | — |
| rmsnorm-qnorm-s128 | rms_norm | (1,16,128,128) x (128) | 2.39 | 1.43 | 1.78 | 1.30 | — | 1.34x | 1.84x | 1.10x | — |
| rmsnorm-qnorm-s32 | rms_norm | (1,16,32,128) x (128) | 1.65 | 1.06 | 1.25 | 1.01 | — | 1.31x | 1.64x | 1.06x | — |
| rmsnorm-qnorm-s512 | rms_norm | (1,16,512,128) x (128) | 5.20 | 2.36 | 4.63 | 2.38 | — | 1.12x | 2.19x | 0.99x | — |
| gated-mlp-s128 | matmul+elementwise | (1,128,1024) x (1024,3072) x (1024,3072) | 29.54 | 27.71 | 439.07 | 40.66 | — | 0.07x | 0.73x | 0.68x | — |
| gated-mlp-s32 | matmul+elementwise | (1,32,1024) x (1024,3072) x (1024,3072) | 27.97 | 20.80 | 117.15 | 117.46 | — | 0.24x | 0.24x | 0.18x | — |
| gated-mlp-s512 | matmul+elementwise | (1,512,1024) x (1024,3072) x (1024,3072) | 102.73 | 90.93 | 1738.40 | 120.02 | — | 0.06x | 0.86x | 0.76x | — |
| matmul-add-downproj-s128 | matmul+elementwise | (1,128,3072) x (3072,1024) x (1,128,1024) | 13.31 | 13.19 | 38.40 | 19.73 | 44.56 | 0.35x | 0.67x | 0.67x | 0.86 |
| matmul-add-downproj-s32 | matmul+elementwise | (1,32,3072) x (3072,1024) x (1,32,1024) | 11.40 | 11.22 | 33.72 | 14.90 | 39.57 | 0.34x | 0.77x | 0.75x | 0.85 |
| matmul-add-downproj-s512 | matmul+elementwise | (1,512,3072) x (3072,1024) x (1,512,1024) | 38.44 | 38.05 | 133.92 | 68.14 | 153.72 | 0.29x | 0.56x | 0.56x | 0.87 |
| matmul-add-oproj-s128 | matmul+elementwise | (1,128,2048) x (2048,1024) x (1,128,1024) | 10.98 | 10.86 | 25.35 | 13.86 | 30.06 | 0.43x | 0.79x | 0.78x | 0.84 |
| matmul-add-oproj-s32 | matmul+elementwise | (1,32,2048) x (2048,1024) x (1,32,1024) | 9.23 | 9.06 | 22.81 | 10.36 | 26.94 | 0.40x | 0.89x | 0.87x | 0.85 |
| matmul-add-oproj-s512 | matmul+elementwise | (1,512,2048) x (2048,1024) x (1,512,1024) | 27.13 | 26.36 | 88.58 | 46.11 | 102.91 | 0.31x | 0.59x | 0.57x | 0.86 |
| matmul-kvproj-s128 | matmul | (1,128,1024) x (1024,1024) | 13.16 | 10.12 | 13.86 | 11.78 | 15.69 | 0.95x | 1.12x | 0.86x | 0.88 |
| matmul-kvproj-s32 | matmul | (1,32,1024) x (1024,1024) | 6.54 | 6.26 | 12.04 | 5.99 | 13.77 | 0.54x | 1.09x | 1.05x | 0.87 |
| matmul-kvproj-s512 | matmul | (1,512,1024) x (1024,1024) | 13.85 | 13.63 | 46.04 | 23.95 | 52.62 | 0.30x | 0.58x | 0.57x | 0.87 |
| matmul-qproj-s128 | matmul | (1,128,1024) x (1024,2048) | 13.07 | 13.06 | 27.62 | 14.51 | 31.25 | 0.47x | 0.90x | 0.90x | 0.88 |
| matmul-qproj-s32 | matmul | (1,32,1024) x (1024,2048) | 8.40 | 8.38 | 12.36 | 8.03 | 14.20 | 0.68x | 1.05x | 1.04x | 0.87 |
| matmul-qproj-s512 | matmul | (1,512,1024) x (1024,2048) | 25.76 | 25.61 | 56.21 | 39.06 | 62.24 | 0.46x | 0.66x | 0.66x | 0.90 |
| sdpa-s128 | sdpa | (1,16,128,128) x (1,8,128,128) x (1,8,128,128) | 26.24 | 24.99 | 1443.90 | 1400.45 | — | 0.02x | 0.02x | 0.02x | — |
| sdpa-s32 | sdpa | (1,16,32,128) x (1,8,32,128) x (1,8,32,128) | 18.70 | 18.40 | 96.17 | 75.27 | — | 0.19x | 0.25x | 0.24x | — |
| sdpa-s512 | sdpa | (1,16,512,128) x (1,8,512,128) x (1,8,512,128) | 88.74 | 86.62 | 18517.66 | 6263.46 | — | 0.00x | 0.01x | 0.01x | — |

Media geométrica vs eager: emmy fijado 0.39x · greedy 0.70x · greedy vs torch.compile 0.59x (greedy gana a torch.compile en 8/27). Reproducimos la referencia CloudRift (±5%) en 12/12 casos con referencia.

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

