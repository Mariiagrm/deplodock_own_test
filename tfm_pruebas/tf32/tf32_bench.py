"""P5: fp32 frente a TF32 frente a 3xTF32 (emulado) en matmul reales.

- fp32  : torch.matmul con TF32 desactivado (por defecto en PyTorch): FMA fp32 en núcleos CUDA (cuBLAS SGEMM).
- tf32  : torch.backends.cuda.matmul.allow_tf32 = True: cuBLAS usa Tensor Cores con mantisa de 10 bits.
- 3xtf32: emulación de 3xTF32: a = a_hi + a_lo (a_hi = a truncado a TF32), igual con b, y
          C = a_hi·b_hi + a_hi·b_lo + a_lo·b_hi con tres matmul TF32. La partición se hace fuera del
          cronómetro (como si los pesos estuvieran preparados); el tiempo incluye las 3 GEMM y las 2 sumas.
          No es el kernel fusionado de CUTLASS: sirve para estimar el coste, no como implementación óptima.
Error: error relativo en norma de Frobenius, ||C - C_fp64|| / ||C_fp64||, en una muestra de filas.
Uso: python tf32_bench.py [tamaños...]
"""

import sys
import time

import torch


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e-3  # mediana en s


def to_tf32(x):
    # truncar los 13 bits bajos de la mantisa: deja 10 bits (TF32)
    return (x.view(torch.int32) & ~0x1FFF).view(torch.float32)


def main():
    sizes = [int(s) for s in sys.argv[1:]] or [1024, 2048, 4096, 8192]
    dev = torch.cuda.get_device_name(0)
    print(f"GPU {dev} · torch {torch.__version__} · cuBLAS allow_tf32 por defecto = {torch.backends.cuda.matmul.allow_tf32}"
          f" · float32_matmul_precision = {torch.get_float32_matmul_precision()}")
    print(f"{'N':>6} {'modo':>7} {'ms':>9} {'TFLOP/s':>9} {'vs fp32':>8} {'err rel (F)':>12}")
    torch.manual_seed(0)
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        rows = torch.arange(0, n, max(1, n // 64), device="cuda")
        ref = (a[rows].double() @ b.double())
        a_hi, b_hi = to_tf32(a), to_tf32(b)
        a_lo, b_lo = a - a_hi, b - b_hi
        flops = 2 * n**3
        res = {}

        def run(mode):
            if mode == "fp32":
                torch.backends.cuda.matmul.allow_tf32 = False
                return lambda: a @ b
            torch.backends.cuda.matmul.allow_tf32 = True
            if mode == "tf32":
                return lambda: a @ b
            return lambda: (a_hi @ b_hi) + (a_hi @ b_lo) + (a_lo @ b_hi)

        for mode in ("fp32", "tf32", "3xtf32"):
            fn = run(mode)
            t = bench(fn, iters=20 if n >= 8192 else 50)
            out = fn()
            err = ((out[rows].double() - ref).norm() / ref.norm()).item()
            res[mode] = t
            print(f"{n:>6} {mode:>7} {t * 1e3:>9.3f} {flops / t / 1e12:>9.1f} {res['fp32'] / t:>7.2f}x {err:>12.2e}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        del a, b, a_hi, b_hi, a_lo, b_lo
        torch.cuda.empty_cache()
        time.sleep(1)


if __name__ == "__main__":
    main()
