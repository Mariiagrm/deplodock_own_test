# tfm_pruebas — suite de emmy (subconjunto TFM) con logs explicativos

Herramientas propias (no forman parte de deplodock) para medir en pascal (RTX 5090, sm_120)
los casos del corpus de emmy que interesan al TFM, y explicar qué optimizaciones lleva cada kernel.

## Ficheros

| fichero | qué hace |
|---|---|
| `lanzar_pascal.sh` | Levanta el contenedor `cutlass-gemm` y ejecuta `correr_suite_tfm.py` dentro, con tu UID y un `HOME` nuevo (base de autotune vacía, como `make bench-kernels`). |
| `correr_suite_tfm.py` | Selecciona los casos, los mide como la suite (`emmy run --golden … --bench --json`), guarda el volcado del compilador y escribe logs + tablas. |
| `tecnicas.py` | Catálogo de técnicas (tu lista de 11, más 2 extra) con su explicación y el detector que las busca en lo que emmy ha generado. |
| `runs/<fecha>/` | Una carpeta por ejecución (ver abajo). |

## Cómo lanzar

```bash
# todo (≈130 casos, ~10 min)
sbatch -p pascal-benchmark --exclusive -J emmy-tfm -o ~/deplodock/tfm_pruebas/slurm_%j.log \
       --wrap 'bash ~/deplodock/tfm_pruebas/lanzar_pascal.sh --workers 4'
# solo una categoría o unos casos
bash ~/deplodock/tfm_pruebas/lanzar_pascal.sh --cat qwen3emb
bash ~/deplodock/tfm_pruebas/lanzar_pascal.sh --solo raster-gm8,splitk
# regenerar logs/tablas de una ejecución sin volver a medir (tras cambiar tecnicas.py)
bash ~/deplodock/tfm_pruebas/lanzar_pascal.sh --reanalizar tfm_pruebas/runs/<fecha>
```

Categorías (`--cat`): `matmul`, `rmsnorm/fused`, `nvfp4`, `qwen3emb`. Un caso puede estar en varias.
En el corpus **no hay MMA fp8 e4m3 pura**: la categoría de baja precisión es NVFP4 (valores fp4 e2m1
con escalas por bloque fp8 e4m3, instrucción `mma.m16n8k64.e2m1`).

`--ncu` añade ocupación, conflictos de banco, % DRAM/FMA por kernel, pero hoy **falla con
`ERR_NVGPUCTRPERM`**: el contenedor no tiene permiso sobre los contadores de la GPU (haría falta
`cap_add: [SYS_ADMIN]` en el compose, o que el admin ponga `NVreg_RestrictProfilingToAdminUsers=0`).

## Qué sale en `runs/<fecha>/`

- `resumen.md`: una tabla por categoría (eager / torch.compile / emmy fijado / emmy greedy / referencia
  CloudRift) con medias geométricas, y la **matriz de técnicas** (✓ · –) por caso.
- `resumen.csv`, `resultados.json`: lo mismo, para gráficas.
- `logs/<familia>__<caso>.log`: el log explicativo de cada caso (tiempos, schedule, y por cada kernel
  las 13 técnicas con `[SI]/[NO]/[N/A]`, **qué se ha hecho** con números concretos, la **evidencia**
  y **qué supone**).
- `casos/<caso>/`: `bench_N.json` (registro de `emmy run`), `bench_N.stdout.txt` y `dump/` (IR de cada
  pasada del compilador + CUDA final en `dump/10_lowering_cuda.kernels/<kernel>.txt`).

### Las dos columnas de emmy

- **emmy (fijado)**: el schedule que fija el caso, el que mide `make bench-kernels` y para el que hay
  referencia CloudRift en una RTX 5090. Sirve para comprobar que pascal reproduce sus números.
- **greedy**: lo que emmy elige solo. Muchos casos del corpus son *de cobertura* (fijan un schedule raro
  para probar un camino del compilador), así que para comparar con PyTorch la columna justa es greedy.
- El análisis de técnicas es del CUDA del schedule **fijado**; del greedy solo se listan los parámetros.

## Cómo leer los parámetros (knobs) del schedule

Gramática en `emmy/compiler/ir/schedule/choices.py`; tutorial en `docs/docs/tutorials/03-forks-and-knobs.md`.

| knob | ejemplo | significado | técnica |
|---|---|---|---|
| `WORK` | `w2x2` / `w2x2+p1` / `t16x8` / `t128` | 2×2 warps por bloque / + 1 warp productor / 16×8 hilos (n×m) / 128 hilos cooperando | factorización, especialización de warps |
| `TILE` | `mma_m16n8k16_f16_f32/f4x8/k2` | átomo Tensor Core m16n8k16 (f16→f32), cada warp 4×8 átomos, BK = 2×16 | Tensor Cores, register tiling |
| `TILE` | `f2x2` / `f4` | camino escalar: cada hilo 2×2 (n×m) / 4 celdas en registros | register tiling |
| `STAGE` | `d2/smem-tma/p2` | 2 etapas global→smem por TMA + ping-pong de 2 fragmentos smem→registros | staging, TMA/cp.async, pipelining |
| | `smem` / `smem-async` / `smem-tma` / vacío | copia síncrona / cp.async / TMA / directo de global | |
| `REDUCE` | `g2k` / `g4a` | split-K en 2 con kernel de combinación / en 4 con atomicAdd | split-K |
| | `coop` / `r4` | reducción cooperativa entre hilos (shuffle + árbol) / 4 acumuladores ILP | reducción cooperativa |
| `RASTER` | `gm8` | 8 baldosas de M seguidas (GROUP_M=8) | CTA swizzle |

El split-K con kernel de combinación se hace además como **corte del grafo** (regla `030_cut`): el caso
pasa a tener un kernel `…__partial` y otro que suma los parciales; el log lo indica.

## Cómo se detecta cada técnica (tecnicas.py)

Siempre mirando el **cuerpo del `__global__`** (el preámbulo trae todas las funciones auxiliares, se usen o no):

| técnica | evidencia |
|---|---|
| Factorización / register tiling | `WORK` + `TILE` → BM×BN×BK, warps, FM×FN, acumuladores por hilo; reutilización = FM·FN/(FM+FN) |
| Vectorización e intercalado | `float4/uint4` (128 bits), `float2/uint2` (64 bits), `emmy_cp_async_cg` (16 B); reglas `050_vectorize_loads`, `080_vectorize_stores`, `095_interleave_loads` con diff |
| Tensor Cores | llamadas `emmy_mma_m16n8k16_*`, `emmy_ldmatrix_*`, `emmy_mma_load_*_gmem` |
| Staging en smem | arrays `__shared__ _a_smem/_b_smem/…` o pool dinámico `_smem_pool` (la smem de reducciones, `acc0_smem`, no cuenta) |
| cp.async / TMA | `emmy_cp_async_cg/ca` + commit/wait; `cp_async_bulk_tensor_Nd` + `mbarrier_*` |
| Pipelining | `STAGE` `d≥2` (anillo global→smem) y/o `p≥2` (ping-pong smem→registros) |
| Especialización de warps | `WORK=…+p<n>`, `setmaxnreg` |
| Padding / swizzle | `emmy_swizzle_b64/b128`; relleno = tamaño del array que no es potencia de 2 (filas × (2^k + p)) |
| CTA swizzle | `RASTER=gm<G>/gn<G>` |
| Split-K | regla `030_cut` (Fold[0..K] → parcial Fold[0..K/n]), kernel `__partial`, `atomicAdd` |
| Reducción cooperativa (extra) | `REDUCE=coop`, `__shfl_xor_sync` |
| PDL (extra) | `griddepcontrol.wait` |
