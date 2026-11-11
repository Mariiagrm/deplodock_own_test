"""Catálogo de técnicas de optimización y detectores sobre lo que emmy ha generado.

Para cada kernel medido se cruzan tres fuentes:
  1. Los parámetros (knobs) del schedule: WORK, TILE, STAGE, REDUCE, RASTER (gramática en
     emmy/compiler/ir/schedule/choices.py).
  2. El CUDA generado (dump/10_lowering_cuda.kernels/<kernel>.txt), solo el cuerpo del
     __global__: el preámbulo trae TODAS las funciones auxiliares, se usen o no.
  3. Las reglas de la bajada a kernel que se han disparado (dump/09_lowering_kernel__*.rules.txt):
     una regla "dispara" si su fichero contiene un diff (líneas "@@ -a,b +c,d @@").

Cada detector devuelve un Resultado con estado SI / NO / N/A, qué se ha hecho en ese kernel
(con números concretos), la evidencia y la explicación de qué supone la técnica.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Parseo de knobs (inverso de los .spell() de emmy)
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(r"mma_m(\d+)n(\d+)k(\d+)_([a-z0-9]+)_([a-z0-9]+)")


@dataclass
class Schedule:
    work_kind: str = "direct"  # direct | warp | thread
    units: tuple[int, int] = (1, 1)  # orden nativo del codec: warp (WM, WN) / thread (N, M)
    producer: int = 0
    atom: str = ""  # p. ej. mma_m16n8k16_f16_f32
    atom_mnk: tuple[int, int, int] = (1, 1, 1)
    atom_in: str = ""
    atom_acc: str = ""
    regs: tuple[int, int] = (1, 1)  # warp (FM, FN) / escalar (fm, fn)
    bk: int = 1
    has_tile: bool = False
    stage_depth: int = 0  # 0 = directo (sin intermedio)
    transport: str = "direct"  # direct | reg | smem | smem-async | smem-tma
    reg_depth: int = 1
    split_k: int = 1
    split_finalize: str = ""  # atomic | kernel
    coop: bool = False
    coop_t: bool = False
    reduce_reg: int = 1
    raster: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_warp(self) -> bool:
        return bool(self.atom)

    @property
    def warps(self) -> int:
        return self.units[0] * self.units[1] if self.work_kind == "warp" else 0

    def block_tile(self) -> tuple[int, int, int] | None:
        """(BM, BN, BK) del bloque, o None si no hay contracción en baldosas."""
        if not self.has_tile:
            return None
        if self.is_warp:
            am, an, ak = self.atom_mnk
            wm, wn = self.units
            fm, fn = self.regs
            return wm * fm * am, wn * fn * an, self.bk * ak
        # escalar: WORK t<N>x<M> (n primero), TILE f<fn>[x<fm>]
        n_thr, m_thr = self.units if self.work_kind == "thread" else (1, 1)
        fm, fn = self.regs
        return m_thr * fm, n_thr * fn, 0


def parse_schedule(knobs: dict) -> Schedule:
    s = Schedule(raw=dict(knobs))
    work = (knobs.get("WORK") or "").strip()
    if work:
        base, _, band = work.partition("+")
        if band.startswith("p") and band[1:].isdigit():
            s.producer = int(band[1:])
        dims = [int(x) for x in base[1:].split("x") if x.isdigit()]
        s.work_kind = {"w": "warp", "t": "thread"}.get(base[:1], "direct")
        s.units = (dims[0], dims[1] if len(dims) > 1 else 1) if dims else (1, 1)
    tile = (knobs.get("TILE") or "").strip()
    if tile:
        s.has_tile = True
        toks = tile.split("/")
        m = _ATOM_RE.fullmatch(toks[0])
        if m:
            s.atom, s.atom_mnk = toks[0], (int(m[1]), int(m[2]), int(m[3]))
            s.atom_in, s.atom_acc = m[4], m[5]
            toks = toks[1:]
        for t in toks:
            if t.startswith("f") and t[1:2].isdigit():
                d = [int(x) for x in t[1:].split("x")]
                a, b = d[0], (d[1] if len(d) > 1 else 1)
                s.regs = (a, b) if s.atom else (b, a)  # warp f<FM>x<FN> / escalar f<fn>[x<fm>]
            elif t.startswith("k") and t[1:].isdigit():
                s.bk = int(t[1:])
    stage = (knobs.get("STAGE") or "").strip()
    if stage:
        s.stage_depth, s.transport = 1, "smem"
        for t in stage.split("/"):
            if t in ("reg", "smem", "smem-async", "smem-tma"):
                s.transport = t
            elif t.startswith("d") and t[1:].isdigit():
                s.stage_depth = int(t[1:])
            elif t.startswith("p") and t[1:].isdigit():
                s.reg_depth = int(t[1:])
    for t in [x for x in (knobs.get("REDUCE") or "").split("/") if x]:
        if t.startswith("g") and t[1:2].isdigit():
            body = t[1:]
            s.split_finalize = "atomic" if body.endswith("a") else "kernel"
            s.split_k = int(body.rstrip("ak"))
        elif t in ("coop", "coop-t"):
            s.coop, s.coop_t = True, t.endswith("-t")
        elif t.startswith("r") and t[1:].isdigit():
            s.reduce_reg = int(t[1:])
    s.raster = (knobs.get("RASTER") or "").strip()
    return s


# ---------------------------------------------------------------------------
# Lectura del volcado (CUDA generado + reglas)
# ---------------------------------------------------------------------------

_TYPE_BYTES = {
    "__half": 2, "half": 2, "__nv_bfloat16": 2, "float": 4, "int": 4, "unsigned": 4, "unsigned int": 4,
    "uint32_t": 4, "unsigned long long": 8, "uint64_t": 8, "double": 8, "unsigned char": 1, "uint8_t": 1,
    "char": 1, "signed char": 1, "int8_t": 1, "__nv_fp8_e4m3": 1, "unsigned short": 2, "uint16_t": 2,
}
_SMEM_RE = re.compile(r"__shared__\s+(?:__align__\(\d+\)\s+)?((?:unsigned\s+long\s+long|unsigned\s+\w+|[\w:]+))\s+(\w+)((?:\[\d+\])+)")


@dataclass
class SmemArray:
    name: str
    ctype: str
    dims: tuple[int, ...]

    @property
    def elems(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.elems * _TYPE_BYTES.get(self.ctype, 4)


@dataclass
class KernelSource:
    name: str
    body: str = ""
    found: bool = False
    launch_bounds: int | None = None
    smem: list[SmemArray] = field(default_factory=list)
    pool_bytes: int = 0  # extern __shared__ _smem_pool[] (smem dinámica troceada por el compilador)

    def count(self, pattern: str) -> int:
        return len(re.findall(pattern, self.body))

    @property
    def operand_smem(self) -> list[SmemArray]:
        """Arrays de staging de operandos (_a_smem, _b_smem, _as_smem...), no el espacio de reducciones."""
        return [a for a in self.smem if a.name.startswith("_") and a.name.endswith("_smem")]

    @property
    def scratch_smem(self) -> list[SmemArray]:
        """smem auxiliar: parciales de una reducción (acc0_smem), estadísticos de una norma, mbarriers."""
        return [a for a in self.smem if a not in self.operand_smem and "mbar" not in a.name]


def read_kernel_source(dump: Path, kernel: str) -> KernelSource:
    ks = KernelSource(name=kernel)
    path = dump / "10_lowering_cuda.kernels" / f"{kernel}.txt"
    if not path.exists():
        return ks
    text = path.read_text(errors="replace")
    idx = text.find('extern "C" __global__')
    ks.body = text[idx:] if idx >= 0 else text
    ks.found = True
    m = re.search(r"__launch_bounds__\((\d+)", ks.body)
    ks.launch_bounds = int(m[1]) if m else None
    for m in _SMEM_RE.finditer(ks.body):
        ctype = " ".join(m[1].split())
        dims = tuple(int(x) for x in re.findall(r"\[(\d+)\]", m[3]))
        ks.smem.append(SmemArray(m[2], ctype, dims))
    m = re.search(r"extern __shared__[^;]*_smem_pool\[\];\s*//\s*(\d+)\s*bytes", ks.body)
    if m:
        ks.pool_bytes = int(m[1])
    return ks


@dataclass
class Corte:
    """Lo que hizo la regla 030_cut (pasada tile_cut): partir una operación en varios kernels."""

    split_k: int = 1
    finalize: str = ""  # kernel | atomic
    place: bool = False  # corte de colocación (__place_): una operación fusionada se reparte en varios kernels


def read_cut(dump: Path) -> Corte:
    c = Corte()
    f = dump / "07_tile_cut__030_cut.rules.txt"
    if not f.exists():
        return c
    text = f.read_text(errors="replace")
    if not re.search(r"^@@ -\d", text, re.M):
        return c
    antes = re.findall(r"^-\s*Fold\[\w+ in 0\.\.(\d+)\] contraction", text, re.M)
    despues = re.findall(r"^\+\s*Fold\[\w+ in 0\.\.(\d+) ⊂ \w+\] contraction", text, re.M)
    if antes and despues and int(despues[0]) and int(antes[0]) > int(despues[0]):
        c.split_k = int(antes[0]) // int(despues[0])
        c.finalize = "atomic" if re.search(r"atomic", text, re.I) and "__partial" not in text else "kernel"
    if "__place" in text:
        c.place = True
    return c


def read_rules(dump: Path) -> tuple[list[str], list[str]]:
    """(reglas que cambiaron el kernel, reglas que coincidieron sin cambiar nada) en la bajada a kernel."""
    fired, noop = [], []
    for f in sorted(dump.glob("09_lowering_kernel__*.rules.txt")):
        rule = f.name.split("__", 1)[1].removesuffix(".rules.txt")
        text = f.read_text(errors="replace")
        (fired if re.search(r"^@@ -\d", text, re.M) else noop).append(rule)
    return fired, noop


# ---------------------------------------------------------------------------
# Resultados y catálogo
# ---------------------------------------------------------------------------

SI, NO, NA = "SI", "NO", "N/A"


@dataclass
class Resultado:
    estado: str
    hecho: str
    evidencia: list[str] = field(default_factory=list)


@dataclass
class Contexto:
    sched: Schedule
    src: KernelSource
    fired: list[str]
    noop: list[str]
    smem_bytes: int | None
    ncu: dict | None  # métricas ncu de este kernel, si se pidieron
    corte: Corte = field(default_factory=Corte)
    op: str = ""  # operaciones del caso (p. ej. "rms_norm+linear")
    dtype: str = ""  # dtypes de las entradas del caso (p. ej. "f32", "f16/f32")


@dataclass
class Tecnica:
    clave: str
    seccion: str
    nombre: str
    abrev: str  # columna corta de la matriz del resumen
    supone: str  # qué supone / por qué mejora
    detectar: object  # Callable[[Contexto], Resultado]


def _ncu(ctx: Contexto, metric: str) -> float | None:
    return None if not ctx.ncu else ctx.ncu.get(metric)


# --- 1. Intensidad aritmética y cómputo ------------------------------------


def _factorizacion(ctx: Contexto) -> Resultado:
    s = ctx.sched
    bt = s.block_tile()
    if not s.has_tile:
        if s.work_kind == "thread":
            return Resultado(NA, f"Kernel sin contracción en baldosas (p. ej. RMSNorm/elementwise): jerarquía bloque → {s.units[0]} hilos que cooperan, sin nivel de registro ni Tensor Core.", [f"WORK={s.raw.get('WORK')}", "TILE vacío"])
        return Resultado(NA, "Sin baldosas: un hilo por celda de salida, sin factorizar el bucle (contracción por celda o elementwise).", [f"WORK={s.raw.get('WORK', '')!r}", "TILE vacío"])
    if s.is_warp:
        bm, bn, bk = bt
        am, an, ak = s.atom_mnk
        wm, wn = s.units
        fm, fn = s.regs
        acc = fm * fn * am * an // 32
        hecho = (
            f"Bloque {bm}×{bn} (BK={bk}) → {wm}×{wn} warps ({wm * wn * 32} hilos) → cada warp {fm}×{fn} átomos "
            f"de {am}×{an}×{ak} → cada hilo {acc} acumuladores {s.atom_acc} en registros → átomo Tensor Core m{am}n{an}k{ak}."
        )
        return Resultado(SI, hecho, [f"WORK={s.raw.get('WORK')}", f"TILE={s.raw.get('TILE')}"])
    fm, fn = s.regs
    if s.work_kind == "thread":
        bm, bn, _ = bt
        n_thr, m_thr = s.units
        hecho = f"Bloque {bm}×{bn} → {m_thr}×{n_thr} hilos ({m_thr * n_thr}) → cada hilo {fm}×{fn} celdas en registros (núcleos CUDA, FMA escalar)."
    else:  # WORK vacío: geometría derivada por el compilador, la leemos de __launch_bounds__
        thr = ctx.src.launch_bounds or 0
        hecho = f"Bloque de {thr or '?'} hilos (geometría derivada, WORK vacío) → cada hilo {fm}×{fn} celdas en registros (núcleos CUDA, FMA escalar); bloque = {thr * fm * fn if thr else '?'} celdas de salida."
    return Resultado(SI, hecho, [f"WORK={s.raw.get('WORK', '')!r}", f"TILE={s.raw.get('TILE')}"] + ([f"__launch_bounds__({ctx.src.launch_bounds})"] if ctx.src.launch_bounds else []))


def _register_tiling(ctx: Contexto) -> Resultado:
    s = ctx.sched
    if not s.has_tile:
        return Resultado(NA, "Sin contracción: no hay producto exterior que reutilizar en registros.")
    fm, fn = s.regs
    if fm * fn == 1:
        if s.is_warp:
            return Resultado(NO, "Cada warp calcula un solo átomo (f1x1): cada fragmento A/B cargado se usa en una única MMA.", [f"TILE={s.raw.get('TILE')}"])
        return Resultado(NO, "Una celda por hilo (f1): cada carga alimenta una sola FMA (kernel limitado por cargas).", [f"TILE={s.raw.get('TILE')}"])
    reuse = fm * fn / (fm + fn)
    unidad = "MMA por fragmento cargado" if s.is_warp else "FMA por valor cargado"
    hecho = f"Sub-rejilla {fm}×{fn} por {'warp' if s.is_warp else 'hilo'}: {fm}+{fn} cargas alimentan {fm * fn} productos → {reuse:.2f} {unidad} (sin tiling: 0.50)."
    ev = [f"TILE={s.raw.get('TILE')}"]
    if s.is_warp:
        ev.append(f"{ctx.src.count(r'emmy_mma_m\d+n\d+k\d+_\w+\(')} llamadas MMA en el cuerpo del kernel")
    return Resultado(SI, hecho, ev)


_VEC128 = r"\b(?:float4|uint4|int4|make_float4|make_uint4|longlong2|double2)\b"
_VEC64 = r"\b(?:float2|uint2|int2|make_float2|make_uint2|__half2_raw2|ulonglong1)\b|reinterpret_cast<(?:unsigned long long|uint64_t)\s*\*>"


def _vectorizacion(ctx: Contexto) -> Resultado:
    v128 = ctx.src.count(_VEC128)
    v64 = ctx.src.count(_VEC64)
    cpa16 = ctx.src.count(r"emmy_cp_async_cg\w*\(")  # cp.async.cg = 16 B por copia
    reglas = [r for r in ctx.fired if r.endswith(("vectorize_loads", "vectorize_stores", "interleave_loads"))]
    interleave = any(r.endswith("interleave_loads") for r in reglas)
    tma = ctx.src.count(r"cp_async_bulk_tensor_\dd\(")
    ldm = ctx.src.count(r"emmy_ldmatrix_\w+\(")
    partes = []
    if v128 or v64:
        partes.append(f"{v128} accesos de 128 bits (float4/uint4, LDG/STG.128) y {v64} de 64 bits (float2/uint2) en el cuerpo")
    if cpa16:
        partes.append(f"{cpa16} cp.async.cg de 16 B (cada hilo copia 128 bits por instrucción)")
    if interleave:
        partes.append("regla interleave_loads: cargas recolocadas junto a su primer uso")
    ev = [f"regla disparada: {r}" for r in reglas] or ["reglas vectorize_*/interleave_loads: sin cambios en este kernel"]
    extra = []
    if tma:
        extra.append(f"además las cargas global→smem van por TMA ({tma} copias de baldosa entera)")
    if ldm:
        extra.append(f"{ldm} ldmatrix.x4 (4 matrices 8×8 de smem a registros por instrucción)")
    if not partes:
        if extra:
            return Resultado(NO, "Sin LDG/STG vectorizados por hilo: " + "; ".join(extra) + ", así que el ancho de la transacción ya lo da el hardware de copia.", ev)
        return Resultado(NO, "Accesos escalares a memoria global (sin LDG.64/LDG.128).", ev)
    return Resultado(SI, "; ".join(partes + extra) + ".", ev)


def _tensor_cores(ctx: Contexto) -> Resultado:
    s = ctx.sched
    mma = ctx.src.count(r"emmy_mma_m\d+n\d+k\d+_\w+\(")
    wg = ctx.src.count(r"wgmma")
    ldm = ctx.src.count(r"emmy_ldmatrix_\w+\(")
    gload = ctx.src.count(r"emmy_mma_load_[ab]_gmem\w*\(")
    if not s.is_warp and not mma:
        if ctx.dtype and all(d == "f32" for d in ctx.dtype.split("/")):
            return Resultado(NA, "Entradas fp32: los Tensor Cores no tienen modo fp32 (solo fp16/bf16/TF32/FP8/FP4), así que se calcula con FMA en núcleos CUDA, igual que PyTorch con su opción por defecto. "
                             "La alternativa sería TF32 (mantisa de 10 bits, pierde precisión), que emmy no ofrece y que en la RTX 5090 rinde lo mismo que fp32 (~87 TFLOP/s a 2 GHz).",
                             [f"dtype de entrada {ctx.dtype}", f"TILE={s.raw.get('TILE', '')!r}"])
        return Resultado(NO, "Camino escalar: FMA en núcleos CUDA, sin mma.sync.", [f"TILE={s.raw.get('TILE', '')!r}"])
    am, an, ak = s.atom_mnk
    hecho = f"mma.sync m{am}n{an}k{ak} ({s.atom_in} → acumulador {s.atom_acc}) emitida como PTX: {mma} instrucciones MMA por iteración desenrollada"
    if ldm:
        hecho += f"; operandos a registros con {ldm} ldmatrix"
    if gload:
        hecho += f"; {gload} fragmentos cargados directamente de global (sin ldmatrix)"
    ev = [f"átomo {s.atom}"]
    if wg:
        ev.append("wgmma presente")
    if s.atom_in in ("e2m1", "e4m3", "e5m2"):
        ev.append(f"baja precisión: entrada {s.atom_in} (k{ak} por instrucción)")
    return Resultado(SI, hecho + ".", ev)


# --- 2. Flujo de datos y solapamiento asíncrono -----------------------------


def _smem_staging(ctx: Contexto) -> Resultado:
    s = ctx.sched
    arrays = ctx.src.operand_smem
    scratch = ctx.src.scratch_smem
    if s.transport in ("direct", "reg") and not arrays and not ctx.src.pool_bytes:
        why = "operandos directos de global a registros" if s.transport == "direct" else "valor retenido en registros (d1/reg)"
        if scratch:
            why += f"; la smem que hay ({', '.join(f'{a.name}{list(a.dims)}' for a in scratch)}) es espacio de trabajo de la reducción entre warps, no staging de operandos"
        return Resultado(NO, f"Sin staging en memoria compartida: {why}.", [f"STAGE={s.raw.get('STAGE', '')!r}"])
    total = sum(a.nbytes for a in arrays) + ctx.src.pool_bytes
    desc = ", ".join(f"{a.name}{list(a.dims)} {a.ctype} ({a.nbytes / 1024:.1f} KB)" for a in arrays)
    if ctx.src.pool_bytes:
        desc = (desc + ", " if desc else "") + f"pool dinámico _smem_pool de {ctx.src.pool_bytes / 1024:.1f} KB troceado por el compilador"
    bt = s.block_tile()
    hecho = f"Baldosas cargadas cooperativamente en smem: {desc or '—'}; total {total / 1024:.1f} KB"
    if ctx.smem_bytes:
        hecho += f" (el lanzamiento reserva {ctx.smem_bytes / 1024:.1f} KB)"
    if bt and bt[2]:
        hecho += f". Cada elemento de A se reutiliza en {bt[1]} columnas y cada uno de B en {bt[0]} filas del bloque, en vez de leerlo de global cada vez"
    return Resultado(SI, hecho + ".", [f"STAGE={s.raw.get('STAGE')}"])


def _cp_async_tma(ctx: Contexto) -> Resultado:
    s = ctx.sched
    tma = re.findall(r"cp_async_bulk_tensor_(\d)d\(", ctx.src.body)
    cg = ctx.src.count(r"emmy_cp_async_cg\w*\(")
    ca = ctx.src.count(r"emmy_cp_async_ca\w*\(")
    mbar = ctx.src.count(r"\bmbarrier_\w+\(")
    commit = ctx.src.count(r"emmy_cp_async_commit\(")
    wait = ctx.src.count(r"emmy_cp_async_wait\w*\(")
    if tma:
        dims = sorted(set(tma))
        return Resultado(SI, f"TMA: {len(tma)} copias de baldosa {'/'.join(d + 'D' for d in dims)} global→smem emitidas por un solo hilo, sin pasar por registros; sincronizadas con {mbar} operaciones mbarrier (arrive_expect_tx / try_wait.parity).", [f"STAGE={s.raw.get('STAGE')}", "descriptores CUtensorMap como argumentos del kernel"])
    if cg or ca or s.transport == "smem-async":
        tipo = []
        if cg:
            tipo.append(f"{cg} cp.async.cg (16 B, solo L2)")
        if ca:
            tipo.append(f"{ca} cp.async.ca (pasa por L1)")
        return Resultado(SI, f"cp.async (Ampere+): {' + '.join(tipo) or 'copias asíncronas'} por hilo en el código (dentro de bucles desenrollados, cada una se repite por fila de la baldosa), agrupadas con {commit} commit_group y {wait} wait_group.", [f"STAGE={s.raw.get('STAGE')}"])
    if s.transport == "smem":
        return Resultado(NO, "Staging síncrono: los hilos cargan a registros y escriben a smem (LDG + STS), seguido de __syncthreads.", [f"STAGE={s.raw.get('STAGE')}"])
    return Resultado(NA if s.transport in ("direct", "reg") else NO, "Sin copias asíncronas (no hay staging en smem).", [f"STAGE={s.raw.get('STAGE', '')!r}"])


def _pipelining(ctx: Contexto) -> Resultado:
    s = ctx.sched
    if s.transport in ("direct", "reg"):
        return Resultado(NA, "Sin staging en smem: no hay baldosas en vuelo que solapar.", [f"STAGE={s.raw.get('STAGE', '')!r}"])
    partes = []
    if s.stage_depth >= 2:
        partes.append(f"anillo de {s.stage_depth} etapas global→smem: prólogo carga {s.stage_depth - 1} baldosa(s) por adelantado y el bucle principal calcula la baldosa k mientras llega la k+{s.stage_depth - 1}")
    if s.reg_depth >= 2:
        partes.append(f"ping-pong de {s.reg_depth} fragmentos smem→registros dentro de cada baldosa (los ldmatrix del paso siguiente se emiten antes de las MMA del actual)")
    if not partes:
        return Resultado(NO, "Un solo búfer (d1): carga → __syncthreads → cálculo, sin solapamiento.", [f"STAGE={s.raw.get('STAGE')}"])
    return Resultado(SI, "; ".join(partes) + ".", [f"STAGE={s.raw.get('STAGE')}"])


def _warp_spec(ctx: Contexto) -> Resultado:
    s = ctx.sched
    setmax = ctx.src.count(r"setmaxnreg")
    if s.producer:
        hecho = f"{s.producer} warp(s) productor(es) dedicados a emitir TMA + {s.warps} warps consumidores que solo hacen MMA, comunicados por un anillo de mbarriers \"lleno/vacío\""
        if setmax:
            hecho += f"; setmaxnreg redistribuye registros entre ambos grupos ({setmax} apariciones)"
        return Resultado(SI, hecho + ".", [f"WORK={s.raw.get('WORK')} (+p{s.producer})"])
    if not s.is_warp:
        return Resultado(NA, "Camino escalar: no aplica.")
    return Resultado(NO, "Todos los warps hacen carga y cálculo (pipeline uniforme SIMT).", [f"WORK={s.raw.get('WORK')}"])


# --- 3. Conflictos de banco -------------------------------------------------


def _padding_de(a: SmemArray, depth: int) -> str | None:
    """Detecta relleno de filas: una baldosa sin relleno ocupa una potencia de 2 elementos; con +p por fila
    queda filas × (2^k + p). Solo para tipos de 2/4 B (los empaquetados fp4/fp8 tienen otra aritmética)."""
    if _TYPE_BYTES.get(a.ctype, 4) not in (2, 4) or a.elems % depth:
        return None
    per = a.elems // depth
    if per & (per - 1) == 0:
        return None
    rows = per & -per  # mayor potencia de 2 que divide
    cols = per // rows
    base = 1 << (cols.bit_length() - 1)
    pad = cols - base
    if pad and pad <= max(8, base // 4):
        return f"relleno en {a.name}: {depth} etapa(s) × {rows} filas × ({base} + {pad} de relleno) = {a.elems} elementos (+{100 * pad / base:.1f}%)"
    return None


def _padding_swizzle(ctx: Contexto) -> Resultado:
    s = ctx.sched
    arrays = ctx.src.operand_smem
    if not arrays and not ctx.src.pool_bytes:
        if ctx.src.scratch_smem:
            return Resultado(NA, "Solo smem de trabajo para la reducción (pocos elementos, un acceso por warp): no hay baldosas de operandos que proteger.")
        return Resultado(NA, "Sin memoria compartida de operandos: no hay bancos que proteger.")
    swz = sorted(set(re.findall(r"emmy_swizzle_(b\d+)\(", ctx.src.body)))
    partes, ev = [], []
    if swz:
        partes.append(f"swizzle XOR de direcciones ({', '.join(m.upper() for m in swz)}) en las lecturas ldmatrix" + (", el mismo patrón con el que el TMA escribe la baldosa" if s.transport == "smem-tma" else ""))
        ev.append(f"{ctx.src.count(r'emmy_swizzle_b\d+\(')} llamadas emmy_swizzle_* en el cuerpo")
    d = max(s.stage_depth, 1)
    for a in arrays:
        p = _padding_de(a, d)
        if p:
            partes.append(p)
    if ctx.src.pool_bytes and not partes:
        ev.append("smem en pool dinámico: la disposición no se ve en las declaraciones")
    conf = _ncu(ctx, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    if conf is not None:
        ev.append(f"ncu: {int(conf):,} conflictos de banco en lecturas de smem")
    if not partes:
        return Resultado(NO, "Baldosas en smem sin relleno ni swizzle detectables: posibles accesos serializados (con --ncu se ve el número real de conflictos).", ev)
    return Resultado(SI, "; ".join(partes) + ".", ev)


# --- 4. Planificación de la rejilla -----------------------------------------


def _cta_swizzle(ctx: Contexto) -> Resultado:
    s = ctx.sched
    if s.raster:
        eje = "M" if s.raster.startswith("gm") else "N"
        g = s.raster[2:]
        return Resultado(SI, f"Rasterizado agrupado {s.raster}: {g} baldosas de {eje} consecutivas se lanzan juntas, de modo que comparten en L2 la franja del otro operando (equivalente a GROUP_M={g} en Triton).", [f"RASTER={s.raster}"])
    if not s.has_tile:
        return Resultado(NA, "Rejilla 1-D sin baldosas 2-D: el orden de los bloques no cambia la reutilización.")
    return Resultado(NO, "Orden plano por filas (N más rápido).", ["RASTER vacío"])


def _split_k(ctx: Contexto) -> Resultado:
    s = ctx.sched
    atom = ctx.src.count(r"\batomicAdd\b|red\.global\.add")
    c = ctx.corte
    if c.split_k > 1:
        # emmy realiza el split-K como reescritura del grafo (regla 030_cut), no como knob REDUCE del kernel
        if ctx.src.name.endswith("__partial"):
            return Resultado(SI, f"Este es el kernel de PARCIALES: K se parte en {c.split_k} trozos, cada bloque reduce 1/{c.split_k} de K y escribe su parcial a un workspace; otro kernel los suma después (sin atómicos, determinista).", ["regla 030_cut (pasada tile_cut): Fold[0..K] → parcial Fold[0..K/n] + reducción", f"kernel {ctx.src.name}"])
        if atom:
            return Resultado(SI, f"K partido en {c.split_k}: cada bloque suma su parcial en la salida con atomicAdd ({atom} atómicos en el código, un solo kernel).", ["regla 030_cut"])
        if not s.has_tile:
            return Resultado(SI, f"Este es el kernel de COMBINACIÓN del split-K: suma los {c.split_k} parciales del workspace.", ["regla 030_cut", f"kernel {ctx.src.name}"])
    if atom and s.has_tile:
        return Resultado(SI, f"La contracción acumula en la salida con atomicAdd ({atom} atómicos): split-K con finalización atómica.", [f"REDUCE={s.raw.get('REDUCE', '')!r}"])
    if s.split_k > 1:
        if s.split_finalize == "atomic":
            return Resultado(SI, f"K repartido entre {s.split_k} bloques; cada uno suma su parcial en la salida con atomicAdd (un solo kernel, {atom} atómicos en el código).", [f"REDUCE={s.raw.get('REDUCE')}"])
        return Resultado(SI, f"K repartido entre {s.split_k} bloques que escriben parciales a un workspace; un segundo kernel los reduce sin atómicos (determinista).", [f"REDUCE={s.raw.get('REDUCE')}"])
    if not s.has_tile:
        return Resultado(NA, "Sin contracción en baldosas.")
    return Resultado(NO, "Toda la reducción en K la hace un mismo bloque.", [f"REDUCE={s.raw.get('REDUCE', '')!r}"])


# --- 5. Extra: técnicas presentes en la suite que no están en la lista -------


def _reduccion_coop(ctx: Contexto) -> Resultado:
    s = ctx.sched
    shfl = ctx.src.count(r"__shfl_(?:xor_|down_)?sync")
    partes = []
    if s.coop:
        partes.append(f"reducción cooperativa: {s.units[0]} hilos se reparten el eje reducido y combinan con mariposa __shfl_xor_sync ({shfl} shuffles) y árbol en smem entre warps" + (" (mapeo transpuesto coop-t)" if s.coop_t else ""))
    if s.reduce_reg > 1:
        partes.append(f"{s.reduce_reg} acumuladores independientes por hilo (ILP) para no encadenar FMA dependientes")
    if not partes and shfl:
        if "rms_norm" in ctx.op or "norm" in ctx.src.name:
            partes.append(f"la parte RMSNorm reduce la suma de cuadrados de cada fila con {shfl} shuffles de warp antes de normalizar, dentro del mismo kernel que la contracción")
        else:
            partes.append(f"{shfl} shuffles de warp en el código")
    if not partes:
        return Resultado(NA if s.has_tile or not s.raw.get("REDUCE") else NO, "Sin reducción cooperativa entre hilos.")
    return Resultado(SI, "; ".join(partes) + ".", [f"REDUCE={s.raw.get('REDUCE', '')!r}", f"WORK={s.raw.get('WORK', '')!r}"])


def _pdl(ctx: Contexto) -> Resultado:
    n = ctx.src.count(r"griddepcontrol")
    if n:
        return Resultado(SI, "Programmatic Dependent Launch (griddepcontrol.wait): el kernel puede lanzarse mientras termina el anterior y solo espera justo antes de leer sus datos, lo que oculta latencia de lanzamiento.", [f"{n} griddepcontrol en el cuerpo"])
    return Resultado(NO, "Sin PDL.")


TECNICAS: list[Tecnica] = [
    Tecnica("factorizacion", "1. Intensidad aritmética y cómputo", "Factorización de bucles", "Fact",
            "Alinea cada nivel del bucle con un nivel del hardware (bloque → warp → hilo → registro → átomo de Tensor Core), de modo que cada nivel reutiliza lo que el nivel superior trajo a una memoria más rápida.", _factorizacion),
    Tecnica("register_tiling", "1. Intensidad aritmética y cómputo", "Register tiling", "RegT",
            "Cada hilo/warp calcula una sub-rejilla FM×FN como producto exterior en registros: con FM+FN cargas hace FM·FN productos, así que el kernel pasa de estar limitado por cargas (load-bound) a estar limitado por FMA/MMA.", _register_tiling),
    Tecnica("vectorizacion", "1. Intensidad aritmética y cómputo", "Cargas vectorizadas e intercalado", "Vec",
            "Una transacción de 64/128 bits (LDG.64/LDG.128) sustituye 2-4 cargas escalares: menos instrucciones de memoria y mejor uso de cada sector de 32 B. Colocar cada carga junto a su primer uso acorta la vida de los registros y mejora la planificación.", _vectorizacion),
    Tecnica("tensor_cores", "1. Intensidad aritmética y cómputo", "Tensor Cores (mma.sync + ldmatrix)", "TC",
            "La instrucción mma.sync hace un bloque 16×8×16 por warp (4096 FMA) en un solo paso; ldmatrix lleva los fragmentos de smem a registros en la disposición que espera la MMA. En una RTX 5090 la potencia en fp16 con Tensor Cores es un orden de magnitud mayor que con núcleos CUDA.", _tensor_cores),
    Tecnica("smem_staging", "2. Flujo de datos y solapamiento asíncrono", "Almacenamiento por fases en smem", "Smem",
            "El bloque carga una vez cada baldosa BK en memoria compartida y todos sus hilos la reutilizan: el tráfico a global baja en un factor ~BM (para B) y ~BN (para A).", _smem_staging),
    Tecnica("cp_async_tma", "2. Flujo de datos y solapamiento asíncrono", "Transferencias asíncronas (cp.async / TMA)", "Async",
            "La copia global→smem no pasa por los registros de los hilos: con cp.async cada hilo emite la copia y sigue; con TMA un solo hilo pide una baldosa 2D/3D completa al motor de copia y una mbarrier avisa cuando llegan los bytes. Libera registros e instrucciones para el cálculo.", _cp_async_tma),
    Tecnica("pipelining", "2. Flujo de datos y solapamiento asíncrono", "Software pipelining / doble búfer", "Pipe",
            "Con varias baldosas en vuelo (prólogo, bucle principal y epílogo), la latencia de memoria de la baldosa siguiente queda oculta tras el cálculo de la actual. d<n> = etapas global→smem; p<n> = ping-pong smem→registros.", _pipelining),
    Tecnica("warp_spec", "2. Flujo de datos y solapamiento asíncrono", "Especialización de warps", "WSpec",
            "Un warp productor solo emite TMA y los consumidores solo hacen MMA, a través de un búfer circular en smem: el cálculo nunca se detiene para emitir cargas, y setmaxnreg da más registros a los consumidores.", _warp_spec),
    Tecnica("padding_swizzle", "3. Eliminación de conflictos de banco", "Padding y swizzling", "Bank",
            "La smem tiene 32 bancos de 4 B; si varios hilos de un warp tocan el mismo banco, el acceso se serializa. El relleno (+1/+8 elementos por fila) o el swizzle XOR (modos B32/B64/B128 del TMA) reparten las direcciones entre bancos distintos.", _padding_swizzle),
    Tecnica("cta_swizzle", "4. Planificación de la rejilla", "CTA swizzle (rasterizado agrupado)", "Rast",
            "Reordenar los bloques (GROUP_M) hace que bloques vecinos en la matriz se ejecuten a la vez y compartan en L2 las franjas de A/B: más aciertos en L2 y menos tráfico a DRAM.", _cta_swizzle),
    Tecnica("split_k", "4. Planificación de la rejilla", "Split-K", "SplitK",
            "En matrices delgadas con K grande hay pocos bloques para llenar los SMs; repartir K entre varios bloques da paralelismo extra, a cambio de combinar los parciales (atomicAdd en un kernel, o un segundo kernel sin atómicos).", _split_k),
    Tecnica("reduccion_coop", "5. Extra (en la suite, fuera de la lista)", "Reducción cooperativa (shuffle / árbol)", "Coop",
            "Para reducciones como RMSNorm: varios hilos se reparten la fila y combinan con __shfl_xor_sync (dentro del warp) y un árbol en smem (entre warps), en vez de un hilo por fila.", _reduccion_coop),
    Tecnica("pdl", "5. Extra (en la suite, fuera de la lista)", "Programmatic Dependent Launch", "PDL",
            "griddepcontrol permite solapar el arranque de un kernel con el final del anterior: importa en kernels de pocos µs, donde el lanzamiento pesa.", _pdl),
]


def analizar_kernel(ctx: Contexto) -> list[tuple[Tecnica, Resultado]]:
    out = []
    for t in TECNICAS:
        try:
            r = t.detectar(ctx)
        except Exception as e:  # noqa: BLE001 — un detector roto no debe tumbar el informe
            r = Resultado("?", f"detector falló: {e!r}")
        out.append((t, r))
    return out


def coherencia(sched: Schedule, src: KernelSource) -> str | None:
    """Aviso si el CUDA volcado no corresponde al schedule fijado (el volcado guarda la última compilación)."""
    if not src.found:
        return "no se encontró el CUDA de este kernel en el volcado"
    if sched.work_kind == "warp" and src.launch_bounds:
        esperado = 32 * (sched.warps + sched.producer)
        if esperado != src.launch_bounds:
            return f"__launch_bounds__({src.launch_bounds}) ≠ {esperado} hilos que implica WORK={sched.raw.get('WORK')}: el código volcado puede ser de otro schedule (greedy)"
    return None
