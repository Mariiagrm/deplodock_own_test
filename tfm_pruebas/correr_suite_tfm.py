"""Subconjunto TFM de la suite de rendimiento de emmy (tests/perf) con logs explicativos.

Qué hace, por cada caso del corpus (tests/compiler/realization/cases) que corre en esta GPU:
  1. Lo mide igual que `make bench-kernels`: `emmy run --golden <caso> --realization <fila> --bench
     --bench-backends eager,tcompile,emmy --json`, con el schedule que fija el caso (el de CloudRift).
     Si sale >5% más lento que la referencia guardada para esta GPU, repite hasta 3 veces y se queda
     con el mínimo (mismo criterio que tests/perf/conftest.py).
  2. Guarda el volcado de la compilación (EMMY_DUMP_DIR) y analiza qué técnicas de optimización
     contiene cada kernel (tecnicas.py), escribiendo <caso>.log.
  3. Al final escribe resumen.md (tablas), resumen.csv y resultados.json.

Categorías: matmul, rmsnorm/fused, nvfp4 (fp4 con escalas fp8; en el corpus no hay MMA fp8 e4m3 pura)
y qwen3emb (las capas de Qwen3-Embedding-0.6B). Un caso puede estar en varias.

Uso (dentro del contenedor, desde la raíz de deplodock):
  PYTHONPATH=. ./venv/bin/python tfm_pruebas/correr_suite_tfm.py [--cat matmul,qwen3emb] [--solo patron]
      [--max N] [--workers 4] [--ncu] [--out tfm_pruebas/runs/<fecha>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import tecnicas as T  # noqa: E402
from tests.compiler.realization import helpers  # noqa: E402

CATEGORIAS = ("matmul", "rmsnorm/fused", "nvfp4", "qwen3emb")
EXPLICA_CAT = {
    "matmul": "multiplicaciones de matrices del corpus (f16/bf16/f32; Tensor Cores y camino escalar)",
    "rmsnorm/fused": "casos con RMSNorm, sola o fusionada con la proyección lineal / atención",
    "nvfp4": "baja precisión NVFP4: valores fp4 e2m1 con escalas por bloque fp8 e4m3 (mma m16n8k64 e2m1); el corpus no tiene MMA fp8 e4m3 pura",
    "qwen3emb": "las 12 operaciones de una capa de Qwen3-Embedding-0.6B (RMSNorm, proyecciones Q/K/V/O, MLP, atención) a seq 32/128/512",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------


@dataclass
class Sel:
    case: object
    id: str
    family: str
    cats: list[str]
    facts: dict
    ref_us: float | None
    ref_tcompile_us: float | None


def seleccionar(cats: set[str], solo: str | None, cap: tuple[int, int]) -> list[Sel]:
    hw = helpers.live_hardware_id()
    out = []
    for path in helpers.case_files():
        case = helpers.load_case(path)
        if case.xfail_stage is not None or tuple(case.compute_cap) != cap:
            continue
        facts = helpers.describe(case)
        fam = facts["family"]
        text = path.read_text().lower()
        c = []
        if fam == "matmul":
            c.append("matmul")
        if "rms_norm" in facts["op"]:
            c.append("rmsnorm/fused")
        if "e2m1" in text or "e4m3" in text or "nvfp4" in path.name or "fp4" in path.name:
            c.append("nvfp4")
        if fam == "qwen3emb":
            c.append("qwen3emb")
        c = [x for x in c if x in cats]
        if not c:
            continue
        cid = f"{fam}/{path.stem}"
        if solo and not any(s in cid for s in solo.split(",")):
            continue
        rec = helpers.recorded_latency(case, hw)
        out.append(Sel(case, cid, fam, c, facts, float(rec.emmy_us) if rec else None,
                       float(getattr(rec, "tcompile_us", None)) if rec and getattr(rec, "tcompile_us", None) else None))
    orden = {c: i for i, c in enumerate(CATEGORIAS)}
    out.sort(key=lambda s: (orden[s.cats[0]], s.id))
    return out


# ---------------------------------------------------------------------------
# Medida
# ---------------------------------------------------------------------------


@dataclass
class Medida:
    sel: Sel
    ok: bool = False
    error: str = ""
    eager_us: float | None = None
    tcompile_us: float | None = None
    emmy_us: float | None = None  # schedule fijado (lo que mide la suite)
    greedy_us: float | None = None
    samples: list[float] = field(default_factory=list)
    record: dict | None = None
    segundos: float = 0.0
    dump: Path | None = None
    tec: dict = field(default_factory=dict)  # clave técnica -> estado agregado del caso
    tune: str = ""  # "" sin tune · "ok" · motivo por el que no se pudo autotunear
    tune_s: float = 0.0


def medir(sel: Sel, outdir: Path, ncu: bool, env_base: dict, tune: int = 0) -> Medida:
    m = Medida(sel)
    cdir = outdir / "casos" / sel.id.replace("/", "__")
    dump = cdir / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    m.dump = dump
    env = {**env_base, "EMMY_NVCC_FLAGS": ""}
    t0 = time.time()
    if tune:
        # Fase 1: autotuning del kernel con base de datos propia del caso; la fase 2 (medida) la lee,
        # así que el greedy pasa a ser "lo que emmy elige tras medir N variantes".
        env["EMMY_TUNE_DB"] = str(cdir / "autotune.db")
        tout = cdir / "tune.json"
        r = subprocess.run(helpers.bench_command(sel.case, tout) + ["--tune", str(tune)], capture_output=True, text=True,
                           timeout=3600, env={**env, "EMMY_DUMP_DIR": str(cdir / "dump_tune")})
        (cdir / "tune.stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        m.tune = _estado_tune(r.stdout, r.stderr)
        m.tune_s = time.time() - t0
        shutil.rmtree(cdir / "dump_tune", ignore_errors=True)
    record = None
    for rep in range(helpers.LATENCY_REPEATS if sel.ref_us else 1):
        out = cdir / f"bench_{rep}.json"
        cmd = helpers.bench_command(sel.case, out) + ["--no-record-evidence"]
        if ncu:
            cmd.append("--profile")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**env, "EMMY_DUMP_DIR": str(dump)})
        (cdir / f"bench_{rep}.stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0 or not out.exists():
            m.error = f"emmy run salió con código {r.returncode}: {_motivo(r.stdout, r.stderr)}"
            break
        rec = json.loads(out.read_text())
        pinned = [p for p in rec.get("pinned", []) if p.get("status") == "ok" and p.get("total_us")]
        if not pinned:
            m.error = f"la fila fijada no midió nada: {rec.get('pinned')}"
            break
        us = float(pinned[0]["total_us"])
        m.samples.append(us)
        if record is None or us <= min(m.samples):
            record = rec
        if not sel.ref_us or us <= sel.ref_us * (1 + helpers.LATENCY_BAND):
            break
    m.segundos = time.time() - t0
    return _rellenar(m, record)


def _estado_tune(stdout: str, stderr: str) -> str:
    """'ok: …' si emmy midió variantes (aunque salga con código 1 porque alguna variante fue irreproducible)."""
    import re

    hits = re.findall(r"\[tune\] (\d+)/(\d+) measured, best ([\d.]+) us", stdout)
    if hits:
        n, tot, best = hits[-1]
        return f"ok: {n}/{tot} variantes medidas, la mejor {float(best):.2f} µs"
    return f"no aplicable/falló: {_motivo(stdout, stderr)[:200]}"


def _motivo(stdout: str, stderr: str) -> str:
    """La razón legible de un fallo de emmy run: emmy avisa en stdout con líneas '!' (p. ej. pin irreproducible)."""
    marcas = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("!") and ":" in ln]
    if marcas:
        txt = marcas[-1].lstrip("! ")
        if "unreproducible pin" in txt:
            txt = "el schedule fijado por el caso ya no se realiza con este compilador (unreproducible pin; emmy no lo mide): " + txt
        return txt
    return stderr.strip().splitlines()[-1] if stderr.strip() else "(sin stderr)"


def _rellenar(m: Medida, record: dict | None) -> Medida:
    if record is not None:
        b = record.get("backends", {})
        m.record = record
        m.ok = True
        m.error = ""
        m.emmy_us = min(m.samples)
        m.eager_us = b.get("Eager PyTorch", {}).get("latency_us")
        m.tcompile_us = b.get("torch.compile", {}).get("latency_us")
        g = record.get("greedy") or {}
        m.greedy_us = g.get("total_us") if g.get("status") == "ok" else None
    return m


def cargar(sel: Sel, outdir: Path) -> Medida:
    """Reconstruye la medida de un caso ya medido (para --reanalizar): bench_*.json + dump/."""
    m = Medida(sel)
    cdir = outdir / "casos" / sel.id.replace("/", "__")
    m.dump = cdir / "dump"
    ts = cdir / "tune.stdout.txt"
    if ts.exists():
        so, _, se = ts.read_text().partition("\n--- stderr ---\n")
        m.tune = _estado_tune(so, se)
    record = None
    for f in sorted(cdir.glob("bench_*.json")):
        rec = json.loads(f.read_text())
        pinned = [p for p in rec.get("pinned", []) if p.get("status") == "ok" and p.get("total_us")]
        if pinned:
            us = float(pinned[0]["total_us"])
            m.samples.append(us)
            if us <= min(m.samples):
                record = rec
    if record is None:
        outs = sorted(cdir.glob("bench_*.stdout.txt"))
        if outs:
            so, _, se = outs[-1].read_text().partition("\n--- stderr ---\n")
            m.error = "emmy run falló: " + _motivo(so, se)
        else:
            m.error = "sin medida válida en " + str(cdir.relative_to(outdir))
    return _rellenar(m, record)


# ---------------------------------------------------------------------------
# Log por caso
# ---------------------------------------------------------------------------


def _f(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _ratio(a: float | None, b: float | None) -> str:
    return "—" if not a or not b else f"{a / b:.2f}x"


def _knobs_str(k: dict) -> str:
    keys = ("WORK", "TILE", "STAGE", "REDUCE", "RASTER")
    return "  ".join(f"{x}={k.get(x, '')!s}" for x in keys if x in k)


def escribir_log(m: Medida, outdir: Path, ncu: bool) -> Path:
    sel = m.sel
    path = outdir / "logs" / f"{sel.id.replace('/', '__')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    w = L.append
    w("=" * 100)
    w(f"CASO {sel.id}")
    w(f"  Categorías : {', '.join(sel.cats)}")
    w(f"  Qué es     : {sel.facts['op']} · dtype {sel.facts['dtype']} · entradas {sel.facts['shape']}")
    w(f"  Fichero    : {sel.case.path.relative_to(HERE.parent)}  (fila fijada: {sel.case.row.name})")
    w("=" * 100)
    if not m.ok:
        w(f"ERROR: {m.error}")
        w(f"Salida completa en {(m.dump.parent).relative_to(outdir)}/bench_*.stdout.txt")
        path.write_text("\n".join(L) + "\n")
        return path
    rec = m.record
    w("")
    w(f"TIEMPOS (µs; {rec.get('iters')} iteraciones tras {rec.get('warmup')} de calentamiento, captura en CUDA graph)")
    w(f"  PyTorch eager ............................ {_f(m.eager_us):>9}")
    w(f"  torch.compile ............................ {_f(m.tcompile_us):>9}   ({_ratio(m.eager_us, m.tcompile_us)} vs eager)")
    w(f"  emmy, schedule fijado (CloudRift) ........ {_f(m.emmy_us):>9}   ({_ratio(m.eager_us, m.emmy_us)} vs eager)  ← lo que mide la suite"
      + (f"; muestras {', '.join(_f(s) for s in m.samples)}" if len(m.samples) > 1 else ""))
    w(f"  emmy, greedy (emmy elige sin ayuda) ...... {_f(m.greedy_us):>9}   ({_ratio(m.eager_us, m.greedy_us)} vs eager)"
      + (f"  ← tras --tune ({m.tune[4:]}" + (f"; {m.tune_s:.0f}s" if m.tune_s else "") + ")" if m.tune.startswith("ok") else ""))
    if m.tune and not m.tune.startswith("ok"):
        w(f"    ! --tune {m.tune}  (el greedy es el de siempre, sin autotuning)")
    g = rec.get("greedy") or {}
    if g.get("status") != "ok" or (g.get("isolated") or {}).get("status") not in (None, "ok"):
        emmy_b = rec.get("backends", {}).get("Emmy", {}).get("latency_us")
        iso = g.get("isolated") or {}
        motivo = (iso.get("error") or iso.get("status")) if iso.get("status") not in (None, "ok") else (g.get("error") or g.get("status"))
        w(f"    ! greedy no medido limpio ({motivo}); latencia de emmy en la comparación global: {_f(emmy_b)} µs")
    if sel.ref_us:
        w(f"  Referencia CloudRift RTX 5090 (emmy) ..... {_f(sel.ref_us):>9}   → aquí tardamos {m.emmy_us / sel.ref_us:.2f}× su tiempo"
          + ("  (dentro de la banda del 5%)" if m.emmy_us <= sel.ref_us * 1.05 else "  (MÁS LENTO que la referencia)"))
    else:
        w("  Referencia CloudRift RTX 5090 ............ (el caso no tiene latencia guardada para esta GPU)")
    corr = {k: v.get("correctness", {}).get("status") for k, v in rec.get("backends", {}).items() if v.get("correctness")}
    if corr:
        w(f"  Corrección frente a eager: {', '.join(f'{k}={v}' for k, v in corr.items())}")
    pinned = [p for p in rec["pinned"] if p.get("status") == "ok"][0]
    greedy = rec.get("greedy") or {}
    w("")
    w("SCHEDULE (parámetros que deciden las optimizaciones; ver tfm_pruebas/README.md)")
    for i, k in enumerate(pinned.get("kernels", [])):
        w(f"  fijado  [{k['kernel']}] {_knobs_str(k.get('knobs', {}))}")
    for k in greedy.get("kernels", []) or []:
        w(f"  greedy  [{k['kernel']}] {_knobs_str(k.get('knobs', {}))}")
    pins = {k: v for k, v in (pinned.get("pinned_knobs") or {}).items() if k in ("FAST_MATH", "FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA", "PLACE")}
    if pins:
        w(f"  interruptores: {', '.join(f'{k}={v}' for k, v in pins.items())}")
    fired, noop = T.read_rules(m.dump)
    corte = T.read_cut(m.dump)
    nk = len(pinned.get("kernels", []))
    if corte.split_k > 1:
        w(f"  corte de grafo (regla 030_cut): split-K en {corte.split_k} trozos → {nk} kernels (parciales + combinación)")
    elif corte.place:
        w(f"  corte de grafo (regla 030_cut): corte de colocación → la operación se reparte en {nk} kernels (lo contrario de fusionar)")
    elif "+" in sel.facts["op"] and nk == 1:
        w(f"  fusión: las operaciones {sel.facts['op']} van en UN solo kernel (sin ida y vuelta a memoria global entre ellas)")
    if m.greedy_us and m.emmy_us and m.emmy_us > 1.5 * m.greedy_us:
        w(f"  NOTA: el schedule fijado es {m.emmy_us / m.greedy_us:.1f}× más lento que el greedy. Los casos del corpus fijan a propósito un schedule")
        w("        para cubrir un camino concreto del compilador (caso de cobertura), no el más rápido: para rendimiento mira la columna greedy.")
    ncu_all = None
    ncu_path = m.dump / "61_ncu_metrics.json"
    if ncu_path.exists():
        try:
            ncu_all = json.loads(ncu_path.read_text())
        except json.JSONDecodeError:
            ncu_all = None
    # emmy run --profile perfila el programa GREEDY (no el schedule fijado). Las métricas solo valen como
    # evidencia del kernel fijado si el greedy eligió exactamente los mismos knobs.
    greedy_knobs = {k["kernel"]: k.get("knobs", {}) for k in (greedy.get("kernels") or [])}

    def _ncu_line(n: dict) -> str:
        return "ocupación {:.0f}% · SM {:.0f}% · Tensor {:.0f}% · FMA {:.0f}% · DRAM {:.0f}% · conflictos smem ld/st {:,}/{:,} · {} regs/hilo · {:.2f} µs".format(
            n.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0), n.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0),
            n.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 0),
            n.get("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", 0), n.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0),
            int(n.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0)), int(n.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0)),
            int(n.get("launch__registers_per_thread", 0)), n.get("gpu__time_duration.sum", 0) / 1000)

    if ncu_all:
        w("")
        w("NCU (Nsight Compute) — emmy perfila el programa GREEDY, no el schedule fijado:")
        for kname, n in ncu_all.items():
            if kname.startswith("k_"):
                w(f"  greedy [{kname}] {_ncu_line(n)}")
    estados: dict[str, list[str]] = {}
    for k in pinned.get("kernels", []):
        name = k["kernel"]
        sched = T.parse_schedule(k.get("knobs", {}))
        src = T.read_kernel_source(m.dump, name)
        mismo = greedy_knobs.get(name) == k.get("knobs", {})
        ctx = T.Contexto(sched, src, fired, noop, k.get("smem_bytes"), (ncu_all or {}).get(name) if mismo else None, corte, sel.facts["op"], sel.facts["dtype"])
        w("")
        w("-" * 100)
        w(f"KERNEL {name}  ·  {_f(k.get('us'))} µs  ·  smem {(_f((k.get('smem_bytes') or 0) / 1024, 1))} KB"
          + (f"  ·  __launch_bounds__({src.launch_bounds})" if src.launch_bounds else ""))
        aviso = T.coherencia(sched, src)
        if aviso:
            w(f"  AVISO: {aviso}")
        if ctx.ncu:
            w(f"  ncu (greedy = mismo schedule que el fijado): {_ncu_line(ctx.ncu)}")
        elif ncu_all:
            w("  ncu: el greedy eligió otro schedule, así que sus métricas (arriba) no describen este kernel")
        seccion = None
        for tec, r in T.analizar_kernel(ctx):
            estados.setdefault(tec.clave, []).append(r.estado)
            if tec.seccion != seccion:
                seccion = tec.seccion
                w("")
                w(f"  {seccion.upper()}")
            w(f"    [{r.estado:^3}] {tec.nombre}")
            w(f"          Qué se ha hecho : {r.hecho}")
            if r.evidencia:
                w(f"          Evidencia       : {' | '.join(r.evidencia)}")
            if r.estado == T.SI:
                w(f"          Qué supone      : {tec.supone}")
    w("")
    w("-" * 100)
    w("REGLAS DE LA BAJADA A KERNEL (pasada 09_lowering_kernel del compilador)")
    w(f"  cambiaron el kernel : {', '.join(fired) or '—'}")
    w(f"  sin efecto          : {', '.join(noop) or '—'}")
    w(f"  Volcado completo (IR de cada pasada + CUDA): {m.dump.relative_to(outdir)}/")
    if ncu and not ncu_all:
        w("  ncu pedido pero sin métricas (¿permisos de contadores en el contenedor?)")
    path.write_text("\n".join(L) + "\n")
    # estado agregado por técnica para la matriz: SI si algún kernel la usa
    for clave, est in estados.items():
        m.tec[clave] = T.SI if T.SI in est else (T.NO if T.NO in est else est[0])
    return path


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------


def escribir_resumen(meds: list[Medida], outdir: Path, meta: dict) -> None:
    L = []
    w = L.append
    w(f"# Suite emmy · subconjunto TFM · {meta['gpu']}")
    w("")
    w(f"- Fecha: {meta['fecha']} · deplodock {meta['git']} · driver {meta['driver']} · torch {meta['torch']}")
    w(f"- Casos: {len(meds)} ({sum(m.ok for m in meds)} medidos, {sum(not m.ok for m in meds)} con error) · categorías: {', '.join(meta['cats'])}")
    w("- `emmy` = schedule fijado por el caso (el de CloudRift; es lo que mide `make bench-kernels`). `greedy` = lo que emmy elige solo.")
    if meta.get("tune"):
        n_ok = sum(m.tune.startswith("ok") for m in meds)
        w(f"- **Ejecución con `--tune {meta['tune']}`**: antes de medir, emmy autotunea cada caso (N variantes del schedule, base de datos propia del caso); "
          f"`greedy` es lo que elige después. Autotuneados {n_ok}/{len(meds)} (el resto tiene varios kernels o falló: ver logs).")
    w("- `vs eager` > 1 significa más rápido que PyTorch eager. `vs ref` = nuestro tiempo / referencia CloudRift (≤ 1.05 = reproduce su número; en **negrita** si es más lento).")
    w("- Ojo: muchos casos del corpus son *de cobertura*: fijan a propósito un schedule para ejercitar un camino del compilador, no el más rápido."
      " Por eso a veces `emmy` es mucho peor que `greedy`. Para comparar rendimiento con PyTorch, la columna justa es `greedy`; para comparar con CloudRift, `emmy`.")
    w("")
    for cat in meta["cats"]:
        ms = [m for m in meds if cat in m.sel.cats]
        if not ms:
            continue
        w(f"## {cat}")
        w("")
        w(f"_{EXPLICA_CAT[cat]}_")
        w("")
        w("| caso | op | entradas | eager µs | t.compile µs | emmy µs | greedy µs | ref CloudRift µs | emmy vs eager | greedy vs eager | greedy vs t.compile | emmy vs ref |")
        w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in ms:
            s = m.sel
            if not m.ok:
                w(f"| {s.id.split('/', 1)[1]} | {s.facts['op']} | {s.facts['shape']} | ERROR: {m.error[:60]} ||||||||")
                continue
            vref = f"{m.emmy_us / s.ref_us:.2f}" if s.ref_us else "—"
            if s.ref_us and m.emmy_us > s.ref_us * 1.05:
                vref = f"**{vref}**"
            w(f"| {s.id.split('/', 1)[1]} | {s.facts['op']} | {s.facts['shape']} | {_f(m.eager_us)} | {_f(m.tcompile_us)} | {_f(m.emmy_us)} | {_f(m.greedy_us)} | {_f(s.ref_us)} | "
              f"{_ratio(m.eager_us, m.emmy_us)} | {_ratio(m.eager_us, m.greedy_us)} | {_ratio(m.tcompile_us, m.greedy_us)} | {vref} |")
        ok = [m for m in ms if m.ok and m.eager_us and m.emmy_us]
        if ok:
            import math

            def gmean(pairs):
                pairs = [(a, b) for a, b in pairs if a and b]
                return (math.exp(sum(math.log(a / b) for a, b in pairs) / len(pairs)), len(pairs)) if pairs else (None, 0)

            ge, _ = gmean((m.eager_us, m.emmy_us) for m in ok)
            gg, _ = gmean((m.eager_us, m.greedy_us) for m in ok)
            ggt, _ = gmean((m.tcompile_us, m.greedy_us) for m in ok)
            wins = sum(1 for m in ok if m.greedy_us and m.tcompile_us and m.greedy_us < m.tcompile_us)
            refs = [m for m in ok if m.sel.ref_us]
            inband = sum(1 for m in refs if m.emmy_us <= m.sel.ref_us * 1.05)
            w("")
            w(f"Media geométrica vs eager: emmy fijado {ge:.2f}x · greedy {_f(gg)}x · greedy vs torch.compile {_f(ggt)}x "
              f"(greedy gana a torch.compile en {wins}/{len(ok)}). Reproducimos la referencia CloudRift (±5%) en {inband}/{len(refs)} casos con referencia.")
        w("")
    w("## Matriz de técnicas (schedule fijado)")
    w("")
    w("✓ = presente en algún kernel del caso · · = no · – = no aplica. Detalle y evidencia en `logs/<caso>.log`.")
    w("")
    w("| caso | " + " | ".join(t.abrev for t in T.TECNICAS) + " |")
    w("|---|" + "---|" * len(T.TECNICAS))
    sym = {T.SI: "✓", T.NO: "·", T.NA: "–"}
    for m in meds:
        if m.ok:
            w(f"| {m.sel.id} | " + " | ".join(sym.get(m.tec.get(t.clave, "?"), "?") for t in T.TECNICAS) + " |")
    w("")
    w("Leyenda: " + " · ".join(f"**{t.abrev}** {t.nombre}" for t in T.TECNICAS))
    w("")
    (outdir / "resumen.md").write_text("\n".join(L) + "\n")

    with open(outdir / "resumen.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["caso", "categorias", "op", "dtype", "entradas", "eager_us", "tcompile_us", "emmy_us", "greedy_us", "ref_cloudrift_us",
                     "emmy_vs_eager", "emmy_vs_tcompile", "emmy_vs_ref", "error"] + [t.clave for t in T.TECNICAS])
        for m in meds:
            s = m.sel
            cw.writerow([s.id, ";".join(s.cats), s.facts["op"], s.facts["dtype"], s.facts["shape"], m.eager_us, m.tcompile_us, m.emmy_us, m.greedy_us, s.ref_us,
                         (m.eager_us / m.emmy_us) if m.ok and m.eager_us else None, (m.tcompile_us / m.emmy_us) if m.ok and m.tcompile_us else None,
                         (m.emmy_us / s.ref_us) if m.ok and s.ref_us else None, m.error] + [m.tec.get(t.clave, "") for t in T.TECNICAS])
    (outdir / "resultados.json").write_text(json.dumps({"meta": meta, "casos": [
        {"caso": m.sel.id, "categorias": m.sel.cats, "facts": m.sel.facts, "ok": m.ok, "error": m.error, "eager_us": m.eager_us, "tcompile_us": m.tcompile_us,
         "emmy_us": m.emmy_us, "emmy_muestras": m.samples, "greedy_us": m.greedy_us, "ref_cloudrift_us": m.sel.ref_us, "segundos": m.segundos, "tecnicas": m.tec, "tune": m.tune, "tune_s": m.tune_s}
        for m in meds]}, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat", default=",".join(CATEGORIAS), help=f"categorías separadas por comas: {', '.join(CATEGORIAS)}")
    ap.add_argument("--solo", help="solo casos cuyo id contenga alguno de estos patrones (separados por comas)")
    ap.add_argument("--max", type=int, help="como mucho N casos (para pruebas rápidas)")
    ap.add_argument("--workers", type=int, default=4, help="casos en paralelo (la GPU se serializa con EMMY_GPU_LOCK; compilar va en paralelo)")
    ap.add_argument("--ncu", action="store_true", help="añadir pasada ncu (ocupación, conflictos de banco, DRAM/FMA%%) a cada kernel")
    ap.add_argument("--tune", type=int, default=0, metavar="N", help="autotunear cada caso midiendo N variantes antes de la medida (solo casos de un kernel)")
    ap.add_argument("--out", help="directorio de salida (por defecto tfm_pruebas/runs/<fecha>)")
    ap.add_argument("--reanalizar", metavar="RUN_DIR", help="no medir: regenerar logs y resumen de una ejecución ya hecha")
    a = ap.parse_args()
    if a.reanalizar:
        a.out = a.reanalizar

    import torch
    cap = torch.cuda.get_device_capability(0)
    gpu = torch.cuda.get_device_name(0)
    cats = {c.strip() for c in a.cat.split(",") if c.strip()}
    bad = cats - set(CATEGORIAS)
    if bad:
        sys.exit(f"categorías desconocidas: {bad}")
    outdir = Path(a.out) if a.out else HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        driver = "?"
    git = subprocess.run(["git", "-C", str(HERE.parent), "log", "-1", "--format=%h %cs"], capture_output=True, text=True).stdout.strip()
    meta = {"fecha": datetime.now().isoformat(timespec="seconds"), "gpu": gpu, "sm": f"sm_{cap[0]}{cap[1]}", "driver": driver, "git": git,
            "torch": torch.__version__, "host": platform.node(), "cats": [c for c in CATEGORIAS if c in cats], "ncu": a.ncu, "tune": a.tune}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))

    log(f"GPU {gpu} (sm_{cap[0]}{cap[1]}) · driver {driver} · deplodock {git} · torch {torch.__version__}")
    log(f"Salida: {outdir}")
    sel = seleccionar(cats, a.solo, cap)
    if a.max:
        sel = sel[: a.max]
    for c in meta["cats"]:
        n = sum(c in s.cats for s in sel)
        log(f"  {c:14s} {n:3d} casos — {EXPLICA_CAT[c]}")
    log(f"Total {len(sel)} casos únicos ({sum(1 for s in sel if s.ref_us)} con referencia CloudRift para esta GPU). Workers: {a.workers}.")
    log("Por caso: compilar con el schedule del caso → medir eager / torch.compile / emmy fijado / emmy greedy → analizar el CUDA volcado.")

    env = dict(os.environ)
    env.setdefault("EMMY_GPU_LOCK", f"/tmp/emmy-gpu-{os.getuid()}.lock")
    meds: list[Medida] = []
    t0 = time.time()
    if a.reanalizar:
        hechos = {p.name for p in (outdir / "casos").iterdir()} if (outdir / "casos").exists() else set()
        sel = [s for s in sel if s.id.replace("/", "__") in hechos]
        log(f"--reanalizar: {len(sel)} casos ya medidos en {outdir}")
        meta = {**json.loads((outdir / "meta.json").read_text()), "reanalizado": datetime.now().isoformat(timespec="seconds")} if (outdir / "meta.json").exists() else meta
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(cargar, s, outdir) if a.reanalizar else ex.submit(medir, s, outdir, a.ncu, env, a.tune): s for s in sel}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                m = fut.result()
            except Exception as e:  # noqa: BLE001
                m = Medida(s, error=repr(e))
            meds.append(m)
            lp = escribir_log(m, outdir, a.ncu)
            if m.ok:
                tecs = ",".join(t.abrev for t in T.TECNICAS if m.tec.get(t.clave) == T.SI) or "—"
                ref = f" ref {s.ref_us:.2f} ({m.emmy_us / s.ref_us:.2f}×)" if s.ref_us else ""
                log(f"[{i:3d}/{len(sel)}] {s.id:58s} eager {_f(m.eager_us):>8} · tcomp {_f(m.tcompile_us):>8} · emmy {_f(m.emmy_us):>8} · greedy {_f(m.greedy_us):>8}{ref} · {m.segundos:5.0f}s · técnicas: {tecs}")
            else:
                log(f"[{i:3d}/{len(sel)}] {s.id:58s} ERROR {m.error[:150]}  (log: {lp.relative_to(outdir)})")
    orden = {s.id: i for i, s in enumerate(sel)}
    meds.sort(key=lambda m: orden[m.sel.id])
    escribir_resumen(meds, outdir, meta)
    log(f"Hecho en {(time.time() - t0) / 60:.1f} min: {sum(m.ok for m in meds)} ok, {sum(not m.ok for m in meds)} con error.")
    log(f"Tabla: {outdir / 'resumen.md'} · logs por caso: {outdir / 'logs'}/")


if __name__ == "__main__":
    main()
