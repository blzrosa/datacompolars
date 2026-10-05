"""Onde vai o tempo FORA da execução das consultas (montagem + planejamento), com cProfile.

No profile_phases.py (1M x 300, janelas de 100k) a fase `details` marca ~1,3 s, mas as consultas dela
(`_diagnose`) levam só ~0,3 s dentro de `LazyFrame.collect`: o resto (~1 s) é montagem de expressões
em Python e/ou planejamento/otimização do Polars. Este script separa as duas coisas.

Modo `real` (precisa do dataset, como no profile_phases.py):
  1. cProfile da comparação inteira: funções Python por tempo próprio (tottime) e acumulado (cumtime),
     filtradas em polars/datacompolars. `LazyFrame.collect` aparece como tempo nativo (otimização + execução);
     tudo o mais é Python (montar expressões, with_columns, select, join, relatório);
  2. para as primeiras `--sample` chamadas de cada ponto do engine.py, mede o PLANEJAMENTO separado da execução:
     `explain(optimized=True)` roda o otimizador sem executar. Mostra também o tamanho do plano (linhas do
     `explain`), porque otimização superlinear no nº de nós é o que a cadeia sequencial de hashes tinha.
     tempo de execução ≈ collect - explain.

Modo `scaling` (sem dataset): monta a MESMA consulta do `_diagnose` (join + 1 comparador por coluna + filtro de
divergência) sobre 100 linhas e N colunas e mede construção, `explain` e `collect`. Se o tempo de `explain` cresce mais
rápido que N (dobrar N mais que dobra o tempo), o plano tem custo superlinear; é o sintoma da cadeia antiga.

    uv run python benchmarks/profile_cpu.py real data/1000000_300cols --window-rows 100000
    uv run python benchmarks/profile_cpu.py real data/1000000_300cols --top 40 --sample 3
    uv run python benchmarks/profile_cpu.py scaling --cols 50,100,200,300,600
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl

from datacompolars import CompareSettings, ReportSettings, compare

# ----------------------------------------------------------------------------- modo real
_ORIG_COLLECT = pl.LazyFrame.collect
_STATS: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: {"plan": [], "total": [], "lines": []})
_SEEN: Dict[Tuple[str, int], int] = defaultdict(int)
_SAMPLE = 3
_IN_PROBE = False


def _caller() -> Tuple[str, int]:
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename.replace("\\", "/").endswith("datacompolars/engine.py"):
            return f.f_code.co_name, f.f_lineno
        f = f.f_back
    return "(fora do engine.py)", 0


def _explain(lf: pl.LazyFrame, kwargs: Dict[str, Any]) -> str:
    """Plano otimizado sem executar; usa o mesmo motor (streaming) do collect quando a versão aceita."""
    engine = kwargs.get("engine")
    if engine is not None:
        try:
            return lf.explain(optimized=True, engine=engine)  # type: ignore[call-arg]
        except TypeError:
            pass
    return lf.explain(optimized=True)


def _probing_collect(self: pl.LazyFrame, *args: Any, **kwargs: Any):
    global _IN_PROBE
    key = _caller()
    if _IN_PROBE or _SEEN[key] >= _SAMPLE:
        return _ORIG_COLLECT(self, *args, **kwargs)
    _SEEN[key] += 1
    _IN_PROBE = True
    try:
        t0 = time.perf_counter()
        text = _explain(self, kwargs)
        t_plan = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = _ORIG_COLLECT(self, *args, **kwargs)
        t_total = time.perf_counter() - t0
    finally:
        _IN_PROBE = False
    st = _STATS[key]
    st["plan"].append(t_plan)
    st["total"].append(t_total)
    st["lines"].append(float(text.count("\n") + 1))
    return out


def run_real(a: argparse.Namespace) -> None:
    global _SAMPLE
    _SAMPLE = a.sample
    base, cmp_ = str(a.dataset / "base"), str(a.dataset / "compare")
    if not (a.dataset / "base").exists():
        raise SystemExit(f"não encontrei {a.dataset / 'base'}; gere o dataset com datagen.py")
    settings = CompareSettings(
        join_columns=["id"], window_rows=a.window_rows, cache_windows=True, columns_per_batch=a.columns_per_batch
    )
    print(f"polars {pl.__version__} · {a.dataset}\n")

    if a.warmup:  # aquece cache de arquivos e imports
        compare(base, cmp_, settings)

    # 1) cProfile (sem a sonda de explain, que distorceria os tempos)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    result = compare(base, cmp_, settings)
    result.report(ReportSettings(print_output=False))
    prof.disable()
    total = time.perf_counter() - t0
    print(f"=== cProfile: total {total:.2f}s · janelas={result.execution.windows} · fases {result.execution.timings}\n")
    ps = pstats.Stats(prof).strip_dirs()
    print(f"--- por tempo PRÓPRIO (tottime), top {a.top}: o que gasta CPU de fato (collect = nativo do Polars) ---")
    ps.sort_stats("tottime").print_stats(a.top)
    print(f"--- por tempo ACUMULADO (cumtime) só em datacompolars/polars, top {a.top} ---")
    ps.sort_stats("cumtime").print_stats(r"datacompolars|polars|frame\.py|expr\.py", a.top)

    # 2) planejamento x execução por ponto do engine.py
    pl.LazyFrame.collect = _probing_collect  # type: ignore[method-assign]
    try:
        compare(base, cmp_, settings)
    finally:
        pl.LazyFrame.collect = _ORIG_COLLECT  # type: ignore[method-assign]
    print(f"\n=== planejamento (explain otimizado) x collect, primeiras {a.sample} chamadas por ponto ===")
    print("collect inclui o planejamento (o Polars otimiza de novo); execução ≈ collect - plano.\n")
    head = f"{'função:linha':<30}{'n':>3}{'plano ms':>10}{'collect ms':>12}{'exec ≈ ms':>11}{'plano %':>9}{'linhas do plano':>17}"
    print(head + "\n" + "-" * len(head))
    for (fn, line), st in sorted(_STATS.items(), key=lambda kv: -sum(kv[1]["total"])):
        plan, tot = statistics.median(st["plan"]) * 1000, statistics.median(st["total"]) * 1000
        print(
            f"{fn + ':' + str(line):<30}{len(st['plan']):>3}{plan:>10.1f}{tot:>12.1f}{max(tot - plan, 0):>11.1f}"
            f"{100 * plan / tot if tot else 0:>8.0f}%{int(statistics.median(st['lines'])):>17,}"
        )
    print("\nLeitura: 'plano %' alto + muitas linhas = o gargalo é o otimizador (reduza nós/expressões por consulta).")
    print("         'plano %' baixo mas tempo FORA do collect alto no cProfile = o gargalo é Python montando expressões.")


# ----------------------------------------------------------------------------- modo scaling
def run_scaling(a: argparse.Namespace) -> None:
    from datacompolars.engine import _Comparison  # privado: reaproveita a consulta real do _diagnose

    rows = a.rows
    print(f"polars {pl.__version__} · consulta do _diagnose, {rows} linhas, mediana de {a.repeats}\n")
    head = f"{'colunas':>8}{'montar ms':>11}{'explain ms':>12}{'collect ms':>12}{'explain/col µs':>16}{'linhas do plano':>17}"
    print(head + "\n" + "-" * len(head))
    prev = None
    for n in a.cols:
        data_l = {"id": list(range(rows))}
        data_r = {"id": list(range(rows))}
        for j in range(n):
            data_l[f"c{j}"] = [float(i + j) for i in range(rows)]
            data_r[f"c{j}"] = [float(i + j) + (1.0 if i % 10 == 0 else 0.0) for i in range(rows)]
        left, right = pl.DataFrame(data_l), pl.DataFrame(data_r)
        cmp_ = _Comparison(left, right, CompareSettings(join_columns=["id"], window_rows=None), None)
        batch = list(cmp_.cols)
        build, expl, coll, lines = [], [], [], 0
        for _ in range(a.repeats):
            cmp_._match_cache.clear()  # mede a montagem das expressões, não o cache
            t0 = time.perf_counter()
            with_m = cmp_._matched(cmp_.left, cmp_.right, batch, None)
            names = [f"__m_{c}" for c in batch]
            q = with_m.filter(pl.any_horizontal([pl.col(x).not_() for x in names])).select([*cmp_.pk, *names])
            build.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            text = q.explain(optimized=True)
            expl.append(time.perf_counter() - t0)
            lines = text.count("\n") + 1
            t0 = time.perf_counter()
            q.collect(engine="streaming")
            coll.append(time.perf_counter() - t0)
        e = statistics.median(expl) * 1000
        grow = f"  (x{e / prev[1]:.1f} p/ x{n / prev[0]:.1f} colunas)" if prev else ""
        print(
            f"{n:>8}{statistics.median(build) * 1000:>11.1f}{e:>12.1f}{statistics.median(coll) * 1000:>12.1f}"
            f"{1000 * e / n:>16.1f}{lines:>17,}{grow}"
        )
        prev = (n, e)
    print("\nSe 'explain/col µs' cresce com as colunas, o otimizador é superlinear nesta consulta (plano em cadeia/aninhado).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("real", help="cProfile + plano x execução num dataset")
    r.add_argument("dataset", type=Path)
    r.add_argument("--window-rows", type=int, default=100_000)
    r.add_argument("--columns-per-batch", type=int, default=50)
    r.add_argument("--top", type=int, default=25)
    r.add_argument("--sample", type=int, default=3, help="chamadas sondadas (explain) por ponto do engine.py")
    r.add_argument("--warmup", action="store_true")
    r.set_defaults(fn=run_real)
    s = sub.add_parser("scaling", help="custo do plano x nº de colunas, sem dataset")
    s.add_argument("--cols", type=lambda t: [int(x) for x in t.split(",")], default=[50, 100, 200, 300, 600])
    s.add_argument("--rows", type=int, default=100)
    s.add_argument("--repeats", type=int, default=5)
    s.set_defaults(fn=run_scaling)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()