"""Onde o tempo vai numa comparação MAIOR (ex.: 1M x 300, janelas de 100k)?

Roda o datacompolars uma vez por configuração e mostra:
  - as fases do motor (execution.timings: open, plan, duplicates, compare, details);
  - cada consulta do Polars (`LazyFrame.collect`) agrupada por quem a chamou no engine.py
    (função:linha, nº de chamadas, tempo total e médio) e quanto sobra de tempo FORA delas (Python);
  - o tempo total e o pico de memória do processo.

Gere antes o dataset (a suíte apaga os dela; sem --tag a pasta é data/<linhas>_<cols>cols):
    uv run python benchmarks/datagen.py --rows 1000000 --cols 300 --divergence 0.01 --out data
Rode:
    uv run python benchmarks/profile_phases.py data/1000000_300cols --window-rows 100000
    uv run python benchmarks/profile_phases.py data/1000000_300cols --window-rows 100000 --variants hash,columnwise
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
import psutil

from datacompolars import CompareSettings, ReportSettings, compare

_CALLS: Dict[Tuple[str, int], List[float]] = defaultdict(list)
_ORIG_COLLECT = pl.LazyFrame.collect


def _caller() -> Tuple[str, int]:
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename.replace("\\", "/").endswith("datacompolars/engine.py"):
            return f.f_code.co_name, f.f_lineno
        f = f.f_back
    return "(fora do engine.py)", 0


def _timed_collect(self, *args: Any, **kwargs: Any):
    key = _caller()
    t0 = time.perf_counter()
    try:
        return _ORIG_COLLECT(self, *args, **kwargs)
    finally:
        _CALLS[key].append(time.perf_counter() - t0)


def _self_peak_mb() -> float:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak:
        return peak / 1024 / 1024
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return info.rss / 1024 / 1024


def run_variant(name: str, base: str, cmp_: str, window_rows: int, batch: int) -> None:
    kw: Dict[str, Any] = {"window_rows": window_rows, "columns_per_batch": batch}
    if name.startswith("columnwise"):
        kw["abs_tol"] = 1e-12
    if name.endswith("_cache"):
        kw["cache_windows"] = True
    if name.endswith("_nocache"):
        kw["cache_windows"] = False
    if name.endswith("_prefetch"):
        kw["cache_windows"] = True
        kw["prefetch_windows"] = True
    settings = CompareSettings(join_columns=["id"], **kw)
    _CALLS.clear()
    t0 = time.perf_counter()
    result = compare(base, cmp_, settings)
    result.report(ReportSettings(print_output=False))
    total = time.perf_counter() - t0
    ex = result.execution
    print(f"=== {name}  (janela={window_rows:,} linhas, lote={batch} colunas) ===")
    print(f"total {total:.2f}s · caminho={ex.path} · janelas={ex.windows} · pico do processo {_self_peak_mb():,.0f} MB")
    print(f"fases do motor: {ex.timings}")
    in_polars = sum(sum(v) for v in _CALLS.values())
    note = "  [com prefetch os collects se sobrepõem: a soma pode passar do total]" if name.endswith("_prefetch") else ""
    print(f"dentro de LazyFrame.collect: {in_polars:.2f}s ({100 * in_polars / total:.0f}%) · fora (Python/relatório): {total - in_polars:.2f}s{note}\n")
    print(f"{'função:linha do engine.py':<34}{'chamadas':>9}{'total s':>10}{'médio ms':>10}{'máx ms':>9}")
    for (fn, line), ts in sorted(_CALLS.items(), key=lambda kv: -sum(kv[1])):
        print(f"{fn + ':' + str(line):<34}{len(ts):>9}{sum(ts):>10.2f}{1000 * sum(ts) / len(ts):>10.1f}{1000 * max(ts):>9.1f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="pasta com base/ e compare/")
    ap.add_argument("--window-rows", type=int, default=100_000)
    ap.add_argument("--columns-per-batch", type=int, default=50)
    ap.add_argument("--variants", default="hash,columnwise", help="hash, columnwise, hash_cache, hash_nocache, hash_cache_prefetch, columnwise_cache")
    ap.add_argument("--warmup", action="store_true", help="roda cada variante uma vez antes (cache de arquivos)")
    a = ap.parse_args()
    if not (a.dataset / "base").exists():
        raise SystemExit(f"não encontrei {a.dataset / 'base'}; gere o dataset com datagen.py (veja o topo do arquivo)")
    base, cmp_ = str(a.dataset / "base"), str(a.dataset / "compare")

    pl.LazyFrame.collect = _timed_collect  # type: ignore[method-assign]
    print(f"polars {pl.__version__} · {a.dataset}\n")
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if a.warmup:
            run_variant(name, base, cmp_, a.window_rows, a.columns_per_batch)
            print("(aquecimento acima; medida a seguir)\n")
        run_variant(name, base, cmp_, a.window_rows, a.columns_per_batch)


if __name__ == "__main__":
    main()
