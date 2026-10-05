"""Onde o datacompolars gasta tempo em datasets pequenos e largos (ex.: 100 linhas x 300 colunas)?

Mede, na mesma máquina e sem o overhead de subprocesso da suíte:
  - datacompolars hash (padrão), hash com column_details=False e columnwise;
  - diffly (compare_frames) para referência;
e mostra o perfil (cProfile) do caminho hash padrão, que diz quais funções dominam.

Gere antes o dataset (a suíte apaga os dela; sem --tag a pasta é data/<linhas>_<cols>cols):
    uv run python benchmarks/datagen.py --rows 100 --cols 300 --divergence 0.1 --out data
Rode:
    uv run python benchmarks/profile_small.py data/100_300cols
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path


def timed(fn, repeats: int):
    fn()  # aquecimento (cache de arquivos, imports tardios)
    times, last = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="pasta com base/ e compare/")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--top", type=int, default=25, help="linhas do perfil")
    a = ap.parse_args()
    base, cmp_ = str(a.dataset / "base"), str(a.dataset / "compare")
    if not (a.dataset / "base").exists():
        raise SystemExit(f"não encontrei {a.dataset / 'base'}; gere o dataset com datagen.py (veja o topo do arquivo)")

    import polars as pl
    from datacompolars import CompareSettings, ReportSettings, compare

    def dcp(**kw):
        settings = CompareSettings(join_columns=["id"], window_rows=None, **kw)

        def go():
            r = compare(base, cmp_, settings)
            r.report(ReportSettings(print_output=False))
            return r

        return go

    variants = {
        "datacompolars hash (padrão)": dcp(),
        "datacompolars hash, column_details=False": dcp(column_details=False),
        "datacompolars columnwise (abs_tol=1e-12)": dcp(abs_tol=1e-12),
    }
    print(f"{a.dataset}  ({a.repeats} execuções, mediana; 1 de aquecimento descartada)\n")
    for name, fn in variants.items():
        med, mn, r = timed(fn, a.repeats)
        print(f"{name:<44} mediana {med:8.3f}s  mín {mn:8.3f}s")
        ex = getattr(r, "execution", None)
        if ex is not None:
            print(f"    caminho={getattr(ex, 'path', '?')}  janelas={getattr(ex, 'windows', '?')}")
            print(f"    fases: {getattr(ex, 'timings', None)}")

    try:
        from diffly import compare_frames

        def dif():
            c = compare_frames(
                pl.scan_parquet(f"{base}/*.parquet"), pl.scan_parquet(f"{cmp_}/*.parquet"),
                primary_key="id", abs_tol=0.0, rel_tol=0.0,
            )
            return str(c.summary())

        med, mn, _ = timed(dif, a.repeats)
        print(f"{'diffly (lazy)':<44} mediana {med:8.3f}s  mín {mn:8.3f}s")
    except ImportError:
        print("diffly não instalado: pulei")

    print(f"\n--- cProfile do datacompolars hash (padrão), top {a.top} por tempo acumulado ---")
    fn = variants["datacompolars hash (padrão)"]
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(a.top)
    print(buf.getvalue())


if __name__ == "__main__":
    main()