"""O hash do Polars já trata -0.0 e NaN como iguais? E quanto custa a normalização de floats?

`hashing.normalized_expr` faz, em cada coluna float, `when(is_nan).then(NaN).otherwise(e + 0.0)` (3 passadas)
para garantir que valores iguais tenham o mesmo hash. Este script mostra, para a SUA versão do Polars:

  1. correção: -0.0 vs 0.0, NaN com sinal/payload diferentes e nulos, com o hash CRU (`.hash()`),
     com a normalização atual e com variantes mais baratas;
  2. custo: tempo do hash de linha (árvore balanceada, como no engine) com cada variante.

Rode:
    uv run python benchmarks/hash_normalization.py
    uv run python benchmarks/hash_normalization.py --rows 1000000 --cols 100 --repeats 7
"""
from __future__ import annotations

import argparse
import statistics
import struct
import time
from typing import Callable, Dict, List, Tuple

import polars as pl

from datacompolars.hashing import HASH_MULTIPLIER, normalized_expr

NAN = float("nan")


def f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# nome -> (a, b); ambos os lados devem produzir o MESMO hash para a comparação ser correta
CASES_F64: Dict[str, Tuple[float | None, float | None]] = {
    "0.0 vs -0.0": (0.0, -0.0),
    "NaN vs NaN de sinal oposto": (f64(0x7FF8000000000000), f64(0xFFF8000000000000)),
    "NaN vs NaN com payload": (f64(0x7FF8000000000000), f64(0x7FF8000000000001)),
    "null vs null": (None, None),
}
CASES_F32: Dict[str, Tuple[float | None, float | None]] = {
    "0.0 vs -0.0": (0.0, -0.0),
    "NaN vs NaN de sinal oposto": (f32(0x7FC00000), f32(0xFFC00000)),
    "NaN vs NaN com payload": (f32(0x7FC00000), f32(0x7FC00001)),
}
# estes devem produzir hashes DIFERENTES (senão o hash esconderia divergências reais)
MUST_DIFFER: Dict[str, Tuple[float | None, float | None]] = {
    "null vs 0.0": (None, 0.0),
    "null vs NaN": (None, NAN),
    "0.0 vs NaN": (0.0, NAN),
    "1.0 vs 1.0000000000000002": (1.0, 1.0000000000000002),
}

VARIANTS: Dict[str, Callable[[str, pl.DataType], pl.Expr]] = {
    "cru (sem normalizar)": lambda c, dt: pl.col(c),
    "atual (when/is_nan + 0.0)": lambda c, dt: normalized_expr(c, dt),
    "so '+ 0.0'": lambda c, dt: pl.col(c) + 0.0,
    "fill_nan(NaN) + 0.0": lambda c, dt: pl.col(c).fill_nan(pl.lit(NAN, dtype=dt)) + 0.0,
}


def _hash_of(value: float | None, dtype: pl.DataType, build: Callable[[str, pl.DataType], pl.Expr]) -> int | None:
    frame = pl.DataFrame({"x": pl.Series([value], dtype=dtype)})
    return frame.select(build("x", dtype).hash(seed=1).alias("h")).item()


def correctness() -> Dict[str, bool]:
    """Devolve {variante: correta?} (iguais onde devem ser iguais E diferentes onde devem ser diferentes)."""
    ok = {name: True for name in VARIANTS}
    for dtype, cases in ((pl.Float64, CASES_F64), (pl.Float32, CASES_F32)):
        print(f"\n--- {dtype} --- (✓ = comportamento correto)")
        header = f"{'caso':<30}" + "".join(f"{n[:24]:<26}" for n in VARIANTS)
        print(header)
        for label, (a, b) in cases.items():
            row = f"{label + ' (iguais)':<30}"
            for name, build in VARIANTS.items():
                good = _hash_of(a, dtype, build) == _hash_of(b, dtype, build)
                ok[name] &= good
                row += f"{'✓' if good else '✗ hash diferente':<26}"
            print(row)
    print("\n--- devem diferir ---")
    for label, (a, b) in MUST_DIFFER.items():
        row = f"{label + ' (difere)':<30}"
        for name, build in VARIANTS.items():
            good = _hash_of(a, pl.Float64, build) != _hash_of(b, pl.Float64, build)
            ok[name] &= good
            row += f"{'✓' if good else '✗ colide':<26}"
        print(row)
    return ok


def _tree(exprs: List[pl.Expr]) -> pl.Expr:  # mesma combinação do engine (hashing.row_hash_expr)
    level = [e.hash(seed=i + 1) for i, e in enumerate(exprs)]
    while len(level) > 1:
        paired = [level[i] * HASH_MULTIPLIER + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    return level[0]


def make_frame(rows: int, cols: int) -> pl.DataFrame:
    idx = pl.int_range(0, rows, eager=True)
    data = {}
    for j in range(cols):
        v = (idx.hash(seed=j) % 100_000).cast(pl.Float64) / 7.0
        data[f"c{j}"] = pl.select(pl.when(idx % 997 == 0).then(NAN).otherwise(v)).to_series()  # alguns NaN
    return pl.DataFrame(data)


def timing(rows: int, cols: int, repeats: int) -> None:
    df = make_frame(rows, cols).lazy()
    print(f"\n--- custo do hash de linha: {rows:,} linhas x {cols} colunas Float64 (melhor de {repeats}, streaming) ---")
    base = None
    for name, build in VARIANTS.items():
        expr = _tree([build(f"c{j}", pl.Float64) for j in range(cols)]).alias("h")
        q = df.select(expr)
        q.collect(engine="streaming")  # aquecimento
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            q.collect(engine="streaming")
            times.append(time.perf_counter() - t)
        best, med = min(times), statistics.median(times)
        base = base if base is not None else best
        print(f"{name:<30} melhor {best * 1000:8.1f} ms   mediana {med * 1000:8.1f} ms   {best / base:5.2f}x vs cru")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    print(f"polars {pl.__version__}")
    ok = correctness()
    timing(a.rows, a.cols, a.repeats)
    print("\n=== veredito de correção ===")
    for name, good in ok.items():
        print(f"  {'OK ' if good else 'FALHA'}  {name}")
    print(
        "\nSe 'cru' estiver OK, a normalização é desnecessária nesta versão do Polars (ainda assim, fixe o teste em CI).\n"
        "Se só uma variante barata estiver OK, troque `normalized_expr` por ela; senão mantenha a atual."
    )


if __name__ == "__main__":
    main()
