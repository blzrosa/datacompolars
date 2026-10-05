"""De onde vem o custo fixo (~1,5 s com 100 linhas x 300 colunas) do hash de linha?

Hipóteses testadas, todas em memória e sem I/O, no mesmo join do motor (chave + hash + flag):
  A. o planejamento/otimização do Polars para a expressão (cadeia de 300 somas aninhadas, profundidade 300);
  B. a profundidade da cadeia: o hash atual (cadeia) contra uma árvore balanceada (profundidade ~9);
  C. otimizações do Polars (eliminação de subexpressões comuns) ligadas/desligadas;
  D. referência "tipo diffly": 300 flags `eq_missing` numa única seleção, sem hash.

Para cada variante imprime o tempo de PLANEJAR (explain) e o de EXECUTAR (collect), medianas.
Uso:
    uv run python benchmarks/microbench_hash.py                 # 100 linhas x 300 colunas
    uv run python benchmarks/microbench_hash.py --rows 100000   # o custo fixo continua pesando?
Não altera nada; o hash em árvore existe só aqui (muda os valores de hash, não a igualdade).
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

import numpy as np
import polars as pl

from datacompolars.hashing import HASH_MULTIPLIER, normalized_expr, row_hash_expr

STRINGS = np.array(["aaa", "bbb", "ccc"])


def make(rows: int, cols: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"id": np.arange(rows, dtype=np.int64)}
    for j in range(cols):
        k = j % 3
        data[f"c{j:03d}"] = rng.integers(0, 10, rows) if k == 0 else rng.random(rows) if k == 1 else STRINGS[rng.integers(0, 3, rows)]
    return pl.DataFrame(data)


def tree_hash(cols: List[str], dtypes) -> pl.Expr:
    leaves = [normalized_expr(c, dtypes[c]).hash(seed=i + 1) for i, c in enumerate(cols)]
    while len(leaves) > 1:
        nxt = [leaves[i] * HASH_MULTIPLIER + leaves[i + 1] for i in range(0, len(leaves) - 1, 2)]
        if len(leaves) % 2:
            nxt.append(leaves[-1])
        leaves = nxt
    return leaves[0]


def timed(fn: Callable[[], object], repeats: int) -> float:
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--cols", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    left = make(a.rows, a.cols, 1)
    right = left.clone()
    cols = [c for c in left.columns if c != "id"]
    dtypes = dict(left.schema)
    print(f"polars {pl.__version__} · {a.rows:,} linhas x {a.cols} colunas · mediana de {a.repeats}\n")

    def hash_query(expr_fn: Callable[[], pl.Expr]) -> pl.LazyFrame:
        def side(df: pl.DataFrame, flag: str, h: str) -> pl.LazyFrame:
            return df.lazy().select(pl.col("id"), expr_fn().alias(h), pl.lit(True).alias(flag))

        j = side(left, "in_l", "h_l").join(side(right, "in_r", "h_r"), on="id", how="full", coalesce=False, nulls_equal=True)
        both = pl.col("in_l").is_not_null() & pl.col("in_r").is_not_null()
        return j.select((both & pl.col("h_l").ne_missing(pl.col("h_r"))).sum().alias("mismatch"))

    def flags_query() -> pl.LazyFrame:
        l = left.lazy()
        r = right.lazy().select(["id", *[pl.col(c).alias("r_" + c) for c in cols]])
        j = l.join(r, on="id", how="inner", nulls_equal=True)
        return j.select([pl.col(c).eq_missing(pl.col("r_" + c)).not_().sum().alias(c) for c in cols])

    variants = {
        "A/B  hash atual (cadeia, profundidade 300)": lambda: hash_query(lambda: row_hash_expr(cols, dtypes)),
        "B    hash em árvore (profundidade ~9)": lambda: hash_query(lambda: tree_hash(cols, dtypes)),
        "D    300 flags eq_missing, sem hash (tipo diffly)": flags_query,
    }
    flags = {"padrão": None}
    try:
        flags["sem CSE de subexpressões"] = pl.QueryOptFlags(comm_subexpr_elim=False)
        flags["sem CSE de subexpr. e subplanos"] = pl.QueryOptFlags(comm_subexpr_elim=False, comm_subplan_elim=False)
        flags["sem nenhuma otimização"] = pl.QueryOptFlags.none()
    except Exception as e:  # versões do Polars com outra API de flags
        print(f"[aviso] QueryOptFlags indisponível: {e}\n")

    print(f"{'variante':<48}{'otimizações':<34}{'planejar':>10}{'executar':>10}")
    for name, build in variants.items():
        for fname, opt in flags.items():
            if name.startswith("D") and fname != "padrão":
                continue
            try:
                lf = build()
                if opt is None:
                    plan = timed(lambda: lf.explain(), a.repeats)
                    run = timed(lambda: lf.collect(engine="streaming"), a.repeats)
                else:
                    plan = timed(lambda: lf.explain(optimizations=opt), a.repeats)
                    run = timed(lambda: lf.collect(engine="streaming", optimizations=opt), a.repeats)
                print(f"{name:<48}{fname:<34}{plan:>9.3f}s{run:>9.3f}s")
            except Exception as e:
                print(f"{name:<48}{fname:<34}  erro: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
