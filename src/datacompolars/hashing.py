"""Hash de linha (64 bits) usado no caminho exato.

Medições (30M x 50 colunas, arquivos com row groups de ~100k linhas): o hash por coluna
combinado custa ~1,7 s e ~1,1 GB por lado; o hash de struct, ~6,5 s e ~2,0 GB.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import polars as pl

HASH_MULTIPLIER = 1_000_003  # ímpar; combinação posicional com aritmética UInt64 (wrap)


def is_hashable(dtype: pl.DataType) -> bool:
    """Tipos aninhados (List/Struct/Array) não entram no caminho de hash."""
    return not dtype.is_nested()


def normalized_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    """Normaliza a coluna para que 'valores iguais' impliquem 'mesmo hash'.

    - Categorical/Enum: o hash depende do mapeamento físico (difere entre frames) -> String.
    - Float: -0.0 vira 0.0 e todo NaN (qualquer sinal/payload) vira um único NaN.
    """
    e = pl.col(col)
    if dtype == pl.Categorical or dtype == pl.Enum:
        return e.cast(pl.String)
    if dtype.is_float():
        # O nome de saída de when/then/otherwise vem do `then` (um literal); sem o alias,
        # todas as colunas float se chamariam "literal".
        return (
            pl.when(e.is_nan())
            .then(pl.lit(float("nan"), dtype=dtype))
            .otherwise(e + 0.0)
            .alias(col)
        )
    return e


def row_hash_expr(columns: Sequence[str], dtypes: Mapping[str, pl.DataType]) -> pl.Expr:
    """Hash por coluna (seed distinta por posição) combinado por multiplicação posicional."""
    if not columns:
        return pl.lit(0, dtype=pl.UInt64)
    acc: Optional[pl.Expr] = None
    for i, col in enumerate(columns):
        h = normalized_expr(col, dtypes[col]).hash(seed=i + 1)
        acc = h if acc is None else acc * HASH_MULTIPLIER + h
    assert acc is not None
    return acc
