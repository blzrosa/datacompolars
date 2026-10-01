"""Janelas de execução por faixa da primeira coluna da PK.

Cada janela é uma comparação independente sobre as linhas com `lo <= pk < hi`. Como a mesma
chave cai sempre na mesma janela, os resultados (contagens, divergências por coluna) somam.
Com arquivos parquet ordenados/particionados por PK o filtro é empurrado para a leitura (poda
de row groups), então cada janela lê só a sua faixa e o pico de memória é limitado por ela.

O planejamento é Python puro (testável sem dados); só `pk_stats` e `Window.predicate` usam Polars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import polars as pl


@dataclass(frozen=True)
class PkStats:
    n: int                 # linhas
    nulls: int             # chaves nulas
    lo: Optional[int]      # menor valor (None se vazio ou só nulos)
    hi: Optional[int]      # maior valor


@dataclass(frozen=True)
class Window:
    index: int
    lo: Optional[int] = None   # inclusivo
    hi: Optional[int] = None   # exclusivo
    null_keys: bool = False    # janela especial: chaves nulas

    @property
    def is_full(self) -> bool:
        return self.lo is None and self.hi is None and not self.null_keys

    def predicate(self, pk: str) -> Optional[pl.Expr]:
        if self.is_full:
            return None
        if self.null_keys:
            return pl.col(pk).is_null()
        return (pl.col(pk) >= self.lo) & (pl.col(pk) < self.hi)

    def describe(self) -> str:
        if self.is_full:
            return "all rows"
        if self.null_keys:
            return "null keys"
        return f"[{self.lo}, {self.hi})"


def pk_stats(lf: pl.LazyFrame, pk: str) -> PkStats:
    """Uma varredura leve (só a coluna da chave)."""
    row = (
        lf.select(
            pl.len().alias("n"),
            pl.col(pk).null_count().alias("nulls"),
            pl.col(pk).min().alias("lo"),
            pl.col(pk).max().alias("hi"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    return PkStats(
        n=int(row["n"]),
        nulls=int(row["nulls"]),
        lo=None if row["lo"] is None else int(row["lo"]),
        hi=None if row["hi"] is None else int(row["hi"]),
    )


def resolve_window_rows(
    setting: Union[int, Literal["auto"], None],
    left: PkStats,
    right: PkStats,
    *,
    threshold: int,
    auto_rows: int,
) -> Optional[int]:
    """Traduz o `window_rows` das configurações para um tamanho de janela (None = sem janelas)."""
    if setting is None:
        return None
    if setting == "auto":
        return auto_rows if max(left.n, right.n) > threshold else None
    return int(setting)


def plan_windows(left: PkStats, right: PkStats, rows_per_window: Optional[int]) -> List[Window]:
    """Divide a faixa [min, max] da chave em janelas de largura igual (~rows_per_window linhas).

    Premissa: ids razoavelmente densos. Em ids muito esparsos/enviesados as janelas ficam
    desiguais (o resultado continua correto; só o pico de memória deixa de ser uniforme).
    """
    n = max(left.n, right.n)
    los = [s.lo for s in (left, right) if s.lo is not None]
    his = [s.hi for s in (left, right) if s.hi is not None]
    if rows_per_window is None or n <= rows_per_window or not los or not his:
        return [Window(index=0)]

    lo, hi = min(los), max(his)
    span = hi - lo + 1
    k = max(1, math.ceil(n / rows_per_window))
    width = max(1, math.ceil(span / k))

    windows: List[Window] = []
    start = lo
    while start <= hi:
        end = min(start + width, hi + 1)
        windows.append(Window(index=len(windows), lo=start, hi=end))
        start = end
    if left.nulls or right.nulls:
        windows.append(Window(index=len(windows), null_keys=True))
    return windows
