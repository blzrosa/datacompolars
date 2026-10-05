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
from typing import List, Literal, Optional, Tuple, Union

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
    window_cells: int,
    min_rows: int,
    n_cols: int = 1,
) -> Optional[int]:
    """Traduz o `window_rows` das configurações para um tamanho de janela (None = sem janelas).

    "auto": a tabela (maior lado) tem `linhas x n_cols` células; até `window_cells` é janela única. Acima disso, as
    janelas têm ~`window_cells` células (`window_cells // n_cols` linhas), com no mínimo `min_rows` linhas.
    """
    if setting is None:
        return None
    if setting == "auto":
        cols = max(n_cols, 1)
        if max(left.n, right.n) * cols <= window_cells:
            return None
        return max(min_rows, window_cells // cols)
    return int(setting)


def key_overlap(left: PkStats, right: PkStats) -> Optional[Tuple[int, int]]:
    """Faixa inclusiva [lo, hi] em que os dois lados têm chaves não nulas.

    None se as faixas são disjuntas ou se algum lado não tem chave não nula. Chaves fora dessa faixa
    não podem ter par no outro lado, então só precisam ser contadas (não comparadas).
    """
    if left.lo is None or left.hi is None or right.lo is None or right.hi is None:
        return None
    lo, hi = max(left.lo, right.lo), min(left.hi, right.hi)
    return (lo, hi) if lo <= hi else None


def rows_in_range(stats: PkStats, lo: int, hi: int) -> int:
    """Estimativa (premissa de ids densos) de linhas com chave não nula em [lo, hi], inclusive."""
    if stats.lo is None or stats.hi is None:
        return 0
    inside = min(hi, stats.hi) - max(lo, stats.lo) + 1
    if inside <= 0:
        return 0
    span = stats.hi - stats.lo + 1
    return -(-((stats.n - stats.nulls) * inside) // span)  # teto, em aritmética inteira


def plan_windows(
    left: PkStats,
    right: PkStats,
    rows_per_window: Optional[int],
    bounds: Optional[Tuple[int, int]] = None,
) -> List[Window]:
    """Divide a faixa [min, max] da chave em janelas de largura igual (~rows_per_window linhas).

    Premissa: ids razoavelmente densos. Em ids muito esparsos/enviesados as janelas ficam
    desiguais (o resultado continua correto; só o pico de memória deixa de ser uniforme).

    Com `bounds` (faixa inclusiva, normalmente `key_overlap`), só essa faixa é dividida e as chaves nulas
    ganham janela própria; o que está fora de `bounds` fica a cargo de quem chama. Mesmo quando cabe
    numa janela só, ela leva o predicado da faixa (para podar row groups na leitura).
    """
    has_nulls = bool(left.nulls or right.nulls)
    if bounds is None:
        n = max(left.n, right.n)
        los = [s.lo for s in (left, right) if s.lo is not None]
        his = [s.hi for s in (left, right) if s.hi is not None]
        if rows_per_window is None or n <= rows_per_window or not los or not his:
            return [Window(index=0)]
        lo, hi = min(los), max(his)
    else:
        lo, hi = bounds
        n = max(rows_in_range(left, lo, hi), rows_in_range(right, lo, hi))
        if rows_per_window is None or n <= rows_per_window:
            single = [Window(index=0, lo=lo, hi=hi + 1)]
            if has_nulls:
                single.append(Window(index=1, null_keys=True))
            return single

    span = hi - lo + 1
    k = max(1, math.ceil(n / rows_per_window))
    width = max(1, math.ceil(span / k))

    windows: List[Window] = []
    start = lo
    while start <= hi:
        end = min(start + width, hi + 1)
        windows.append(Window(index=len(windows), lo=start, hi=end))
        start = end
    if has_nulls:
        windows.append(Window(index=len(windows), null_keys=True))
    return windows
