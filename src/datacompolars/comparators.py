"""Comparadores por coluna. Todos devolvem uma expressão booleana lazy (pl.Expr)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseComparator(ABC):
    """Interface dos comparadores.

    Recebe os NOMES das colunas esquerda/direita (já presentes no frame unido) e devolve uma
    expressão booleana: True = valores considerados iguais. O motor aplica a regra de nulos
    por cima (nulo/nulo = igual; nulo de um lado só = diferente), então o comparador não
    precisa se preocupar com isso.

    O motor repassa em **kwargs: abs_tol, rel_tol, ignore_spaces, ignore_case, is_float.
    """

    @abstractmethod
    def compare(self, col_l: str, col_r: str, **kwargs: Any) -> pl.Expr: ...


class NumericComparator(BaseComparator):
    """Igualdade exata ou com tolerância: |l - r| <= abs_tol + rel_tol * |r|."""

    def compare(
        self,
        col_l: str,
        col_r: str,
        *,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        is_float: bool = False,
        **kwargs: Any,
    ) -> pl.Expr:
        left, right = pl.col(col_l), pl.col(col_r)
        exact = left.eq_missing(right)
        if is_float:
            exact = exact | (left.is_nan() & right.is_nan())  # NaN == NaN
        if abs_tol == 0.0 and rel_tol == 0.0:
            return exact
        within = ((left - right).abs() <= abs_tol + rel_tol * right.abs()).fill_null(False)
        return exact | within


class StringComparator(BaseComparator):
    """Igualdade de strings, opcionalmente ignorando caixa e/ou espaços em branco."""

    def compare(
        self,
        col_l: str,
        col_r: str,
        *,
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        **kwargs: Any,
    ) -> pl.Expr:
        # cast: o namespace .str não existe em Categorical/Enum
        left = pl.col(col_l).cast(pl.String)
        right = pl.col(col_r).cast(pl.String)
        if ignore_case:
            left, right = left.str.to_lowercase(), right.str.to_lowercase()
        if ignore_spaces:
            left = left.str.replace_all(r"\s+", "")
            right = right.str.replace_all(r"\s+", "")
        return left.eq_missing(right)


class ArrayComparator(BaseComparator):
    """Listas comparadas sem considerar a ordem dos elementos."""

    def compare(self, col_l: str, col_r: str, **kwargs: Any) -> pl.Expr:
        left = pl.col(col_l).list.eval(pl.element().sort())
        right = pl.col(col_r).list.eval(pl.element().sort())
        return left.eq_missing(right)
