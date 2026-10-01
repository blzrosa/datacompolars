"""Helpers e modos de execução compartilhados (importados pelos testes: `from helpers import ...`).

PREMISSA DE API (única linha a ajustar se o nome for outro): `compare(left, right, settings)`
devolve um `ComparisonResult`. Todo o resto usa os modelos de results.py / settings.py.
"""
from __future__ import annotations

import polars as pl
import pytest

from datacompolars import CompareSettings, compare

NAN = float("nan")

# Caminho exato (hash de linha) e caminho com comparadores por coluna (qualquer tolerância > 0
# tira o motor do hash). Cada um, com janela única e forçando várias janelas (window_rows=2).
EXACT = [
    pytest.param({"window_rows": None}, id="hash"),
    pytest.param({"window_rows": 2}, id="hash-windows"),
]
COLUMNWISE = [
    pytest.param({"abs_tol": 1e-12, "window_rows": None}, id="columnwise"),
    pytest.param({"abs_tol": 1e-12, "window_rows": 2}, id="columnwise-windows"),
]
ALL_MODES = EXACT + COLUMNWISE


def run(left, right, **kwargs):
    kwargs.setdefault("join_columns", ["id"])
    return compare(left, right, CompareSettings(**kwargs))


def df(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols)
