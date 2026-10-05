"""A expressão de igualdade enxuta (sem o invólucro de nulos) é equivalente à antiga em TODOS os pares de valores."""
import itertools

import polars as pl
import pytest

from datacompolars import CompareSettings
from datacompolars.comparators import NumericComparator, StringComparator
from datacompolars.engine import _Comparison

NAN = float("nan")
FLOATS = [None, 0.0, -0.0, 1.0, 1.0 + 1e-9, 2.5, NAN, -NAN]
INTS = [None, 0, 1, 2, -3]
STRS = [None, "", "a", "A", " a ", "b"]


def _old(comp_expr: pl.Expr) -> pl.Expr:
    a, b = pl.col("x"), pl.col("__r_x")
    return (a.is_null() & b.is_null()) | (a.is_not_null() & b.is_not_null() & comp_expr.fill_null(False))


def _frame(values, dtype):
    pairs = list(itertools.product(values, values))
    return pl.DataFrame(
        {"x": pl.Series([p[0] for p in pairs], dtype=dtype), "__r_x": pl.Series([p[1] for p in pairs], dtype=dtype)}
    )


def _engine(dtype, **kw):
    one = pl.DataFrame({"id": [1], "x": pl.Series([None], dtype=dtype)})
    return _Comparison(one, one, CompareSettings(join_columns=["id"], **kw), None)


@pytest.mark.parametrize("tol", [{}, {"abs_tol": 1e-6}, {"rel_tol": 1e-3}, {"abs_tol": 0.5, "rel_tol": 1e-3}])
@pytest.mark.parametrize("dtype,values", [(pl.Float64, FLOATS), (pl.Float32, FLOATS), (pl.Int64, INTS)])
def test_numeric_match_equals_old_formula(dtype, values, tol):
    eng = _engine(dtype, **tol)
    df = _frame(values, dtype)
    is_float = dtype.is_float()
    m = NumericComparator().compare(
        "x", "__r_x", abs_tol=tol.get("abs_tol", 0.0), rel_tol=tol.get("rel_tol", 0.0), is_float=is_float
    )
    got = df.select(eng._build_match_expr("x").alias("m")).to_series()
    want = df.select(_old(m).alias("m")).to_series()
    assert got.null_count() == 0
    assert got.to_list() == want.to_list()


@pytest.mark.parametrize("opts", [{}, {"ignore_case": True}, {"ignore_spaces": True}, {"ignore_case": True, "ignore_spaces": True}])
def test_string_match_equals_old_formula(opts):
    eng = _engine(pl.String, **opts)
    df = _frame(STRS, pl.String)
    m = StringComparator().compare("x", "__r_x", **opts)
    got = df.select(eng._build_match_expr("x").alias("m")).to_series()
    want = df.select(_old(m).alias("m")).to_series()
    assert got.to_list() == want.to_list()
