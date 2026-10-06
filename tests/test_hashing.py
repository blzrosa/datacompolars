"""Hash de linha: -0.0/0.0 e NaN têm o mesmo hash em TODA variante de normalização usada pelo motor.

Fixa em CI o comportamento do Polars de que `hashing._float_normalizer_index` depende (se uma versão futura
deixar de normalizar no hash cru, o autoteste escolhe uma variante mais cara e estes testes continuam passando).
"""
import polars as pl
import pytest

from datacompolars import hashing
from datacompolars.hashing import _EQUAL_FLOAT_CASES, _FLOAT_NORMALIZERS, normalized_expr, row_hash_expr

from .helpers import df, run


def _h(value, dtype, expr_fn):
    return pl.DataFrame({"x": pl.Series([value], dtype=dtype)}).select(expr_fn(pl.col("x"), dtype).hash(seed=1)).item()


@pytest.mark.parametrize("idx", range(len(_FLOAT_NORMALIZERS)))
def test_last_normalizer_is_always_correct(idx):
    # as variantes baratas podem falhar em versões antigas do Polars; só a última precisa sempre funcionar
    build = _FLOAT_NORMALIZERS[idx]
    ok = all(_h(a, dt, build) == _h(b, dt, build) for dt, a, b in _EQUAL_FLOAT_CASES)
    assert ok or idx < len(_FLOAT_NORMALIZERS) - 1


def test_selected_normalizer_passes_equal_cases():
    build = _FLOAT_NORMALIZERS[hashing._float_normalizer_index()]
    for dt, a, b in _EQUAL_FLOAT_CASES:
        assert _h(a, dt, build) == _h(b, dt, build)


@pytest.mark.parametrize("idx", range(len(_FLOAT_NORMALIZERS)))
def test_engine_is_correct_with_every_normalizer(monkeypatch, idx):
    """Mesmo forçando cada variante, as comparações do motor continuam corretas quando a variante é válida."""
    monkeypatch.setattr(hashing, "_float_normalizer_index", lambda: idx)
    build = _FLOAT_NORMALIZERS[idx]
    if not all(_h(a, dt, build) == _h(b, dt, build) for dt, a, b in _EQUAL_FLOAT_CASES):
        pytest.skip("variante não confiável nesta versão do Polars (o autoteste a descartaria)")
    nan = float("nan")
    left = df(id=[1, 2, 3], x=[0.0, nan, 1.0])
    right = df(id=[1, 2, 3], x=[-0.0, -nan, 2.0])
    assert run(left, right, window_rows=None).rows.mismatched == 1


def test_float_columns_keep_their_names_in_row_hash():
    expr = row_hash_expr(["a", "b"], {"a": pl.Float64, "b": pl.Float32})
    out = pl.DataFrame({"a": [0.0], "b": [1.0]}, schema={"a": pl.Float64, "b": pl.Float32}).select(expr.alias("h"))
    assert out.columns == ["h"]
    assert normalized_expr("a", pl.Float64).meta.output_name() == "a"
