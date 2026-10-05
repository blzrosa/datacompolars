"""Lote adaptativo dos detalhes e prefetch de janelas: não podem mudar o resultado, só o caminho até ele."""
import polars as pl
import pytest

from datacompolars import CompareSettings
from datacompolars import engine as engine_mod
from datacompolars.engine import _Comparison

from .helpers import run

N_COLS = 120


def _cols(n_rows: int, n_cols: int = N_COLS) -> dict:
    data = {"id": list(range(n_rows))}
    for j in range(n_cols):
        data[f"c{j:03d}"] = [(i * 7 + j) % 11 for i in range(n_rows)]
    return data


def _pair(n_rows: int = 40):
    left, right = _cols(n_rows), _cols(n_rows)
    for i, j in [(1, 0), (5, 59), (13, 60), (14, 119), (22, 3), (39, 118)]:  # espalhado por várias janelas
        right[f"c{j:03d}"][i] += 100
    return pl.DataFrame(left), pl.DataFrame(right)


def _signature(r):
    return r.rows, r.column_stats, r.samples


# ------------------------------------------------------------------ lote adaptativo
def test_batch_size_unit():
    d = pl.DataFrame(_cols(10))
    c = _Comparison(d, d, CompareSettings(join_columns=["id"]), None)
    assert [len(b) for b in c._batches(None)] == [50, 50, 20]  # janela inteira: piso columns_per_batch
    assert [len(b) for b in c._batches(10)] == [N_COLS]  # poucas linhas: um lote só
    assert [len(b) for b in c._batches(10_000_000)] == [50, 50, 20]  # muitas linhas: volta ao piso
    c2 = _Comparison(d, d, CompareSettings(join_columns=["id"], columns_per_batch=7), None)
    assert max(len(b) for b in c2._batches(10_000_000)) == 7  # o piso é respeitado


@pytest.mark.parametrize("window_rows", [None, 10])
def test_result_does_not_depend_on_batch_budget(monkeypatch, window_rows):
    left, right = _pair()
    wide = run(left, right, window_rows=window_rows)
    monkeypatch.setattr(engine_mod, "SUBSET_CELLS_BUDGET", 1)  # força o caminho de vários lotes
    narrow = run(left, right, window_rows=window_rows)
    assert wide.rows.mismatched == 6
    assert _signature(wide) == _signature(narrow)


# ------------------------------------------------------------------ prefetch
def test_prefetch_setting_default_is_off():
    assert CompareSettings(join_columns=["id"]).prefetch_windows is False


@pytest.mark.parametrize("extra", [{}, {"common_keys_only": True}, {"column_details": False}])
def test_prefetch_matches_no_prefetch(extra):
    left, right = _pair()
    base = run(left, right, window_rows=7, cache_windows=True, **extra)
    pre = run(left, right, window_rows=7, cache_windows=True, prefetch_windows=True, **extra)
    assert base.execution.windows > 1
    assert _signature(base) == _signature(pre)
    assert "prefetch_wait" in pre.execution.timings
    assert "cache_bg" in pre.execution.timings


def test_prefetch_without_windows_or_cache_is_a_noop():
    left, right = _pair()
    base = run(left, right, window_rows=None)
    assert _signature(base) == _signature(run(left, right, window_rows=None, prefetch_windows=True))
    no_cache = run(left, right, window_rows=7, cache_windows=False, prefetch_windows=True)
    assert _signature(base) == _signature(no_cache)
    assert "prefetch_wait" not in no_cache.execution.timings


def test_prefetch_with_common_keys_and_disjoint_windows():
    left = pl.DataFrame(_cols(30))
    right_data = _cols(30)
    right_data["id"] = [i + 10 for i in range(30)]  # só 10..29 em comum
    right = pl.DataFrame(right_data)
    base = run(left, right, window_rows=5, cache_windows=True, common_keys_only=True)
    pre = run(left, right, window_rows=5, cache_windows=True, common_keys_only=True, prefetch_windows=True)
    assert _signature(base) == _signature(pre)
    assert (base.rows.left_only, base.rows.right_only) == (pre.rows.left_only, pre.rows.right_only)
