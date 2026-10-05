import polars as pl
import pytest

from datacompolars.settings import AUTO_WINDOW_CELLS, AUTO_WINDOW_MIN_ROWS
from datacompolars.windows import PkStats, Window, pk_stats, plan_windows, resolve_window_rows


def stats(n, lo, hi, nulls=0):
    return PkStats(n=n, nulls=nulls, lo=lo, hi=hi)


# ------------------------------------------------------------------ planejamento (Python puro)
def test_no_windows_when_small_or_disabled():
    s = stats(100, 0, 99)
    assert plan_windows(s, s, None) == [Window(index=0)]
    assert plan_windows(s, s, 100) == [Window(index=0)]
    assert plan_windows(s, s, 1_000) == [Window(index=0)]
    assert Window(index=0).is_full


def test_empty_or_all_null_keys_give_single_window():
    empty = stats(0, None, None)
    assert plan_windows(empty, empty, 10) == [Window(index=0)]


@pytest.mark.parametrize("n,rows", [(1_000, 100), (1_001, 100), (7, 2), (10_000, 3_333)])
def test_windows_tile_the_key_range_without_gaps_or_overlap(n, rows):
    s = stats(n, 10, 10 + n - 1)
    ws = plan_windows(s, s, rows)
    assert len(ws) > 1
    assert ws[0].lo == 10 and ws[-1].hi == 10 + n
    for a, b in zip(ws, ws[1:]):
        assert a.hi == b.lo
    assert [w.index for w in ws] == list(range(len(ws)))


def test_range_covers_both_sides():
    ws = plan_windows(stats(100, 0, 99), stats(100, 50, 149), 50)
    assert ws[0].lo == 0 and ws[-1].hi == 150


def test_null_window_added_only_with_null_keys():
    assert not any(w.null_keys for w in plan_windows(stats(100, 0, 99), stats(100, 0, 99), 30))
    ws = plan_windows(stats(100, 0, 99, nulls=2), stats(100, 0, 99), 30)
    assert ws[-1].null_keys and ws[-1].lo is None and not ws[-1].is_full


def test_describe():
    assert Window(index=0).describe() == "all rows"
    assert Window(index=1, lo=0, hi=10).describe() == "[0, 10)"
    assert Window(index=2, null_keys=True).describe() == "null keys"


# ------------------------------------------------------------------ resolve_window_rows
@pytest.mark.parametrize(
    "setting,n,expected",
    [
        (None, 10**9, None),
        (500, 10, 500),
        ("auto", 1_000, None),
        ("auto", 20_000_000, None),  # no limiar (células) não liga; só acima
        ("auto", 20_000_001, 20_000_000),  # 1 coluna: janela de window_cells // 1 linhas (acima do piso min_rows)
    ],
)
def test_resolve_window_rows(setting, n, expected):
    out = resolve_window_rows(
        setting, stats(n, 0, n), stats(5, 0, 5), window_cells=20_000_000, min_rows=10_000_000
    )
    assert out == expected


def test_auto_defaults_are_30m_cells_and_100k_rows():
    assert (AUTO_WINDOW_CELLS, AUTO_WINDOW_MIN_ROWS) == (30_000_000, 100_000)
    small = stats(5, 0, 5)

    def auto(rows, cols):
        return resolve_window_rows(
            "auto", stats(rows, 0, rows), small, window_cells=AUTO_WINDOW_CELLS, min_rows=AUTO_WINDOW_MIN_ROWS, n_cols=cols
        )

    assert auto(1_000_000, 10) is None  # 10M células: janela única
    assert auto(3_000_000, 10) is None  # 30M: no limiar não liga
    assert auto(10_000_000, 10) == 3_000_000  # janelas de ~30M células
    assert auto(1_000_000, 50) == 600_000
    assert auto(1_000_000, 100) == 300_000
    assert auto(1_000_000, 300) == 100_000
    assert auto(10_000_000, 300) == 100_000
    assert auto(100_000, 300) is None  # 30M células
    assert auto(100_000, 301) == 100_000  # um pouco acima: piso de 100 mil linhas
    assert auto(1_000_000, 3_000) == 100_000  # muito largo: o piso manda
    other = resolve_window_rows(
        "auto", small, stats(1_000_000, 0, 1_000_000), window_cells=AUTO_WINDOW_CELLS, min_rows=AUTO_WINDOW_MIN_ROWS, n_cols=50
    )
    assert other == 600_000  # vale o maior lado


def test_auto_uses_the_larger_side():
    out = resolve_window_rows("auto", stats(5, 0, 5), stats(30, 0, 30), window_cells=20, min_rows=10)
    assert out == 20


def test_auto_counts_columns():
    kw = {"window_cells": 100, "min_rows": 10}
    assert resolve_window_rows("auto", stats(30, 0, 30), stats(5, 0, 5), n_cols=3, **kw) is None  # 90 <= 100
    assert resolve_window_rows("auto", stats(30, 0, 30), stats(5, 0, 5), n_cols=4, **kw) == 25  # 120 > 100 -> 100 // 4


# ------------------------------------------------------------------ com Polars
def test_pk_stats():
    lf = pl.LazyFrame({"id": [5, None, 1, 9, None]})
    assert pk_stats(lf, "id") == PkStats(n=5, nulls=2, lo=1, hi=9)
    assert pk_stats(pl.LazyFrame({"id": []}, schema={"id": pl.Int64}), "id") == PkStats(0, 0, None, None)


def test_predicates_partition_the_rows_exactly():
    ids = [None, None] + list(range(97))
    lf = pl.LazyFrame({"id": ids})
    s = pk_stats(lf, "id")
    ws = plan_windows(s, s, 10)
    counts = [lf.filter(w.predicate("id")).select(pl.len()).collect().item() for w in ws]
    assert sum(counts) == len(ids)  # cada linha (inclusive nula) cai em exatamente uma janela
    assert Window(index=0).predicate("id") is None