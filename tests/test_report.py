"""Relatório (todos os formatos) e ponta a ponta com arquivos gerados por benchmarks/datagen.py."""
import json

import polars as pl
import pytest

from benchmarks.datagen import DatasetSpec, generate
from datacompolars import CompareSettings, ReportSettings, compare
from helpers import df, run

FORMATS = ["text", "markdown", "html", "json"]


def _mismatching():
    left = df(id=list(range(30)), a=list(range(30)), s=["x"] * 30)
    right = df(id=list(range(1, 31)), a=[v + 1 if v < 12 else v for v in range(1, 31)], s=["x"] * 29 + ["y"])
    return run(left, right, left_name="base", right_name="novo", sample_count=3)


def _quiet(**kw):
    return ReportSettings(print_output=False, **kw)


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_renders_names_and_columns(fmt):
    text = _mismatching().report(_quiet(format=fmt))
    assert "base" in text and "novo" in text and "a" in text
    if fmt == "json":
        data = json.loads(text)
        assert data["rows"]["mismatched"] > 0 and data["left_name"] == "base"


@pytest.mark.parametrize("fmt", FORMATS)
def test_match_and_aborted_results_render(fmt):
    d = df(id=[1, 2], a=[1, 2])
    assert run(d, d).report(_quiet(format=fmt))
    aborted = run(d, df(other=[1], a=[1]))
    assert aborted.aborted and aborted.report(_quiet(format=fmt))


def test_ascii_style_has_no_unicode():
    assert _mismatching().report(_quiet(style="ascii")).isascii()


def test_print_output(capsys):
    text = _mismatching().report(ReportSettings(print_output=True))
    assert text.strip() in capsys.readouterr().out
    _mismatching().report(_quiet())
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ext,fmt", [("md", "markdown"), ("html", "html"), ("json", "json"), ("txt", "text")])
def test_save_path_infers_format_and_writes_utf8(tmp_path, ext, fmt):
    path = tmp_path / "out" / f"report.{ext}"
    r = _mismatching()
    text = r.report(_quiet(save_path=path))
    assert path.read_text(encoding="utf-8") == text
    assert text == r.report(_quiet(format=fmt))


def test_max_columns_limits_text_but_json_keeps_all():
    n_cols = 30
    left = df(id=[1, 2], **{f"c{i:02d}": [1, 2] for i in range(n_cols)})
    right = df(id=[1, 2], **{f"c{i:02d}": [1, 3] for i in range(n_cols)})
    r = run(left, right)
    text = r.report(_quiet(max_columns=5))
    assert "c29" not in text or "c05" not in text  # lista truncada
    assert len(json.loads(r.report(_quiet(format="json")))["column_stats"]) == n_cols


def test_to_json_roundtrip():
    r = _mismatching()
    data = json.loads(r.to_json())
    assert data["rows"]["mismatched"] == r.rows.mismatched
    assert data["is_match"] is False and data["aborted"] is False


# ------------------------------------------------------------------ ponta a ponta (arquivos)
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    spec = DatasetSpec(
        rows=6_000, cols=9, partitions=6, divergence=0.05, null_rate=0.05,
        only_left=0.01, only_right=0.01, left_only_cols=1, right_only_cols=2, row_group_size=500,
    )
    root, exp = generate(spec, tmp_path_factory.mktemp("data"))
    return root, exp


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"window_rows": None}, id="hash"),
        pytest.param({"window_rows": 1_000}, id="hash-windows"),
        pytest.param({"window_rows": None, "abs_tol": 1e-12}, id="columnwise"),
        pytest.param({"window_rows": 1_000, "abs_tol": 1e-12}, id="columnwise-windows"),
    ],
)
def test_generated_parquet_matches_expected(dataset, kw):
    root, exp = dataset
    r = compare(str(root / "base"), str(root / "compare"), CompareSettings(join_columns=["id"], **kw))
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (
        exp.common, exp.mismatched, exp.left_only, exp.right_only,
    )
    assert (r.rows.left_total, r.rows.right_total) == (exp.left_rows, exp.right_rows)
    assert {s.column: s.mismatches for s in r.column_stats} == {
        c: n for c, n in exp.column_mismatches.items() if n
    }
    assert r.columns.left_only == ["lo_0"] and r.columns.right_only == ["ro_0", "ro_1"]
    if kw["window_rows"]:
        assert r.execution.windows > 1


def test_csv_and_parquet_sources_agree(tmp_path):
    left = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "s": ["x", "y", "z"]})
    right = pl.DataFrame({"id": [1, 2, 3], "a": [1, 9, 3], "s": ["x", "y", "z"]})
    for ext, write in (("parquet", "write_parquet"), ("csv", "write_csv")):
        getattr(left, write)(tmp_path / f"l.{ext}")
        getattr(right, write)(tmp_path / f"r.{ext}")
        r = compare(str(tmp_path / f"l.{ext}"), str(tmp_path / f"r.{ext}"), CompareSettings(join_columns=["id"]))
        assert (r.rows.common, r.rows.mismatched) == (3, 1), ext
