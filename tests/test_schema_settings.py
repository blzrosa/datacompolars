from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from datacompolars.comparators import NumericComparator
from datacompolars.schema import align_schemas, normalize_schema
from datacompolars.settings import CompareSettings, ReportSettings


# ------------------------------------------------------------------ schema
def test_normalize_schema_lowercases_and_maps_renames():
    out, rename = normalize_schema({"ID": pl.Int64, "a": pl.Utf8}, lowercase=True)
    assert list(out) == ["id", "a"] and rename == {"ID": "id"}


def test_normalize_schema_off_and_collision():
    out, rename = normalize_schema({"ID": pl.Int64}, lowercase=False)
    assert list(out) == ["ID"] and rename == {}
    with pytest.raises(ValueError, match="colidem"):
        normalize_schema({"A": pl.Int64, "a": pl.Int64}, lowercase=True)


def test_align_basic_columns():
    left = {"id": pl.Int64, "a": pl.Int64, "x": pl.Int64}
    right = {"id": pl.Int64, "a": pl.Int64, "y": pl.Int64}
    aln = align_schemas(left, right, ["id"])
    assert aln.compare_columns == ["a"]
    assert (aln.left_only, aln.right_only) == (["x"], ["y"])
    assert not aln.fatal_errors


def test_align_missing_keys_is_fatal():
    aln = align_schemas({"id": pl.Int64}, {"other": pl.Int64}, ["id"])
    assert len(aln.fatal_errors) == 1 and "right" in aln.fatal_errors[0]


def test_align_type_mismatch_policies():
    left = {"id": pl.Int64, "a": pl.Int64}
    right = {"id": pl.Int64, "a": pl.String}
    none = align_schemas(left, right, ["id"], casting="none")
    assert none.compare_columns == [] and none.type_mismatches == [("a", "Int64", "String")]

    cl = align_schemas(left, right, ["id"], casting="left")
    assert cl.compare_columns == ["a"] and cl.right_casts == {"a": pl.Int64}

    cr = align_schemas(left, right, ["id"], casting="right")
    assert cr.compare_columns == ["a"] and cr.left_casts == {"a": pl.String}


def test_align_key_type_mismatch():
    left, right = {"id": pl.Int64, "a": pl.Int64}, {"id": pl.Int32, "a": pl.Int64}
    assert align_schemas(left, right, ["id"], casting="none").fatal_errors
    assert not align_schemas(left, right, ["id"], casting="left").fatal_errors


# ------------------------------------------------------------------ settings
def test_defaults_use_exact_hash():
    s = CompareSettings(join_columns=["id"])
    assert s.uses_exact_hash() and s.window_rows == "auto" and s.lowercase_columns


@pytest.mark.parametrize(
    "kw",
    [
        {"abs_tol": 0.1},
        {"rel_tol": {"a": 0.1}},
        {"ignore_spaces": True},
        {"ignore_case": True},
        {"custom_comparators": {"a": NumericComparator()}},
    ],
)
def test_options_that_disable_the_hash_path(kw):
    assert not CompareSettings(join_columns=["id"], **kw).uses_exact_hash()


def test_zero_tolerance_dict_keeps_hash_path():
    assert CompareSettings(join_columns=["id"], abs_tol={"a": 0.0}).uses_exact_hash()


@pytest.mark.parametrize(
    "kw",
    [
        {"join_columns": []},
        {"join_columns": ["id", "ID"]},
        {"join_columns": [" "]},
        {"join_columns": ["id"], "typo_option": 1},  # extra="forbid"
        {"join_columns": ["id"], "abs_tol": -1.0},
        {"join_columns": ["id"], "rel_tol": float("inf")},
        {"join_columns": ["id"], "abs_tol": {"a": float("nan")}},
        {"join_columns": ["id"], "window_rows": 0},
        {"join_columns": ["id"], "window_rows": True},
        {"join_columns": ["id"], "window_rows": "weekly"},
        {"join_columns": ["id"], "columns_per_batch": 0},
        {"join_columns": ["id"], "sample_count": -1},
        {"join_columns": ["id"], "custom_comparators": {"ID": NumericComparator()}},
    ],
)
def test_invalid_settings_are_rejected(kw):
    with pytest.raises(ValidationError):
        CompareSettings(**kw)


def test_valid_window_rows():
    for v in (None, "auto", 1, 10_000_000):
        assert CompareSettings(join_columns=["id"], window_rows=v).window_rows == v


# ------------------------------------------------------------------ report settings
@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, "text"),
        ({"save_path": "r.md"}, "markdown"),
        ({"save_path": "r.HTML"}, "html"),
        ({"save_path": "r.json"}, "json"),
        ({"save_path": "r.csv"}, "text"),
        ({"save_path": "r.md", "format": "json"}, "json"),  # explícito vence a extensão
    ],
)
def test_resolved_format(kw, expected):
    assert ReportSettings(**kw).resolved_format() == expected


def test_report_settings_limits():
    assert ReportSettings(save_path="x/y.txt").save_path == Path("x/y.txt")
    with pytest.raises(ValidationError):
        ReportSettings(width=20)
    with pytest.raises(ValidationError):
        ReportSettings(unknown=1)
