"""Leitura de SAS pelo extra opcional `sas`.

Rode com:  uv run --extra sas pytest -m sas -v -rs

Fixture: um .sas7bdat REAL (não dá para gerar .sas7bdat em Python: o pyreadstat só escreve .xpt).
Ordem de busca:
  1. tests/fixtures/sample.sas7bdat  (recomendado: versione um arquivo pequeno no repositório)
  2. o test1.sas7bdat que o próprio pandas distribui em pandas/tests/io/sas/data/
  3. senão, os testes que dependem dele são pulados (veja o motivo com -rs)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.sas

pytest.importorskip("polars_readstat")

from datacompolars import CompareSettings, SourceOptions, compare  # noqa: E402
from datacompolars.io import open_source  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- fixture e helpers
@pytest.fixture(scope="module")
def sample_sas() -> Path:
    fixture = FIXTURES / "sample.sas7bdat"
    if fixture.is_file():
        return fixture
    pd = pytest.importorskip("pandas")
    bundled = Path(pd.__file__).parent / "tests" / "io" / "sas" / "data" / "test1.sas7bdat"
    if bundled.is_file():
        return bundled
    pytest.skip("sem .sas7bdat de exemplo: coloque um em tests/fixtures/sample.sas7bdat")


def _pick_key(df: pl.DataFrame) -> str:
    """Primeira coluna sem nulos e com valores únicos (a amostra não precisa ter uma PK declarada)."""
    for name in df.columns:
        s = df[name]
        if s.null_count() == 0 and s.n_unique() == df.height:
            return name
    pytest.skip("a amostra não tem coluna única para usar como chave")


def _pick_float_column(df: pl.DataFrame, key: str) -> str:
    for name, dtype in df.schema.items():
        if name != key and dtype.is_float() and df[name].null_count() < df.height:
            return name
    pytest.skip("a amostra não tem coluna numérica para alterar")


# --------------------------------------------------------------------------- .sas7bdat
def test_open_sas7bdat_returns_lazyframe_with_rows(sample_sas):
    lf = open_source(sample_sas)

    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert df.height > 0
    assert df.width > 1


def test_sas7bdat_identical_files_match(sample_sas):
    """O mesmo arquivo dos dois lados tem de dar igualdade total (nulos, texto, datas e floats)."""
    key = _pick_key(open_source(sample_sas).collect())

    r = compare(sample_sas, sample_sas, CompareSettings(join_columns=[key]))

    assert not r.aborted, r.fatal_errors
    assert r.is_match
    assert r.rows.left_only == 0 and r.rows.right_only == 0 and r.rows.mismatched == 0


def test_sas7bdat_vs_parquet_detects_single_changed_cell(sample_sas, tmp_path):
    """SAS de um lado, parquet do outro, com UMA célula alterada: tem de achar exatamente essa."""
    df = open_source(sample_sas).collect()
    key = _pick_key(df)
    col = _pick_float_column(df, key)

    idx = df[col].is_not_null().arg_max()  # primeira linha com valor
    changed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == idx).then(pl.col(col) + 1.0).otherwise(pl.col(col)).alias(col)
    )
    right = tmp_path / "right.parquet"
    changed.write_parquet(right)

    r = compare(sample_sas, right, CompareSettings(join_columns=[key]))

    assert not r.aborted, r.fatal_errors
    assert r.rows.common == df.height
    assert r.rows.mismatched == 1
    stats = {s.column: s.mismatches for s in r.column_stats}
    assert stats == {col.lower(): 1}


def test_directory_of_sas_files_is_concatenated(sample_sas, tmp_path):
    n = open_source(sample_sas).collect().height
    shutil.copy(sample_sas, tmp_path / "a.sas7bdat")
    (tmp_path / "sub").mkdir()
    shutil.copy(sample_sas, tmp_path / "sub" / "b.sas7bdat")

    assert open_source(tmp_path).collect().height == 2 * n
    assert open_source(tmp_path, SourceOptions(recursive=False)).collect().height == n


def test_glob_of_sas_files_is_expanded(sample_sas, tmp_path):
    """O glob precisa casar vários arquivos. Se falhar aqui, o scan_readstat não expande padrões
    (diferente do parquet/csv do Polars) e o io.py precisa expandir o glob antes de chamá-lo."""
    n = open_source(sample_sas).collect().height
    shutil.copy(sample_sas, tmp_path / "a.sas7bdat")
    shutil.copy(sample_sas, tmp_path / "b.sas7bdat")

    lf = open_source(str(tmp_path / "*.sas7bdat"))

    assert lf.collect().height == 2 * n


# --------------------------------------------------------------------------- .xpt
def test_xpt_read_and_compare(tmp_path):
    """README e io.py prometem .xpt. O polars-readstat documenta só sas7bdat/dta/sav: este teste
    confirma (ou derruba) a promessa. Os arquivos .xpt são gerados com pyreadstat."""
    pd = pytest.importorskip("pandas")
    pyreadstat = pytest.importorskip("pyreadstat")

    left, right = tmp_path / "left.xpt", tmp_path / "right.xpt"
    pyreadstat.write_xport(
        pd.DataFrame({"id": [1.0, 2.0, 3.0], "valor": [10.0, 20.0, 30.0]}), str(left), table_name="DADOS"
    )
    pyreadstat.write_xport(
        pd.DataFrame({"id": [1.0, 2.0, 4.0], "valor": [10.0, 20.5, 40.0]}), str(right), table_name="DADOS"
    )

    r = compare(left, right, CompareSettings(join_columns=["id"]))

    assert not r.aborted, r.fatal_errors
    assert r.rows.common == 2
    assert r.rows.mismatched == 1
    assert r.rows.left_only == 1
    assert r.rows.right_only == 1
