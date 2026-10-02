"""open_source: erros e comportamentos que NÃO dependem do extra `sas` (rodam no núcleo)."""
from __future__ import annotations

import sys

import polars as pl
import pytest

from datacompolars import SourceOptions
from datacompolars.io import open_source


def test_sas_without_extra_raises_friendly_error(tmp_path, monkeypatch):
    """Sem polars-readstat instalado, a mensagem deve ensinar a instalar o extra."""
    f = tmp_path / "dados.sas7bdat"
    f.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "polars_readstat", None)  # força ImportError no import

    with pytest.raises(ImportError, match=r"datacompolars\[sas\]"):
        open_source(f)


def test_unsupported_source_type():
    with pytest.raises(TypeError, match="Unsupported source type"):
        open_source(123)  # type: ignore[arg-type]


def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "nao_existe.parquet")


def test_unknown_extension_asks_for_format(tmp_path):
    f = tmp_path / "dados.txt"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="SourceOptions.format"):
        open_source(f)


def test_empty_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No parquet/csv/sas files"):
        open_source(tmp_path)


def test_directory_ignores_hidden_and_underscore_files(tmp_path):
    pl.DataFrame({"id": [1, 2]}).write_parquet(tmp_path / "part-0.parquet")
    pl.DataFrame({"id": [99]}).write_parquet(tmp_path / "_temp.parquet")
    (tmp_path / ".hidden").mkdir()
    pl.DataFrame({"id": [98]}).write_parquet(tmp_path / ".hidden" / "x.parquet")

    assert open_source(tmp_path).collect()["id"].to_list() == [1, 2]


def test_csv_separator_and_forced_format(tmp_path):
    f = tmp_path / "dados.txt"  # extensão desconhecida: só abre porque o formato foi forçado
    f.write_text("id;valor\n1;10\n2;20\n")

    df = open_source(f, SourceOptions(format="csv", csv_separator=";")).collect()

    assert df.columns == ["id", "valor"]
    assert df.height == 2

def test_glob_without_matches_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "*.parquet"))