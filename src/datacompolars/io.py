"""Abertura de fontes de dados como LazyFrame.

Aceita pl.LazyFrame, pl.DataFrame, caminho de arquivo, glob (`dados/*.parquet`) ou pasta
(parquet particionado; procura em subpastas se `SourceOptions.recursive`).
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

from .settings import SourceOptions

Source = Union[pl.LazyFrame, pl.DataFrame, str, Path]

_EXTENSIONS = {
    "parquet": (".parquet", ".pq"),
    "csv": (".csv",),
    "sas": (".sas7bdat", ".xpt"),
}
_FORMAT_BY_EXT = {ext: fmt for fmt, exts in _EXTENSIONS.items() for ext in exts}
_GLOB_CHARS = set("*?[")


def open_source(source: Source, options: Optional[SourceOptions] = None) -> pl.LazyFrame:
    opts = options or SourceOptions()
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        return _open_path(str(source), opts)
    raise TypeError(
        f"Unsupported source type: {type(source).__name__} (use a path, pl.DataFrame or pl.LazyFrame)"
    )


def _open_path(raw: str, opts: SourceOptions) -> pl.LazyFrame:
    if _GLOB_CHARS & set(raw):
        fmt = opts.format or _FORMAT_BY_EXT.get(Path(raw).suffix.lower())
        if fmt is None:
            raise ValueError(f"Cannot infer the file format from the pattern '{raw}'; set SourceOptions.format")
        return _scan(fmt, _expand_globs([raw]), opts)

    path = Path(raw)
    if path.is_dir():
        fmt, files = _files_in_directory(path, opts)
        return _scan(fmt, [str(f) for f in files], opts)
    if path.is_file():
        fmt = opts.format or _FORMAT_BY_EXT.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"Unsupported file format: '{path.name}'. Set SourceOptions.format.")
        return _scan(fmt, [str(path)], opts)
    raise FileNotFoundError(f"Source not found: {raw}")


def _files_in_directory(folder: Path, opts: SourceOptions):
    formats = [opts.format] if opts.format else list(_EXTENSIONS)
    for fmt in formats:
        found: List[Path] = []
        for ext in _EXTENSIONS[fmt]:
            it = folder.rglob(f"*{ext}") if opts.recursive else folder.glob(f"*{ext}")
            for f in it:
                parts = f.relative_to(folder).parts
                if f.is_file() and not any(p.startswith((".", "_")) for p in parts):
                    found.append(f)
        if found:
            return fmt, sorted(found)
    raise FileNotFoundError(f"No parquet/csv/sas files found in '{folder}'")

def _expand_globs(paths: List[str]) -> List[str]:
    """Expande padrões glob em caminhos concretos, para que todos os formatos falhem cedo
    com FileNotFoundError quando nada casa (scan_readstat também não expande padrões)."""
    expanded: List[str] = []
    for p in paths:
        if _GLOB_CHARS & set(p):
            matches = sorted(m for m in glob.glob(p, recursive=True) if Path(m).is_file())
            if not matches:
                raise FileNotFoundError(f"No files match pattern: {p}")
            expanded.extend(matches)
        else:
            expanded.append(p)
    return expanded

def _scan(fmt: str, paths: List[str], opts: SourceOptions) -> pl.LazyFrame:
    if fmt == "parquet":
        return pl.scan_parquet(paths if len(paths) > 1 else paths[0])
    if fmt == "csv":
        return pl.scan_csv(
            paths if len(paths) > 1 else paths[0],
            infer_schema_length=opts.csv_infer_schema_length,
            separator=opts.csv_separator,
        )
    if fmt == "sas":
        try:
            from polars_readstat import scan_readstat  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading SAS files requires: pip install 'datacompolars[sas]'") from e
        frames = [scan_readstat(p) for p in paths]
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
    raise ValueError(f"Unknown format: {fmt}")
