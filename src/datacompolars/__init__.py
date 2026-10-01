"""datacompolars: comparação de DataFrames/arquivos em Polars (hash de linha, janelas por PK, relatório)."""
from .comparators import ArrayComparator, BaseComparator, NumericComparator, StringComparator
from .engine import compare
from .results import ComparisonResult
from .settings import CompareSettings, ReportSettings, SourceOptions

__all__ = [
    "compare",
    "CompareSettings",
    "ReportSettings",
    "SourceOptions",
    "ComparisonResult",
    "BaseComparator",
    "NumericComparator",
    "StringComparator",
    "ArrayComparator",
]
