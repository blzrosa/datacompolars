"""Modelos de resultado (pydantic): serializáveis em JSON e consumíveis por código ou relatório."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

if TYPE_CHECKING:  # pragma: no cover
    from .settings import ReportSettings


class RowSummary(BaseModel):
    left_total: int = Field(ge=0, description="Linhas no lado esquerdo (comum + só esquerda).")
    right_total: int = Field(ge=0, description="Linhas no lado direito (comum + só direita).")
    common: int = Field(ge=0, description="Linhas cuja chave existe nos dois lados.")
    left_only: int = Field(ge=0)
    right_only: int = Field(ge=0)
    mismatched: int = Field(ge=0, description="Linhas em comum com ao menos uma coluna diferente.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matched(self) -> int:
        return self.common - self.mismatched

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mismatch_rate(self) -> float:
        return self.mismatched / self.common if self.common else 0.0


class TypeMismatch(BaseModel):
    column: str
    left: str
    right: str


class ColumnSummary(BaseModel):
    join_columns: List[str]
    compared: List[str] = Field(default_factory=list, description="Colunas comparadas (exclui a chave).")
    left_only: List[str] = Field(default_factory=list)
    right_only: List[str] = Field(default_factory=list)
    type_mismatches: List[TypeMismatch] = Field(
        default_factory=list, description="Colunas com tipos diferentes que NÃO foram comparadas."
    )


class ColumnStat(BaseModel):
    column: str
    dtype: str
    mismatches: int = Field(ge=0)
    rate: float = Field(ge=0.0, description="mismatches / linhas em comum")


class SampleTable(BaseModel):
    column: str
    total_mismatches: int
    headers: List[str] = Field(description="Chave(s), valor à esquerda, valor à direita.")
    rows: List[List[Any]]


class ExecutionInfo(BaseModel):
    path: Optional[Literal["hash", "columnwise"]] = Field(
        default=None, description="'hash': igualdade exata por hash de linha; 'columnwise': comparadores por coluna."
    )
    path_reason: str = ""
    windows: int = 1
    rows_per_window: Optional[int] = None
    column_details_computed: bool = False
    polars_version: str = ""
    total_seconds: float = 0.0
    timings: Dict[str, float] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_name: str
    right_name: str
    join_columns: List[str]
    rows: RowSummary
    columns: ColumnSummary
    column_stats: List[ColumnStat] = Field(default_factory=list)
    samples: List[SampleTable] = Field(default_factory=list)
    fatal_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def aborted(self) -> bool:
        return bool(self.fatal_errors)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_match(self) -> bool:
        """True se os dois lados são iguais: mesmas linhas, mesmos valores e mesmas colunas."""
        return (
            not self.aborted
            and self.rows.mismatched == 0
            and self.rows.left_only == 0
            and self.rows.right_only == 0
            and not self.columns.left_only
            and not self.columns.right_only
            and not self.columns.type_mismatches
        )

    def report(self, settings: Optional["ReportSettings"] = None) -> str:
        """Gera (e opcionalmente imprime/salva) o relatório. Devolve o texto."""
        from .report import emit
        from .settings import ReportSettings

        return emit(self, settings or ReportSettings())

    def to_json(self, indent: Optional[int] = 2) -> str:
        return self.model_dump_json(indent=indent)
