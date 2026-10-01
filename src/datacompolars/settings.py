"""Configurações (pydantic). Tudo que o usuário controla passa por aqui, com validação."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .comparators import BaseComparator

# Janelas de execução automáticas (window_rows="auto").
# Medido com 30M x 50 colunas (row groups de ~100k linhas): a query única de hash usa ~3,7 GB.
# Acima do limiar, o motor fatia a comparação em janelas de ~AUTO_WINDOW_ROWS linhas por faixa
# de PK, e o pico de memória passa a depender do tamanho da janela, não do dataset.
AUTO_WINDOW_THRESHOLD_ROWS = 20_000_000
AUTO_WINDOW_ROWS = 10_000_000

PathLike = Union[str, Path]


def _has_tolerance(value: Union[float, Dict[str, float]]) -> bool:
    if isinstance(value, dict):
        return any(v != 0.0 for v in value.values())
    return value != 0.0


class CompareSettings(BaseModel):
    """Como comparar. `extra="forbid"`: um parâmetro com nome errado falha, em vez de ser ignorado."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    join_columns: List[str] = Field(min_length=1, description="Colunas da chave (PK) do join.")
    left_name: str = Field(default="left", min_length=1, description="Rótulo do lado esquerdo no relatório.")
    right_name: str = Field(default="right", min_length=1, description="Rótulo do lado direito no relatório.")
    lowercase_columns: bool = Field(default=True, description="Compara nomes de colunas sem diferenciar caixa.")
    casting: Literal["left", "right", "none"] = Field(
        default="none",
        description=(
            "Tipos divergentes: 'left' converte a direita para o tipo da esquerda, 'right' o inverso, "
            "'none' apenas reporta (coluna ignorada; chave com tipo divergente é erro fatal)."
        ),
    )
    abs_tol: Union[float, Dict[str, float]] = Field(default=0.0, description="Tolerância absoluta (global ou por coluna).")
    rel_tol: Union[float, Dict[str, float]] = Field(default=0.0, description="Tolerância relativa (global ou por coluna).")
    ignore_spaces: bool = Field(default=False, description="Ignora espaços em branco em strings.")
    ignore_case: bool = Field(default=False, description="Ignora caixa em strings.")
    custom_comparators: Dict[str, BaseComparator] = Field(
        default_factory=dict,
        description="Comparador próprio por coluna ({'coluna': MeuComparador()}).",
    )
    check_duplicate_keys: bool = Field(
        default=True,
        description="Valida unicidade da PK em cada lado (se desligado e houver duplicatas, as contagens inflam).",
    )
    column_details: bool = Field(
        default=True,
        description=(
            "Calcula divergências por coluna e amostras. No caminho exato custa uma leitura extra "
            "das janelas que têm divergência; desligue para só contar linhas."
        ),
    )
    sample_count: int = Field(default=5, ge=0, description="Amostras de linhas divergentes por coluna (0 desativa).")
    columns_per_batch: int = Field(default=50, gt=0, description="Colunas avaliadas por lote no caminho coluna a coluna.")
    window_rows: Union[int, Literal["auto"], None] = Field(
        default="auto",
        description=(
            "Janelas de execução por faixa da primeira coluna da PK (precisa ser inteira). "
            "'auto' liga acima de ~20M linhas; um inteiro força janelas desse tamanho; None desliga."
        ),
    )

    @field_validator("join_columns")
    @classmethod
    def _valid_join_columns(cls, v: List[str]) -> List[str]:
        if any(not c.strip() for c in v):
            raise ValueError("join_columns não pode conter nomes vazios")
        if len({c.lower() for c in v}) != len(v):
            raise ValueError("join_columns contém colunas repetidas")
        return v

    @field_validator("abs_tol", "rel_tol")
    @classmethod
    def _valid_tolerance(cls, v: Union[float, Dict[str, float]]) -> Union[float, Dict[str, float]]:
        values = list(v.values()) if isinstance(v, dict) else [v]
        for x in values:
            if not math.isfinite(x) or x < 0:
                raise ValueError("tolerâncias devem ser números finitos e >= 0")
        return v

    @field_validator("window_rows", mode="before")
    @classmethod
    def _valid_window_rows(cls, v: Union[int, str, None]) -> Union[int, str, None]:
        if isinstance(v, bool):
            raise ValueError("window_rows deve ser um inteiro positivo, 'auto' ou None")
        if isinstance(v, int) and v <= 0:
            raise ValueError("window_rows deve ser > 0")
        return v

    @model_validator(mode="after")
    def _comparators_for_known_keys(self) -> "CompareSettings":
        keys = {c.lower() if self.lowercase_columns else c for c in self.custom_comparators}
        pk = {c.lower() if self.lowercase_columns else c for c in self.join_columns}
        clash = keys & pk
        if clash:
            raise ValueError(f"custom_comparators não pode conter colunas da chave: {sorted(clash)}")
        return self

    def uses_exact_hash(self) -> bool:
        """True se a comparação pode usar o hash de linha (igualdade exata em todas as colunas)."""
        return not (
            _has_tolerance(self.abs_tol)
            or _has_tolerance(self.rel_tol)
            or self.ignore_spaces
            or self.ignore_case
            or bool(self.custom_comparators)
        )


class SourceOptions(BaseModel):
    """Como abrir arquivos (parquet, csv, SAS)."""

    model_config = ConfigDict(extra="forbid")

    format: Optional[Literal["parquet", "csv", "sas"]] = Field(
        default=None, description="Força o formato; por padrão é deduzido da extensão."
    )
    csv_infer_schema_length: Optional[int] = Field(
        default=10_000, ge=0, description="Linhas usadas para inferir tipos do CSV (None = arquivo inteiro)."
    )
    csv_separator: str = Field(default=",", min_length=1, max_length=1)
    recursive: bool = Field(default=True, description="Ao receber um diretório, procura arquivos em subpastas.")


class ReportSettings(BaseModel):
    """Como apresentar o relatório."""

    model_config = ConfigDict(extra="forbid")

    print_output: bool = Field(default=True, description="Imprime o relatório no console.")
    format: Optional[Literal["text", "markdown", "html", "json"]] = Field(
        default=None,
        description="Formato. Se None, é deduzido da extensão de save_path (.md/.html/.json); senão 'text'.",
    )
    style: Literal["auto", "unicode", "ascii"] = Field(
        default="auto", description="Bordas e barras do formato texto. 'auto' usa ascii se o console não suportar unicode."
    )
    width: int = Field(default=100, ge=60, le=200, description="Largura do formato texto.")
    max_columns: int = Field(default=20, ge=1, description="Máximo de colunas divergentes listadas (o JSON/HTML traz todas).")
    max_sample_columns: int = Field(default=10, ge=0, description="Máximo de colunas com tabela de amostras.")
    save_path: Optional[Path] = Field(default=None, description="Salva o relatório neste arquivo (UTF-8).")

    def resolved_format(self) -> str:
        if self.format is not None:
            return self.format
        if self.save_path is not None:
            return {".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html", ".json": "json"}.get(
                Path(self.save_path).suffix.lower(), "text"
            )
        return "text"
