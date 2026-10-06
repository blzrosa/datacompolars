"""Configurações (pydantic). Tudo que o usuário controla passa por aqui, com validação."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .comparators import BaseComparator

# Janelas de execução automáticas (window_rows="auto"), dimensionadas em CÉLULAS (linhas x colunas), não só em linhas:
# cada janela tem custo fixo (consultas, cache) que só compensa se a janela for grande, e o pico de memória é
# proporcional às células da janela. Regras:
#   - tabela (maior lado) com até AUTO_WINDOW_CELLS células: janela única (sem janelas, sem cache);
#   - acima disso: janelas de ~AUTO_WINDOW_CELLS células, nunca com menos de AUTO_WINDOW_MIN_ROWS linhas
#     (300 colunas -> 100 mil linhas; 100 colunas -> 300 mil; 10 colunas -> 3 milhões).
# Medido (benchmarks/, polars 1.44.2, janelas fixas de 100 mil linhas): 1M x 10 colunas, janela única 0,16 s x
# janelas+cache 0,31 s; 10M x 10, 1,5 s x 3,8 s (muitas janelas pequenas); 1M x 50 empate; 1M x 300 janelas 2,1 s com
# ~1,2 GB (10M x 300 em ~22 s com ~1,3 GB). Janelas menores que o row group dos arquivos decodificam o mesmo row group
# mais de uma vez (o ideal é um múltiplo do tamanho do row group).
AUTO_WINDOW_CELLS = 30_000_000
AUTO_WINDOW_MIN_ROWS = 100_000

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
            "'auto' liga quando linhas x colunas passa de 30 milhões de células, com janelas de ~30 milhões de células (mín. 100 mil linhas); um inteiro força janelas desse tamanho; None desliga."
        ),
    )
    cache_windows: Union[bool, Literal["auto"]] = Field(
        default="auto",
        description=(
            "Carrega as duas metades de cada janela em memória uma vez e roda hash, detalhes e amostras sobre "
            "elas, em vez de reler/decodificar o parquet a cada passada. 'auto' (padrão) liga no caminho exato "
            "(hash) quando há janelas; True liga também no caminho columnwise (mais memória, ganho pequeno); "
            "False desliga. Custa ~2 x janela x colunas x 8 bytes e é ignorado quando a janela passa de ~100M "
            "células (linhas x colunas x 2). Sem janelas (window_rows) não tem efeito."
        ),
    )
    prefetch_windows: bool = Field(
        default=False,
        description=(
            "Decodifica a janela k+1 numa thread de fundo enquanto a k é processada (hash, detalhes). Só tem "
            "efeito com o cache de janelas ligado e mais de uma janela. Dobra o pico de memória (~2 janelas em "
            "memória) em troca de sobrepor leitura e cálculo; meça antes de ligar. Fases de fundo aparecem em "
            "`timings` como `cache_bg`/`keys_bg` (não somam no total); `prefetch_wait` é o que não foi sobreposto."
        ),
    )
    common_keys_only: bool = Field(
        default=False,
        description=(
            "Para tabelas com ids muito diferentes: antes de comparar, restringe cada janela às chaves presentes "
            "nos dois lados (varredura só das chaves + semi join), de modo que o hash/a comparação por coluna "
            "processe só as linhas em comum. Se a 1ª coluna da PK for inteira, a comparação também fica limitada à "
            "interseção das faixas de ids dos dois lados. As linhas exclusivas continuam no relatório, contadas como "
            "(total - em comum), o que pressupõe chaves únicas (veja check_duplicate_keys). Com muita sobreposição de "
            "ids, custa uma varredura extra só das chaves."
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
