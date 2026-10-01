"""Alinhamento de schemas: nomes, colunas exclusivas, tipos divergentes e casting.

São funções puras sobre schemas (dict nome -> dtype); não tocam nos dados.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import polars as pl


@dataclass
class SchemaAlignment:
    join_columns: List[str]
    left_schema: Dict[str, pl.DataType]
    right_schema: Dict[str, pl.DataType]
    compare_columns: List[str] = field(default_factory=list)
    left_only: List[str] = field(default_factory=list)
    right_only: List[str] = field(default_factory=list)
    type_mismatches: List[Tuple[str, str, str]] = field(default_factory=list)  # (coluna, esquerda, direita)
    left_casts: Dict[str, pl.DataType] = field(default_factory=dict)
    right_casts: Dict[str, pl.DataType] = field(default_factory=dict)
    fatal_errors: List[str] = field(default_factory=list)


def normalize_schema(
    schema: Mapping[str, pl.DataType], lowercase: bool
) -> Tuple[Dict[str, pl.DataType], Dict[str, str]]:
    """Devolve (schema normalizado, mapa de renomeação). Erro se dois nomes colidirem."""
    if not lowercase:
        return dict(schema), {}
    out: Dict[str, pl.DataType] = {}
    rename: Dict[str, str] = {}
    for name, dtype in schema.items():
        low = name.lower()
        if low in out:
            raise ValueError(f"Colunas colidem após converter nomes para minúsculas: '{low}'")
        out[low] = dtype
        if low != name:
            rename[name] = low
    return out, rename


def align_schemas(
    left: Mapping[str, pl.DataType],
    right: Mapping[str, pl.DataType],
    join_columns: Sequence[str],
    casting: Literal["left", "right", "none"] = "none",
) -> SchemaAlignment:
    pk = list(join_columns)
    aln = SchemaAlignment(join_columns=pk, left_schema=dict(left), right_schema=dict(right))

    aln.left_only = [c for c in left if c not in right]
    aln.right_only = [c for c in right if c not in left]

    missing_left = [c for c in pk if c not in left]
    missing_right = [c for c in pk if c not in right]
    if missing_left:
        aln.fatal_errors.append(f"Join keys missing from the left table: {missing_left}")
    if missing_right:
        aln.fatal_errors.append(f"Join keys missing from the right table: {missing_right}")
    if aln.fatal_errors:
        return aln

    def align(col: str, is_key: bool) -> bool:
        """True se a coluna ficou com o mesmo dtype nos dois lados."""
        dl, dr = aln.left_schema[col], aln.right_schema[col]
        if dl == dr:
            return True
        if casting == "left":
            aln.right_casts[col] = dl
            aln.right_schema[col] = dl
            return True
        if casting == "right":
            aln.left_casts[col] = dr
            aln.left_schema[col] = dr
            return True
        if is_key:
            aln.fatal_errors.append(f"Join key '{col}' has different types: left {dl} vs right {dr}")
        else:
            aln.type_mismatches.append((col, str(dl), str(dr)))
        return False

    for k in pk:
        align(k, is_key=True)
    if aln.fatal_errors:
        return aln

    for col in left:
        if col in right and col not in pk and align(col, is_key=False):
            aln.compare_columns.append(col)
    return aln
