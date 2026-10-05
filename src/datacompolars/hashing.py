"""Hash de linha (64 bits) usado no caminho exato.

Medições (30M x 50 colunas, arquivos com row groups de ~100k linhas): o hash por coluna
combinado custa ~1,7 s e ~1,1 GB por lado; o hash de struct, ~6,5 s e ~2,0 GB.
A combinação das colunas é uma árvore balanceada (ver `row_hash_expr`), não uma cadeia.
"""
from __future__ import annotations

import struct
from functools import lru_cache
from typing import Callable, List, Mapping, Sequence, Tuple

import polars as pl

HASH_MULTIPLIER = 1_000_003  # ímpar; combinação posicional com aritmética UInt64 (wrap)


def is_hashable(dtype: pl.DataType) -> bool:
    """Tipos aninhados (List/Struct/Array) não entram no caminho de hash."""
    return not dtype.is_nested()


def _bits64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _bits32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# (dtype, a, b): os dois valores são "iguais" e precisam gerar o MESMO hash
_EQUAL_FLOAT_CASES = (
    (pl.Float64, 0.0, -0.0),
    (pl.Float64, _bits64(0x7FF8000000000000), _bits64(0xFFF8000000000000)),  # NaN de sinal oposto
    (pl.Float64, _bits64(0x7FF8000000000000), _bits64(0x7FF8000000000001)),  # NaN com payload
    (pl.Float32, 0.0, -0.0),
    (pl.Float32, _bits32(0x7FC00000), _bits32(0xFFC00000)),
    (pl.Float32, _bits32(0x7FC00000), _bits32(0x7FC00001)),
)

# Da mais barata para a mais cara; a primeira que passar no autoteste é usada.
# (benchmarks/hash_normalization.py: cru 1,00x, '+ 0.0' 1,14x, when/is_nan 1,67x, no Polars 1.44.2)
_FLOAT_NORMALIZERS: Tuple[Callable[[pl.Expr, pl.DataType], pl.Expr], ...] = (
    lambda e, dt: e,
    lambda e, dt: e + 0.0,
    lambda e, dt: pl.when(e.is_nan()).then(pl.lit(float("nan"), dtype=dt)).otherwise(e + 0.0),
)


def _hash_value(value: float, dtype: pl.DataType, build: Callable[[pl.Expr, pl.DataType], pl.Expr]) -> object:
    frame = pl.DataFrame({"x": pl.Series([value], dtype=dtype)})
    return frame.select(build(pl.col("x"), dtype).hash(seed=1).alias("h")).item()


@lru_cache(maxsize=None)
def _float_normalizer_index() -> int:
    """Índice da normalização de float mais barata que funciona na versão do Polars instalada.

    Versões recentes do Polars já geram o mesmo hash para -0.0/0.0 e para qualquer NaN; então o hash cru basta
    e poupa até ~40% do custo do hash de linha. Como isso é comportamento interno do Polars, em vez de
    assumi-lo o motor confere (uma vez por processo, com 6 linhas) e cai para uma variante mais cara se preciso.
    """
    for i, build in enumerate(_FLOAT_NORMALIZERS[:-1]):
        try:
            if all(_hash_value(a, dt, build) == _hash_value(b, dt, build) for dt, a, b in _EQUAL_FLOAT_CASES):
                return i
        except Exception:  # noqa: BLE001 - qualquer falha do autoteste = variante não confiável
            continue
    return len(_FLOAT_NORMALIZERS) - 1


def normalized_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    """Normaliza a coluna para que 'valores iguais' impliquem 'mesmo hash'.

    - Categorical/Enum: o hash depende do mapeamento físico (difere entre frames) -> String.
    - Float: -0.0 e 0.0, e todo NaN (qualquer sinal/payload), precisam ter o mesmo hash. Usa a normalização
      mais barata que passa no autoteste (ver `_float_normalizer_index`): normalmente nenhuma.
    """
    e = pl.col(col)
    if dtype == pl.Categorical or dtype == pl.Enum:
        return e.cast(pl.String)
    if dtype.is_float():
        # alias: when/then/otherwise herda o nome do `then` (um literal) e todas as colunas se chamariam "literal"
        return _FLOAT_NORMALIZERS[_float_normalizer_index()](e, dtype).alias(col)
    return e


def row_hash_expr(columns: Sequence[str], dtypes: Mapping[str, pl.DataType]) -> pl.Expr:
    """Hash por coluna (seed distinta por posição) combinado em ÁRVORE BALANCEADA.

    A combinação é `esq * HASH_MULTIPLIER + dir` (UInt64, com wrap). Antes era uma cadeia
    `((h0 * M + h1) * M + h2) ...`: com 300 colunas a expressão tem profundidade 300 e o Polars gasta
    ~0,9 s planejando e ~1,5 s executando mesmo com 100 linhas (microbenchmark, polars 1.44), contra
    ~0,04 s com a árvore (profundidade ~log2(n)). O valor do hash muda em relação à cadeia, mas só a
    igualdade entre os dois lados importa, e ambos usam esta mesma função. A sensibilidade à posição
    vem da seed de cada coluna (`seed=i + 1`) e da multiplicação.
    """
    if not columns:
        return pl.lit(0, dtype=pl.UInt64)
    level: List[pl.Expr] = [normalized_expr(col, dtypes[col]).hash(seed=i + 1) for i, col in enumerate(columns)]
    while len(level) > 1:
        paired = [level[i] * HASH_MULTIPLIER + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    return level[0]