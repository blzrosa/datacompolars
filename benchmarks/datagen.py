"""Gerador de datasets sintéticos (parquet particionado) com resultado ESPERADO conhecido.

Layout (compatível com o antigo):
    {out}/{rows}_{cols}cols[_{tag}]/base/part-00000.parquet ...
    {out}/{rows}_{cols}cols[_{tag}]/compare/part-00000.parquet ... (+ extra-00000.parquet)
    {out}/{rows}_{cols}cols[_{tag}]/expected.json

Diferenças em relação ao gerador antigo:
  * O `compare` é o `base` com mutações controladas (taxa de divergência configurável), em vez de
    dados independentes (que davam ~100% de linhas divergentes).
  * Uma partição por vez na memória; ids contíguos e ordenados por arquivo, então os min/max de
    cada arquivo permitem podar arquivos/row groups nas janelas por PK.
  * Grava o resultado esperado (linhas em comum, divergentes, exclusivas e divergências por
    coluna), que serve de gabarito para testes e benchmarks.
  * `--id-overlap` (< 1) deixa os ids do `compare` deslocados em relação ao `base`: só essa fração dos
    ids é comum, como uma faixa contígua (ids iniciais saem do compare; o mesmo número de ids novos
    entra acima do maior id do base). Serve para medir `common_keys_only`. Para sobreposição
    espalhada (sem faixa), use `--only-left`/`--only-right`.

Uso:
    uv run python benchmarks/datagen.py --rows 10_000_000 --cols 50 --divergence 0.01 --tag div1pct
    uv run python benchmarks/datagen.py --rows 10_000_000 --cols 50 --id-overlap 0.1 --tag ov10pct
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import string
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

STRINGS = np.array(["aaa", "bbb", "ccc"])


def column_names(n: int) -> List[str]:
    names = ["".join(p) for p in product(string.ascii_lowercase, repeat=2)]
    if not 1 <= n <= len(names):
        raise ValueError(f"cols deve estar entre 1 e {len(names)}")
    return names[:n]


@dataclass(frozen=True)
class DatasetSpec:
    rows: int = 1_000
    cols: int = 9                 # colunas de dados em comum (fora o id); tipos ciclam int/float/str
    partitions: int = 4
    divergence: float = 0.01      # fração de linhas em comum com ao menos uma célula alterada
    cell_rate: float = 0.10       # em linha divergente, chance de CADA outra coluna também mudar
    null_rate: float = 0.02       # nulos (iguais nos dois lados)
    only_left: float = 0.0        # fração de linhas removidas do compare
    only_right: float = 0.0       # fração de linhas extras (ids novos) só no compare
    id_overlap: float = 1.0       # fração dos ids do base presentes no compare, em faixa contígua (1.0 = todos)
    left_only_cols: int = 0       # colunas só no base
    right_only_cols: int = 0      # colunas só no compare
    seed: int = 42
    row_group_size: int = 100_000
    tag: str = ""

    def __post_init__(self) -> None:
        if self.rows < 1 or self.partitions < 1:
            raise ValueError("rows e partitions devem ser >= 1")
        column_names(self.cols)
        for name in ("divergence", "cell_rate", "null_rate", "only_left", "only_right", "id_overlap"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} deve estar em [0, 1]")

    @property
    def label(self) -> str:
        return f"{self.rows}_{self.cols}cols" + (f"_{self.tag}" if self.tag else "")


@dataclass
class Expected:
    left_rows: int = 0
    right_rows: int = 0
    common: int = 0
    left_only: int = 0
    right_only: int = 0
    mismatched: int = 0
    column_mismatches: Dict[str, int] = field(default_factory=dict)


def _values(rng: np.random.Generator, j: int, n: int) -> np.ndarray:
    kind = j % 3
    if kind == 0:
        return rng.integers(0, 10, n)
    if kind == 1:
        return rng.random(n)
    return rng.integers(0, 3, n)  # índice em STRINGS


def _frame(
    ids: np.ndarray,
    names: List[str],
    arrays: Dict[str, np.ndarray],
    nulls: Dict[str, Optional[np.ndarray]],
    extra: Dict[str, np.ndarray],
) -> pl.DataFrame:
    cols: List[pl.Series] = [pl.Series("id", ids)]
    for j, name in enumerate(names):
        arr = arrays[name]
        s = pl.Series(name, STRINGS[arr] if j % 3 == 2 else arr)
        mask = nulls.get(name)
        if mask is not None and mask.any():
            s = pl.select(pl.when(pl.Series(~mask)).then(s)).to_series().alias(name)
        cols.append(s)
    cols += [pl.Series(k, v) for k, v in extra.items()]
    return pl.DataFrame(cols)


def _partition(
    spec: DatasetSpec, idx: int, start: int, n: int, extra_start: int
) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame], Expected]:
    rng = np.random.default_rng([spec.seed, idx])
    names = column_names(spec.cols)
    ids = np.arange(start, start + n, dtype=np.int64)

    base = {name: _values(rng, j, n) for j, name in enumerate(names)}
    nulls: Dict[str, Optional[np.ndarray]] = {
        name: (rng.random(n) < spec.null_rate) if spec.null_rate > 0 else None for name in names
    }
    divergent = rng.random(n) < spec.divergence
    forced = rng.integers(0, spec.cols, n)  # coluna que sempre muda numa linha divergente
    keep = rng.random(n) >= spec.only_left
    id_offset = int(round(spec.rows * (1.0 - spec.id_overlap)))  # ids [0, id_offset) saem do compare
    keep &= ids >= id_offset

    exp = Expected(left_rows=n, common=int(keep.sum()), left_only=int((~keep).sum()))
    row_mismatch = np.zeros(n, dtype=bool)
    cmp_arrays: Dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        v = base[name]
        mask = divergent & ((forced == j) | (rng.random(n) < spec.cell_rate))
        new = (v + 1) % 3 if j % 3 == 2 else v + 1  # sempre diferente do original
        cmp_arrays[name] = np.where(mask, new, v)
        effective = mask & keep
        if nulls[name] is not None:
            effective &= ~nulls[name]  # célula nula nos dois lados = igual
        row_mismatch |= effective
        exp.column_mismatches[name] = int(effective.sum())
    exp.mismatched = int(row_mismatch.sum())

    left_extra = {f"lo_{k}": rng.integers(0, 10, n) for k in range(spec.left_only_cols)}
    base_df = _frame(ids, names, base, nulls, left_extra)

    right_extra = {
        f"ro_{k}": rng.integers(0, 10, int(keep.sum())) for k in range(spec.right_only_cols)
    }
    cmp_df = _frame(
        ids[keep],
        names,
        {k: a[keep] for k, a in cmp_arrays.items()},
        {k: (m[keep] if m is not None else None) for k, m in nulls.items()},
        right_extra,
    )

    extra_df: Optional[pl.DataFrame] = None
    r = int(round(n * spec.only_right)) + int(round(n * (1.0 - spec.id_overlap)))
    if r:
        extra_df = _frame(
            np.arange(extra_start, extra_start + r, dtype=np.int64),
            names,
            {name: _values(rng, j, r) for j, name in enumerate(names)},
            {},
            {f"ro_{k}": rng.integers(0, 10, r) for k in range(spec.right_only_cols)},
        )
        exp.right_only = r
    exp.right_rows = exp.common + exp.right_only
    return base_df, cmp_df, extra_df, exp


def _sizes(rows: int, parts: int) -> List[int]:
    parts = min(parts, rows)
    q, r = divmod(rows, parts)
    return [q + (1 if i < r else 0) for i in range(parts)]


def _write(df: pl.DataFrame, path: Path, row_group_size: int) -> None:
    df.write_parquet(path, compression="zstd", row_group_size=row_group_size, statistics=True)


def generate(
    spec: DatasetSpec,
    out_dir: str | Path,
    overwrite: bool = True,
    log: Callable[[str], None] = lambda _: None,
) -> Tuple[Path, Expected]:
    """Gera o dataset em `out_dir/<label>` e devolve (pasta, resultado esperado)."""
    root = Path(out_dir) / spec.label
    if overwrite and root.exists():
        shutil.rmtree(root)
    base_dir, cmp_dir = root / "base", root / "compare"
    base_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    total = Expected()
    start, extra_start = 0, spec.rows  # ids "só direita" ficam acima de todos os ids do base
    for i, n in enumerate(_sizes(spec.rows, spec.partitions)):
        b, c, x, e = _partition(spec, i, start, n, extra_start)
        _write(b, base_dir / f"part-{i:05d}.parquet", spec.row_group_size)
        _write(c, cmp_dir / f"part-{i:05d}.parquet", spec.row_group_size)
        if x is not None:
            _write(x, cmp_dir / f"extra-{i:05d}.parquet", spec.row_group_size)
            extra_start += x.height
        for k in ("left_rows", "right_rows", "common", "left_only", "right_only", "mismatched"):
            setattr(total, k, getattr(total, k) + getattr(e, k))
        for col, v in e.column_mismatches.items():
            total.column_mismatches[col] = total.column_mismatches.get(col, 0) + v
        start += n
        log(f"  partição {i:05d}: ids [{start - n}, {start})")

    (root / "expected.json").write_text(
        json.dumps({"spec": asdict(spec), "expected": asdict(total)}, indent=2), encoding="utf-8"
    )
    return root, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--cols", type=int, default=50)
    ap.add_argument("--partitions", type=int, default=0, help="0 = ~100k linhas por partição")
    ap.add_argument("--divergence", type=float, default=0.01)
    ap.add_argument("--cell-rate", type=float, default=0.10)
    ap.add_argument("--null-rate", type=float, default=0.02)
    ap.add_argument("--only-left", type=float, default=0.0)
    ap.add_argument("--only-right", type=float, default=0.0)
    ap.add_argument(
        "--id-overlap", type=float, default=1.0, help="fração dos ids em comum, em faixa contígua (1.0 = todos)"
    )
    ap.add_argument("--left-only-cols", type=int, default=0)
    ap.add_argument("--right-only-cols", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    spec = DatasetSpec(
        rows=a.rows,
        cols=a.cols,
        partitions=a.partitions or max(1, math.ceil(a.rows / 100_000)),
        divergence=a.divergence,
        cell_rate=a.cell_rate,
        null_rate=a.null_rate,
        only_left=a.only_left,
        only_right=a.only_right,
        id_overlap=a.id_overlap,
        left_only_cols=a.left_only_cols,
        right_only_cols=a.right_only_cols,
        seed=a.seed,
        tag=a.tag,
    )
    print(f"Gerando {spec.label} ({spec.partitions} partições)...")
    root, exp = generate(spec, a.out, log=print)
    print(f"Pronto: {root}\nEsperado: {exp.common:,} em comum, {exp.mismatched:,} divergentes, "
          f"{exp.left_only:,} só esquerda, {exp.right_only:,} só direita")


if __name__ == "__main__":
    main()
