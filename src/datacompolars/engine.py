"""Motor de comparação.

Fluxo (decidido pelos benchmarks de memória/tempo):

  1. abre as fontes (lazy), normaliza nomes, alinha schemas (schema.py) e escolhe o caminho;
  2. planeja janelas por faixa da 1ª coluna da PK (windows.py) -- o pico de memória passa a
     depender do tamanho da janela, não do dataset;
     (common_keys_only: se a 1ª coluna é inteira, só a interseção das faixas de ids dos dois lados
     é comparada; as chaves fora dela não têm par e são apenas contadas, lendo só a coluna da chave);
  3. para cada janela (a mesma chave cai sempre na mesma janela, então os resultados somam):
       a. checa PK duplicada (group_by só da chave, sequencial: rodar junto com o join multiplica a memória);
       a'. (common_keys_only) varre só as chaves, materializa as chaves em comum e filtra os dois lados
          com semi join; só essas linhas seguem. Exclusivas = total - em comum;
       b. caminho "hash":     UMA query de join (chave + hash por coluna + flag) -> contagens e nº de divergentes;
          caminho "columnwise": join de chaves + comparadores por coluna em lotes;
       c. detalhes (opcional): no hash, só se a janela tem divergência, relê SÓ as linhas divergentes
          (semi join nas chaves divergentes) para contar por coluna e coletar amostras.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import polars as pl

from .comparators import ArrayComparator, BaseComparator, NumericComparator, StringComparator
from .hashing import is_hashable, row_hash_expr
from .io import Source, open_source
from .results import (
    ColumnStat,
    ColumnSummary,
    ComparisonResult,
    ExecutionInfo,
    RowSummary,
    SampleTable,
    TypeMismatch,
)
from .schema import SchemaAlignment, align_schemas, normalize_schema
from .settings import AUTO_WINDOW_ROWS, AUTO_WINDOW_THRESHOLD_ROWS, CompareSettings, SourceOptions
from .windows import PkStats, Window, key_overlap, pk_stats, plan_windows, resolve_window_rows

_IN_L, _IN_R = "__in_l", "__in_r"
_H_L, _H_R = "__h_l", "__h_r"
_R = "__r_"  # prefixo das colunas do lado direito no frame unido
_M = "__m_"  # prefixo das colunas booleanas "igual?"

_NUMERIC = NumericComparator()
_STRING = StringComparator()
_ARRAY = ArrayComparator()


def compare(
    left: Source,
    right: Source,
    settings: Union[CompareSettings, Dict[str, Any]],
    source: Optional[SourceOptions] = None,
) -> ComparisonResult:
    """Compara duas fontes (caminho, pasta/glob de parquet, DataFrame ou LazyFrame)."""
    if isinstance(settings, dict):
        settings = CompareSettings(**settings)
    return _Comparison(left, right, settings, source).run()


class _Comparison:
    def __init__(self, left: Source, right: Source, settings: CompareSettings, source: Optional[SourceOptions]):
        self.s = settings
        self.t0 = time.perf_counter()
        self.timings: Dict[str, float] = defaultdict(float)
        self.fatal: List[str] = []
        self.warnings: List[str] = []
        self.aln: Optional[SchemaAlignment] = None
        self.path: Optional[str] = None
        self.path_reason = ""
        # common_keys_only: com as duas tabelas já restritas às chaves comuns, o join é inner.
        self._how = "inner" if settings.common_keys_only else "full"
        # Linhas de chave não nula fora da interseção das faixas (só com common_keys_only): sem par possível.
        self._outside_l: Optional[pl.LazyFrame] = None
        self._outside_r: Optional[pl.LazyFrame] = None
        self._outside_n = (0, 0)

        lower = settings.lowercase_columns
        self.pk = [c.lower() if lower else c for c in settings.join_columns]
        self.abs_tol = self._norm_tol(settings.abs_tol, lower)
        self.rel_tol = self._norm_tol(settings.rel_tol, lower)
        self.custom: Dict[str, BaseComparator] = {
            (k.lower() if lower else k): v for k, v in settings.custom_comparators.items()
        }

        with self._timed("open"):
            self.left = open_source(left, source)
            self.right = open_source(right, source)
            lschema, rschema = dict(self.left.collect_schema()), dict(self.right.collect_schema())
            try:
                lschema, lren = normalize_schema(lschema, lower)
                rschema, rren = normalize_schema(rschema, lower)
            except ValueError as e:
                self.fatal.append(str(e))
                return
            if lren:
                self.left = self.left.rename(lren)
            if rren:
                self.right = self.right.rename(rren)

            self.aln = align_schemas(lschema, rschema, self.pk, settings.casting)
            self.fatal += self.aln.fatal_errors
            if self.fatal:
                return
            if self.aln.left_casts:
                self.left = self.left.with_columns([pl.col(c).cast(t) for c, t in self.aln.left_casts.items()])
            if self.aln.right_casts:
                self.right = self.right.with_columns([pl.col(c).cast(t) for c, t in self.aln.right_casts.items()])
            self._choose_path()

    # ------------------------------------------------------------------ utilidades
    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] += time.perf_counter() - t

    @staticmethod
    def _norm_tol(value: Union[float, Dict[str, float]], lower: bool) -> Union[float, Dict[str, float]]:
        if isinstance(value, dict):
            return {(k.lower() if lower else k): v for k, v in value.items()}
        return value

    @staticmethod
    def _tol(value: Union[float, Dict[str, float]], col: str) -> float:
        return value.get(col, 0.0) if isinstance(value, dict) else value

    @property
    def cols(self) -> List[str]:
        assert self.aln is not None
        return self.aln.compare_columns

    def _choose_path(self) -> None:
        assert self.aln is not None
        nested = [c for c in self.cols if not is_hashable(self.aln.left_schema[c])]
        if not self.s.uses_exact_hash():
            self.path, self.path_reason = "columnwise", "tolerances, string options or custom comparators are enabled"
        elif nested:
            self.path = "columnwise"
            self.path_reason = f"nested columns can't be hashed: {', '.join(nested)}"
        else:
            self.path, self.path_reason = "hash", "exact equality on all columns (row hash)"

    # ------------------------------------------------------------------ execução
    def run(self) -> ComparisonResult:
        if self.fatal:
            return self._result(aborted=True)
        assert self.aln is not None
        s = self.s

        with self._timed("plan"):
            windows, rows_per_window = self._plan_windows()
        if not self.cols:
            self.warnings.append("No common columns besides the join key: only row membership was compared.")

        # a. PK duplicada (por janela, sequencial)
        if s.check_duplicate_keys:
            with self._timed("duplicates"):
                dup_l = dup_r = 0
                for w in windows:
                    dup_l += self._duplicates(self._window(self.left, w))
                    dup_r += self._duplicates(self._window(self.right, w))
                # fora da interseção das faixas não há janela, mas a unicidade da PK vale para a tabela toda
                if self._outside_l is not None:
                    dup_l += self._duplicates(self._outside_l)
                if self._outside_r is not None:
                    dup_r += self._duplicates(self._outside_r)
            if dup_l:
                self.fatal.append(f"Duplicate join keys in the left table: {dup_l:,} surplus rows")
            if dup_r:
                self.fatal.append(f"Duplicate join keys in the right table: {dup_r:,} surplus rows")
            if self.fatal:
                return self._result(aborted=True, windows=len(windows), rows_per_window=rows_per_window)

        tot = {"only_left": self._outside_n[0], "only_right": self._outside_n[1], "both": 0, "mismatch": 0}
        col_counts: Dict[str, int] = defaultdict(int)
        samples: Dict[str, List[List[Any]]] = {}
        details = s.column_details

        for w in windows:
            lw, rw = self._window(self.left, w), self._window(self.right, w)
            subset: Optional[pl.DataFrame] = None
            counts: Dict[str, int] = {}
            only_l = only_r = 0

            if s.common_keys_only:
                with self._timed("keys"):
                    restricted_l, restricted_r, only_l, only_r = self._restrict_to_common(lw, rw)
                if restricted_l is None or restricted_r is None:  # nenhuma chave em comum nesta janela
                    tot["only_left"] += only_l
                    tot["only_right"] += only_r
                    continue
                lw, rw = restricted_l, restricted_r

            with self._timed("compare"):
                if self.path == "hash":
                    stats = self._membership(lw, rw, with_hash=True)
                else:
                    stats = self._membership(lw, rw, with_hash=False)
            if s.common_keys_only:  # lw/rw só têm chaves em comum: as exclusivas vieram da varredura de chaves
                stats["only_left"], stats["only_right"] = only_l, only_r

            if self.path == "hash":
                if details and stats["mismatch"] > 0:
                    with self._timed("details"):
                        subset = self._mismatching_keys(lw, rw)
                        counts, _ = self._diagnose(lw, rw, subset, need_union=False)
            elif stats["both"] > 0 and self.cols:
                with self._timed("compare"):
                    counts, subset = self._diagnose(lw, rw, None, need_union=True)
                stats["mismatch"] = subset.height if subset is not None else 0

            for k in tot:
                tot[k] += stats[k]
            if details and counts:
                for c, n in counts.items():
                    col_counts[c] += n
                if s.sample_count > 0 and subset is not None and subset.height:
                    with self._timed("details"):
                        self._collect_samples(lw, rw, subset, counts, samples)

        return self._result(
            aborted=False,
            tot=tot,
            col_counts=dict(col_counts) if details else {},
            samples=samples if details else {},
            windows=len(windows),
            rows_per_window=rows_per_window,
        )

    # ------------------------------------------------------------------ janelas
    def _plan_windows(self) -> Tuple[List[Window], Optional[int]]:
        assert self.aln is not None
        setting = self.s.window_rows
        ranged = self.s.common_keys_only
        if setting is None and not ranged:
            return [Window(index=0)], None
        pk0 = self.pk[0]
        if not (self.aln.left_schema[pk0].is_integer() and self.aln.right_schema[pk0].is_integer()):
            if isinstance(setting, int):
                self.warnings.append(
                    f"window_rows ignored: the first join key '{pk0}' is not an integer column (single window used)"
                )
            return [Window(index=0)], None
        ls: PkStats = pk_stats(self.left, pk0)
        rs: PkStats = pk_stats(self.right, pk0)
        rows = resolve_window_rows(
            setting, ls, rs, threshold=AUTO_WINDOW_THRESHOLD_ROWS, auto_rows=AUTO_WINDOW_ROWS
        )
        if not ranged:
            return plan_windows(ls, rs, rows), rows

        # common_keys_only: só a interseção das faixas de ids precisa ser comparada
        bounds = key_overlap(ls, rs)
        self._outside_l, n_l = self._outside_of(self.left, ls, bounds)
        self._outside_r, n_r = self._outside_of(self.right, rs, bounds)
        self._outside_n = (n_l, n_r)
        if bounds is None:
            # nenhuma chave não nula em comum; as chaves nulas (que casam entre si) ainda passam por uma janela
            return ([Window(index=0, null_keys=True)] if (ls.nulls or rs.nulls) else []), rows
        return plan_windows(ls, rs, rows, bounds=bounds), rows

    def _outside_of(
        self, lf: pl.LazyFrame, st: PkStats, bounds: Optional[Tuple[int, int]]
    ) -> Tuple[Optional[pl.LazyFrame], int]:
        """Linhas de chave não nula fora de `bounds` (frame lazy, contagem). Só a coluna da chave é lida."""
        non_null = st.n - st.nulls
        if non_null == 0 or st.lo is None or st.hi is None:
            return None, 0
        key = pl.col(self.pk[0])
        if bounds is None:  # sem faixa em comum: toda chave não nula está fora
            return lf.filter(key.is_not_null()), non_null
        lo, hi = bounds
        if st.lo >= lo and st.hi <= hi:  # esse lado está inteiro dentro da faixa: nada a contar
            return None, 0
        outside = lf.filter(key.is_not_null() & ((key < lo) | (key > hi)))
        n = int(outside.select(pl.len()).collect(engine="streaming").item())
        return outside, n

    def _window(self, lf: pl.LazyFrame, w: Window) -> pl.LazyFrame:
        pred = w.predicate(self.pk[0])
        return lf if pred is None else lf.filter(pred)

    # ------------------------------------------------------------------ PK duplicada / membership
    def _duplicates(self, lf: pl.LazyFrame) -> int:
        out = (
            lf.select(self.pk)
            .group_by(self.pk)
            .agg(pl.len().alias("n"))
            .select((pl.col("n") - 1).sum().alias("excess"))
            .collect(engine="streaming")
        )
        return int(out.item() or 0)

    def _side(self, lf: pl.LazyFrame, flag: str, hash_name: Optional[str]) -> pl.LazyFrame:
        assert self.aln is not None
        exprs: List[pl.Expr] = [pl.col(c) for c in self.pk]
        if hash_name:
            exprs.append(row_hash_expr(self.cols, self.aln.left_schema).alias(hash_name))
        exprs.append(pl.lit(True).alias(flag))
        return lf.select(exprs)

    def _joined(self, lw: pl.LazyFrame, rw: pl.LazyFrame, with_hash: bool) -> pl.LazyFrame:
        left = self._side(lw, _IN_L, _H_L if with_hash else None)
        right = self._side(rw, _IN_R, _H_R if with_hash else None)
        if self._how == "inner":  # common_keys_only: lw/rw só têm chaves em comum
            return left.join(right, on=self.pk, how="inner", nulls_equal=True)
        return left.join(right, on=self.pk, how="full", coalesce=False, nulls_equal=True)

    def _restrict_to_common(
        self, lw: pl.LazyFrame, rw: pl.LazyFrame
    ) -> Tuple[Optional[pl.LazyFrame], Optional[pl.LazyFrame], int, int]:
        """common_keys_only: restringe a janela às chaves presentes nos dois lados.

        Uma varredura só das colunas da chave conta as linhas de cada lado e materializa as chaves em
        comum (só chaves: ~janela x largura da PK). Depois, um semi join filtra os dois lados, e é só nessas
        linhas que o hash/os comparadores trabalham. Devolve (lw, rw, só_esquerda, só_direita); lw e rw
        são None quando não há chave em comum. Exclusivas = total - em comum (pressupõe chaves únicas).
        """
        pk = self.pk
        lk, rk = lw.select(pk), rw.select(pk)
        n_l, n_r, common = pl.collect_all(
            [
                lk.select(pl.len().alias("n")),
                rk.select(pl.len().alias("n")),
                lk.join(rk, on=pk, how="semi", nulls_equal=True),
            ],
            engine="streaming",
        )
        n_common = common.height
        only_l, only_r = int(n_l.item()) - n_common, int(n_r.item()) - n_common
        if n_common == 0:
            return None, None, only_l, only_r
        keys = common.lazy()
        return (
            lw.join(keys, on=pk, how="semi", nulls_equal=True),
            rw.join(keys, on=pk, how="semi", nulls_equal=True),
            only_l,
            only_r,
        )

    @staticmethod
    def _both() -> pl.Expr:
        return pl.col(_IN_L).is_not_null() & pl.col(_IN_R).is_not_null()

    def _membership(self, lw: pl.LazyFrame, rw: pl.LazyFrame, with_hash: bool) -> Dict[str, int]:
        both = self._both()
        aggs = [
            pl.col(_IN_R).is_null().sum().alias("only_left"),
            pl.col(_IN_L).is_null().sum().alias("only_right"),
            both.sum().alias("both"),
        ]
        if with_hash:
            aggs.append((both & pl.col(_H_L).ne_missing(pl.col(_H_R))).sum().alias("mismatch"))
        row = self._joined(lw, rw, with_hash).select(aggs).collect(engine="streaming").row(0, named=True)
        out = {k: int(v or 0) for k, v in row.items()}
        out.setdefault("mismatch", 0)
        return out

    def _mismatching_keys(self, lw: pl.LazyFrame, rw: pl.LazyFrame) -> pl.DataFrame:
        joined = self._joined(lw, rw, with_hash=True)
        return (
            joined.filter(self._both() & pl.col(_H_L).ne_missing(pl.col(_H_R)))
            .select(self.pk)
            .collect(engine="streaming")
        )

    # ------------------------------------------------------------------ comparação por coluna
    def _match_expr(self, col: str) -> pl.Expr:
        assert self.aln is not None
        a, b = pl.col(col), pl.col(_R + col)
        dtype = self.aln.left_schema[col]
        comp = self.custom.get(col)
        if comp is None:
            if dtype.is_numeric():
                comp = _NUMERIC
            elif dtype == pl.String or dtype == pl.Categorical or dtype == pl.Enum:
                comp = _STRING
            elif dtype == pl.List:
                comp = _ARRAY
            else:
                return a.eq_missing(b)
        m = comp.compare(
            col,
            _R + col,
            abs_tol=self._tol(self.abs_tol, col),
            rel_tol=self._tol(self.rel_tol, col),
            ignore_spaces=self.s.ignore_spaces,
            ignore_case=self.s.ignore_case,
            is_float=dtype.is_float(),
        )
        # regra de nulos aplicada pelo motor: nulo/nulo = igual; nulo de um lado só = diferente
        return (a.is_null() & b.is_null()) | (a.is_not_null() & b.is_not_null() & m.fill_null(False))

    def _matched(
        self, lw: pl.LazyFrame, rw: pl.LazyFrame, batch: List[str], subset: Optional[pl.DataFrame]
    ) -> pl.LazyFrame:
        pk = self.pk
        left = lw.select([*pk, *batch])
        right = rw.select([*pk, *[pl.col(c).alias(_R + c) for c in batch]])
        if subset is not None:
            keys = subset.lazy()
            left = left.join(keys, on=pk, how="semi", nulls_equal=True)
            right = right.join(keys, on=pk, how="semi", nulls_equal=True)
        merged = left.join(right, on=pk, how="inner", nulls_equal=True)
        return merged.with_columns([self._match_expr(c).alias(_M + c) for c in batch])

    def _batches(self) -> Iterator[List[str]]:
        n = self.s.columns_per_batch
        for i in range(0, len(self.cols), n):
            yield self.cols[i : i + n]

    def _diagnose(
        self, lw: pl.LazyFrame, rw: pl.LazyFrame, subset: Optional[pl.DataFrame], need_union: bool
    ) -> Tuple[Dict[str, int], Optional[pl.DataFrame]]:
        """Divergências por coluna (e, se pedido, a união das chaves divergentes) em lotes de colunas."""
        counts: Dict[str, int] = {}
        unions: List[pl.DataFrame] = []
        for batch in self._batches():
            with_m = self._matched(lw, rw, batch, subset)
            names = [_M + c for c in batch]
            counts_lf = with_m.select(pl.len().alias("__n"), *[pl.col(n).not_().sum().alias(n) for n in names])
            if need_union:
                pks_lf = (
                    with_m.filter(pl.any_horizontal([pl.col(n).not_() for n in names])).select(self.pk).unique()
                )
                cdf, pdf = pl.collect_all([counts_lf, pks_lf], engine="streaming")
                unions.append(pdf)
            else:
                cdf = counts_lf.collect(engine="streaming")
            row = cdf.row(0, named=True)
            for c, n in zip(batch, names):
                counts[c] = int(row[n] or 0)
        union = None
        if need_union:
            union = pl.concat(unions, how="vertical").unique() if unions else pl.DataFrame()
        return counts, union

    def _collect_samples(
        self,
        lw: pl.LazyFrame,
        rw: pl.LazyFrame,
        subset: pl.DataFrame,
        counts: Dict[str, int],
        samples: Dict[str, List[List[Any]]],
    ) -> None:
        n = self.s.sample_count
        for col in self.cols:
            if counts.get(col, 0) == 0:
                continue
            have = samples.setdefault(col, [])
            need = n - len(have)
            if need <= 0:
                continue
            rows = (
                self._matched(lw, rw, [col], subset)
                .filter(pl.col(_M + col).not_())
                .select([*self.pk, pl.col(col), pl.col(_R + col)])
                .sort(self.pk)
                .head(need)
                .collect(engine="streaming")
                .rows()
            )
            have.extend([list(r) for r in rows])

    # ------------------------------------------------------------------ resultado
    def _result(
        self,
        aborted: bool,
        tot: Optional[Dict[str, int]] = None,
        col_counts: Optional[Dict[str, int]] = None,
        samples: Optional[Dict[str, List[List[Any]]]] = None,
        windows: int = 1,
        rows_per_window: Optional[int] = None,
    ) -> ComparisonResult:
        s, aln = self.s, self.aln
        tot = tot or {"only_left": 0, "only_right": 0, "both": 0, "mismatch": 0}
        col_counts = col_counts or {}
        samples = samples or {}
        common = tot["both"]

        rows = RowSummary(
            left_total=common + tot["only_left"],
            right_total=common + tot["only_right"],
            common=common,
            left_only=tot["only_left"],
            right_only=tot["only_right"],
            mismatched=tot["mismatch"],
        )
        columns = ColumnSummary(
            join_columns=self.pk,
            compared=list(aln.compare_columns) if aln else [],
            left_only=list(aln.left_only) if aln else [],
            right_only=list(aln.right_only) if aln else [],
            type_mismatches=[TypeMismatch(column=c, left=l, right=r) for c, l, r in (aln.type_mismatches if aln else [])],
        )
        stats = [
            ColumnStat(
                column=c,
                dtype=str(aln.left_schema[c]) if aln else "",
                mismatches=n,
                rate=n / common if common else 0.0,
            )
            for c, n in sorted(col_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if n > 0
        ]
        tables = [
            SampleTable(
                column=st.column,
                total_mismatches=st.mismatches,
                headers=[*self.pk, f"{st.column} ({s.left_name})", f"{st.column} ({s.right_name})"],
                rows=samples[st.column],
            )
            for st in stats
            if samples.get(st.column)
        ]
        total_s = time.perf_counter() - self.t0
        execution = ExecutionInfo(
            path=self.path if self.path in ("hash", "columnwise") else None,  # type: ignore[arg-type]
            path_reason=self.path_reason,
            windows=windows,
            rows_per_window=rows_per_window,
            common_keys_only=s.common_keys_only,
            column_details_computed=bool(s.column_details and not aborted),
            polars_version=pl.__version__,
            total_seconds=round(total_s, 4),
            timings={k: round(v, 4) for k, v in self.timings.items()},
        )
        return ComparisonResult(
            left_name=s.left_name,
            right_name=s.right_name,
            join_columns=self.pk,
            rows=rows,
            columns=columns,
            column_stats=stats,
            samples=tables,
            fatal_errors=list(self.fatal),
            warnings=list(self.warnings),
            execution=execution,
        )
