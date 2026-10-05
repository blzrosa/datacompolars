"""Benchmark datacompolars x datacompy (Polars e pandas) x diffly.

Cada execução roda em um SUBPROCESSO novo (o alocador do Polars não devolve memória ao SO, então
medir várias execuções no mesmo processo acumula RSS). O processo pai amostra o RSS do filho e o
encerra se passar do teto, se a RAM livre do sistema cair abaixo do piso ou se passar do tempo
limite (--timeout-s); o caso é registrado como `killed` e o benchmark segue.

O que é cronometrado: leitura dos dados + comparação + geração do relatório. Os imports pesados
(polars, pandas, datacompy, diffly, datacompolars) são feitos ANTES do cronômetro e da leitura do
RSS-base; sem isso, em tabelas pequenas o tempo medido seria quase só tempo de import.

Gabarito: se existir `expected.json` na pasta do dataset (gerado por datagen.py), as contagens de
cada engine são conferidas contra ele (coluna `ok`).

Uso:
    uv run python benchmarks/run.py data/1000000_50cols_div1pct
    uv run python benchmarks/run.py data/30000000_50cols --engines hash,hash_windows --repeats 2
    uv run python benchmarks/run.py data/* --engines datacompy_polars,hash --window-rows 5_000_000
    uv run python benchmarks/run.py data/1000_10cols_grid --engines diffly --repeats 1 --warmup 0

Engines:
    datacompy_polars    baseline: datacompy PolarsCompare (DataFrames Polars em RAM). Alias: `datacompy`
    datacompy_pandas    baseline: datacompy PandasCompare (DataFrames pandas em RAM)
    diffly              diffly.compare_frames sobre LazyFrames (scan_parquet); tolerâncias zeradas
    diffly_eager        o mesmo, sobre DataFrames (read_parquet)
    hash                datacompolars: caminho exato, janela única
    hash_windows        datacompolars: caminho exato, janelas por faixa de PK (--window-rows)
    columnwise          datacompolars: comparadores por coluna (abs_tol), janela única
    columnwise_windows  datacompolars: comparadores por coluna, janelas por faixa de PK
    default             datacompolars com as configurações PADRÃO da biblioteca (window_rows="auto" por nº de células,
                        cache_windows="auto"). É a engine que representa o que o usuário recebe; use nos resultados finais.
    hash_windows_cache  caminho exato, janelas FIXAS de --window-rows e cache_windows=True (janela decodificada uma vez)
    columnwise_windows_cache
                        o mesmo, no caminho columnwise (mais memória, ganho pequeno)
    <engine>_common     o mesmo da engine, com common_keys_only=True (restringe às chaves em comum);
                        ex.: hash_common, hash_windows_common. Compare com a engine sem o sufixo em datasets
                        gerados com `datagen.py --id-overlap` ou `--only-left/--only-right`.

Todas as engines usam igualdade exata (tolerância zero) e a configuração padrão de relatório de
cada biblioteca.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import psutil

MB = 1024 * 1024
GB = 1024 * MB
BASE_ENGINES = (
    "hash", "hash_windows", "hash_windows_cache", "columnwise", "columnwise_windows", "columnwise_windows_cache"
)
LIB_DEFAULT = "default"  # datacompolars com as configurações PADRÃO da biblioteca (janelas e cache decididos pelo motor)
EXTERNAL_ENGINES = ("datacompy_polars", "datacompy_pandas", "diffly", "diffly_eager")
ENGINE_ALIASES = {"datacompy": "datacompy_polars"}
ALL_ENGINES = (LIB_DEFAULT, *EXTERNAL_ENGINES, *BASE_ENGINES, *(f"{e}_common" for e in BASE_ENGINES))
DEFAULT_ENGINES = "default,datacompy_polars,datacompy_pandas,diffly"  # datacompolars com os padrões da biblioteca

# O nome das classes do datacompy mudou entre versões; a primeira que importar é usada e o nome
# efetivo vai para o resultado (campo `impl`).
POLARS_COMPARE_CANDIDATES = ("datacompy.polars:PolarsCompare", "datacompy:PolarsCompare")
PANDAS_COMPARE_CANDIDATES = (
    "datacompy.pandas:PandasCompare",
    "datacompy:PandasCompare",
    "datacompy.core:Compare",
    "datacompy:Compare",
)


def engine_kwargs(engine: str, window_rows: int, columns_per_batch: int = 50) -> Dict[str, Any]:
    if engine == LIB_DEFAULT:  # sem nenhum override: é o que um usuário obtém com CompareSettings(join_columns=[...])
        return {}
    parts = engine.split("_")
    kw: Dict[str, Any] = {"window_rows": window_rows if "windows" in parts else None}
    if parts[0] == "columnwise":
        kw["abs_tol"] = 1e-12  # qualquer tolerância > 0 tira o motor do caminho de hash
        kw["columns_per_batch"] = columns_per_batch  # só vale no caminho columnwise
    if "windows" in parts:  # explícito: as engines *_windows medem SEM cache; *_cache, COM (o padrão da lib é "auto")
        kw["cache_windows"] = "cache" in parts
    if "common" in parts:
        kw["common_keys_only"] = True
    return kw


# ----------------------------------------------------------------------------- worker
def _self_peak_bytes() -> int:
    """Pico de RSS (high-water mark) do PRÓPRIO processo, só da imagem atual.

    No Linux usa `VmHWM` de /proc/self/status. Não usar `ru_maxrss` aqui: depois de fork+exec ele
    herda o pico do processo pai (o pai da suíte importa polars/numpy e tem centenas de MB), o que
    colocaria um piso falso em toda medição.
    """
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak:
        return int(peak)
    try:  # Linux
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError):
        pass
    try:  # macOS e demais (ru_maxrss em bytes no macOS, em KB no restante)
        import resource

        peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak_raw if sys.platform == "darwin" else peak_raw * 1024
    except Exception:
        return int(info.rss)


def _resolve_class(candidates: Sequence[str]) -> Any:
    """Devolve a primeira classe 'modulo:Classe' que puder ser importada."""
    errors: List[str] = []
    for cand in candidates:
        module, _, name = cand.partition(":")
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{cand}: {exc}")
    raise ImportError("nenhuma classe do datacompy encontrada -> " + "; ".join(errors))


def _nrows(df: Any) -> int:
    """Nº de linhas de um DataFrame Polars (`height`) ou pandas (`len`)."""
    height = getattr(df, "height", None)
    return int(height) if height is not None else int(len(df))


def _prepare(
    engine: str, base: str, cmp_: str, window_rows: Optional[int], columns_per_batch: int = 50
) -> Callable[[], Dict[str, Any]]:
    """Faz os imports pesados e devolve a função cronometrada (leitura + comparação + relatório)."""
    import polars as pl

    if engine in ("datacompy_polars", "datacompy_pandas"):
        use_pandas = engine == "datacompy_pandas"
        cls = _resolve_class(PANDAS_COMPARE_CANDIDATES if use_pandas else POLARS_COMPARE_CANDIDATES)
        if use_pandas:
            import pandas as pd

        def run_datacompy() -> Dict[str, Any]:
            if use_pandas:
                left, right = pd.read_parquet(base), pd.read_parquet(cmp_)
            else:
                left, right = pl.read_parquet(f"{base}/*.parquet"), pl.read_parquet(f"{cmp_}/*.parquet")
            comp = cls(left, right, join_columns="id")
            report = comp.report()
            common = _nrows(comp.intersect_rows)
            return {
                "common": common,
                "mismatched": common - int(comp.count_matching_rows()),
                "left_only": _nrows(comp.df1_unq_rows),
                "right_only": _nrows(comp.df2_unq_rows),
                "report_chars": len(report),
                "impl": f"{cls.__module__}.{cls.__name__}",
            }

        return run_datacompy

    if engine in ("diffly", "diffly_eager"):
        from diffly import compare_frames

        lazy = engine == "diffly"

        def run_diffly() -> Dict[str, Any]:
            if lazy:
                left, right = pl.scan_parquet(f"{base}/*.parquet"), pl.scan_parquet(f"{cmp_}/*.parquet")
            else:
                left, right = pl.read_parquet(f"{base}/*.parquet"), pl.read_parquet(f"{cmp_}/*.parquet")
            # O diffly tem tolerância padrão (abs_tol=1e-08, rel_tol=1e-05): zerada para igualdade exata.
            comparison = compare_frames(left, right, primary_key="id", abs_tol=0.0, rel_tol=0.0)
            report = str(comparison.summary())
            return {
                "common": int(comparison.num_rows_joined()),
                "mismatched": int(comparison.num_rows_joined_unequal()),
                "left_only": int(comparison.num_rows_left_only()),
                "right_only": int(comparison.num_rows_right_only()),
                "report_chars": len(report),
                "impl": "diffly.compare_frames",
            }

        return run_diffly

    from datacompolars import CompareSettings, ReportSettings, compare

    settings = CompareSettings(join_columns=["id"], **engine_kwargs(engine, window_rows, columns_per_batch))

    def run_datacompolars() -> Dict[str, Any]:
        result = compare(base, cmp_, settings)
        result.report(ReportSettings(print_output=False))
        return {
            "common": result.rows.common,
            "mismatched": result.rows.mismatched,
            "left_only": result.rows.left_only,
            "right_only": result.rows.right_only,
            "windows": result.execution.windows,
            "path": result.execution.path,
        }

    return run_datacompolars


def worker(spec: Dict[str, Any]) -> None:
    go = _prepare(
        spec["engine"], spec["base"], spec["compare"], spec["window_rows"], spec.get("columns_per_batch", 50)
    )
    rss0 = psutil.Process().memory_info().rss  # RSS-base: já com as bibliotecas importadas

    t0 = time.perf_counter()
    out = go()
    out["total_s"] = time.perf_counter() - t0

    peak = _self_peak_bytes()
    out["peak_mb"] = peak / MB
    out["delta_mb"] = max(0.0, peak - rss0) / MB
    print("RESULT " + json.dumps(out))


# ----------------------------------------------------------------------------- pai
def measure_once(
    spec: Dict[str, Any], max_rss: int, min_free: int, timeout_s: Optional[float] = None
) -> Dict[str, Any]:
    # stdout/stderr do filho vão para arquivos temporários: com PIPE, um filho que escreva mais que
    # o buffer do pipe (~64 KB) travaria esperando o pai, que só sonda o RSS.
    with tempfile.TemporaryFile("w+b") as out_f, tempfile.TemporaryFile("w+b") as err_f:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            stdout=out_f,
            stderr=err_f,
        )
        started = time.monotonic()
        ps = psutil.Process(proc.pid)
        peak, killed = 0, None
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
            except psutil.Error:
                break
            peak = max(peak, rss)
            if rss > max_rss:
                killed = f"rss>{max_rss / GB:.1f}GB"
            elif psutil.virtual_memory().available < min_free:
                killed = f"free<{min_free / GB:.1f}GB"
            elif timeout_s and time.monotonic() - started > timeout_s:
                killed = f"timeout>{timeout_s:.0f}s"
            if killed:
                proc.kill()
                break
            time.sleep(0.02)
        proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")

    if killed:
        return {"killed": killed, "peak_mb": round(peak / MB, 1)}
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["peak_mb"] = max(r["peak_mb"], peak / MB)
            return r
    return {"error": (stderr.strip().splitlines() or ["sem saída"])[-1][:300]}


def _quartiles(values: List[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[2]


def run_case(ds: Path, engine: str, args: argparse.Namespace, max_rss: int, min_free: int) -> Dict[str, Any]:
    spec = {
        "engine": engine,
        "base": str(ds / "base"),
        "compare": str(ds / "compare"),
        "window_rows": args.window_rows,
        "columns_per_batch": getattr(args, "columns_per_batch", 50),
    }
    timeout_s = getattr(args, "timeout_s", None)
    row: Dict[str, Any] = {"dataset": ds.name, "engine": engine}
    runs: List[Dict[str, Any]] = []
    for i in range(args.warmup + args.repeats):
        r = measure_once(spec, max_rss, min_free, timeout_s)
        if "killed" in r or "error" in r:
            row.update({k: r[k] for k in ("killed", "error", "peak_mb") if k in r})
            return row
        if i == 0 and args.warmup:
            row["cold_s"] = round(r["total_s"], 4)
        if i >= args.warmup:
            runs.append(r)

    times = [r["total_s"] for r in runs]
    q1, q3 = _quartiles(times)
    last = runs[-1]
    row.update(
        min_s=round(min(times), 4),
        median_s=round(statistics.median(times), 4),
        q1_s=round(q1, 4),
        q3_s=round(q3, 4),
        delta_mb=round(statistics.median(r["delta_mb"] for r in runs), 1),
        peak_mb=round(max(r["peak_mb"] for r in runs), 1),
        peak_median_mb=round(statistics.median(r["peak_mb"] for r in runs), 1),
        n_runs=len(runs),
        times_s=[round(t, 5) for t in times],
        peaks_mb=[round(r["peak_mb"], 1) for r in runs],
        common=last["common"],
        mismatched=last["mismatched"],
        left_only=last["left_only"],
        right_only=last["right_only"],
        windows=last.get("windows"),
        path=last.get("path"),
    )
    for extra in ("impl", "report_chars"):
        if extra in last:
            row[extra] = last[extra]
    expected_file = ds / "expected.json"
    if expected_file.exists():
        exp = json.loads(expected_file.read_text(encoding="utf-8"))["expected"]
        row["ok"] = all(row[k] == exp[k] for k in ("common", "mismatched", "left_only", "right_only"))
    return row


def print_table(rows: List[Dict[str, Any]]) -> None:
    head = f"{'dataset':<28}{'engine':<27}{'min s':>9}{'med s':>9}{'Δ MB':>8}{'pico MB':>9}{'jan.':>5}{'ok':>4}  nota"
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        note = r.get("killed") or r.get("error") or ""
        ok = {True: "✓", False: "✗", None: "-"}[r.get("ok")]
        print(
            f"{r['dataset']:<28}{r['engine']:<27}"
            f"{r.get('min_s', '-'):>9}{r.get('median_s', '-'):>9}{r.get('delta_mb', '-'):>8}"
            f"{r.get('peak_mb', '-'):>9}{str(r.get('windows') or '-'):>5}{ok:>4}  {note}"
        )


def _csv_cell(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path, help="pastas com base/ e compare/")
    ap.add_argument("--engines", default=DEFAULT_ENGINES)
    ap.add_argument("--window-rows", type=lambda s: int(s.replace("_", "")), default=5_000_000)
    ap.add_argument("--columns-per-batch", type=int, default=50, help="lote de colunas do caminho columnwise")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="execuções descartadas (aquecem o cache do SO)")
    ap.add_argument("--max-rss-gb", type=float, default=None, help="default: 70%% da RAM")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument(
        "--timeout-s", type=float, default=None, help="tempo máximo de UMA execução; ao estourar, o caso vira `killed`"
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    engines = [ENGINE_ALIASES.get(e.strip(), e.strip()) for e in args.engines.split(",") if e.strip()]
    bad = set(engines) - set(ALL_ENGINES)
    if bad:
        ap.error(f"engines desconhecidas: {sorted(bad)}")
    max_rss = int((args.max_rss_gb * GB) if args.max_rss_gb else psutil.virtual_memory().total * 0.7)
    min_free = int(args.min_free_gb * GB)

    rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        if not (ds / "base").is_dir() or not (ds / "compare").is_dir():
            print(f"[pulado] {ds}: sem base/ e compare/")
            continue
        for engine in engines:
            print(f"-> {ds.name} / {engine}", flush=True)
            rows.append(run_case(ds, engine, args, max_rss, min_free))

    print_table(rows)
    if rows:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "engine"), k))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: _csv_cell(v) for k, v in r.items()} for r in rows)
        print(f"\nCSV salvo em {path}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
