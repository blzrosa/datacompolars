"""Benchmark datacompy x datacompolars.

Cada execução roda em um SUBPROCESSO novo (o alocador do Polars não devolve memória ao SO, então
medir várias execuções no mesmo processo acumula RSS). O processo pai amostra o RSS do filho e o
encerra se passar do teto ou se a RAM livre do sistema cair abaixo do piso; o caso é registrado
como `killed` e o benchmark segue.

Gabarito: se existir `expected.json` na pasta do dataset (gerado por datagen.py), as contagens de
cada engine são conferidas contra ele (coluna `ok`).

Uso:
    uv run python benchmarks/run.py data/1000000_50cols_div1pct
    uv run python benchmarks/run.py data/30000000_50cols --engines hash,hash_windows --repeats 2
    uv run python benchmarks/run.py data/* --engines datacompy,hash --window-rows 5_000_000

Engines:
    datacompy           baseline (carrega tudo em RAM)
    hash                caminho exato, janela única
    hash_windows        caminho exato, janelas por faixa de PK (--window-rows)
    columnwise          comparadores por coluna (abs_tol), janela única
    columnwise_windows  comparadores por coluna, janelas por faixa de PK
    <engine>_common     o mesmo da engine, com common_keys_only=True (restringe às chaves em comum);
                        ex.: hash_common, hash_windows_common. Compare com a engine sem o sufixo em datasets
                        gerados com `datagen.py --id-overlap` ou `--only-left/--only-right`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

MB = 1024 * 1024
GB = 1024 * MB
BASE_ENGINES = ("hash", "hash_windows", "columnwise", "columnwise_windows")
ALL_ENGINES = ("datacompy", *BASE_ENGINES, *(f"{e}_common" for e in BASE_ENGINES))


def engine_kwargs(engine: str, window_rows: int) -> Dict[str, Any]:
    parts = engine.split("_")
    kw: Dict[str, Any] = {"window_rows": window_rows if "windows" in parts else None}
    if parts[0] == "columnwise":
        kw["abs_tol"] = 1e-12  # qualquer tolerância > 0 tira o motor do caminho de hash
    if "common" in parts:
        kw["common_keys_only"] = True
    return kw


# ----------------------------------------------------------------------------- worker
def _self_peak_bytes() -> int:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak:
        return int(peak)
    try:
        import resource  # Linux

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return int(info.rss)


def worker(spec: Dict[str, Any]) -> None:
    import polars as pl

    base, cmp_ = spec["base"], spec["compare"]
    rss0 = psutil.Process().memory_info().rss

    t0 = time.perf_counter()
    if spec["engine"] == "datacompy":
        import datacompy

        left = pl.read_parquet(f"{base}/*.parquet")
        right = pl.read_parquet(f"{cmp_}/*.parquet")
        comp = datacompy.polars.PolarsCompare(left, right, join_columns="id")
        comp.report()
        common = comp.intersect_rows.height
        out = {
            "common": common,
            "mismatched": common - int(comp.count_matching_rows()),
            "left_only": comp.df1_unq_rows.height,
            "right_only": comp.df2_unq_rows.height,
        }
    else:
        from datacompolars import CompareSettings, ReportSettings, compare

        settings = CompareSettings(join_columns=["id"], **engine_kwargs(spec["engine"], spec["window_rows"]))
        result = compare(base, cmp_, settings)
        result.report(ReportSettings(print_output=False))
        out = {
            "common": result.rows.common,
            "mismatched": result.rows.mismatched,
            "left_only": result.rows.left_only,
            "right_only": result.rows.right_only,
            "windows": result.execution.windows,
            "path": result.execution.path,
        }
    out["total_s"] = time.perf_counter() - t0
    out["peak_mb"] = _self_peak_bytes() / MB
    out["delta_mb"] = max(0.0, _self_peak_bytes() - rss0) / MB
    print("RESULT " + json.dumps(out))


# ----------------------------------------------------------------------------- pai
def measure_once(spec: Dict[str, Any], max_rss: int, min_free: int) -> Dict[str, Any]:
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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
        if killed:
            proc.kill()
            break
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    if killed:
        return {"killed": killed, "peak_mb": peak / MB}
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["peak_mb"] = max(r["peak_mb"], peak / MB)
            return r
    return {"error": (stderr.strip().splitlines() or ["sem saída"])[-1][:200]}


def run_case(ds: Path, engine: str, args: argparse.Namespace, max_rss: int, min_free: int) -> Dict[str, Any]:
    spec = {
        "engine": engine,
        "base": str(ds / "base"),
        "compare": str(ds / "compare"),
        "window_rows": args.window_rows,
    }
    row: Dict[str, Any] = {"dataset": ds.name, "engine": engine}
    runs: List[Dict[str, Any]] = []
    for i in range(args.warmup + args.repeats):
        r = measure_once(spec, max_rss, min_free)
        if "killed" in r or "error" in r:
            row.update({k: r[k] for k in ("killed", "error", "peak_mb") if k in r})
            return row
        if i == 0 and args.warmup:
            row["cold_s"] = round(r["total_s"], 2)
        if i >= args.warmup:
            runs.append(r)

    times = [r["total_s"] for r in runs]
    last = runs[-1]
    row.update(
        min_s=round(min(times), 2),
        median_s=round(statistics.median(times), 2),
        delta_mb=round(statistics.median(r["delta_mb"] for r in runs)),
        peak_mb=round(max(r["peak_mb"] for r in runs)),
        common=last["common"],
        mismatched=last["mismatched"],
        left_only=last["left_only"],
        right_only=last["right_only"],
        windows=last.get("windows"),
        path=last.get("path"),
    )
    expected_file = ds / "expected.json"
    if expected_file.exists():
        exp = json.loads(expected_file.read_text(encoding="utf-8"))["expected"]
        row["ok"] = all(row[k] == exp[k] for k in ("common", "mismatched", "left_only", "right_only"))
    return row


def print_table(rows: List[Dict[str, Any]]) -> None:
    head = f"{'dataset':<28}{'engine':<27}{'min s':>8}{'med s':>8}{'Δ MB':>8}{'pico MB':>9}{'jan.':>5}{'ok':>4}  nota"
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        note = r.get("killed") or r.get("error") or ""
        ok = {True: "✓", False: "✗", None: "-"}[r.get("ok")]
        print(
            f"{r['dataset']:<28}{r['engine']:<27}"
            f"{r.get('min_s', '-'):>8}{r.get('median_s', '-'):>8}{r.get('delta_mb', '-'):>8}"
            f"{r.get('peak_mb', '-'):>9}{str(r.get('windows') or '-'):>5}{ok:>4}  {note}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path, help="pastas com base/ e compare/")
    ap.add_argument("--engines", default="datacompy,hash,hash_windows,columnwise")
    ap.add_argument("--window-rows", type=lambda s: int(s.replace("_", "")), default=5_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="execuções descartadas (aquecem o cache do SO)")
    ap.add_argument("--max-rss-gb", type=float, default=None, help="default: 70%% da RAM")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
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
            w.writerows(rows)
        print(f"\nCSV salvo em {path}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
