"""Suíte de benchmarks: grade (linhas x colunas) x engines, com N repetições por caso.

Para cada (linhas, colunas) o dataset sintético é gerado (datagen.py), todas as engines rodam
sobre ele (cada execução em subprocesso novo, via run.py) e o dataset é apagado em seguida.

Resultados em `benchmarks/results/<name>/`:
    results.jsonl   uma linha por (linhas, colunas, engine), gravada ao fim de cada caso (retomável)
    results.csv     o mesmo, achatado, regenerado ao final
    meta.json       máquina, versões das bibliotecas, commit do git e argumentos de cada invocação

Retomada: rode de novo com o mesmo --name. Casos já medidos são pulados. Use --retry-failed para
refazer os que terminaram em `killed`/`error`.

Falhas: se uma engine falha (estouro de RAM, timeout ou erro) em (R, C), ela é pulada em todo
(R', C') com R' >= R e C' >= C, porque custa pelo menos o mesmo. Isso evita horas em casos que
já se sabe que não cabem. Falhas ficam registradas no results.jsonl.

Divergência: a taxa de linhas divergentes é `--divergence` (1%), mas nunca menos que
`--min-divergent` linhas (10). Sem isso, em 100 linhas a taxa de 1% daria 0 divergências em ~37%
das vezes e o caminho "tudo igual" das bibliotecas (que costuma ser mais rápido) contaminaria a grade.

Uso:
    # piloto para validar a lógica
    uv run python benchmarks/suite.py --name pilot --rows 100,1000 --cols 10,50 --repeats 3
    # grade completa
    uv run python benchmarks/suite.py --name full
    # só ver o plano (nº de execuções, disco estimado), sem rodar nada
    uv run python benchmarks/suite.py --name full --dry-run

Depois:
    uv run python benchmarks/plot.py benchmarks/results/full
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

try:  # `python -m benchmarks.suite` (raiz do repositório no sys.path)
    from benchmarks import datagen
    from benchmarks import run as bench_run
except ImportError:  # `python benchmarks/suite.py` (só a pasta benchmarks/ no sys.path)
    import datagen  # type: ignore[no-redef]
    import run as bench_run  # type: ignore[no-redef]

DEFAULT_ROWS = (100, 1_000, 10_000, 100_000, 1_000_000)
DEFAULT_COLS = (10, 50, 100, 150, 200, 250, 300)
DEFAULT_ENGINES = ("default", "datacompy_polars", "datacompy_pandas", "diffly")
LABELS = {
    "default": "datacompolars",  # configurações padrão da biblioteca: é "o" datacompolars nos gráficos
    "hash_windows_cache": "datacompolars (janelas + cache)",
    "datacompy_polars": "datacompy (Polars)",
    "datacompy_pandas": "datacompy (pandas)",
    "diffly": "diffly",
    "diffly_eager": "diffly (eager)",
    "hash_windows": "datacompolars (janelas)",
    "hash": "datacompolars (janela única)",
    "columnwise": "datacompolars (columnwise)",
    "columnwise_windows": "datacompolars (columnwise + janelas)",
    "columnwise_windows_cache": "datacompolars (columnwise + janelas + cache)",
}
FIRST_COLUMNS = ("rows", "cols", "engine", "label", "status", "ok")
GB = bench_run.GB
# Interrupção por "RAM livre baixa" com o processo de teste usando menos que isto não é culpa da engine:
# o sistema já estava sem memória. O caso não é gravado e a suíte para.
ENV_PEAK_MB = 300


# ----------------------------------------------------------------------------- utilidades
def parse_ints(text: str) -> List[int]:
    return sorted({int(tok.replace("_", "")) for tok in text.split(",") if tok.strip()})


def lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("datacompolars", "datacompy", "diffly", "polars", "pandas", "pyarrow", "numpy", "psutil"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "não instalado"
    return out


def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_model(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / GB, 1),
        "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
    }


def git_info(root: Path) -> Dict[str, Any]:
    def git(*cmd: str) -> str:
        return subprocess.check_output(["git", *cmd], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except Exception:
        return {}


def read_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # linha truncada por queda no meio da escrita
    return rows


def append_result(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def latest_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    """Última linha de cada (linhas, colunas, engine): execuções posteriores substituem as antigas."""
    return {(r["rows"], r["cols"], r["engine"]): r for r in rows}


def export_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    latest = list(latest_by_key(rows).values())
    if not latest:
        return
    fields = sorted(
        {k for r in latest for k in r},
        key=lambda k: (0, FIRST_COLUMNS.index(k), k) if k in FIRST_COLUMNS else (1, 0, k),
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(latest, key=lambda r: (r["rows"], r["cols"], r["engine"])):
            w.writerow({k: bench_run._csv_cell(v) for k, v in r.items()})


def blocked_by(
    failures: Dict[str, List[Tuple[int, int, str]]], engine: str, rows: int, cols: int
) -> Optional[Tuple[int, int, str]]:
    """Falha anterior da engine em um caso menor ou igual (em linhas E colunas), se houver."""
    for f_rows, f_cols, why in failures.get(engine, []):
        if rows >= f_rows and cols >= f_cols:
            return f_rows, f_cols, why
    return None


def estimated_disk_bytes(rows: int, cols: int) -> int:
    # ~8 bytes por célula (parquet/zstd comprime pouco os floats aleatórios), base + compare, com folga
    return int(rows * cols * 8 * 2 * 1.3)


def fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="nome da suíte (pasta em --out); default: suite_<data>")
    ap.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)), help="lista separada por vírgula")
    ap.add_argument("--cols", default=",".join(map(str, DEFAULT_COLS)), help="lista separada por vírgula")
    ap.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    ap.add_argument("--repeats", type=int, default=10, help="execuções medidas por caso")
    ap.add_argument("--warmup", type=int, default=1, help="execuções descartadas antes das medidas")
    ap.add_argument("--divergence", type=float, default=0.01)
    ap.add_argument("--min-divergent", type=int, default=10, help="mínimo de linhas divergentes esperadas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--window-rows", type=lambda s: int(s.replace("_", "")), default=100_000,
        help="tamanho da janela (linhas) das engines *_windows; com N linhas há ceil(N/janela) janelas",
    )
    ap.add_argument("--columns-per-batch", type=int, default=50, help="lote de colunas das engines columnwise*")
    ap.add_argument("--max-rss-gb", type=float, default=None, help="default: 70%% da RAM")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=1800.0, help="tempo máximo de UMA execução (0 = sem limite)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--keep-data", action="store_true", help="não apagar os datasets gerados")
    ap.add_argument(
        "--known-failures", default="",
        help="falhas já conhecidas, para NÃO rodar: 'engine@LINHASxCOLUNAS,...' (ex.: diffly@10000000x100). "
        "Vale para esse tamanho e todos os maiores/mais largos, como as falhas medidas",
    )
    ap.add_argument(
        "--failures-from", type=Path, default=None,
        help="pasta de OUTRA suíte (com results.jsonl): as falhas (killed/error) dela entram como falhas conhecidas, "
        "e esses tamanhos (e os maiores) não são rodados de novo",
    )
    ap.add_argument("--retry-failed", action="store_true", help="refazer casos killed/error de execuções anteriores")
    ap.add_argument("--dry-run", action="store_true", help="mostra o plano e sai")
    args = ap.parse_args()

    rows_list, cols_list = parse_ints(args.rows), parse_ints(args.cols)
    engines = [bench_run.ENGINE_ALIASES.get(e.strip(), e.strip()) for e in args.engines.split(",") if e.strip()]
    bad = set(engines) - set(bench_run.ALL_ENGINES)
    if bad:
        ap.error(f"engines desconhecidas: {sorted(bad)}")

    name = args.name or f"suite_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out / name
    results_path = out_dir / "results.jsonl"
    meta_path = out_dir / "meta.json"

    latest = latest_by_key(read_results(results_path))

    def is_done(key: Tuple[int, int, str]) -> bool:
        r = latest.get(key)
        return r is not None and (r.get("status") == "ok" or not args.retry_failed)

    failures: Dict[str, List[Tuple[int, int, str]]] = {}
    if not args.retry_failed:
        for (r, c, e), row in latest.items():
            if row.get("status") != "ok":
                failures.setdefault(e, []).append((r, c, row.get("killed") or row.get("error") or "falha"))

    if args.failures_from is not None:
        src_file = args.failures_from / "results.jsonl"
        if not src_file.exists():
            ap.error(f"--failures-from: não achei {src_file}")
        for (r, c, e), row in latest_by_key(read_results(src_file)).items():
            if row.get("status") != "ok":
                failures.setdefault(e, []).append((r, c, f"falhou em '{args.failures_from.name}'"))
    for item in [t.strip() for t in args.known_failures.split(",") if t.strip()]:
        try:
            eng_part, size = item.split("@")
            r_part, c_part = size.lower().split("x")
            failures.setdefault(bench_run.ENGINE_ALIASES.get(eng_part, eng_part), []).append(
                (int(r_part.replace("_", "")), int(c_part), "falha conhecida (--known-failures)")
            )
        except ValueError:
            ap.error(f"--known-failures: formato inválido em '{item}' (esperado engine@LINHASxCOLUNAS)")

    runs_per_case = args.warmup + args.repeats
    todo = [
        (r, c, e)
        for r in rows_list
        for c in cols_list
        for e in engines
        if not is_done((r, c, e)) and not blocked_by(failures, e, r, c)
    ]
    biggest = max(((r, c) for r in rows_list for c in cols_list), key=lambda rc: rc[0] * rc[1])
    print(
        f"Suíte '{name}': {len(rows_list)} tamanhos x {len(cols_list)} colunas x {len(engines)} engines "
        f"= {len(rows_list) * len(cols_list) * len(engines)} casos "
        f"({len(todo)} a executar), {runs_per_case} execuções por caso."
    )
    print(f"Engines: {', '.join(engines)}")
    print(f"Maior dataset: {biggest[0]:,} x {biggest[1]} (~{estimated_disk_bytes(*biggest) / GB:.1f} GB em disco)")
    if args.dry_run:
        print(f"Execuções a realizar (aquecimento + medidas): {len(todo) * runs_per_case}")
        return

    available = psutil.virtual_memory().available
    if available < int(args.min_free_gb * GB):
        raise SystemExit(
            f"RAM disponível agora ({available / GB:.1f} GB) está abaixo do piso de segurança "
            f"--min-free-gb ({args.min_free_gb:g} GB), e o teste seria interrompido de imediato.\n"
            "Feche programas (navegador, IDE, Docker) ou baixe o piso, por exemplo --min-free-gb 0.5."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    invocation = {
        "started_at": started_at,
        "argv": sys.argv[1:],
        "machine": machine_info(),
        "libs": lib_versions(),
        "git": git_info(Path(__file__).resolve().parent.parent),
        "settings": {
            "rows": rows_list,
            "cols": cols_list,
            "engines": engines,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "divergence": args.divergence,
            "min_divergent": args.min_divergent,
            "seed": args.seed,
            "timeout_s": args.timeout_s,
            "window_rows": args.window_rows,
            "columns_per_batch": args.columns_per_batch,
        },
    }
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"name": name, "invocations": []}
    meta["invocations"].append(invocation)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    max_rss = int(args.max_rss_gb * GB) if args.max_rss_gb else int(psutil.virtual_memory().total * 0.7)
    min_free = int(args.min_free_gb * GB)
    case_args = argparse.Namespace(
        window_rows=args.window_rows,
        columns_per_batch=args.columns_per_batch,
        warmup=args.warmup,
        repeats=args.repeats,
        timeout_s=args.timeout_s or None,
    )

    t_suite = time.monotonic()
    n_configs = len(rows_list) * len(cols_list)
    config_idx = 0
    try:
        for rows_n in rows_list:
            for cols_n in cols_list:
                config_idx += 1
                pending, skipped = [], []
                for eng in engines:
                    if is_done((rows_n, cols_n, eng)):
                        continue
                    why = blocked_by(failures, eng, rows_n, cols_n)
                    if why:
                        skipped.append(f"{eng} (falhou em {why[0]:,}x{why[1]}: {why[2]})")
                    else:
                        pending.append(eng)
                if skipped:
                    print(f"[pulado em {rows_n:,}x{cols_n}] " + "; ".join(skipped))
                if not pending:
                    continue

                if shutil.disk_usage(args.data_dir).free < estimated_disk_bytes(rows_n, cols_n) * 1.2:
                    print(f"[sem disco] {rows_n:,}x{cols_n} precisa de ~{estimated_disk_bytes(rows_n, cols_n) / GB:.1f} GB livres; pulando.")
                    continue

                divergence = min(1.0, max(args.divergence, args.min_divergent / rows_n))
                spec = datagen.DatasetSpec(
                    rows=rows_n,
                    cols=cols_n,
                    partitions=max(1, math.ceil(rows_n / 100_000)),
                    divergence=divergence,
                    seed=args.seed,
                    tag="grid",
                )
                print(
                    f"\n=== [{config_idx}/{n_configs}] {rows_n:,} linhas x {cols_n} colunas "
                    f"(divergência {divergence:.2%}) | decorrido {fmt_hms(time.monotonic() - t_suite)} ===",
                    flush=True,
                )
                t_gen = time.monotonic()
                ds, _ = datagen.generate(spec, args.data_dir)
                gen_s = time.monotonic() - t_gen
                print(f"dataset gerado em {gen_s:.1f}s: {ds}", flush=True)

                config_rows: List[Dict[str, Any]] = []
                try:
                    for eng in pending:
                        print(f"-> {eng}", flush=True)
                        row = bench_run.run_case(ds, eng, case_args, max_rss, min_free)
                        if str(row.get("killed", "")).startswith("free<") and row.get("peak_mb", 0) < ENV_PEAK_MB:
                            raise SystemExit(
                                f"\nRAM livre do sistema caiu abaixo de {args.min_free_gb:g} GB enquanto o teste usava só "
                                f"{row.get('peak_mb', 0):.0f} MB: problema do ambiente, não da engine. Nada foi gravado para "
                                f"{eng} em {rows_n:,}x{cols_n}. Feche programas ou baixe --min-free-gb e rode de novo "
                                "com o mesmo --name para continuar."
                            )
                        status = "killed" if "killed" in row else "error" if "error" in row else "ok"
                        row.update(
                            rows=rows_n,
                            cols=cols_n,
                            label=LABELS.get(eng, eng),
                            window_rows=args.window_rows if "windows" in eng else None,
                            cache_windows=True if eng.endswith("_cache") else None,
                            columns_per_batch=args.columns_per_batch if eng.startswith("columnwise") else None,
                            status=status,
                            divergence=round(divergence, 6),
                            finished_at=datetime.now().isoformat(timespec="seconds"),
                        )
                        append_result(results_path, row)
                        latest[(rows_n, cols_n, eng)] = row
                        config_rows.append(row)
                        if status != "ok":
                            why_failed = row.get("killed") or row.get("error") or "falha"
                            failures.setdefault(eng, []).append((rows_n, cols_n, why_failed))
                            print(f"   !! {eng} falhou em {rows_n:,}x{cols_n}: {why_failed}", flush=True)
                        elif row.get("ok") is False:
                            print(f"   !! {eng}: contagens diferentes do gabarito (ok=✗)", flush=True)
                finally:
                    if config_rows:
                        bench_run.print_table(config_rows)
                    if not args.keep_data:
                        shutil.rmtree(ds, ignore_errors=True)
    finally:
        all_rows = read_results(results_path)
        export_csv(all_rows, out_dir / "results.csv")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["invocations"][-1]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["invocations"][-1]["elapsed_s"] = round(time.monotonic() - t_suite, 1)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados em {out_dir}/ (results.jsonl, results.csv, meta.json)")
        print(f"Gráficos e tabelas: uv run python benchmarks/plot.py {out_dir}")


if __name__ == "__main__":
    main()
