"""Benchmark FINAL, padronizado: todas as engines em toda a grade, numa suíte nova (nada é reaproveitado).

Engines (ids de run.py):
    default             datacompolars com as configurações padrão da biblioteca (é o "datacompolars" dos gráficos)
    hash                datacompolars, caminho exato com janela única (mostra o custo/benefício do janelamento)
    columnwise_windows  datacompolars, caminho com comparadores por coluna (tolerância > 0) e janelas
    datacompy_polars, datacompy_pandas, diffly

Grade: 100 a 1M linhas (10 repetições por caso) e 10M linhas (3 repetições) x 10..300 colunas.
Falhas que já se sabe que acontecem (datacompy e diffly com 10M x 100 colunas ou mais, por falta de RAM) são
PULADAS, não rodadas: ficam como "—" nas tabelas. Elas vêm de (a) --failures-from, as falhas registradas numa suíte
anterior (padrão: benchmarks/results/final, se existir) e (b) --known-failures, a lista fixa abaixo.

É retomável: se cair no meio, rode de novo com o mesmo --name. No fim gera gráficos e summary.md.

    uv run python benchmarks/final_benchmark.py                 # tudo (várias horas; precisa de ~60 GB livres p/ 10M x 300)
    uv run python benchmarks/final_benchmark.py --skip-10m      # só 100..1M linhas
    uv run python benchmarks/final_benchmark.py --only-10m      # só 10M (depois de rodar o resto)
    uv run python benchmarks/final_benchmark.py --dry-run       # mostra o plano e os comandos
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES = "default,hash,columnwise_windows,datacompy_polars,datacompy_pandas,diffly"
KNOWN_FAILURES = "datacompy_polars@10000000x100,datacompy_pandas@10000000x100,diffly@10000000x100"


def sh(cmd: list[str], dry: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="final_version", help="nome da suíte (pasta em --results)")
    ap.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--engines", default=ENGINES)
    ap.add_argument("--known-failures", default=KNOWN_FAILURES)
    ap.add_argument("--failures-from", default="final", help="suíte anterior (em --results) de onde ler as falhas; '' desliga")
    ap.add_argument("--window-rows", default="100000", help="janela das engines *_windows (a `default` decide sozinha)")
    ap.add_argument("--min-free-gb", default="0")
    ap.add_argument("--skip-10m", action="store_true")
    ap.add_argument("--only-10m", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    common = [
        sys.executable, "benchmarks/suite.py", "--name", a.name, "--engines", a.engines, "--out", str(a.results),
        "--window-rows", a.window_rows, "--min-free-gb", a.min_free_gb, "--known-failures", a.known_failures,
    ]
    prev = a.results / a.failures_from if a.failures_from else None
    if prev is not None and (ROOT / prev / "results.jsonl").exists():
        common += ["--failures-from", str(prev)]
    elif prev is not None:
        print(f"[aviso] {prev}/results.jsonl não existe: usando só --known-failures")
    if not a.only_10m:
        sh([*common, "--rows", "100,1000,10000,100000,1000000"], a.dry_run)
    if not a.skip_10m:
        sh([*common, "--rows", "10000000", "--repeats", "3"], a.dry_run)
    sh([sys.executable, "benchmarks/plot.py", str(a.results / a.name)], a.dry_run)
    print(f"\nPronto: {a.results / a.name}/summary.md e os PNGs na mesma pasta.")


if __name__ == "__main__":
    main()
