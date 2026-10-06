"""Comparação final SEM recalcular datacompy e diffly.

1. copia os resultados de datacompy (Polars/pandas) e diffly de uma suíte já medida (`--source`, padrão `final`)
   para a suíte de destino (`--name`, padrão `final2`), sem rodar nada deles de novo (merge_results.py);
2. mede só o datacompolars no modo padrão (`hash_windows_cache`: janelas + cache, sem prefetch);
   a grade 100..1M linhas usa 10 repetições e os 10M linhas, 3 (como na suíte anterior);
3. gera gráficos e summary.md (plot.py).

É retomável: se cair no meio, rode de novo com o mesmo --name (casos já medidos são pulados, e o merge nunca
sobrescreve). Use a MESMA máquina e as mesmas versões de datacompy/diffly/polars da suíte de origem; o script
avisa se as versões divergirem.

    uv run python benchmarks/final_comparison.py
    uv run python benchmarks/final_comparison.py --skip-10m              # só 100..1M linhas
    uv run python benchmarks/final_comparison.py --only-10m              # só 10M (depois de rodar o resto)
    uv run python benchmarks/final_comparison.py --skip-run              # só merge + gráficos, sem medir nada
    uv run python benchmarks/final_comparison.py --dry-run               # mostra os comandos
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXTERNAL = "datacompy_polars,datacompy_pandas,diffly"
OURS = "hash_windows_cache"


def sh(cmd: list[str], dry: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def first_libs(suite_dir: Path) -> dict:
    meta = suite_dir / "meta.json"
    if not meta.exists():
        return {}
    inv = (json.loads(meta.read_text(encoding="utf-8")).get("invocations") or [{}])
    return inv[0].get("libs", {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="final", help="suíte com datacompy/diffly já medidos (em benchmarks/results)")
    ap.add_argument("--name", default="final2", help="suíte de destino")
    ap.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--window-rows", default="100000")
    ap.add_argument("--min-free-gb", default="0")
    ap.add_argument("--skip-10m", action="store_true")
    ap.add_argument("--only-10m", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="não medir nada (só merge e gráficos)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src, dst = a.results / a.source, a.results / a.name
    if not (src / "results.jsonl").exists():
        sys.exit(f"não achei {src / 'results.jsonl'}")
    libs_src = first_libs(src)
    print(f"versões na origem: { {k: libs_src.get(k) for k in ('datacompy', 'diffly', 'polars')} }")

    py = [sys.executable]
    # 1) copia os externos (nunca sobrescreve o que já existe no destino)
    sh([*py, "benchmarks/merge_results.py", str(src), "--into", str(dst), "--engines", EXTERNAL], a.dry_run)

    # 2) mede só o datacompolars
    if not a.skip_run:
        common = ["--name", a.name, "--engines", OURS, "--window-rows", a.window_rows, "--min-free-gb", a.min_free_gb,
                  "--out", str(a.results)]
        if not a.only_10m:
            sh([*py, "benchmarks/suite.py", *common, "--rows", "100,1000,10000,100000,1000000"], a.dry_run)
        if not a.skip_10m:
            sh([*py, "benchmarks/suite.py", *common, "--rows", "10000000", "--repeats", "3"], a.dry_run)

    # 3) gráficos + summary.md
    if not a.dry_run:
        libs_dst = first_libs(dst)
        for lib in ("datacompy", "diffly", "polars"):
            if libs_src.get(lib) and libs_dst.get(lib) and libs_src[lib] != libs_dst[lib]:
                print(f"[aviso] {lib}: origem {libs_src[lib]} x agora {libs_dst[lib]} — comparação não é maçã com maçã")
    sh([*py, "benchmarks/plot.py", str(dst)], a.dry_run)
    print(f"\nPronto: {dst}/summary.md e os PNGs em {dst}/")


if __name__ == "__main__":
    main()
