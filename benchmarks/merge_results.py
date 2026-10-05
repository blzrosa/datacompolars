"""Copia resultados de OUTRA suíte para esta, sem rodar nada de novo.

Útil quando só uma parte das engines mudou (ex.: o código do datacompolars) e as outras (datacompy, diffly)
já foram medidas, na mesma máquina, com o mesmo `suite.py`. Só entram casos (linhas, colunas, engine) que
ainda não existem no destino; o que já está lá nunca é sobrescrito. O destino ganha, em meta.json, um
registro `merged` com a origem, e o results.csv é regenerado.

    uv run python benchmarks/merge_results.py benchmarks/results/full --into benchmarks/results/final \
        --engines datacompy_polars,datacompy_pandas,diffly
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from benchmarks import suite
except ImportError:  # `python benchmarks/merge_results.py`
    import suite  # type: ignore[no-redef]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="pasta da suíte de origem")
    ap.add_argument("--into", type=Path, required=True, help="pasta da suíte de destino")
    ap.add_argument("--engines", required=True, help="engines a copiar, separadas por vírgula")
    a = ap.parse_args()

    wanted = {e.strip() for e in a.engines.split(",") if e.strip()}
    src_rows = suite.latest_by_key(suite.read_results(a.source / "results.jsonl"))
    dest_file = a.into / "results.jsonl"
    a.into.mkdir(parents=True, exist_ok=True)
    have = suite.latest_by_key(suite.read_results(dest_file))

    copied = skipped = 0
    for key, row in sorted(src_rows.items()):
        if key[2] not in wanted:
            continue
        if key in have:
            skipped += 1
            continue
        suite.append_result(dest_file, row)
        copied += 1
    missing = wanted - {k[2] for k in src_rows}
    if missing:
        print(f"[aviso] engines sem nenhum resultado na origem: {sorted(missing)}")

    suite.export_csv(suite.read_results(dest_file), a.into / "results.csv")

    meta_file = a.into / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {"name": a.into.name, "invocations": []}
    src_meta_file = a.source / "meta.json"
    src_meta = json.loads(src_meta_file.read_text(encoding="utf-8")) if src_meta_file.exists() else {}
    first = (src_meta.get("invocations") or [{}])[0]
    meta.setdefault("merged", []).append(
        {
            "merged_at": datetime.now().isoformat(timespec="seconds"),
            "from": str(a.source),
            "engines": sorted(wanted),
            "copied": copied,
            "source_libs": first.get("libs"),
            "source_machine": first.get("machine"),
            "source_started_at": first.get("started_at"),
        }
    )
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"copiados {copied} casos de {a.source} para {a.into} ({skipped} já existiam)")


if __name__ == "__main__":
    main()
