# Estrutura e uso

```
datacompolars/
├─ pyproject.toml
├─ src/datacompolars/
│  ├─ __init__.py      # exporta compare, CompareSettings, ReportSettings, SourceOptions  (premissa)
│  ├─ settings.py  schema.py  hashing.py  windows.py  comparators.py  results.py   (já existentes)
│  ├─ engine.py        # compare(): ainda não enviado
│  ├─ io.py            # abertura de parquet/csv/sas: ainda não enviado
│  └─ report.py        # emit(): ainda não enviado
├─ tests/
│  ├─ helpers.py  test_engine.py  test_windows.py  test_schema_settings.py  test_report.py
└─ benchmarks/
   ├─ datagen.py       # gerador (módulo + CLI)
   └─ run.py           # benchmark datacompy x datacompolars
```

## pyproject.toml (alterações)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["tests", "."]      # `from helpers import ...` e `from benchmarks.datagen import ...`
markers = [
    "sas: testes que exigem polars-readstat/pyreadstat",
    "slow: testes mais lentos (datasets maiores)",
]
```

`pytest-benchmark` deixa de ser necessário (o `run.py` mede em subprocessos). Dev deps: `pytest`, `psutil`, `datacompy`.

## Comandos

```powershell
uv run pytest                                   # testes
uv run python benchmarks/datagen.py --rows 1000000 --cols 50 --divergence 0.01 --tag div1pct
uv run python benchmarks/datagen.py --rows 30000000 --cols 50 --partitions 300 --divergence 0.01 --tag div1pct
uv run python benchmarks/run.py data/1000000_50cols_div1pct
uv run python benchmarks/run.py data/30000000_50cols_div1pct --engines hash,hash_windows --repeats 2 --window-rows 5_000_000
```
