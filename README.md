# DataComPolars

Comparação de DataFrames e arquivos em **Polars**, no estilo do `datacompy`, mas pensada para tabelas grandes: usa **hash de linha**, **janelas por faixa da chave primária** (o pico de memória depende do tamanho da janela, não do dataset) e gera um **relatório completo** em texto, Markdown, HTML ou JSON.

- Fontes: caminho de arquivo (parquet, csv, sas), pasta/glob de parquet, `pl.DataFrame` ou `pl.LazyFrame`.
- Execução lazy e em *streaming*; nenhuma das duas tabelas precisa caber inteira na memória.
- Tipos de divergência cobertos: linhas só de um lado, valores diferentes por coluna, colunas só de um lado, tipos incompatíveis e chave duplicada.

## Sumário

- [Instalação](#instalação)
- [Início rápido](#início-rápido)
- [Como funciona](#como-funciona)
- [API](#api)
- [Exemplo de relatório](#exemplo-de-relatório)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Testes](#testes)
- [Benchmarks](#benchmarks)

## Instalação

Requisitos: Python **>= 3.11**, `polars >= 1.30`, `pydantic >= 2.7`.

```bash
git clone https://github.com/blzrosa/datacompolars.git
cd datacompolars
uv sync                 # núcleo + grupo dev (pytest, psutil, datacompy)
uv sync --extra sas     # opcional: leitura de .sas7bdat/.xpt (polars-readstat, pyreadstat, pandas)
```

## Início rápido

```python
import polars as pl
from datacompolars import CompareSettings, ReportSettings, compare
from datacompolars.report import emit

left = pl.DataFrame({"id": [1, 2, 3], "valor": [10.0, 20.0, 30.0]})
right = pl.DataFrame({"id": [1, 2, 4], "valor": [10.0, 20.5, 40.0]})

result = compare(
    left,
    right,
    CompareSettings(join_columns=["id"], left_name="producao", right_name="homologacao"),
)

print(result.is_match)                  # False
print(result.rows.common)               # 2  (ids 1 e 2)
print(result.rows.mismatched)           # 1  (id 2: valor diferente)
emit(result, ReportSettings(print_output=True))   # imprime o relatório
```

Há uma demonstração maior, com igualdade exata, tolerância e relatório HTML, em [`examples/demo.py`](examples/demo.py):

```bash
uv run python examples/demo.py
```

## Como funciona

`compare()` segue este fluxo (detalhado no docstring de `engine.py`):

1. **Abre as fontes** (lazy), normaliza nomes de colunas, alinha os schemas e escolhe o caminho de comparação.
2. **Planeja janelas** por faixa da 1ª coluna da chave (`join_columns[0]`, precisa ser inteira). A mesma chave cai sempre na mesma janela, então os resultados de cada janela simplesmente se somam.
3. **Para cada janela:**
   1. checa chave duplicada (só se `check_duplicate_keys` estiver ligado);
   2. compara com um dos dois caminhos abaixo;
   3. se `column_details` estiver ligado, conta divergências por coluna e coleta amostras.

### Caminhos de comparação

| Caminho | Quando é usado | Como compara |
|---|---|---|
| `hash` | Igualdade exata em todas as colunas e nenhuma coluna aninhada | Uma única query de *full join* (chave + hash por linha). Os detalhes por coluna só são lidos nas janelas que têm divergência, e só para as linhas divergentes. |
| `columnwise` | Há tolerância, opções de string (`ignore_spaces`, `ignore_case`), comparadores customizados ou colunas aninhadas (não hasheáveis) | Join de chaves e comparadores por coluna, em lotes de `columns_per_batch` colunas. |

O caminho escolhido e o motivo aparecem no relatório (seção *Execution*) e em `result.execution`.

### Regras de comparação

- **Nulos:** nulo com nulo é igual; nulo de um lado só é diferença.
- **Números:** tolerância absoluta (`abs_tol`) e relativa (`rel_tol`), globais ou por coluna.
- **Strings:** opções `ignore_spaces` e `ignore_case`.
- **Listas:** comparador próprio para colunas `List`.
- **Chave duplicada:** se `check_duplicate_keys` estiver ligado e houver duplicidade em qualquer lado, a comparação é **abortada** (`result.aborted == True`) e o erro vai para `result.fatal_errors`.
- **Colunas com tipos diferentes:** são listadas no relatório e **não comparadas**, a menos que o `casting` do `CompareSettings` resolva o alinhamento.
- **Nomes de colunas:** com `lowercase_columns`, nomes de colunas, chave, tolerâncias e comparadores customizados são normalizados para minúsculas.

## API

### `compare(left, right, settings, source=None) -> ComparisonResult`

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `left`, `right` | caminho, pasta/glob de parquet, `pl.DataFrame` ou `pl.LazyFrame` | As duas fontes. |
| `settings` | `CompareSettings` ou `dict` | Configuração da comparação. Um `dict` é convertido com `CompareSettings(**settings)`. |
| `source` | `SourceOptions` (opcional) | Opções de leitura dos arquivos (parquet/csv/sas). |

### `CompareSettings`

Modelo Pydantic. Campos usados pelo motor:

| Campo | Descrição |
|---|---|
| `join_columns` | Colunas da chave primária. A **primeira** é usada para o janelamento. |
| `left_name`, `right_name` | Rótulos dos dois lados, usados no relatório. |
| `lowercase_columns` | Normaliza nomes de colunas para minúsculas antes de alinhar os schemas. |
| `abs_tol`, `rel_tol` | Tolerância absoluta/relativa. Aceita um `float` (todas as colunas) ou `dict[str, float]` (por coluna). |
| `ignore_spaces`, `ignore_case` | Opções de comparação para strings. |
| `custom_comparators` | `dict[str, BaseComparator]` com comparadores por coluna (ver `comparators.py`). |
| `casting` | Política de conversão de tipos ao alinhar os schemas. |
| `check_duplicate_keys` | Verifica chave duplicada em cada lado antes de comparar. |
| `column_details` | Calcula divergências por coluna e amostras. Desligado, o motor só informa contagens de linhas. |
| `sample_count` | Máximo de linhas de amostra por coluna divergente. |
| `columns_per_batch` | Tamanho do lote de colunas no caminho `columnwise`. |
| `window_rows` | Linhas por janela. `None` = janela única; um inteiro fixa o tamanho; o modo automático decide pelo tamanho dos dados. Ignorado (com aviso) se a 1ª coluna da chave não for inteira. |

A referência completa de tipos e valores padrão está em [`src/datacompolars/settings.py`](src/datacompolars/settings.py).

### `ComparisonResult`

Objeto devolvido por `compare()` (definido em `results.py`).

| Atributo | Conteúdo |
|---|---|
| `is_match` | `True` quando linhas, valores e colunas são idênticos. |
| `aborted` | `True` quando a comparação de linhas não pôde ser feita (erro fatal). |
| `rows` | `RowSummary`: `left_total`, `right_total`, `common`, `left_only`, `right_only`, `mismatched`, além de `matched` e `mismatch_rate`. |
| `columns` | `ColumnSummary`: `join_columns`, `compared`, `left_only`, `right_only`, `type_mismatches`. |
| `column_stats` | Lista de `ColumnStat(column, dtype, mismatches, rate)`, ordenada da coluna com mais divergências para a com menos. |
| `samples` | Lista de `SampleTable(column, total_mismatches, headers, rows)`: linhas divergentes (chave + valor de cada lado). |
| `fatal_errors`, `warnings` | Mensagens de erro e aviso. |
| `execution` | `ExecutionInfo`: `path`, `path_reason`, `windows`, `rows_per_window`, `column_details_computed`, `polars_version`, `total_seconds`, `timings` (por fase). |
| `created_at` | Data/hora UTC da comparação. |
| `to_json()` | Serializa o resultado completo em JSON. |

### `emit(result, settings=None) -> str`

Em `datacompolars.report`. Renderiza o relatório, salva em UTF-8 se `save_path` estiver definido, imprime se `print_output` estiver ligado e **devolve o texto**.

Formatos: `text`, `markdown`, `html` e `json`. O conteúdo é montado uma única vez (`build_blocks`) e cada renderizador só decide a apresentação.

| Campo de `ReportSettings` | Descrição |
|---|---|
| `save_path` | Se definido, grava o relatório nesse caminho (cria as pastas que faltarem). |
| `print_output` | Imprime o relatório no stdout. |
| `style` | Texto: `unicode`, `ascii` ou `auto`. O modo `auto` cai para ASCII se o terminal não suportar os caracteres. |
| `width` | Largura do relatório em texto. |
| `max_columns` | Máximo de colunas na tabela *Differences by column* (texto e Markdown). |
| `max_sample_columns` | Máximo de colunas com amostras (texto e Markdown). |

HTML e JSON sempre trazem **tudo**, sem os limites acima. O HTML é um arquivo único, sem dependências externas, com tema claro/escuro automático.

## Exemplo de relatório

Saída em texto de uma comparação de 1.000.000 de linhas (valores ilustrativos do formato; rode `examples/demo.py` para ver o seu):

```text
════════════════════════════════════════════════════════════════════════════════════════════════════
  DataComPolars Comparison Report
════════════════════════════════════════════════════════════════════════════════════════════════════

  Compared     producao  vs  homologacao
  Join key(s)  id
  Generated    2026-10-01 22:10:43 UTC

┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ✘ DIFFERENCES FOUND                                                                            │
│ 9,870 of 999,990 common rows differ (0.99%); 10 rows only in producao; the column sets differ; │
│ 1 columns have different types.                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘

ROWS
────

  ┌──────────────────────┬───────────┬─────────────┬──────────────┐
  │ Category             │ Rows      │ % of common │              │
  ├──────────────────────┼───────────┼─────────────┼──────────────┤
  │ In common            │   999,990 │             │              │
  │   equal              │   990,120 │      99.01% │ ████████████ │
  │   with differences   │     9,870 │       0.99% │ █░░░░░░░░░░░ │
  │ Only in producao     │        10 │             │              │
  │ Only in homologacao  │         0 │             │              │
  │ Total in producao    │ 1,000,000 │             │              │
  │ Total in homologacao │   999,990 │             │              │
  └──────────────────────┴───────────┴─────────────┴──────────────┘

COLUMNS
───────

  Compared                   3
  Only in producao           1
  Only in homologacao        0
  Different types (skipped)  1
  · Only in producao: flag_legado

  Columns with different types (not compared)
  ┌────────┬──────────────────┬─────────────────────┐
  │ Column │ Type in producao │ Type in homologacao │
  ├────────┼──────────────────┼─────────────────────┤
  │ data   │ Date             │ String              │
  └────────┴──────────────────┴─────────────────────┘

DIFFERENCES BY COLUMN
─────────────────────

  ┌────────┬─────────┬─────────────┬─────────────┬──────────────┐
  │ Column │ Type    │ Differences │ % of common │              │
  ├────────┼─────────┼─────────────┼─────────────┼──────────────┤
  │ valor  │ Float64 │       9,500 │       0.95% │ █░░░░░░░░░░░ │
  │ status │ String  │         420 │       0.04% │ █░░░░░░░░░░░ │
  │ nome   │ String  │           3 │      <0.01% │ █░░░░░░░░░░░ │
  └────────┴─────────┴─────────────┴─────────────┴──────────────┘

SAMPLE DIFFERING ROWS
─────────────────────

  valor: 9,500 differing rows, showing 3
  ┌─────┬──────────────────┬─────────────────────┐
  │ id  │ valor (producao) │ valor (homologacao) │
  ├─────┼──────────────────┼─────────────────────┤
  │ 17  │ 10.5             │ 10.7                │
  │ 203 │ 99.99            │ 100                 │
  │ 998 │ null             │ 0                   │
  └─────┴──────────────────┴─────────────────────┘

  status: 420 differing rows, showing 2
  ┌────┬───────────────────┬──────────────────────┐
  │ id │ status (producao) │ status (homologacao) │
  ├────┼───────────────────┼──────────────────────┤
  │ 5  │ ATIVO             │ INATIVO              │
  │ 88 │ ATIVO             │ null                 │
  └────┴───────────────────┴──────────────────────┘

EXECUTION
─────────

  Method             hash  (exact equality on all columns (row hash))
  Execution windows  1
  Column details     computed
  Total time         1.23 s
  Time by phase      open 10 ms, plan 0 ms, duplicates 310 ms, compare 620 ms, details 290 ms
```

Quando a comparação é abortada (por exemplo, chave duplicada), o veredito vira `COMPARISON ABORTED` e a seção *Errors* lista os motivos.

## Estrutura do repositório

```text
datacompolars/
├─ pyproject.toml
├─ examples/
│  └─ demo.py              # demonstração de uso
├─ src/datacompolars/
│  ├─ __init__.py          # exporta compare, CompareSettings, ReportSettings, SourceOptions
│  ├─ engine.py            # compare(): motor (janelas, hash, columnwise, amostras)
│  ├─ io.py                # abertura lazy de parquet/csv/sas, DataFrame e LazyFrame
│  ├─ report.py            # emit(): relatório em text, markdown, html e json
│  ├─ settings.py          # CompareSettings, ReportSettings, SourceOptions
│  ├─ results.py           # ComparisonResult e modelos do resultado
│  ├─ schema.py            # normalização e alinhamento de schemas
│  ├─ hashing.py           # hash de linha
│  ├─ windows.py           # planejamento de janelas por PK
│  └─ comparators.py       # comparadores numérico, string e lista
├─ tests/
│  ├─ helpers.py
│  ├─ test_engine.py  test_windows.py  test_schema_settings.py  test_report.py
└─ benchmarks/
   ├─ datagen.py           # gerador de dados (módulo + CLI)
   ├─ run.py               # benchmark datacompy x datacompolars
   └─ results/             # saídas dos benchmarks
```

Os diretórios `data/` (datasets gerados) e `reports/` (relatórios salvos) são locais.

## Testes

```bash
uv run pytest
```

Marcadores disponíveis: `sas` (exige `polars-readstat`/`pyreadstat`) e `slow` (datasets maiores). Exemplo: `uv run pytest -m "not slow"`.

## Benchmarks

O baseline é o `datacompy` (instalado no grupo `dev`).

**Geração de dados:**

```bash
uv run python benchmarks/datagen.py --rows 1000000 --cols 50 --divergence 0.01 --tag div1pct
uv run python benchmarks/datagen.py --rows 5000000 --cols 100 --divergence 0.01 --tag div1pct
uv run python benchmarks/datagen.py --rows 30000000 --cols 50 --partitions 300 --divergence 0.01 --tag div1pct
```

**Execução:**

```bash
uv run python benchmarks/run.py data/1000000_50cols_div1pct
uv run python benchmarks/run.py data/30000000_50cols_div1pct --engines hash,hash_windows --repeats 2 --window-rows 5_000_000
uv run python benchmarks/run.py data/5000000_100cols_div1pct --engines hash_windows --repeats 2 --window-rows 100_000 --min-free-gb 0
```