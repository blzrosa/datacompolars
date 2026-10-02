[![CI](https://github.com/blzrosa/datacompolars/actions/workflows/ci.yml/badge.svg)](https://github.com/blzrosa/datacompolars/actions/workflows/ci.yml)

# DataComPolars

Comparação de DataFrames e arquivos em **Polars**, no estilo do `datacompy`, mas pensada para tabelas grandes: usa **hash de linha**, **janelas por faixa da chave primária** (o pico de memória depende do tamanho da janela, não do dataset) e gera um **relatório completo** em texto, Markdown, HTML ou JSON.

- Fontes: caminho de arquivo, pasta ou glob (parquet, csv ou sas), `pl.DataFrame` ou `pl.LazyFrame`.
- Execução lazy e em *streaming*; nenhuma das duas tabelas precisa caber inteira na memória.
- Tipos de divergência cobertos: linhas só de um lado, valores diferentes por coluna, colunas só de um lado, tipos incompatíveis e chave duplicada.

## Sumário

- [Instalação](#instalação)
- [Início rápido](#início-rápido)
- [Como funciona](#como-funciona)
- [API](#api)
- [Configuração](#configuração)
- [Exemplo de relatório](#exemplo-de-relatório)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Testes](#testes)
- [Benchmarks](#benchmarks)
- [Licença](#licença)

## Instalação

Requisitos: Python **>= 3.11**, `polars >= 1.30`, `pydantic >= 2.7`.

```bash
pip install datacompolars
pip install "datacompolars[sas]"   # opcional: leitura de .sas7bdat/.xpt
```

Para desenvolver (núcleo + grupo dev com pytest, psutil e datacompy):

```bash
git clone https://github.com/blzrosa/datacompolars.git
cd datacompolars
uv sync                 # ou: uv sync --extra sas
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

Há uma demonstração maior, com igualdade exata, tolerância e relatório HTML, em [`examples/demo.py`](https://github.com/blzrosa/datacompolars/blob/main/examples/demo.py) (no repositório clonado):

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

### Ids muito diferentes (`common_keys_only`)

Por padrão, cada janela faz um *full join* que carrega as linhas dos dois lados, inclusive as sem par (e, no caminho `hash`, calcula o hash de linha de todas elas). Quando as tabelas têm **poucos ids em comum**, ligue `common_keys_only=True`:

1. Se a 1ª coluna da chave é inteira, só a **interseção das faixas** de ids dos dois lados (`[max(mínimos), min(máximos)]`) é dividida em janelas. As chaves fora dela não podem ter par e são apenas contadas, lendo só a coluna da chave; em parquet, o predicado de faixa poda row groups.
2. Em cada janela, uma varredura **só das colunas da chave** encontra as chaves em comum e um *semi join* filtra os dois lados. O hash e os comparadores processam apenas essas linhas, com *inner join*.
3. As linhas exclusivas continuam no relatório: só esquerda = total da esquerda − em comum (idem à direita).

O resultado é o mesmo do modo padrão; muda só o custo. Pontos de atenção:

- A contagem de exclusivas pressupõe chaves únicas. `check_duplicate_keys` (padrão) garante isso, inclusive fora da interseção; com ele desligado e havendo duplicatas, as contagens já eram infladas.
- Com muita sobreposição de ids a flag só acrescenta uma varredura das chaves, por isso é opt-in.
- O semi join, sozinho, não evita ler as colunas de valor (o filtro é aplicado depois da leitura). Quem reduz a leitura é a restrição por faixa, que rende mais quando as faixas de ids diferem; com sobreposição espalhada, o ganho vem de não hashear nem comparar as linhas sem par.
- As chaves em comum de cada janela ficam em memória (só as colunas da chave).
- Chaves nulas casam entre si, como no modo padrão.

O ganho depende de quanto os ids se sobrepõem e de quantas colunas a tabela tem; meça com os [benchmarks](#benchmarks) (`--id-overlap` e as engines `*_common`).

## API

### `compare(left, right, settings, source=None) -> ComparisonResult`

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `left`, `right` | caminho, pasta ou glob (parquet, csv ou sas), `pl.DataFrame` ou `pl.LazyFrame` | As duas fontes. |
| `settings` | [`CompareSettings`](#comparesettings) ou `dict` | Configuração da comparação. Um `dict` é convertido com `CompareSettings(**settings)`. |
| `source` | [`SourceOptions`](#sourceoptions) (opcional) | Opções de leitura dos arquivos (parquet/csv/sas). |

> `CompareSettings` e `SourceOptions` estão documentados na seção [Configuração](#configuração).

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

Todas as opções do `ReportSettings` estão em [Configuração](#reportsettings).

HTML e JSON sempre trazem **tudo**, sem os limites acima. O HTML é um arquivo único, sem dependências externas, com tema claro/escuro automático.

## Configuração

Tudo que o usuário controla passa por três modelos Pydantic, definidos em [`settings.py`](https://github.com/blzrosa/datacompolars/blob/main/src/datacompolars/settings.py):

| Modelo | Controla | Onde é usado |
|---|---|---|
| [`CompareSettings`](#comparesettings) | **Como comparar** (chave, tolerâncias, tipos, desempenho) | 3º argumento de `compare()` |
| [`SourceOptions`](#sourceoptions) | **Como abrir arquivos** (formato, separador do CSV etc.) | `source=` de `compare()` |
| [`ReportSettings`](#reportsettings) | **Como apresentar o relatório** (formato, largura, destino) | 2º argumento de `emit()` |

Os três usam `extra="forbid"`: um parâmetro com nome errado levanta `ValidationError` em vez de ser ignorado silenciosamente. Onde `compare()` aceita `CompareSettings`, também aceita um `dict` equivalente.

```python
from datacompolars import CompareSettings, ReportSettings, SourceOptions, compare
from datacompolars.report import emit

result = compare(
    "dados/producao.csv",
    "dados/homologacao.csv",
    CompareSettings(
        join_columns=["id"],
        left_name="producao",
        right_name="homologacao",
        abs_tol={"valor": 0.01},      # tolerância só na coluna "valor"
        ignore_case=True,
        sample_count=10,
    ),
    source=SourceOptions(csv_separator=";"),
)

emit(result, ReportSettings(save_path="reports/comparacao.html", print_output=False))
```

### `CompareSettings`

#### Chave e rótulos

| Campo | Padrão | Descrição |
|---|---|---|
| `join_columns` | *obrigatório* | Colunas da chave primária (PK) do join. Precisa ter ao menos uma. A **primeira** é usada para o janelamento e, nesse caso, precisa ser inteira. |
| `left_name` | `"left"` | Rótulo do lado esquerdo no relatório. |
| `right_name` | `"right"` | Rótulo do lado direito no relatório. |

#### Alinhamento de schemas

| Campo | Padrão | Descrição |
|---|---|---|
| `lowercase_columns` | `True` | Compara nomes de colunas sem diferenciar caixa. Normaliza para minúsculas os nomes de colunas, a chave, as tolerâncias por coluna e os comparadores customizados. |
| `casting` | `"none"` | O que fazer quando a mesma coluna tem tipos diferentes nos dois lados (ver tabela abaixo). |

Valores de `casting`:

| Valor | Efeito |
|---|---|
| `"none"` | Só reporta. A coluna com tipos divergentes é listada no relatório e **não é comparada**. Se o tipo divergente for de uma coluna da **chave**, é erro fatal. |
| `"left"` | Converte a coluna da direita para o tipo da esquerda. |
| `"right"` | Converte a coluna da esquerda para o tipo da direita. |

#### Regras de igualdade

| Campo | Padrão | Descrição |
|---|---|---|
| `abs_tol` | `0.0` | Tolerância absoluta. Um `float` vale para todas as colunas; um `dict[str, float]` define por coluna. |
| `rel_tol` | `0.0` | Tolerância relativa, com o mesmo formato de `abs_tol`. |
| `ignore_spaces` | `False` | Ignora espaços em branco em strings. |
| `ignore_case` | `False` | Ignora maiúsculas/minúsculas em strings. |
| `custom_comparators` | `{}` | `dict[str, BaseComparator]` com um comparador próprio por coluna, por exemplo `{"valor": MeuComparador()}`. Veja `comparators.py`. Não pode conter colunas da chave. |

Nulo com nulo é sempre igual; nulo de um lado só é diferença.

#### Integridade da chave

| Campo | Padrão | Descrição |
|---|---|---|
| `check_duplicate_keys` | `True` | Valida a unicidade da PK em cada lado. Se encontrar duplicatas, a comparação é **abortada** (`result.aborted`). Desligado, e havendo duplicatas, as contagens ficam infladas. |

#### Nível de detalhe

| Campo | Padrão | Descrição |
|---|---|---|
| `column_details` | `True` | Calcula as divergências por coluna e as amostras. No caminho `hash`, custa uma leitura extra das janelas que têm divergência. Desligue para apenas contar linhas. |
| `sample_count` | `5` | Máximo de linhas de amostra por coluna divergente. `0` desativa as amostras. |

#### Desempenho e memória

| Campo | Padrão | Descrição |
|---|---|---|
| `columns_per_batch` | `50` | Quantas colunas são avaliadas por lote no caminho `columnwise`. Reduza se faltar memória em tabelas muito largas. |
| `window_rows` | `"auto"` | Janelas de execução por faixa da 1ª coluna da PK (veja abaixo). |
| `common_keys_only` | `False` | Restringe a comparação às chaves presentes nos dois lados e, com PK inteira, à interseção das faixas de ids. Para tabelas com ids muito diferentes; veja [Ids muito diferentes](#ids-muito-diferentes-common_keys_only). |

**`window_rows`** controla o fatiamento da comparação:

| Valor | Efeito |
|---|---|
| `"auto"` | Liga as janelas acima de **20 milhões de linhas**, com janelas de ~**10 milhões** (constantes `AUTO_WINDOW_THRESHOLD_ROWS` e `AUTO_WINDOW_ROWS` em `settings.py`). Abaixo disso, usa janela única. |
| inteiro `> 0` | Força janelas desse tamanho. |
| `None` | Desliga o janelamento (janela única). |

A mesma chave cai sempre na mesma janela, então o pico de memória passa a depender do tamanho da janela, não do dataset. Se a 1ª coluna da chave não for inteira, o janelamento é ignorado e o motor emite um aviso. Para referência, no benchmark de 30M linhas × 50 colunas a query única de hash usou cerca de 3,7 GB.

#### Validações

O `CompareSettings` recusa configurações inválidas na criação:

- `join_columns` vazio, com nomes em branco ou com colunas repetidas (sem diferenciar caixa);
- `abs_tol` / `rel_tol` negativos, `NaN` ou infinitos (inclusive dentro do `dict`);
- `window_rows` igual a zero, negativo, `bool` ou uma string diferente de `"auto"`;
- `sample_count < 0` ou `columns_per_batch <= 0`;
- `custom_comparators` contendo colunas da chave.

#### Qual caminho de comparação será usado

O método `uses_exact_hash()` informa se a comparação pode usar o **hash de linha** (mais rápido). Ele devolve `False`, e o motor usa o caminho `columnwise`, se qualquer uma destas opções estiver ativa:

| Opção | Faz o motor usar `columnwise` quando... |
|---|---|
| `abs_tol` / `rel_tol` | algum valor for diferente de zero (global ou por coluna) |
| `ignore_spaces` | for `True` |
| `ignore_case` | for `True` |
| `custom_comparators` | tiver ao menos um comparador |

Colunas aninhadas (não hasheáveis) também forçam `columnwise`. O caminho escolhido e o motivo aparecem no relatório e em `result.execution`.

### `SourceOptions`

Como abrir arquivos. Só afeta fontes dadas como caminho; `pl.DataFrame` e `pl.LazyFrame` não são alterados.

| Campo | Padrão | Descrição |
|---|---|---|
| `format` | `None` | Força o formato: `"parquet"`, `"csv"` ou `"sas"`. Por padrão, é deduzido da extensão do arquivo. |
| `csv_infer_schema_length` | `10_000` | Linhas usadas para inferir os tipos do CSV. `None` lê o arquivo inteiro (mais lento, porém mais seguro se os tipos mudam ao longo do arquivo). |
| `csv_separator` | `","` | Separador do CSV. Precisa ter exatamente 1 caractere. |
| `recursive` | `True` | Ao receber um diretório, procura arquivos também nas subpastas. |

A leitura de SAS exige o extra opcional: `pip install "datacompolars[sas]"` (ou `uv sync --extra sas` no repositório).

### `ReportSettings`

Como apresentar o relatório. É passado a `emit()`.

| Campo | Padrão | Descrição |
|---|---|---|
| `print_output` | `True` | Imprime o relatório no console. |
| `format` | `None` | `"text"`, `"markdown"`, `"html"` ou `"json"`. Se `None`, é deduzido da extensão de `save_path` (veja abaixo); sem `save_path`, usa `"text"`. |
| `save_path` | `None` | Salva o relatório neste arquivo (UTF-8). |
| `style` | `"auto"` | Bordas e barras do formato texto: `"unicode"`, `"ascii"` ou `"auto"` (cai para ASCII se o console não suportar unicode). |
| `width` | `100` | Largura do formato texto. Aceita de 60 a 200. |
| `max_columns` | `20` | Máximo de colunas divergentes listadas na tabela *Differences by column* (texto e Markdown). |
| `max_sample_columns` | `10` | Máximo de colunas com tabela de amostras (texto e Markdown). `0` oculta as amostras. |

Dedução do formato a partir de `save_path` (quando `format` é `None`):

| Extensão | Formato |
|---|---|
| `.md`, `.markdown` | `markdown` |
| `.html`, `.htm` | `html` |
| `.json` | `json` |
| qualquer outra | `text` |

`max_columns`, `max_sample_columns` e `width` só valem para texto e Markdown. HTML e JSON sempre trazem **tudo**.

## Exemplo de relatório

Saída em texto de uma comparação de 1.000.000 de linhas (valores ilustrativos do formato; rode o `examples/demo.py` do repositório para ver o seu):

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
│  ├─ __init__.py
│  ├─ helpers.py
│  ├─ fixtures/
│  │  └─ sample.sas7bdat   # amostra real para os testes de SAS
│  ├─ test_engine.py  test_windows.py  test_schema_settings.py  test_report.py  test_common_keys.py
│  └─ test_io.py  test_sas.py
└─ benchmarks/
   ├─ __init__.py
   ├─ datagen.py           # gerador de dados (módulo + CLI)
   ├─ run.py               # benchmark datacompy x datacompolars
   └─ results/             # saídas dos benchmarks
```

`tests/`, `benchmarks/` e `examples/` existem apenas no repositório do GitHub; o pacote publicado no PyPI contém só `src/datacompolars`.

Os diretórios `data/` (datasets gerados) e `reports/` (relatórios salvos) são locais.

## Testes

```bash
uv run --extra sas pytest      # suíte completa (inclui SAS)
uv run --exact pytest          # núcleo; os testes de SAS são pulados sem o extra
```

Marcadores disponíveis: `sas` (exige `polars-readstat`/`pyreadstat`) e `slow` (datasets maiores). Exemplo: `uv run --extra sas pytest -m "not slow"`.

## Benchmarks

O baseline é o `datacompy` (instalado no grupo `dev`).

**Geração de dados:**

```bash
uv run python benchmarks/datagen.py --rows 1000000 --cols 50 --divergence 0.01 --tag div1pct
uv run python benchmarks/datagen.py --rows 5000000 --cols 100 --divergence 0.01 --tag div1pct
uv run python benchmarks/datagen.py --rows 30000000 --cols 50 --partitions 300 --divergence 0.01 --tag div1pct
# ids parcialmente em comum (aqui, só 10% dos ids do base existem no compare), para medir common_keys_only
uv run python benchmarks/datagen.py --rows 5000000 --cols 100 --id-overlap 0.1 --tag ov10pct
```

**Execução:**

```bash
uv run python benchmarks/run.py data/1000000_50cols_div1pct
uv run python benchmarks/run.py data/30000000_50cols_div1pct --engines hash,hash_windows --repeats 2 --window-rows 5_000_000
uv run python benchmarks/run.py data/5000000_100cols_div1pct --engines hash_windows --repeats 2 --window-rows 100_000 --min-free-gb 0
# com e sem common_keys_only (sufixo _common) em ids pouco sobrepostos
uv run python benchmarks/run.py data/5000000_100cols_ov10pct --engines hash,hash_common --repeats 2
```

## Licença

Distribuído sob a [Apache License 2.0](https://github.com/blzrosa/datacompolars/blob/main/LICENSE).