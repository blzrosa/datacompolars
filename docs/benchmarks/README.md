# Benchmarks

Comparação de desempenho do **datacompolars** com o `datacompy` (Polars e pandas) e o `diffly`, de 100 a 10 milhões de linhas e de 10 a 300 colunas. Todas as engines fazem **igualdade exata** (tolerância zero) e a configuração padrão de relatório de cada biblioteca.

## Tempo

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/blzrosa/datacompolars/main/docs/benchmarks/images/time_vs_rows_dark.png">
  <img alt="Tempo de comparação x tamanho do dataset" src="https://raw.githubusercontent.com/blzrosa/datacompolars/main/docs/benchmarks/images/time_vs_rows_light.png">
</picture>

## Memória

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/blzrosa/datacompolars/main/docs/benchmarks/images/memory_vs_rows_dark.png">
  <img alt="Pico de memória x tamanho do dataset" src="https://raw.githubusercontent.com/blzrosa/datacompolars/main/docs/benchmarks/images/memory_vs_rows_light.png">
</picture>

Eixos log-log; a faixa clara em cada curva é o intervalo interquartil. As tabelas completas (todas as combinações de linhas e colunas) estão em [`summary.md`](summary.md).

## Principais números

Tempo mediano · pico de memória (e quantas vezes mais lento que o datacompolars). `—` = não medido: datacompy e diffly já falham por falta de memória com 10M de linhas e 100 colunas.

| Linhas | Colunas | datacompolars | datacompy (Polars) | datacompy (pandas) | diffly |
|---:|---:|---:|---:|---:|---:|
| 1.000.000 | 10 | **0,16 s** · 477 MB | 0,56 s · 1,0 GB (3,5×) | 0,79 s · 756 MB (4,9×) | 0,40 s · 867 MB (2,5×) |
| 1.000.000 | 50 | **0,27 s** · 922 MB | 1,88 s · 2,5 GB (6,9×) | 4,81 s · 2,9 GB (18×) | 1,10 s · 2,8 GB (4,0×) |
| 1.000.000 | 100 | **0,51 s** · 966 MB | 4,20 s · 4,6 GB (8,3×) | 12,6 s · 5,6 GB (25×) | 1,97 s · 5,1 GB (3,9×) |
| 1.000.000 | 300 | **1,45 s** · 1,2 GB | 32,1 s · 10,6 GB (22×) | 82,0 s · 10,9 GB (56×) | 54,6 s · 10,1 GB (38×) |
| 10.000.000 | 10 | **0,97 s** · 1,4 GB | 6,49 s · 7,5 GB (6,7×) | 8,63 s · 5,4 GB (8,9×) | 4,04 s · 4,9 GB (4,2×) |
| 10.000.000 | 50 | **2,96 s** · 1,4 GB | 133 s · 11,5 GB (45×) | 123 s · 9,7 GB (42×) | 184 s · 11,7 GB (62×) |
| 10.000.000 | 100 | **5,11 s** · 1,4 GB | — | — | — |
| 10.000.000 | 300 | **22,2 s** · 1,2 GB | — | — | — |

- Em 1M × 300, o datacompolars leva **1,45 s com ~1,2 GB**; as outras engines levam de 32 s a 82 s e usam ~10 GB.
- Com 10M de linhas, o pico de memória do datacompolars fica em **~1,2 a 1,5 GB** para qualquer número de colunas: o consumo depende do tamanho da janela, não do dataset.
- Em tabelas pequenas (até ~10 mil linhas) todas as engines terminam em menos de ~3 s, e a diferença é menor (o datacompolars ainda é de ~1,1× a ~16× mais rápido, e a diferença cresce com o número de colunas).

## Como o datacompolars foi configurado

A curva **datacompolars** usa as configurações **padrão** da biblioteca, sem nenhum ajuste (`CompareSettings(join_columns=["id"])`): o motor decide sozinho se usa janelas. Até 30 milhões de células (linhas × colunas) a comparação roda em janela única; acima disso, em janelas de ~30 milhões de células (no mínimo 100 mil linhas), com cada janela carregada em memória uma vez. Detalhes em [`window_rows`](../../README.md#desempenho-e-memória).

As outras duas curvas do datacompolars mostram o custo/benefício dessas escolhas:

| Curva | O que é |
|---|---|
| **datacompolars** | Configuração padrão (janelas decididas pelo motor). |
| **datacompolars (janela única)** | Caminho exato sem janelas: mais rápido só em tabelas pequenas. e a memória cresce com o dataset (6.9 GB em 1M × 300; 11.8 GB em 10M × 300). |
| **datacompolars (columnwise + janelas)** | Caminho com comparadores por coluna (usado com tolerância. `ignore_case` etc.): mais lento. mas com o menor consumo de memória (~0.8 GB). |

## Observações

- **100 mil linhas × 300 colunas é mais rápido que × 250** no datacompolars: é exatamente onde a configuração padrão passa a usar janela + cache em memória (30,1 milhões de células), que evita reler o parquet para os detalhes por coluna. Abaixo do limiar a janela única não faz cache.
- Os dados são sintéticos (parquet, ~1% de linhas divergentes e ~2% de nulos). Resultados com outros perfis de dados, outros formatos de arquivo ou outro hardware podem diferir.
- O que é cronometrado: leitura dos dados + comparação + geração do relatório. Os imports ficam fora do cronômetro.
- Cada caso roda em um processo novo (o alocador do Polars não devolve memória ao sistema operacional): 10 execuções medidas por ponto (3 a partir de 10M linhas), mais 1 de aquecimento descartada. A memória é o pico de RSS do processo.

## Ambiente

Intel Core (10 núcleos físicos / 16 lógicos), 15,7 GB de RAM, Windows 11 · Python 3.12 · polars 1.44.2, datacompy 1.1.0, diffly 1.4.1, pandas 3.0.6, pyarrow 25.0.1. Versões e argumentos de cada execução em [`data/meta.json`](data/meta.json); medições brutas em [`data/results.csv`](data/results.csv) e [`data/results.jsonl`](data/results.jsonl).
Obs.: Código medido: commit `47951b2589`. O data/meta.json registra o commit-base `1294d362c5`, pois a execução foi feita com as alterações ainda locais (pré-commit).

## Como reproduzir

```bash
uv run python benchmarks/final_benchmark.py                # grade completa + gráficos e summary.md
uv run python benchmarks/final_benchmark.py --skip-10m     # só 100 a 1M linhas
uv run python benchmarks/plot.py benchmarks/results/final_version   # refaz os gráficos a partir dos resultados
```

A geração dos dados e os demais scripts estão descritos na seção [Benchmarks](../../README.md#benchmarks) do README principal.
