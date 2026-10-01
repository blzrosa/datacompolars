"""Demonstração mínima do DataComPolars.

Rodar a partir da raiz do repositório:
    uv run python examples/demo.py
"""
import polars as pl

from datacompolars import CompareSettings, ReportSettings, compare
from datacompolars.report import emit

# --- duas tabelas pequenas com divergências conhecidas ---------------------------------------
left = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "nome": ["Ana", "Bruno", "Carla", "Diego", "Elisa"],
        "valor": [10.0, 20.0, 30.0, 40.0, 50.0],
        "status": ["ATIVO", "ATIVO", "ATIVO", "INATIVO", "ATIVO"],
        "legado": [1, 1, 1, 1, 1],  # só existe na esquerda
    }
)
right = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 6],  # 5 só na esquerda, 6 só na direita
        "nome": ["Ana", "Bruno", "Carla", "Diego", "Fabio"],
        "valor": [10.0, 20.5, 30.0, 40.0, 60.0],  # id=2 diverge em 0.5
        "status": ["ATIVO", "ATIVO", "INATIVO", "INATIVO", "ATIVO"],  # id=3 diverge
        "obs": ["", "", "", "", ""],  # só existe na direita
    }
)

# --- 1) igualdade exata (caminho "hash") -----------------------------------------------------
result = compare(
    left,
    right,
    CompareSettings(join_columns=["id"], left_name="producao", right_name="homologacao"),
)
print(f"is_match={result.is_match}  aborted={result.aborted}")
print(f"linhas em comum={result.rows.common}  divergentes={result.rows.mismatched}")
emit(result, ReportSettings(print_output=True))

# --- 2) com tolerância (caminho "columnwise"): 0.5 em 'valor' deixa de ser divergência -------
result_tol = compare(
    left,
    right,
    CompareSettings(
        join_columns=["id"],
        left_name="producao",
        right_name="homologacao",
        abs_tol={"valor": 1.0},  # também aceita um float único para todas as colunas
    ),
)
emit(result_tol, ReportSettings(print_output=True))

# --- 3) relatório HTML em arquivo ------------------------------------------------------------
emit(result, ReportSettings(save_path="reports/demo.html", print_output=False))
print("HTML salvo em reports/demo.html")