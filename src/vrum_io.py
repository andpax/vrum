"""Carga e padronização das bases brutas do VRUM.

Fonte única de leitura dos 5 CSVs brutos (em `docs/`), compartilhada por todos
os scripts de ingestão para evitar divergência de schema entre etapas:

  - cadastro_chassi_mock.csv             -> metadados do chassi (sem data)
  - financiamentos_chassi_2026_{01..03}.csv -> propostas de financiamento
  - eventos_target_chassi_90d.csv        -> eventos de risco / target 90d
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "docs"
OUTPUT_DIR = REPO_ROOT / "output"

SEP = ";"
ENC = "utf-8-sig"
MESES_FINANCIAMENTOS = (1, 2, 3)


def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding=ENC, sep=SEP)


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BOM, espaços e maiúsculas dos nomes de coluna."""
    df = df.copy()
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    )
    return df


def limpar_chave(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Normaliza colunas de join (chassi/id) para string sem espaços."""
    df = df.copy()
    df[coluna] = df[coluna].astype("string").str.strip()
    return df


def carregar_bases() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lê e padroniza as 3 fontes: (cadastro, propostas, eventos).

    As chaves de join (`chassi_id_sintetico`, `id_proposta`) são normalizadas
    aqui para que toda etapa downstream use o mesmo contrato.
    """
    cadastro = padronizar_colunas(ler_csv(DATA_DIR / "cadastro_chassi_mock.csv"))
    propostas = pd.concat(
        [
            padronizar_colunas(
                ler_csv(DATA_DIR / f"financiamentos_chassi_2026_{i:02d}.csv")
            )
            for i in MESES_FINANCIAMENTOS
        ],
        ignore_index=True,
    )
    eventos = padronizar_colunas(
        ler_csv(DATA_DIR / "eventos_target_chassi_90d.csv")
    )

    cadastro = limpar_chave(cadastro, "chassi_id_sintetico")
    propostas = limpar_chave(propostas, "chassi_id_sintetico")
    propostas = limpar_chave(propostas, "id_proposta")
    eventos = limpar_chave(eventos, "id_proposta")
    return cadastro, propostas, eventos
