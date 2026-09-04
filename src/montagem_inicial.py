"""Montagem da timeline unificada e do dataset de modelagem do VRUM.

Etapa 1 do pipeline: empilha propostas (3 meses) + eventos de risco em uma
timeline cronológica por chassi, calcula features temporais e consolida a
base de modelagem no nível chassi.

Execução: `python src/montagem_inicial.py`
Saídas em `output/`:
  - vrum_timeline_completa.csv  (consumida por src/modelo_xgboost.py)
  - vrum_dataset_modelagem.csv  (nível chassi, para EDA/dashboard)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrum_io import OUTPUT_DIR, carregar_bases


def construir_timeline(df_prop: pd.DataFrame, df_evt: pd.DataFrame) -> pd.DataFrame:
    """Empilha propostas + eventos de risco e padroniza coluna temporal.

    A tabela de eventos NÃO possui data nativa: a data canônica do evento de
    risco é derivada como dt_evento = data_hora_proposta + dias_ate_evento
    via merge com as propostas. Apenas eventos reais (tipo_evento !=
    'SEM_EVENTO') entram como ponto cronológico; SEM_EVENTO fica apenas como
    atributo target anexado à proposta (não tem data própria).
    """
    df_financia = df_prop.copy()
    df_financia["dt_evento"] = pd.to_datetime(df_financia["data_hora_proposta"])
    df_financia["tipo_evento"] = "PROPOSTA_FINANCIAMENTO"
    df_financia["subtipo_evento"] = pd.NA
    df_financia["origem_registro"] = "PROPOSTA_FINANCIAMENTO"
    df_financia["prioridade_tipo"] = 1

    risco = df_evt[df_evt["tipo_evento"] != "SEM_EVENTO"].copy()
    if not risco.empty:
        anchor = df_financia[
            ["id_proposta", "chassi_id_sintetico", "data_hora_proposta"]
        ].copy()
        risco = risco.merge(anchor, on="id_proposta", how="left", indicator=True)
        dt_prop = pd.to_datetime(risco["data_hora_proposta"], errors="coerce")
        risco["dt_evento"] = dt_prop + pd.to_timedelta(
            risco["dias_ate_evento"], unit="D", errors="coerce"
        )
        risco["subtipo_evento"] = risco["tipo_evento"]
        risco["tipo_evento"] = "EVENTO_RISCO"
        risco["origem_registro"] = "EVENTO_RISCO"
        risco["prioridade_tipo"] = 2
        # Mantém só colunas compatíveis com df_financia para o concat
        cols_comuns = [
            "chassi_id_sintetico", "id_proposta", "dt_evento", "tipo_evento",
            "subtipo_evento", "origem_registro", "prioridade_tipo",
            "evento_risco_chassi_90d",
        ]
        df_full = pd.concat(
            [df_financia, risco[cols_comuns].copy()], ignore_index=True
        )
    else:
        df_full = df_financia.copy()

    # Ordenação determinística (evita leakage): dt_evento -> prioridade_tipo
    # (proposta antes de risco derivado) -> id_proposta (chave estável).
    return df_full.sort_values(
        by=["chassi_id_sintetico", "dt_evento", "prioridade_tipo", "id_proposta"],
        na_position="last",
    ).reset_index(drop=True)


def extrair_features_vrum(df_tl: pd.DataFrame) -> pd.DataFrame:
    """Calcula janelas de tempo, permanência de posse e flags de risco por chassi."""
    df = df_tl.copy()

    df["dt_evento_anterior"] = df.groupby("chassi_id_sintetico")[
        "dt_evento"
    ].shift(1)
    df["dias_desde_ultima_passagem"] = (
        df["dt_evento"] - df["dt_evento_anterior"]
    ).dt.days

    if "tipo_proponente" in df.columns:
        df["proponente_anterior"] = df.groupby("chassi_id_sintetico")[
            "tipo_proponente"
        ].shift(1)
        df["flag_alternancia_pf_pj"] = np.where(
            (df["tipo_proponente"].notna())
            & (df["proponente_anterior"].notna())
            & (df["tipo_proponente"] != df["proponente_anterior"]),
            1,
            0,
        )

    if "modalidade_aquisicao" in df.columns:
        df["modalidade_anterior"] = df.groupby("chassi_id_sintetico")[
            "modalidade_aquisicao"
        ].shift(1)
        df["flag_a_vista_para_financiado_30d"] = np.where(
            (df["modalidade_anterior"] == "A_VISTA")
            & (df["modalidade_aquisicao"] == "FINANCIADA")
            & (df["dias_desde_ultima_passagem"] <= 30),
            1,
            0,
        )

    return df


def calcular_janela_móvel(
    df: pd.DataFrame, dias: int, col_nome: str
) -> pd.Series:
    """Contagem retroativa precisa por janela temporal sem vazamento de dados."""
    df_sub = (
        df[["chassi_id_sintetico", "dt_evento"]]
        .reset_index()
        .sort_values(by="dt_evento")  # merge_asof exige ordenação por data
        .reset_index(drop=True)
    )

    df_merged = pd.merge_asof(
        df_sub,
        df_sub,
        on="dt_evento",
        by="chassi_id_sintetico",
        tolerance=pd.Timedelta(days=dias),
        direction="backward",
        allow_exact_matches=False,  # exclui o evento atual da contagem prévia
    )

    return (
        df_merged.groupby("index_x")["index_y"]
        .count()
        .rename(col_nome)
    )


def consolidar_dataset_chassi(
    df_tl: pd.DataFrame, df_cad: pd.DataFrame
) -> pd.DataFrame:
    """Agrega o histórico por chassi e jun ao cadastro oficial."""
    agg = {
        "qtd_total_passagens": ("dt_evento", "count"),
        "primeira_passagem": ("dt_evento", "min"),
        "ultima_passagem": ("dt_evento", "max"),
        "media_dias_entre_passagens": ("dias_desde_ultima_passagem", "mean"),
        "mediana_dias_entre_passagens": ("dias_desde_ultima_passagem", "median"),
        "qtd_passagens_7d_max": ("qtd_passagens_7d", "max"),
        "qtd_passagens_30d_max": ("qtd_passagens_30d", "max"),
        "qtd_passagens_90d_max": ("qtd_passagens_90d", "max"),
    }
    if "flag_alternancia_pf_pj" in df_tl.columns:
        agg["flag_alternancia_pf_pj_total"] = ("flag_alternancia_pf_pj", "sum")

    df_agregado = (
        df_tl.groupby("chassi_id_sintetico").agg(**agg).reset_index()
    )

    df_modelagem = df_cad.merge(
        df_agregado, on="chassi_id_sintetico", how="left"
    )

    # Chassis cadastrados sem movimentação: contadores/flags zerados, não nulos
    cols_zerar = [c for c in df_modelagem.columns if "qtd_" in c or "flag_" in c]
    df_modelagem[cols_zerar] = df_modelagem[cols_zerar].fillna(0)
    return df_modelagem


def main() -> None:
    df_cad, df_prop, df_evt = carregar_bases()
    print("Todas as 5 bases foram carregadas e padronizadas.")

    df_timeline = construir_timeline(df_prop, df_evt)
    df_timeline_features = extrair_features_vrum(df_timeline)

    for dias in [7, 15, 30, 90]:
        df_timeline_features[f"qtd_passagens_{dias}d"] = calcular_janela_móvel(
            df_timeline_features, dias, f"qtd_passagens_{dias}d"
        )

    df_modelagem = consolidar_dataset_chassi(df_timeline_features, df_cad)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df_timeline_features.to_csv(
        OUTPUT_DIR / "vrum_timeline_completa.csv", sep=";", index=False
    )
    df_modelagem.to_csv(
        OUTPUT_DIR / "vrum_dataset_modelagem.csv", sep=";", index=False
    )
    print(f"Pipeline do VRUM concluído. Datasets salvos em '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
