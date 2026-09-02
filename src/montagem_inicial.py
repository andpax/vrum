from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E CARGA DE DADOS
# ==============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "docs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8-sig", sep=";")


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


# Carga das 5 bases (Ajustado typo no arquivo 03)
df_propostas_01 = padronizar_colunas(
    ler_csv(DATA_DIR / "financiamentos_chassi_2026_01.csv")
)
df_propostas_02 = padronizar_colunas(
    ler_csv(DATA_DIR / "financiamentos_chassi_2026_02.csv")
)
df_propostas_03 = padronizar_colunas(
    ler_csv(DATA_DIR / "financiamentos_chassi_2026_03.csv")
)
df_eventos = padronizar_colunas(ler_csv(DATA_DIR / "eventos_target_chassi_90d.csv"))
df_chassis = padronizar_colunas(ler_csv(DATA_DIR / "cadastro_chassi_mock.csv"))

print("✅ Todas as 5 bases foram carregadas e padronizadas com sucesso.")


# ==============================================================================
# 2. CONSTRUÇÃO DA TIMELINE UNIFICADA (PROJETO VRUM)
# ==============================================================================
def construir_timeline(df_p1, df_p2, df_p3, df_evt) -> pd.DataFrame:
    """Empilha propostas + eventos de risco e padroniza coluna temporal.

    Correção: a tabela de eventos NÃO possui coluna de data nativa. A data
    canônica do evento de risco é derivada como
        dt_evento = data_hora_proposta + dias_ate_evento
    via merge com as propostas. Apenas eventos de risco reais
    (tipo_evento != 'SEM_EVENTO') entram como ponto cronológico; SEM_EVENTO
    fica apenas como atributo target anexado à proposta (não tem data própria).
    """
    # Empilha as tabelas de propostas/financiamentos
    df_financia = pd.concat([df_p1, df_p2, df_p3], ignore_index=True)

    # Coluna de data canônica das propostas
    col_data_financia = [c for c in df_financia.columns if "data" in c or "dt" in c][0]
    df_financia["dt_evento"] = pd.to_datetime(df_financia[col_data_financia])
    df_financia["tipo_evento"] = "PROPOSTA_FINANCIAMENTO"
    df_financia["subtipo_evento"] = pd.NA
    df_financia["origem_registro"] = "PROPOSTA_FINANCIAMENTO"
    df_financia["prioridade_tipo"] = 1

    # Eventos de risco: derivar data via merge com propostas.
    # SEM_EVENTO (sem dias_ate_evento) não gera ponto cronológico.
    risco = df_evt[df_evt["tipo_evento"] != "SEM_EVENTO"].copy()
    if not risco.empty:
        anchor = df_financia[["id_proposta", "chassi_id_sintetico", col_data_financia]].copy()
        risco = risco.merge(anchor, on="id_proposta", how="left", indicator=True)
        dt_prop = pd.to_datetime(risco[col_data_financia], errors="coerce")
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
        risco_out = risco[cols_comuns].copy()
        df_full = pd.concat([df_financia, risco_out], ignore_index=True)
    else:
        df_full = df_financia.copy()

    # ORDENAÇÃO CRONOLÓGICA RIGOROSA E DETERMINÍSTICA (evita leakage)
    # Critério de desempate: dt_evento -> prioridade_tipo (proposta antes de
    # risco derivado) -> id_proposta (chave estável).
    df_full = df_full.sort_values(
        by=["chassi_id_sintetico", "dt_evento", "prioridade_tipo", "id_proposta"],
        na_position="last",
    ).reset_index(drop=True)

    return df_full


df_timeline = construir_timeline(
    df_propostas_01, df_propostas_02, df_propostas_03, df_eventos
)


# ==============================================================================
# 3. ENGENHARIA DE VARIÁVEIS TEMPORAIS E COMPORTAMENTAIS
# ==============================================================================
def extrair_features_vrum(df_tl: pd.DataFrame) -> pd.DataFrame:
    """Calcula janelas de tempo, permanência de posse e flags de risco por chassi."""
    df = df_tl.copy()

    # A. Intervalo entre passagens (Dias desde a última ocorrência)
    df["dt_evento_anterior"] = df.groupby("chassi_id_sintetico")[
        "dt_evento"
    ].shift(1)
    df["dias_desde_ultima_passagem"] = (
        df["dt_evento"] - df["dt_evento_anterior"]
    ).dt.days

    # B. Sequências e Alternância Atípica PF <-> PJ
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

    # C. Modalidades de Aquisição (À vista vs Financiado em curto janela)
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


df_timeline_features = extrair_features_vrum(df_timeline)


# ==============================================================================
# 4. ROLLING WINDOWS (CONTAGEM DE PASSAGENS EM 7d, 15d, 30d, 90d)
# ==============================================================================
def calcular_janela_móvel(
    df: pd.DataFrame, dias: int, col_nome: str
) -> pd.Series:
    """Contagem retroativa precisa por janela temporal sem vazamento de dados."""
    # 1. Filtra as colunas essenciais e garante ordenação rigorosa por data
    df_sub = (
        df[["chassi_id_sintetico", "dt_evento"]]
        .reset_index()
        .sort_values(by="dt_evento")  # CORREÇÃO: Garante ordenação exigida pelo merge_asof
        .reset_index(drop=True)
    )

    # 2. Realiza a junção temporal retroativa
    df_merged = pd.merge_asof(
        df_sub,
        df_sub,
        on="dt_evento",
        by="chassi_id_sintetico",
        tolerance=pd.Timedelta(days=dias),
        direction="backward",
        allow_exact_matches=False,  # Exclui o evento atual da contagem prévia
    )

    # 3. Mapeia as contagens de volta para os índices originais do dataframe
    contagens = (
        df_merged.groupby("index_x")["index_y"]
        .count()
        .rename(col_nome)
    )
    return contagens

for dias in [7, 15, 30, 90]:
    df_timeline_features[f"qtd_passagens_{dias}d"] = calcular_janela_móvel(
        df_timeline_features, dias, f"qtd_passagens_{dias}d"
    )

# ==============================================================================
# 5. CONSOLIDAÇÃO FINAL DA BASE DE MODELAGEM (LEVEL: CHASSI)
# ==============================================================================
# Agrega o histórico completo de cada chassi para o dataset final
df_agregado_chassi = (
    df_timeline_features.groupby("chassi_id_sintetico")
    .agg(
        qtd_total_passagens=("dt_evento", "count"),
        primeira_passagem=("dt_evento", "min"),
        ultima_passagem=("dt_evento", "max"),
        media_dias_entre_passagens=("dias_desde_ultima_passagem", "mean"),
        mediana_dias_entre_passagens=("dias_desde_ultima_passagem", "median"),
        qtd_passagens_7d_max=("qtd_passagens_7d", "max"),
        qtd_passagens_30d_max=("qtd_passagens_30d", "max"),
        qtd_passagens_90d_max=("qtd_passagens_90d", "max"),
        flag_alternancia_pf_pj_total=("flag_alternancia_pf_pj", "sum")
        if "flag_alternancia_pf_pj" in df_timeline_features.columns
        else ("dt_evento", "count"),
    )
    .reset_index()
)

# Merge final no cadastro oficial de chassis
df_modelagem_vrum = df_chassis.merge(
    df_agregado_chassi, on="chassi_id_sintetico", how="left"
)

# Preenchimento de nulos para chassis sem movimentações
cols_zerar = [c for c in df_modelagem_vrum.columns if "qtd_" in c or "flag_" in c]
df_modelagem_vrum[cols_zerar] = df_modelagem_vrum[cols_zerar].fillna(0)

# Salvando artefatos na pasta de saída
df_timeline_features.to_csv(
    OUTPUT_DIR / "vrum_timeline_completa.csv", sep=";", index=False
)
df_modelagem_vrum.to_csv(
    OUTPUT_DIR / "vrum_dataset_modelagem.csv", sep=";", index=False
)

print(
    f"🚀 Pipeline do Projeto VRUM concluído! Datasets salvos em '{OUTPUT_DIR}'."
)