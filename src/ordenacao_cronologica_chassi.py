"""Ordenação cronológica de transações por chassi — Projeto VRUM.

Reconstrói a linha do tempo de propriedade/eventos de cada veículo
identificado por `chassi_id_sintetico`, pronta para alimentar a etapa de
detecção de fraude.

Fontes (5 CSVs em DATA_DIR):
  - financiamentos_chassi_2026_{01,02,03}.csv  -> eventos PROPOSTA_FINANCIAMENTO
  - eventos_target_chassi_90d.csv               -> eventos EVENTO_RISCO (data derivada)
  - cadastro_chassi_mock.csv                    -> metadados do chassi (sem data)

Decisões de modelagem temporal:
  * A única coluna de timestamp nativa é `data_hora_proposta` (propostas).
  * A tabela de eventos NÃO possui data própria; a data canônica do evento
    de risco é derivada como  dt_evento = data_hora_proposta + dias_ate_evento
    (validado: dias_ate_evento em [1,90] e sempre >= dt_proposta).
  * Eventos SEM_EVENTO (98,53%) não geram ponto na timeline de risco; ficam
    apenas como atributo target anexado à proposta correspondente.

Regras obrigatórias atendidas:
  - Agrupamento por chassi_id_sintetico (chave única).
  - Ordenação determinística por (dt_evento, prioridade_tipo, id_proposta).
  - Datas ausentes/inválidas/fora de janela são SINALIZADAS, nunca descartadas.
  - Saída estruturada reutilizável: DataFrame long + JSON por chassi + sumário.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vrum_io import DATA_DIR, OUTPUT_DIR, SEP, carregar_bases

# --------------------------------------------------------------------------- #
# 0. CONFIGURAÇÃO
# --------------------------------------------------------------------------- #

# Janela plausível (conforme inspeção: propostas jan-mar/2026; risco até +90d).
JANELA_INICIO = pd.Timestamp("2026-01-01")
JANELA_FIM_PROPOSTA = pd.Timestamp("2026-03-31 23:59:59")
JANELA_FIM_RISCO = pd.Timestamp("2026-06-30 00:00:00")  # max proposta + 90d ~ 2026-06-29

# Prioridade de desempate quando dt_evento empatar:
#   1 = PROPOSTA_FINANCIAMENTO (ocorre primeiro, âncora)
#   2 = EVENTO_RISCO           (derivado da proposta, em/after dt_proposta)
PRIORIDADE_TIPO = {
    "PROPOSTA_FINANCIAMENTO": 1,
    "EVENTO_RISCO": 2,
}


# --------------------------------------------------------------------------- #
# 1. CONSTRUÇÃO DOS EVENTOS
# --------------------------------------------------------------------------- #
def construir_eventos_proposta(
    df_fin: pd.DataFrame, df_cad: pd.DataFrame
) -> pd.DataFrame:
    """Cada proposta vira um evento PROPOSTA_FINANCIAMENTO, com metadados do
    cadastro mesclados e data canônica = data_hora_proposta."""
    df = df_fin.copy()
    df["dt_evento"] = pd.to_datetime(df["data_hora_proposta"], errors="coerce")
    df["tipo_evento"] = "PROPOSTA_FINANCIAMENTO"
    df["subtipo_evento"] = pd.NA
    df["origem_registro"] = "PROPOSTA_FINANCIAMENTO"
    df["prioridade_tipo"] = PRIORIDADE_TIPO["PROPOSTA_FINANCIAMENTO"]

    # Metadados do cadastro (marca, ano, uf_registro, valor_fipe) sem data.
    cad = df_cad[[
        "chassi_id_sintetico",
        "marca",
        "ano_modelo",
        "uf_registro",
        "valor_fipe_referencia",
    ]].copy()
    out = df.merge(cad, on="chassi_id_sintetico", how="left")
    return out


def construir_eventos_risco(
    df_fin: pd.DataFrame, df_ev: pd.DataFrame
) -> pd.DataFrame:
    """Eventos de risco ancorados no tempo via dt_proposta + dias_ate_evento.

    Regras:
      - Apenas linhas com tipo_evento != 'SEM_EVENTO' viram ponto de timeline.
      - SEM_EVENTO fica como atributo target anexado às propostas (não vira
        ponto cronológico, pois não tem data própria).
      - dt_evento ausente/inválida é mantida e sinalizada (status_data).
    """
    fin = df_fin[["id_proposta", "chassi_id_sintetico", "data_hora_proposta"]].copy()
    ev = df_ev.copy()
    # Join traz a data âncora da proposta para o evento.
    m = ev.merge(fin, on="id_proposta", how="left", indicator=True)

    # Marca propostas sem evento correspondente (órfãs de evento).
    propostas_sem_evento = set(df_fin["id_proposta"]) - set(ev["id_proposta"])
    if propostas_sem_evento:
        print(f"  [aviso] {len(propostas_sem_evento)} proposta(s) sem evento (órfãs).")

    # Eventos sem proposta (órfãos de proposta) — sinalizados, não descartados.
    m["evento_orfao_proposta"] = m["_merge"] == "left_only"

    # Apenas eventos de risco reais entram como ponto cronológico.
    risco = m[m["tipo_evento"] != "SEM_EVENTO"].copy()
    risco["dt_proposta"] = pd.to_datetime(risco["data_hora_proposta"], errors="coerce")
    risco["dt_evento"] = risco["dt_proposta"] + pd.to_timedelta(
        risco["dias_ate_evento"], unit="D", errors="coerce"
    )
    risco["tipo_evento_timeline"] = "EVENTO_RISCO"
    risco["subtipo_evento"] = risco["tipo_evento"]  # FRAUDE_CONFIRMADA / NEVER_PAY / ...
    risco["origem_registro"] = "EVENTO_RISCO"
    risco["prioridade_tipo"] = PRIORIDADE_TIPO["EVENTO_RISCO"]
    return risco


def consolidar_timeline(
    eventos_proposta: pd.DataFrame, eventos_risco: pd.DataFrame
) -> pd.DataFrame:
    """Empilha eventos de proposta e de risco num DataFrame long padronizado."""

    colunas_comuns = [
        "chassi_id_sintetico",
        "id_proposta",
        "dt_evento",
        "tipo_evento",
        "subtipo_evento",
        "origem_registro",
        "prioridade_tipo",
    ]

    ep = eventos_proposta[colunas_comuns].copy()
    er = eventos_risco[colunas_comuns].copy()

    tl = pd.concat([ep, er], ignore_index=True)

    # ---- Status da data: VALIDA | AUSENTE | FORA_JANELA ------------------- #
    def classificar_data(dt: pd.Timestamp, origem: str) -> str:
        if pd.isna(dt):
            return "AUSENTE"
        fim = JANELA_FIM_RISCO if origem == "EVENTO_RISCO" else JANELA_FIM_PROPOSTA
        if dt < JANELA_INICIO or dt > fim:
            return "FORA_JANELA"
        return "VALIDA"

    tl["status_data"] = [
        classificar_data(dt, og) for dt, og in zip(tl["dt_evento"], tl["origem_registro"])
    ]

    # ---- Ordenação cronológica determinística por chassi ----------------- #
    # Critério: dt_evento ASC; empate -> prioridade_tipo ASC (proposta antes
    # de risco derivado); empate -> id_proposta ASC (chave estável).
    tl = tl.sort_values(
        by=["chassi_id_sintetico", "dt_evento", "prioridade_tipo", "id_proposta"],
        na_position="last",
    ).reset_index(drop=True)

    # Ordem do evento dentro da linha do tempo do chassi (1-based).
    tl["ordem_evento"] = tl.groupby("chassi_id_sintetico").cumcount() + 1

    return tl


# --------------------------------------------------------------------------- #
# 2. ENRIQUECIMENTO + ATRIBUTOS DA PROPOSTA
# --------------------------------------------------------------------------- #
def anexar_atributos_proposta(
    tl: pd.DataFrame, eventos_proposta: pd.DataFrame, eventos_risco: pd.DataFrame
) -> pd.DataFrame:
    """Anexa à timeline os atributos da proposta (tipo_proponente, valores,
    canal, uf) e o target evento_risco_chassi_90d + flags de risco."""
    tl = tl.copy()

    # Atributos da proposta (presentes em eventos_proposta).
    attr_prop = eventos_proposta[
        [
            "id_proposta",
            "tipo_proponente",
            "cpf_cnpj_proponente_sintetico",
            "if_id_sintetico",
            "valor_financiado",
            "valor_entrada",
            "prazo_meses",
            "canal",
            "uf_proposta",
            "marca",
            "ano_modelo",
            "uf_registro",
            "valor_fipe_referencia",
        ]
    ].copy()
    tl = tl.merge(attr_prop, on="id_proposta", how="left")

    # Target binário e subtipo de risco (vindo de eventos_risco).
    risco_keys = eventos_risco[["id_proposta", "evento_risco_chassi_90d"]].copy()
    tl = tl.merge(risco_keys, on="id_proposta", how="left")
    tl["evento_risco_chassi_90d"] = tl["evento_risco_chassi_90d"].fillna(0).astype("int8")

    # Flag de evento de risco para a linha de EVENTO_RISCO.
    tl["flag_evento_risco"] = (tl["origem_registro"] == "EVENTO_RISCO").astype("int8")

    # LTV na proposta (proteção contra fipe zero/ausente).
    fipe = pd.to_numeric(tl["valor_fipe_referencia"], errors="coerce")
    vf = pd.to_numeric(tl["valor_financiado"], errors="coerce")
    tl["ltv_proposta"] = np.where(fipe > 0, vf / fipe, np.nan)

    return tl


# --------------------------------------------------------------------------- #
# 3. SAÍDA ESTRUTURADA (DataFrame + JSON por chassi + sumário)
# --------------------------------------------------------------------------- #
def gerar_json_linha_do_tempo(tl: pd.DataFrame, caminho: Path, limite_chassis: int | None = None) -> None:
    """Serializa a linha do tempo por chassi em JSON.

    Estrutura por chassi:
      {
        "chassi_id_sintetico": "...",
        "eventos": [
          {"ordem": 1, "dt_evento": "...", "tipo": "...", "subtipo": ...,
           "id_proposta": "...", "status_data": "VALIDA", ...},
          ...
        ]
      }

    `limite_chassis` permite gerar um subconjunto para inspeção rápida sem
    produzir um JSON gigantesco (a base tem ~583k chassis). Passe None para
    exportar tudo (arquivo pode ser grande).
    """
    colunas_data = ["chassi_id_sintetico", "ordem_evento", "dt_evento",
                    "tipo_evento", "subtipo_evento", "id_proposta",
                    "status_data", "tipo_proponente",
                    "evento_risco_chassi_90d", "flag_evento_risco"]
    df = tl[colunas_data].copy()
    df["dt_evento"] = df["dt_evento"].apply(
        lambda d: d.isoformat() if pd.notna(d) else None
    )

    chassis = df["chassi_id_sintetico"].astype(str).unique()
    if limite_chassis is not None:
        chassis = chassis[:limite_chassis]

    saida = {}
    for ch in chassis:
        sub = df[df["chassi_id_sintetico"].astype(str) == ch]
        saida[ch] = {
            "chassi_id_sintetico": ch,
            "eventos": sub.drop(columns=["chassi_id_sintetico"]).to_dict(orient="records"),
        }

    with caminho.open("w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)


def gerar_sumario_chassi(tl: pd.DataFrame, df_cad: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores por chassi reutilizáveis pela etapa de detecção."""
    g = tl.groupby("chassi_id_sintetico")
    sumario = g.agg(
        qtd_eventos=("id_proposta", "count"),
        qtd_propostas=(
            "id_proposta",
            lambda s: tl.loc[s.index, "origem_registro"]
            .eq("PROPOSTA_FINANCIAMENTO")
            .sum(),
        ),
        qtd_eventos_risco=("flag_evento_risco", "sum"),
        target_fraude_90d=("evento_risco_chassi_90d", "max"),
        primeira_passagem=("dt_evento", "min"),
        ultima_passagem=("dt_evento", "max"),
        qtd_status_ausente=("status_data", lambda s: (s == "AUSENTE").sum()),
        qtd_status_fora_janela=("status_data", lambda s: (s == "FORA_JANELA").sum()),
        qtd_tipo_proponente_distintos=("tipo_proponente", "nunique"),
    ).reset_index()

    # Chassis cadastrados sem nenhuma movimentação (sem proposta/evento).
    sem_mov = df_cad[~df_cad["chassi_id_sintetico"].isin(sumario["chassi_id_sintetico"])]
    if not sem_mov.empty:
        vazio = sem_mov[["chassi_id_sintetico"]].copy()
        for c in sumario.columns.drop("chassi_id_sintetico"):
            vazio[c] = 0
        sumario = pd.concat([sumario, vazio], ignore_index=True)

    return sumario


# --------------------------------------------------------------------------- #
# 4. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------- #
def main() -> None:
    print("== VRUM: ordenação cronológica por chassi ==")

    # Carga
    print("[1/5] Carga dos 5 CSVs...")
    df_cad, df_fin, df_ev = carregar_bases()
    print(f"  cadastro: {df_cad.shape} | financiamentos: {df_fin.shape} | eventos: {df_ev.shape}")

    # Eventos
    print("[2/5] Construção de eventos (proposta + risco derivado)...")
    ev_prop = construir_eventos_proposta(df_fin, df_cad)
    ev_risco = construir_eventos_risco(df_fin, df_ev)
    print(f"  eventos proposta: {len(ev_prop)} | eventos risco: {len(ev_risco)}")

    # Timeline ordenada
    print("[3/5] Consolidação + ordenação cronológica determinística...")
    tl = consolidar_timeline(ev_prop, ev_risco)
    n_status = tl["status_data"].value_counts().to_dict()
    print(f"  timeline: {tl.shape} | status_data: {n_status}")

    # Enriquecimento
    print("[4/5] Anexando atributos de proposta + target...")
    tl = anexar_atributos_proposta(tl, ev_prop, ev_risco)

    # Saídas
    print("[5/5] Gravando artefatos...")
    tl_path = OUTPUT_DIR / "vrum_timeline_cronologica.csv"
    tl.to_csv(tl_path, sep=SEP, index=False)
    print(f"  timeline long -> {tl_path}")

    sumario = gerar_sumario_chassi(tl, df_cad)
    sumario_path = OUTPUT_DIR / "vrum_sumario_chassi.csv"
    sumario.to_csv(sumario_path, sep=SEP, index=False)
    print(f"  sumario por chassi -> {sumario_path}  ({sumario.shape})")

    # JSON por chassi: amostra dos primeiros 500 para inspeção (base cheia ~583k
    # chassis produziria JSON enorme; passe limite=None para exportar tudo).
    json_path = OUTPUT_DIR / "vrum_linha_do_tempo_chassi_amostra.json"
    gerar_json_linha_do_tempo(tl, json_path, limite_chassis=500)
    print(f"  json por chassi (amostra 500) -> {json_path}")

    # Relatório de saneamento
    print("\n== Relatório de saneamento ==")
    print(f"  eventos com data AUSENTE:        {int((tl['status_data']=='AUSENTE').sum())}")
    print(f"  eventos FORA_DE_JANELA:          {int((tl['status_data']=='FORA_JANELA').sum())}")
    print(f"  chassis na timeline:             {tl['chassi_id_sintetico'].nunique()}")
    print(f"  chassis cadastrados s/ moviment: {sumario['qtd_eventos'].eq(0).sum()}")
    print(f"  target positivo (chassi):        {int(sumario['target_fraude_90d'].sum())}")
    print("== Concluído. ==")


if __name__ == "__main__":
    main()
