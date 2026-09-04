"""Motor de decisão operacional das propostas VRUM.

Aplica flags comportamentais/financeiras e zona de decisão (APROVAR /
INVESTIGAR) sobre um DataFrame já enriquecido com features históricas.
Limites (P90/P95) são calculados SOMENTE nas linhas de treino para evitar
leakage; BLOQUEAR permanece desabilitado por política (ver
docs/flags_para_mesa.txt).

Contrato de entrada (colunas em REQUIRED_COLUMNS, mais uma de split):
  - features históricas: geradas por src/modelo_xgboost.py::construir_features
    (atenção: `tipo_proponente_anterior` é dropada lá; recriá-la ou montar o
    DataFrame via notebook se for usar este motor fora dos testes)
  - `ltv_fipe`: valor_financiado / valor_fipe_referencia (presente no notebook
    notebooks/processo_completo_vrum.ipynb; não é gerada pelo pipeline src)
  - coluna de split: `split_group` (Train_Set/Validation_Set/OOT_Production)
    ou `safra` (treino/validacao/oot)
  - opcional: `score_modelo` (saída de src/politica_operacional_vrum.py)
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "valor_financiado",
    "ltv_fipe",
    "indice_rotatividade_vrum",
    "transferencias_ultimos_45d",
    "dias_desde_ultima_proposta",
    "tempo_posse_mediano_acumulado",
    "tipo_proponente",
    "tipo_proponente_anterior",
    "qtd_propostas_historicas",
}


def build_vrum_decision_engine(
    df: pd.DataFrame,
    threshold_score_investigar: float | None = None,
) -> pd.DataFrame:
    """Aplica flags e zona de decisão sem calcular limites com dados futuros."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    result = df.copy()
    split_column = "split_group" if "split_group" in result else "safra"
    if split_column not in result:
        raise ValueError("Coluna de split obrigatoria: split_group ou safra")

    split_values = result[split_column].astype("string").str.lower()
    train_mask = split_values.isin(["train", "train_set", "treino"])
    if not train_mask.any():
        raise ValueError("Nenhuma linha de treino encontrada para calcular limites")
    train = result.loc[train_mask]

    p95_index = train["indice_rotatividade_vrum"].quantile(0.95)
    p90_exposure = train["valor_financiado"].quantile(0.90)
    p90_ltv = train["ltv_fipe"].quantile(0.90)
    p95_frequency = train["transferencias_ultimos_45d"].quantile(0.95)
    frequency_limit = max(1, int(p95_frequency))

    result["flag_indice_rotatividade_alto"] = (
        result["indice_rotatividade_vrum"] > p95_index
        if p95_index == 0
        else result["indice_rotatividade_vrum"] >= p95_index
    )
    result["flag_frequencia_alta"] = (
        result["transferencias_ultimos_45d"] >= frequency_limit
    )
    result["flag_intervalo_curto"] = (
        result["dias_desde_ultima_proposta"].notna()
        & (result["dias_desde_ultima_proposta"] <= 5)
    )
    result["flag_posse_historica_curta"] = (
        result["tempo_posse_mediano_acumulado"] > 0
    ) & (result["tempo_posse_mediano_acumulado"] <= 15)
    result["flag_alternancia_pf_pj"] = (
        result["tipo_proponente_anterior"].notna()
        & result["tipo_proponente"].notna()
        & (
            result["tipo_proponente_anterior"]
            != result["tipo_proponente"]
        )
    )
    result["flag_historico_insuficiente"] = (
        result["qtd_propostas_historicas"] < 2
    )

    result["flag_exposicao_alta"] = (
        result["valor_financiado"] >= p90_exposure
    )
    result["flag_ltv_alto"] = result["ltv_fipe"] >= p90_ltv
    result["flag_exposicao_e_ltv_altos"] = (
        result["flag_exposicao_alta"] & result["flag_ltv_alto"]
    )

    behavior_flags = [
        "flag_indice_rotatividade_alto",
        "flag_frequencia_alta",
        "flag_intervalo_curto",
        "flag_posse_historica_curta",
        "flag_alternancia_pf_pj",
    ]
    result["qtd_sinais_mesa"] = result[behavior_flags].sum(axis=1)
    result["zona_decisao"] = "APROVAR"

    investigate = (
        result["flag_historico_insuficiente"]
        & result["flag_exposicao_alta"]
    ) | (result["qtd_sinais_mesa"] >= 2)
    if threshold_score_investigar is None and "score_modelo" in result:
        threshold_score_investigar = train["score_modelo"].quantile(0.95)
    if threshold_score_investigar is not None and "score_modelo" in result:
        investigate |= result["score_modelo"] >= threshold_score_investigar
    result.loc[investigate, "zona_decisao"] = "INVESTIGAR"

    result["zona_decisao_bloqueio_ativo"] = False
    result["_ordem_zona"] = result["zona_decisao"].map(
        {"BLOQUEAR": 0, "INVESTIGAR": 1, "APROVAR": 2}
    )
    score_column = "score_modelo" if "score_modelo" in result else "valor_financiado"
    return result.sort_values(
        by=[
            "_ordem_zona",
            "qtd_sinais_mesa",
            "flag_exposicao_e_ltv_altos",
            "valor_financiado",
            score_column,
        ],
        ascending=[True, False, False, False, False],
    ).drop(columns="_ordem_zona").reset_index(drop=True)
