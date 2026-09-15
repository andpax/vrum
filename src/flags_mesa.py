"""Dataset por proposta com as flags da mesa de análise — Projeto VRUM.

Etapa 4 do pipeline (após src/politica_operacional_vrum.py): gera a base
linha a linha consumida pelo dashboard da mesa, com features históricas,
score do modelo, as 13 flags de docs/flags_para_mesa.txt, `qtd_sinais_mesa`
e `zona_decisao`.

Regra de zona_decisao (calibração operacional, ver docstring de aplicar_flags):
  INVESTIGAR se (histórico insuficiente E exposição alta) OU
  (score >= limiar da validação); caso contrário APROVAR provisoriamente.
  BLOQUEAR permanece desabilitado por política (AUC OOT ~ 0,5 nesta base).

As flags comportamentais/financeiras e `qtd_sinais_mesa` NÃO mudam a zona:
nesta base sintética o acionamento delas é proporcional à captura (sem
discriminação), então servem como EVIDÊNCIA para o analista e para ordenar a
fila da mesa — não como gatilho de zona (docs/flags_para_mesa.txt prevê o
somatório como gatilho; desvio documentado por inviabilidade operacional:
qtd_sinais >= 2 encaminharia ~75% das propostas).

Diferença para src/vrum_decision_engine.py (protótipo): aqui as 4 flags
restantes da documentação (muitos_proponentes, muitas_ifs, mudanca_canal,
mudanca_uf) também entram no `qtd_sinais_mesa`, e o score vem do modelo
treinado em src/modelo_xgboost.py.

Limites P90/P95 são calculados SOMENTE nas linhas de treino (anti-leakage) e
persistidos em output/limiares_flags_mesa.csv para auditoria do dashboard.

Execução: `python src/flags_mesa.py`
Requisitos: output/modelo_xgboost_vrum.json e output/limiares_politica_vrum.csv
Saídas: output/vrum_propostas_flags.csv, output/limiares_flags_mesa.csv
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
from xgboost import XGBClassifier

try:
    from .modelo_xgboost import (
        TARGET,
        carregar_base,
        construir_features,
        preparar_modelagem,
    )
except ImportError:
    from modelo_xgboost import (
        TARGET,
        carregar_base,
        construir_features,
        preparar_modelagem,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "modelo_xgboost_vrum.json"
LIMIARES_POLITICA_PATH = OUTPUT_DIR / "limiares_politica_vrum.csv"
PROPOSTAS_FLAGS_PATH = OUTPUT_DIR / "vrum_propostas_flags.csv"
LIMIARES_FLAGS_PATH = OUTPUT_DIR / "limiares_flags_mesa.csv"

# Ordem da fila da mesa (docs/flags_para_mesa.txt): sinais, depois
# exposição+LTV, depois valor, depois score.
FLAGS_COMPORTAMENTAIS = [
    "flag_indice_rotatividade_alto",
    "flag_frequencia_alta",
    "flag_intervalo_curto",
    "flag_posse_historica_curta",
    "flag_alternancia_pf_pj",
    "flag_muitos_proponentes",
    "flag_muitas_ifs",
    "flag_mudanca_canal",
    "flag_mudanca_uf",
]


def contar_distintos_janela(
    df: pl.DataFrame, coluna: str, dias: int, nome: str
) -> pl.DataFrame:
    """Valores distintos de `coluna` entre propostas anteriores do chassi.

    Janela histórica [t - dias, t): exclui a proposta atual, mesmo critério
    das janelas de transferências do modelo (anti-leakage).
    """
    base = df.select(["chassi_id_sintetico", "data_hora_proposta", coluna])
    contagens = (
        base.join(base, on="chassi_id_sintetico", suffix="_ant")
        .filter(
            (pl.col("data_hora_proposta_ant") < pl.col("data_hora_proposta"))
            & (
                pl.col("data_hora_proposta") - pl.col("data_hora_proposta_ant")
                <= pl.duration(days=dias)
            )
        )
        .group_by(["chassi_id_sintetico", "data_hora_proposta"])
        .agg(pl.col(f"{coluna}_ant").n_unique().alias(nome))
    )
    return df.join(
        contagens, on=["chassi_id_sintetico", "data_hora_proposta"], how="left"
    ).with_columns(pl.col(nome).fill_null(0))


def adicionar_features_mesa(base: pl.DataFrame) -> pl.DataFrame:
    """Acrescenta LTV, contagens distintas por janela e mudança canal/UF."""
    base = base.with_columns(
        # FIPE ausente/zero não gera LTV (fica nulo, não infinito)
        pl.when(pl.col("valor_fipe_referencia") > 0)
        .then(pl.col("valor_financiado") / pl.col("valor_fipe_referencia"))
        .otherwise(None)
        .alias("ltv_fipe")
    )
    for coluna, prefixo in [
        ("cpf_cnpj_proponente_sintetico", "proponentes_distintos"),
        ("if_id_sintetico", "ifs_distintas"),
    ]:
        for dias in (45, 90):
            base = contar_distintos_janela(
                base, coluna, dias, f"{prefixo}_{dias}d"
            )
    base = (
        base.sort(["chassi_id_sintetico", "data_hora_proposta", "id_proposta"])
        .with_columns(
            pl.col("canal").shift(1).over("chassi_id_sintetico").alias("canal_anterior"),
            pl.col("uf_proposta").shift(1).over("chassi_id_sintetico").alias("uf_anterior"),
        )
        .with_columns(
            (
                pl.col("canal").is_not_null()
                & pl.col("canal_anterior").is_not_null()
                & (pl.col("canal") != pl.col("canal_anterior"))
            )
            .cast(pl.Int8)
            .alias("flag_mudanca_canal"),
            (
                pl.col("uf_proposta").is_not_null()
                & pl.col("uf_anterior").is_not_null()
                & (pl.col("uf_proposta") != pl.col("uf_anterior"))
            )
            .cast(pl.Int8)
            .alias("flag_mudanca_uf"),
        )
        .drop(["canal_anterior", "uf_anterior"])
    )
    return base


def marcar_safra(base: pl.DataFrame) -> pl.DataFrame:
    """Rotula treino/validacao/oot nos mesmos blocos de 30 dias do modelo."""
    inicio = base.select(pl.col("data_hora_proposta").min()).item()
    marco_30 = inicio + timedelta(days=30)
    marco_60 = inicio + timedelta(days=60)
    return base.with_columns(
        pl.when(pl.col("data_hora_proposta") < marco_30)
        .then(pl.lit("treino"))
        .when(pl.col("data_hora_proposta") < marco_60)
        .then(pl.lit("validacao"))
        .otherwise(pl.lit("oot"))
        .alias("safra")
    )


def pontuar(base: pl.DataFrame) -> pl.DataFrame:
    """Aplica o modelo treinado a TODAS as propostas (uso diário da mesa:
    proposta nova ainda não tem target observado)."""
    treino = base.filter(pl.col("safra") == "treino")
    _, features, categorias = preparar_modelagem(treino)
    codificada, _, _ = preparar_modelagem(base, categorias)

    modelo = XGBClassifier()
    modelo.load_model(MODEL_PATH)
    assert modelo.get_booster().num_features() == len(features)
    score = modelo.predict_proba(codificada.select(features).to_numpy())[:, 1]
    return base.with_columns(pl.Series("score_modelo", score))


def calcular_limiares(base: pl.DataFrame) -> dict[str, float]:
    """Limites P90/P95 calculados somente no treino (anti-leakage)."""
    treino = base.filter(pl.col("safra") == "treino")

    def quantil(coluna: str, p: float) -> float:
        return float(treino.select(pl.col(coluna).quantile(p)).item())

    return {
        "p95_indice_rotatividade": quantil("indice_rotatividade_vrum", 0.95),
        "p95_transferencias_45d": max(
            1, int(quantil("transferencias_ultimos_45d", 0.95))
        ),
        "p90_exposicao": quantil("valor_financiado", 0.90),
        "p90_ltv": quantil("ltv_fipe", 0.90),
        "p95_proponentes_45d": max(
            1, int(quantil("proponentes_distintos_45d", 0.95))
        ),
        "p95_proponentes_90d": max(
            1, int(quantil("proponentes_distintos_90d", 0.95))
        ),
        "p95_ifs_45d": max(1, int(quantil("ifs_distintas_45d", 0.95))),
        "p95_ifs_90d": max(1, int(quantil("ifs_distintas_90d", 0.95))),
    }


def aplicar_flags(base: pl.DataFrame, limiares: dict[str, float]) -> pl.DataFrame:
    """Calcula as 13 flags, qtd_sinais_mesa (informativos) e zona_decisao.

    Zona usa apenas histórico insuficiente + exposição alta e o limiar de
    score da validação; o somatório de sinais não é gatilho de zona nesta
    calibração (base sintética: acionamento das flags não discrimina risco).
    """
    p95_index = limiares["p95_indice_rotatividade"]
    # P95 == 0: usar > para não flagar toda a base de índice nulo
    cond_indice = (
        pl.col("indice_rotatividade_vrum") > p95_index
        if p95_index == 0
        else pl.col("indice_rotatividade_vrum") >= p95_index
    )
    limiar_investigar = float(
        pl.read_csv(LIMIARES_POLITICA_PATH, separator=";")
        .filter(pl.col("regra") == "investigar")["limiar_score"][0]
    )

    return (
        base.with_columns(
            cond_indice.cast(pl.Boolean).alias("flag_indice_rotatividade_alto"),
            (pl.col("transferencias_ultimos_45d") >= limiares["p95_transferencias_45d"])
            .alias("flag_frequencia_alta"),
            (
                pl.col("dias_desde_ultima_proposta").is_not_null()
                & (pl.col("dias_desde_ultima_proposta") <= 5)
            ).alias("flag_intervalo_curto"),
            # Limite fixo de 15 dias (precedente do decision engine); a doc
            # prevê limite calibrado no treino quando houver maturação de target.
            (
                (pl.col("tempo_posse_mediano_acumulado") > 0)
                & (pl.col("tempo_posse_mediano_acumulado") <= 15)
            ).alias("flag_posse_historica_curta"),
            pl.col("flag_alternancia_pf_pj").cast(pl.Boolean),
            (
                (pl.col("proponentes_distintos_45d") >= limiares["p95_proponentes_45d"])
                | (pl.col("proponentes_distintos_90d") >= limiares["p95_proponentes_90d"])
            ).alias("flag_muitos_proponentes"),
            (
                (pl.col("ifs_distintas_45d") >= limiares["p95_ifs_45d"])
                | (pl.col("ifs_distintas_90d") >= limiares["p95_ifs_90d"])
            ).alias("flag_muitas_ifs"),
            pl.col("flag_mudanca_canal").cast(pl.Boolean),
            pl.col("flag_mudanca_uf").cast(pl.Boolean),
            (pl.col("qtd_propostas_historicas") < 2)
            .alias("flag_historico_insuficiente"),
            (pl.col("valor_financiado") >= limiares["p90_exposicao"])
            .alias("flag_exposicao_alta"),
            (pl.col("ltv_fipe") >= limiares["p90_ltv"]).alias("flag_ltv_alto"),
        )
        .with_columns(
            (pl.col("flag_exposicao_alta") & pl.col("flag_ltv_alto"))
            .alias("flag_exposicao_e_ltv_altos"),
            pl.sum_horizontal([pl.col(f) for f in FLAGS_COMPORTAMENTAIS])
            .alias("qtd_sinais_mesa"),
        )
        .with_columns(
            pl.when(
                (pl.col("flag_historico_insuficiente") & pl.col("flag_exposicao_alta"))
                | (pl.col("score_modelo") >= limiar_investigar)
            )
            .then(pl.lit("INVESTIGAR"))
            .otherwise(pl.lit("APROVAR"))
            .alias("zona_decisao")
        )
    )


def ordenar_fila_mesa(df: pl.DataFrame) -> pl.DataFrame:
    """Ordena a fila: BLOQUEAR (reservado), INVESTIGAR, APROVAR; dentro da
    zona, prioridade por sinais, exposição+LTV, valor e score."""
    return (
        df.with_columns(
            pl.col("zona_decisao")
            .replace_strict({"BLOQUEAR": 0, "INVESTIGAR": 1, "APROVAR": 2})
            .alias("_ordem_zona")
        )
        .sort(
            [
                "_ordem_zona",
                "qtd_sinais_mesa",
                "flag_exposicao_e_ltv_altos",
                "valor_financiado",
                "score_modelo",
            ],
            descending=[False, True, True, True, True],
        )
        .drop("_ordem_zona")
    )


def main() -> None:
    print("== VRUM: flags para a mesa de análise ==")
    base = marcar_safra(adicionar_features_mesa(construir_features(carregar_base())))
    base = pontuar(base)
    print(f"  propostas pontuadas: {base.height}")

    limiares = calcular_limiares(base)
    df = aplicar_flags(base, limiares)

    colunas_saida = [
        "id_proposta", "data_hora_proposta", "chassi_id_sintetico",
        "if_id_sintetico", "cpf_cnpj_proponente_sintetico", "tipo_proponente",
        "canal", "uf_proposta", "safra", "score_modelo",
        "target_observado", "target_risco_90d",
        "valor_financiado", "valor_entrada", "prazo_meses",
        "valor_fipe_referencia", "ltv_fipe",
        "dias_desde_ultima_proposta", "qtd_propostas_historicas",
        "tempo_posse_mediano_acumulado",
        "transferencias_ultimos_7d", "transferencias_ultimos_15d",
        "transferencias_ultimos_30d", "transferencias_ultimos_45d",
        "transferencias_ultimos_90d", "indice_rotatividade_vrum",
        "proponentes_distintos_45d", "proponentes_distintos_90d",
        "ifs_distintas_45d", "ifs_distintas_90d",
        *FLAGS_COMPORTAMENTAIS,
        "flag_historico_insuficiente", "flag_exposicao_alta",
        "flag_ltv_alto", "flag_exposicao_e_ltv_altos",
        "qtd_sinais_mesa", "zona_decisao",
    ]
    df = ordenar_fila_mesa(df.select(colunas_saida))

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.write_csv(PROPOSTAS_FLAGS_PATH, separator=";")
    pl.DataFrame(
        {"limiar": list(limiares), "valor": [float(v) for v in limiares.values()]}
    ).write_csv(LIMIARES_FLAGS_PATH, separator=";")

    print(f"  dataset com flags -> {PROPOSTAS_FLAGS_PATH}  ({df.shape})")
    print(f"  limiares (treino) -> {LIMIARES_FLAGS_PATH}")
    print("\nZonas de decisão:")
    print(df.group_by("zona_decisao").agg(pl.len().alias("n")).sort("zona_decisao"))
    print("\nFlags mais acionadas:")
    print(
        pl.DataFrame(
            {
                "flag": FLAGS_COMPORTAMENTAIS,
                "acionamentos": [
                    df.get_column(f).sum() for f in FLAGS_COMPORTAMENTAIS
                ],
            }
        ).sort("acionamentos", descending=True)
    )
    print("\nCaptura de risco em INVESTIGAR (target observado):")
    obs = df.filter(pl.col("target_observado") == 1)
    positivos_total = obs.get_column(TARGET).sum()
    positivos_inv = obs.filter(pl.col("zona_decisao") == "INVESTIGAR").get_column(TARGET).sum()
    print(f"  {positivos_inv}/{positivos_total} positivos em INVESTIGAR")


if __name__ == "__main__":
    main()
