"""Treina modelo XGBoost com features históricas e split temporal de 30 dias."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "output" / "vrum_timeline_completa.csv"
OUTPUT_DIR = REPO_ROOT / "output"

TARGET = "target_risco_90d"
FEATURES_NUMERICOS = [
    "valor_financiado",
    "valor_entrada",
    "prazo_meses",
    "dias_desde_ultima_proposta",
    "tempo_posse_mediano_acumulado",
    "transferencias_ultimos_7d",
    "transferencias_ultimos_15d",
    "transferencias_ultimos_30d",
    "transferencias_ultimos_45d",
    "transferencias_ultimos_90d",
    "indice_rotatividade_vrum",
    "flag_alternancia_pf_pj",
]
FEATURES_CATEGORICOS = ["tipo_proponente", "canal", "uf_proposta"]


def carregar_base(caminho: Path = INPUT_PATH) -> pl.DataFrame:
    """Lê propostas e liga target pela linha de evento de risco correspondente."""
    timeline = pl.read_csv(
        caminho,
        separator=";",
        null_values=[""],
        try_parse_dates=False,
        infer_schema_length=10_000,
    ).with_columns(
        pl.col("data_hora_proposta").str.strptime(
            pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
        )
    )

    propostas = (
        timeline.filter(pl.col("origem_registro") == "PROPOSTA_FINANCIAMENTO")
        .select(
            [
                "id_proposta",
                "data_hora_proposta",
                "chassi_id_sintetico",
                "if_id_sintetico",
                "tipo_proponente",
                "valor_financiado",
                "valor_entrada",
                "prazo_meses",
                "canal",
                "uf_proposta",
            ]
        )
        .sort(["chassi_id_sintetico", "data_hora_proposta", "id_proposta"])
    )
    eventos = timeline.filter(pl.col("origem_registro") == "EVENTO_RISCO").select(
        "id_proposta"
    ).unique()

    return propostas.join(
        eventos.with_columns(pl.lit(1).cast(pl.Int8).alias(TARGET)),
        on="id_proposta",
        how="left",
    ).with_columns(pl.col(TARGET).fill_null(0).cast(pl.Int8))


def adicionar_janela(df: pl.DataFrame, dias: int) -> pl.DataFrame:
    """Conta propostas anteriores no intervalo [agora - dias, agora)."""
    nome = f"transferencias_ultimos_{dias}d"
    janela = (
        df.select(["chassi_id_sintetico", "data_hora_proposta"])
        .rolling(
            index_column="data_hora_proposta",
            period=f"{dias}d",
            group_by="chassi_id_sintetico",
            closed="left",
        )
        .agg(pl.col("data_hora_proposta").count().alias(nome))
        .unique(["chassi_id_sintetico", "data_hora_proposta"])
    )
    return df.join(janela, on=["chassi_id_sintetico", "data_hora_proposta"])


def construir_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula somente histórico anterior à proposta atual."""
    df = df.sort(["chassi_id_sintetico", "data_hora_proposta", "id_proposta"])
    df = df.with_columns(
        (
            (pl.col("data_hora_proposta") - pl.col("data_hora_proposta").shift(1).over("chassi_id_sintetico"))
            .dt.total_seconds()
            / 86_400
        ).alias("tempo_posse_dias")
    )
    df = df.with_columns(
        pl.col("tipo_proponente")
        .shift(1)
        .over("chassi_id_sintetico")
        .alias("tipo_proponente_anterior")
    ).with_columns(
        (
            pl.col("tipo_proponente").is_not_null()
            & pl.col("tipo_proponente_anterior").is_not_null()
            & (pl.col("tipo_proponente") != pl.col("tipo_proponente_anterior"))
        )
        .cast(pl.Int8)
        .alias("flag_alternancia_pf_pj")
    )
    df = df.with_columns(
        pl.col("tempo_posse_dias")
        .shift(1)
        .cumulative_eval(
            pl.element().drop_nulls().quantile(0.5, interpolation="linear"),
            min_samples=1,
        )
        .over("chassi_id_sintetico")
        .fill_null(0)
        .alias("tempo_posse_mediano_acumulado")
    )
    for dias in [7, 15, 30, 45, 90]:
        df = adicionar_janela(df, dias)
    df = df.with_columns(
        (
            pl.col("transferencias_ultimos_45d")
            / (pl.col("tempo_posse_mediano_acumulado") + 1)
        ).alias("indice_rotatividade_vrum"),
        pl.col("tempo_posse_dias").alias("dias_desde_ultima_proposta"),
    )
    return df.drop(["tempo_posse_dias", "tipo_proponente_anterior"])


def preparar_modelagem(
    df: pl.DataFrame, categorias: dict[str, list[str]] | None = None
) -> tuple[pl.DataFrame, list[str], dict[str, list[str]]]:
    """Codifica categorias usando somente categorias conhecidas no treino."""
    df = df.with_columns(
        [pl.col(coluna).fill_null("DESCONHECIDO") for coluna in FEATURES_CATEGORICOS]
    )
    if categorias is None:
        categorias = {
            coluna: df.get_column(coluna).unique().sort().to_list()
            for coluna in FEATURES_CATEGORICOS
        }
    nomes_dummies = []
    for coluna in FEATURES_CATEGORICOS:
        for categoria in categorias[coluna][1:]:
            nome = f"{coluna}_{categoria}"
            df = df.with_columns((pl.col(coluna) == categoria).cast(pl.Int8).alias(nome))
            nomes_dummies.append(nome)
    features = FEATURES_NUMERICOS + nomes_dummies
    return df.drop(FEATURES_CATEGORICOS), features, categorias


def split_temporal_30_dias(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Divide em blocos consecutivos de 30 dias, sem embaralhamento."""
    inicio = df.select(pl.col("data_hora_proposta").min()).item()
    marco_30 = inicio + timedelta(days=30)
    marco_60 = inicio + timedelta(days=60)
    treino = df.filter(pl.col("data_hora_proposta") < marco_30)
    validacao = df.filter(
        (pl.col("data_hora_proposta") >= marco_30)
        & (pl.col("data_hora_proposta") < marco_60)
    )
    oot = df.filter(pl.col("data_hora_proposta") >= marco_60)
    return treino, validacao, oot


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def escolher_limiar(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def avaliar(
    nome: str, y_true: np.ndarray, score: np.ndarray, limiar: float
) -> dict[str, float | str]:
    pred = (score >= limiar).astype(np.int8)
    return {
        "split": nome,
        "n": float(len(y_true)),
        "positivos": float(y_true.sum()),
        "taxa_positiva": float(y_true.mean()),
        "auc_roc": float(roc_auc_score(y_true, score)),
        "ks": ks_statistic(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "limiar": float(limiar),
    }


def main() -> None:
    base = construir_features(carregar_base())
    treino, validacao, oot = split_temporal_30_dias(base)
    treino, features, categorias = preparar_modelagem(treino)
    validacao, _, _ = preparar_modelagem(validacao, categorias)
    oot, _, _ = preparar_modelagem(oot, categorias)
    proibidas = {TARGET, "tipo_risco", "dt_risco"}
    assert proibidas.isdisjoint(features)
    assert treino.height and validacao.height and oot.height
    assert treino["data_hora_proposta"].max() < validacao["data_hora_proposta"].min()
    assert validacao["data_hora_proposta"].max() < oot["data_hora_proposta"].min()

    x_treino = treino.select(features).to_numpy()
    x_validacao = validacao.select(features).to_numpy()
    x_oot = oot.select(features).to_numpy()
    y_treino = treino[TARGET].to_numpy()
    y_validacao = validacao[TARGET].to_numpy()
    y_oot = oot[TARGET].to_numpy()
    peso_positivo = float((y_treino == 0).sum() / (y_treino == 1).sum())

    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=peso_positivo,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    modelo.fit(x_treino, y_treino, eval_set=[(x_validacao, y_validacao)], verbose=False)

    scores = {
        "treino": modelo.predict_proba(x_treino)[:, 1],
        "validacao": modelo.predict_proba(x_validacao)[:, 1],
        "oot": modelo.predict_proba(x_oot)[:, 1],
    }
    limiar = escolher_limiar(y_validacao, scores["validacao"])
    metricas = pl.DataFrame(
        [
            avaliar("treino", y_treino, scores["treino"], limiar),
            avaliar("validacao", y_validacao, scores["validacao"], limiar),
            avaliar("oot", y_oot, scores["oot"], limiar),
        ]
    )
    importancia = pl.DataFrame(
        {"feature": features, "importance": modelo.feature_importances_}
    ).sort("importance", descending=True)
    resumo_split = pl.DataFrame(
        {
            "split": ["treino", "validacao", "oot"],
            "linhas": [treino.height, validacao.height, oot.height],
            "positivos": [
                int(treino[TARGET].sum()),
                int(validacao[TARGET].sum()),
                int(oot[TARGET].sum()),
            ],
            "inicio": [
                treino["data_hora_proposta"].min(),
                validacao["data_hora_proposta"].min(),
                oot["data_hora_proposta"].min(),
            ],
            "fim": [
                treino["data_hora_proposta"].max(),
                validacao["data_hora_proposta"].max(),
                oot["data_hora_proposta"].max(),
            ],
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    modelo.save_model(OUTPUT_DIR / "modelo_xgboost_vrum.json")
    metricas.write_csv(OUTPUT_DIR / "metricas_xgboost_vrum.csv")
    importancia.write_csv(OUTPUT_DIR / "importancia_xgboost_vrum.csv")
    resumo_split.write_csv(OUTPUT_DIR / "split_temporal_30d_vrum.csv")

    print("Split temporal de 30 dias:")
    print(resumo_split)
    print("\nMétricas:")
    print(metricas)
    print("\nFeatures mais importantes:")
    print(importancia.head(15))
    print(f"\nLimiar escolhido na validação: {limiar:.6f}")


if __name__ == "__main__":
    main()
