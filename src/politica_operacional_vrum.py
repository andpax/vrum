"""Avalia política operacional Aprovar/Investigar/Bloquear para o VRUM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from .modelo_xgboost import (
        TARGET,
        construir_features,
        carregar_base,
        preparar_modelagem,
        split_temporal_30_dias,
    )
except ImportError:
    from modelo_xgboost import (
        TARGET,
        construir_features,
        carregar_base,
        preparar_modelagem,
        split_temporal_30_dias,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "modelo_xgboost_vrum.json"


def pontuar_splits(base: pl.DataFrame) -> tuple[pl.DataFrame, float, float]:
    base = base.filter(pl.col("target_observado") == 1)
    treino, validacao, oot = split_temporal_30_dias(base)
    treino_modelo, features, categorias = preparar_modelagem(treino)
    validacao_modelo, _, _ = preparar_modelagem(validacao, categorias)
    oot_modelo, _, _ = preparar_modelagem(oot, categorias)

    modelo = XGBClassifier()
    modelo.load_model(MODEL_PATH)
    assert modelo.get_booster().num_features() == len(features)

    pontuados = []
    for nome, original, codificado in [
        ("treino", treino, treino_modelo),
        ("validacao", validacao, validacao_modelo),
        ("oot", oot, oot_modelo),
    ]:
        score = modelo.predict_proba(codificado.select(features).to_numpy())[:, 1]
        pontuados.append(
            original.with_columns(
                pl.Series("score_modelo", score), pl.lit(nome).alias("safra")
            )
        )

    validacao_score = pontuados[1].get_column("score_modelo").to_numpy()
    return (
        pl.concat(pontuados),
        float(np.quantile(validacao_score, 0.95)),
        float(np.quantile(validacao_score, 0.99)),
    )


def aplicar_zonas(
    df: pl.DataFrame, limiar_investigar: float, limiar_bloquear: float
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("score_modelo") < limiar_investigar)
        .then(pl.lit("APROVAR"))
        .when(pl.col("score_modelo") < limiar_bloquear)
        .then(pl.lit("INVESTIGAR"))
        .otherwise(pl.lit("BLOQUEAR"))
        .alias("zona_decisao")
    )


def resumo_zonas(df: pl.DataFrame, grupo: str) -> pl.DataFrame:
    positivos_total = int(df[TARGET].sum())
    negativos_total = df.height - positivos_total
    resumo = (
        df.with_columns(pl.col("valor_financiado").fill_null(0))
        .group_by("zona_decisao")
        .agg(
            [
                pl.len().alias("n"),
                pl.col(TARGET).sum().alias("positivos"),
                pl.col("valor_financiado").sum().alias("exposicao_total"),
                pl.when(pl.col(TARGET) == 1)
                .then(pl.col("valor_financiado"))
                .otherwise(0)
                .sum()
                .alias("exposicao_risco"),
            ]
        )
        .with_columns(
            pl.lit(grupo).alias("grupo"),
            pl.when(pl.col("zona_decisao") == "APROVAR")
            .then(0)
            .otherwise(pl.col("n") - pl.col("positivos"))
            .alias("falsos_positivos"),
            (pl.col("positivos") / pl.col("n")).alias("taxa_risco_zona"),
            pl.when(pl.col("zona_decisao") == "APROVAR")
            .then(0)
            .otherwise(pl.col("positivos") / positivos_total)
            .alias("captura_risco"),
            (
                pl.when(pl.col("zona_decisao") == "APROVAR")
                .then(0)
                .otherwise((pl.col("n") - pl.col("positivos")) / negativos_total)
            ).alias("falso_positivo_global"),
        )
        .with_columns(
            ((pl.col("n") - pl.col("positivos")) / pl.col("n")).alias(
                "proporcao_legitimos_zona"
            ),
            pl.when(pl.col("zona_decisao") == "APROVAR")
            .then(0)
            .otherwise(pl.col("exposicao_risco") * 0.25)
            .alias("saving_proxy_25pct"),
            pl.when(pl.col("zona_decisao") == "APROVAR")
            .then(0)
            .otherwise(pl.col("exposicao_risco") * 0.50)
            .alias("saving_proxy_50pct"),
            pl.when(pl.col("zona_decisao") == "APROVAR")
            .then(0)
            .otherwise(pl.col("exposicao_risco") * 1.00)
            .alias("saving_proxy_100pct"),
        )
        .with_columns(
            pl.col("zona_decisao")
            .replace({"APROVAR": 0, "INVESTIGAR": 1, "BLOQUEAR": 2})
            .cast(pl.Int8)
            .alias("ordem_zona")
        )
        .sort("ordem_zona")
        .drop("ordem_zona")
    )
    return resumo


def estabilidade_por_if(df: pl.DataFrame) -> pl.DataFrame:
    linhas = []
    grupos = df.partition_by(["safra", "if_id_sintetico"], as_dict=True)
    for (safra, if_id), grupo in grupos.items():
        y = grupo[TARGET].to_numpy()
        score = grupo["score_modelo"].to_numpy()
        n = grupo.height
        linhas.append(
            {
                "safra": safra,
                "if_id_sintetico": if_id,
                "n": n,
                "positivos": int(y.sum()),
                "taxa_risco": float(y.mean()),
                "score_mediano": float(np.median(score)),
                "taxa_aprovar": float(
                    (grupo["zona_decisao"] == "APROVAR").sum() / n
                ),
                "taxa_investigar": float(
                    (grupo["zona_decisao"] == "INVESTIGAR").sum() / n
                ),
                "taxa_bloquear": float(
                    (grupo["zona_decisao"] == "BLOQUEAR").sum() / n
                ),
                "auc_roc": float(roc_auc_score(y, score))
                if np.unique(y).size == 2
                else float("nan"),
                "pr_auc": float(average_precision_score(y, score))
                if y.sum() > 0
                else float("nan"),
            }
        )
    return pl.DataFrame(linhas).sort(["safra", "if_id_sintetico"])


def main() -> None:
    base = construir_features(carregar_base())
    pontuado, limiar_investigar, limiar_bloquear = pontuar_splits(base)
    pontuado = aplicar_zonas(pontuado, limiar_investigar, limiar_bloquear)

    resumo = pl.concat(
        [resumo_zonas(pontuado.filter(pl.col("safra") == safra), safra) for safra in ["treino", "validacao", "oot"]]
    )
    estabilidade = estabilidade_por_if(pontuado)
    limiares = pl.DataFrame(
        {
            "regra": ["investigar", "bloquear"],
            "quantil_validacao": [0.95, 0.99],
            "limiar_score": [limiar_investigar, limiar_bloquear],
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    resumo.write_csv(OUTPUT_DIR / "politica_zonas_vrum.csv")
    estabilidade.write_csv(OUTPUT_DIR / "estabilidade_safra_if_vrum.csv")
    limiares.write_csv(OUTPUT_DIR / "limiares_politica_vrum.csv")

    print("Limiar de investigação:", f"{limiar_investigar:.6f}")
    print("Limiar de bloqueio:", f"{limiar_bloquear:.6f}")
    print("\nPolítica por safra:")
    print(resumo)
    print("\nEstabilidade por IF:")
    print(estabilidade)
    print(
        "\nSaving é proxy de exposição de risco; não representa perda observada."
    )


if __name__ == "__main__":
    main()
