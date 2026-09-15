"""Comparativo de modelos de classificação na base do VRUM.

Experimento de apoio (fora do pipeline principal): treina Dummy, regressão
logística, árvore, Random Forest, HistGradientBoosting e XGBoost (produção)
na mesma base/split do pipeline (features só com passado, split temporal de
30 dias, limiar por F1 na validação) e compara AUC/KS/PR-AUC/Precision/Recall
no OOT.

Conclusão da base atual (sintética, target independente das features): todos
os modelos em AUC OOT ~ 0,50 e PR-AUC igual à taxa base — a escolha de
algoritmo é irrelevante; o ganho depende de dados reais. O XGBoost permanece
na produção por convenção (NaN nativo, peso de classe), não por superioridade
medida.

Detalhe metodológico: NaN em `dias_desde_ultima_proposta` (primeira proposta
do chassi) é preservado para XGBoost/HistGradientBoosting (suporte nativo) e
imputado com sentinela -1 nos demais modelos sklearn.

Execução: `python src/comparativo_modelos.py`
Requisito: output/vrum_timeline_completa.csv + docs/eventos_target_chassi_90d.csv
Saída: apenas console.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from .modelo_xgboost import (
        TARGET,
        carregar_base,
        construir_features,
        escolher_limiar,
        ks_statistic,
        preparar_modelagem,
        split_temporal_30_dias,
    )
except ImportError:
    from modelo_xgboost import (
        TARGET,
        carregar_base,
        construir_features,
        escolher_limiar,
        ks_statistic,
        preparar_modelagem,
        split_temporal_30_dias,
    )


def _avaliar(nome: str, modelo, dados: dict[str, tuple[np.ndarray, np.ndarray]], tempo: float) -> None:
    """Imprime a linha do modelo: AUC val/OOT, KS/PR-AUC/Precision/Recall OOT."""
    x_val, y_val = dados["val"]
    x_oot, y_oot = dados["oot"]
    score_val = modelo.predict_proba(x_val)[:, 1]
    score_oot = modelo.predict_proba(x_oot)[:, 1]
    if nome.startswith("Dummy"):
        limiar = 0.5
    else:
        limiar = escolher_limiar(y_val, score_val)
    pred_oot = (score_oot >= limiar).astype(int)
    print(
        f"{nome:24s} {roc_auc_score(y_val, score_val):8.3f} "
        f"{roc_auc_score(y_oot, score_oot):8.3f} {ks_statistic(y_oot, score_oot):7.3f} "
        f"{average_precision_score(y_oot, score_oot):10.3f} "
        f"{precision_score(y_oot, pred_oot, zero_division=0):8.3f} "
        f"{recall_score(y_oot, pred_oot, zero_division=0):7.3f} {tempo:6.0f}s"
    )


def main() -> None:
    base = construir_features(carregar_base()).filter(pl.col("target_observado") == 1)
    treino, validacao, oot = split_temporal_30_dias(base)
    treino_m, features, categorias = preparar_modelagem(treino)
    validacao_m, _, _ = preparar_modelagem(validacao, categorias)
    oot_m, _, _ = preparar_modelagem(oot, categorias)

    dados = {
        "treino": (
            treino_m.select(features).to_numpy(),
            treino_m[TARGET].to_numpy().astype(int),
        ),
        "val": (
            validacao_m.select(features).to_numpy(),
            validacao_m[TARGET].to_numpy().astype(int),
        ),
        "oot": (
            oot_m.select(features).to_numpy(),
            oot_m[TARGET].to_numpy().astype(int),
        ),
    }
    # Sentinela -1 apenas para modelos sklearn sem suporte nativo a NaN
    dados_imputados = {
        k: (np.nan_to_num(x, nan=-1.0), y) for k, (x, y) in dados.items()
    }

    y_treino = dados["treino"][1]
    peso_positivo = float((y_treino == 0).sum() / (y_treino == 1).sum())

    modelos = [
        ("Dummy (aleatorio)", DummyClassifier(strategy="stratified", random_state=42), True),
        ("Regressao logistica", LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        ("Arvore decisao (d=4)", DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced", random_state=42), True),
        ("Random Forest (100)", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=4, random_state=42), True),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=20, random_state=42), False),
        ("XGBoost (producao)", XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", n_estimators=200,
            max_depth=4, learning_rate=0.05, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=peso_positivo,
            tree_method="hist", n_jobs=4, random_state=42,
        ), False),
    ]

    print(f"base: {base.height} propostas | features: {len(features)} | splits 30d (treino/val/oot)")
    print(f"{'modelo':24s} {'AUC val':>8s} {'AUC oot':>8s} {'KS oot':>7s} {'PR-AUC oot':>10s} {'Prec oot':>8s} {'Rec oot':>7s} {'tempo':>6s}")
    for nome, modelo, imputa in modelos:
        t0 = time.time()
        x_treino, y_treino_fit = (
            dados_imputados["treino"] if imputa else dados["treino"]
        )
        modelo.fit(x_treino, y_treino_fit)
        _avaliar(nome, modelo, dados_imputados if imputa else dados, time.time() - t0)
    print(
        "\nNota: PR-AUC igual à taxa base (~1,5%) em todos os modelos — nesta base\n"
        "sintética o target é independente das features; ganho depende de dados reais."
    )


if __name__ == "__main__":
    main()
