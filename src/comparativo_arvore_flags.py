"""Comparativo: árvore de decisão sobre as flags vs. score XGBoost.

Experimento de apoio (fora do pipeline principal): testa se uma árvore de
decisão treinada sobre as 13 flags da mesa agrega poder de triagem além do
score do modelo. Conclusão da base atual (sintética, AUC OOT ~ 0,5): não
agrega — todas as folhas da árvore convergem para a classe majoritária e a
AUC OOT fica em ~0,50, igual à referência. As flags seguem como evidência e
priorização de fila (docs/flags_para_mesa.txt), não como classificador.

Execução: `python src/comparativo_arvore_flags.py`
Requisito: output/vrum_propostas_flags.csv (etapa src/flags_mesa.py)
Saída: apenas console (tabela de métricas + regras da árvore depth=3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_text

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAGS_PATH = REPO_ROOT / "output" / "vrum_propostas_flags.csv"

FLAGS = [
    "flag_indice_rotatividade_alto",
    "flag_frequencia_alta",
    "flag_intervalo_curto",
    "flag_posse_historica_curta",
    "flag_alternancia_pf_pj",
    "flag_muitos_proponentes",
    "flag_muitas_ifs",
    "flag_mudanca_canal",
    "flag_mudanca_uf",
    "flag_historico_insuficiente",
    "flag_exposicao_alta",
    "flag_ltv_alto",
    "flag_exposicao_e_ltv_altos",
]


def ks(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def linha(nome: str, y_val, s_val, y_oot, s_oot) -> str:
    return (
        f"{nome:28s} {roc_auc_score(y_val, s_val):8.3f} {ks(y_val, s_val):7.3f} "
        f"{roc_auc_score(y_oot, s_oot):8.3f} {ks(y_oot, s_oot):7.3f} "
        f"{average_precision_score(y_oot, s_oot):10.3f}"
    )


def main() -> None:
    if not FLAGS_PATH.exists():
        raise FileNotFoundError(f"Base de flags não encontrada: {FLAGS_PATH}")
    df = pl.read_csv(FLAGS_PATH, separator=";")
    # Avaliação exige target conhecido; propostas sem evento não têm rótulo
    obs = df.filter(pl.col("target_observado") == 1)
    treino = obs.filter(pl.col("safra") == "treino")
    validacao = obs.filter(pl.col("safra") == "validacao")
    oot = obs.filter(pl.col("safra") == "oot")

    x = lambda d: d.select(FLAGS).to_numpy()
    y = lambda d: d.get_column("target_risco_90d").to_numpy().astype(int)
    y_val, y_oot = y(validacao), y(oot)

    print(f"{'modelo':28s} {'AUC val':>8s} {'KS val':>7s} {'AUC oot':>8s} {'KS oot':>7s} {'PR-AUC oot':>10s}")
    print(
        linha(
            "XGBoost score (referencia)",
            y_val,
            validacao.get_column("score_modelo").to_numpy(),
            y_oot,
            oot.get_column("score_modelo").to_numpy(),
        )
    )

    # min_samples_leaf alto evita folhas absurdas com target ~1,5%
    for depth in (2, 3, 4):
        arvore = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=5000, random_state=42
        )
        arvore.fit(x(treino), y(treino))
        print(
            linha(
                f"arvore flags depth={depth}",
                y_val,
                arvore.predict_proba(x(validacao))[:, 1],
                y_oot,
                arvore.predict_proba(x(oot))[:, 1],
            )
        )

    print("\n--- regras da arvore depth=3 ---")
    arvore3 = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5000, random_state=42
    )
    arvore3.fit(x(treino), y(treino))
    print(export_text(arvore3, feature_names=FLAGS, max_depth=3))


if __name__ == "__main__":
    main()
