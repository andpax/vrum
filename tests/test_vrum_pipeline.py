import unittest

import pandas as pd
import polars as pl

from src.modelo_xgboost import anexar_target_observado, construir_features
from src.vrum_decision_engine import build_vrum_decision_engine
from src.vrum_split_temporal import apply_temporal_split


class TestTarget(unittest.TestCase):
    def test_preserva_target_nao_observado(self):
        propostas = pl.DataFrame({"id_proposta": ["p1", "p2", "p3"]})
        eventos = pl.DataFrame(
            {
                "id_proposta": ["p1", "p2"],
                "evento_risco_chassi_90d": [0, 1],
                "tipo_evento": ["SEM_EVENTO", "NEVER_PAY"],
            }
        )

        resultado = anexar_target_observado(propostas, eventos)

        self.assertEqual(resultado["target_observado"].to_list(), [1, 1, 0])
        self.assertEqual(resultado["target_risco_90d"].to_list(), [0, 1, None])


class TestFeatures(unittest.TestCase):
    def test_historico_e_janela_excluem_proposta_atual(self):
        base = pl.DataFrame(
            {
                "id_proposta": ["p1", "p2"],
                "data_hora_proposta": pl.datetime_range(
                    pl.datetime(2026, 1, 1),
                    pl.datetime(2026, 1, 2),
                    interval="1d",
                    eager=True,
                ),
                "chassi_id_sintetico": ["c1", "c1"],
                "tipo_proponente": ["PF", "PJ"],
            }
        )

        resultado = construir_features(base)

        self.assertEqual(resultado["qtd_propostas_historicas"].to_list(), [0, 1])
        self.assertEqual(resultado["transferencias_ultimos_45d"].to_list(), [0, 1])
        self.assertEqual(resultado["flag_alternancia_pf_pj"].to_list(), [0, 1])
        self.assertEqual(resultado["indice_rotatividade_vrum"].to_list(), [0.0, 1.0])


class TestSplit(unittest.TestCase):
    def test_split_temporal_forma_tres_blocos_de_30_dias(self):
        base = pd.DataFrame(
            {
                "timestamp_proposta": pd.date_range("2026-01-01", periods=90),
                "valor": range(90),
            }
        )

        treino, validacao, oot, completo = apply_temporal_split(base)

        self.assertEqual([len(treino), len(validacao), len(oot)], [30, 30, 30])
        self.assertEqual(
            completo["split_group"].value_counts().to_dict(),
            {"Train_Set": 30, "Validation_Set": 30, "OOT_Production": 30},
        )


class TestDecisionEngine(unittest.TestCase):
    def test_usa_schema_real_e_detecta_alternancia(self):
        base = pd.DataFrame(
            {
                "valor_financiado": [50_000, 150_000, 60_000],
                "ltv_fipe": [0.8, 1.4, 0.9],
                "indice_rotatividade_vrum": [0.0, 1.0, 0.2],
                "transferencias_ultimos_45d": [0, 1, 0],
                "dias_desde_ultima_proposta": [None, 3.0, 20.0],
                "tempo_posse_mediano_acumulado": [0.0, 3.0, 20.0],
                "tipo_proponente": ["PF", "PJ", "PF"],
                "tipo_proponente_anterior": [None, "PF", "PF"],
                "qtd_propostas_historicas": [0, 1, 2],
                "score_modelo": [0.1, 0.8, 0.2],
                "split_group": ["train", "train", "validation"],
            }
        )

        resultado = build_vrum_decision_engine(base)
        linha_pj = resultado.loc[resultado["tipo_proponente"] == "PJ"].iloc[0]

        self.assertTrue(linha_pj["flag_alternancia_pf_pj"])
        self.assertEqual(linha_pj["zona_decisao"], "INVESTIGAR")
        self.assertIn("flag_exposicao_alta", resultado)


if __name__ == "__main__":
    unittest.main()
