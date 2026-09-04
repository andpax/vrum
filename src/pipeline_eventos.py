"""EDA de sanidade das bases brutas do VRUM (fora do pipeline principal).

Verifica duplicatas de chaves, cardinalidades, órfãos entre propostas/eventos/
cadastro e monta um dataset agregado por chassi para inspeção. Não grava
artefatos; toda saída é em console.

Execução: `python src/pipeline_eventos.py`
"""

from __future__ import annotations

import pandas as pd

from vrum_io import carregar_bases


def checagens_sanidade(
    df_propostas: pd.DataFrame, df_eventos: pd.DataFrame, df_chassis: pd.DataFrame
) -> None:
    """Printa duplicatas, cardinalidades e órfãos entre as 3 fontes."""
    duplicadas = df_propostas[df_propostas["id_proposta"].duplicated(keep=False)]
    print("Propostas duplicadas:", len(duplicadas))

    for nome, df in [
        ("PROPOSTAS", df_propostas),
        ("EVENTOS", df_eventos),
        ("CHASSIS", df_chassis),
    ]:
        print(f"\n=== {nome} ===")
        print("Linhas:", len(df))
        print("Colunas:", list(df.columns))

    print("\n=== CARDINALIDADE ===")
    print("ID propostas únicos:", df_propostas["id_proposta"].nunique())
    print("ID eventos únicos:", df_eventos["id_proposta"].nunique())
    print("Chassis únicos nas propostas:", df_propostas["chassi_id_sintetico"].nunique())
    print("Chassis únicos no cadastro:", df_chassis["chassi_id_sintetico"].nunique())

    print("\n=== EVENTOS POR PROPOSTA ===")
    print(
        df_eventos.groupby("id_proposta").size().value_counts().sort_index()
    )

    propostas_ids = set(df_propostas["id_proposta"])
    eventos_ids = set(df_eventos["id_proposta"])
    eventos_sem_proposta = len(eventos_ids - propostas_ids)
    propostas_sem_evento = len(propostas_ids - eventos_ids)
    print(f"\nEventos sem proposta cadastrada: {eventos_sem_proposta} ({eventos_sem_proposta/len(eventos_ids):.1%})")
    print(f"Propostas sem nenhum evento: {propostas_sem_evento} ({propostas_sem_evento/len(propostas_ids):.1%})")

    chassis_orfaos = len(
        set(df_propostas["chassi_id_sintetico"]) - set(df_chassis["chassi_id_sintetico"])
    )
    print(f"Chassis em propostas sem cadastro em CHASSIS: {chassis_orfaos}")


def montar_dataset_chassi(
    df_propostas: pd.DataFrame, df_eventos: pd.DataFrame, df_chassis: pd.DataFrame
) -> pd.DataFrame:
    """Agrega propostas e eventos por chassi sobre o cadastro oficial."""
    df_prop_agg = df_propostas.groupby("chassi_id_sintetico").agg(
        qtd_propostas=("id_proposta", "count"),
        qtd_ifs_distintas=("if_id_sintetico", "nunique"),
        qtd_proponentes_distintos=("cpf_cnpj_proponente_sintetico", "nunique"),
        valor_financiado_max=("valor_financiado", "max"),
        prazo_meses_medio=("prazo_meses", "mean"),
    ).reset_index()

    # Traz o chassi para a tabela de eventos via propostas
    df_eventos_chassi = df_eventos.merge(
        df_propostas[["id_proposta", "chassi_id_sintetico"]],
        on="id_proposta",
        how="inner",
    )
    df_eventos_agg = df_eventos_chassi.groupby("chassi_id_sintetico").agg(
        target_fraude_90d=("evento_risco_chassi_90d", "max"),  # 1 se qualquer evento de risco
        qtd_eventos_total=("tipo_evento", "count"),
        dias_ate_evento_min=("dias_ate_evento", "min"),
    ).reset_index()

    df_modelagem = df_chassis.merge(df_prop_agg, on="chassi_id_sintetico", how="left")
    df_modelagem = df_modelagem.merge(df_eventos_agg, on="chassi_id_sintetico", how="left")

    # Chassis sem propostas/eventos: contadores zerados
    df_modelagem["qtd_propostas"] = df_modelagem["qtd_propostas"].fillna(0)
    df_modelagem["target_fraude_90d"] = df_modelagem["target_fraude_90d"].fillna(0)

    df_modelagem["ltv_max"] = (
        df_modelagem["valor_financiado_max"] / df_modelagem["valor_fipe_referencia"]
    )
    return df_modelagem


def main() -> None:
    df_chassis, df_propostas, df_eventos = carregar_bases()
    checagens_sanidade(df_propostas, df_eventos, df_chassis)
    df_modelagem = montar_dataset_chassi(df_propostas, df_eventos, df_chassis)
    print(f"\nDataset agregado por chassi: {df_modelagem.shape}")


if __name__ == "__main__":
    main()
