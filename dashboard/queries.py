"""Parameterized read queries for the Streamlit fraud dashboard."""

from __future__ import annotations

import sqlite3
from typing import Any


FILTER_COLUMNS = {
    "canal": "p.canal",
    "uf_proposta": "p.uf_proposta",
    "safra": "p.safra",
    "zona_decisao": "p.zona_decisao",
}


def filter_sql(filters: dict[str, str | None], alias: str = "p") -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    for key, value in filters.items():
        if value and key in FILTER_COLUMNS:
            column = FILTER_COLUMNS[key].replace("p.", f"{alias}.")
            clauses.append(f"{column} = ?")
            params.append(value)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def options(connection: sqlite3.Connection, column: str) -> list[str]:
    if column not in FILTER_COLUMNS:
        raise ValueError(f"Filtro não permitido: {column}")
    rows = connection.execute(
        f"SELECT DISTINCT {FILTER_COLUMNS[column]} FROM propostas p "
        f"WHERE {FILTER_COLUMNS[column]} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return [row[0] for row in rows]


def kpis(connection: sqlite3.Connection, filters: dict[str, str | None]) -> dict[str, Any]:
    suffix, params = filter_sql(filters)
    total_propostas = connection.execute(
        f"SELECT COUNT(*) FROM propostas p WHERE 1=1{suffix}", params
    ).fetchone()[0]
    fraud_params = [*params]
    fraude = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(f.valor_financiado), 0)
        FROM (
            SELECT id_proposta, MAX(valor_financiado) AS valor_financiado
            FROM eventos_risco
            WHERE tipo_evento = 'FRAUDE_CONFIRMADA'
            GROUP BY id_proposta
        ) f
        JOIN propostas p ON p.id_proposta = f.id_proposta
        WHERE 1=1
        """ + suffix,
        fraud_params,
    ).fetchone()
    fila_suffix, fila_params = filter_sql(filters)
    fila = connection.execute(
        """
        SELECT COUNT(*)
        FROM propostas p
        WHERE p.zona_decisao IN ('INVESTIGAR', 'BLOQUEAR')
        """ + fila_suffix,
        fila_params,
    ).fetchone()[0]
    casos, exposicao = fraude
    return {
        "propostas": total_propostas,
        "fraudes": casos,
        "exposicao": exposicao,
        "taxa": (casos / total_propostas * 100) if total_propostas else 0,
        "fila": fila,
    }


def monthly(connection: sqlite3.Connection, filters: dict[str, str | None]):
    suffix, params = filter_sql(filters)
    return connection.execute(
        """
        SELECT substr(e.dt_evento, 1, 7) AS mes,
               COUNT(DISTINCT e.id_proposta) AS casos,
               COALESCE(SUM(e.valor_financiado), 0) AS exposicao
        FROM eventos_risco e
        JOIN propostas p ON p.id_proposta = e.id_proposta
        WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
        """ + suffix + " GROUP BY mes ORDER BY mes",
        params,
    ).fetchall()


def ranking(connection: sqlite3.Connection, dimension: str, filters: dict[str, str | None]):
    dimensions = {
        "canal": "COALESCE(e.canal, p.canal)",
        "uf_proposta": "COALESCE(e.uf_proposta, p.uf_proposta)",
        "marca": "e.marca",
    }
    if dimension not in dimensions:
        raise ValueError(f"Dimensão não permitida: {dimension}")
    group_column = dimensions[dimension]
    suffix, params = filter_sql(filters)
    return connection.execute(
        f"""
        SELECT {group_column} AS dimensao,
               COUNT(DISTINCT e.id_proposta) AS casos,
               COALESCE(SUM(e.valor_financiado), 0) AS exposicao
        FROM eventos_risco e
        JOIN propostas p ON p.id_proposta = e.id_proposta
        WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
        {suffix}
        GROUP BY {group_column}
        ORDER BY exposicao DESC
        LIMIT 10
        """,
        params,
    ).fetchall()


def investigation_queue(
    connection: sqlite3.Connection,
    filters: dict[str, str | None],
    limit: int = 100,
):
    suffix, params = filter_sql(filters)
    return connection.execute(
        """
        SELECT p.id_proposta,
               p.data_hora_proposta,
               p.chassi_id_sintetico,
               p.canal,
               p.uf_proposta,
               p.tipo_proponente,
               p.score_modelo,
               p.qtd_sinais_mesa,
               p.valor_financiado,
               p.ltv_fipe,
               p.zona_decisao
        FROM propostas p
        WHERE p.zona_decisao IN ('INVESTIGAR', 'BLOQUEAR')
        """ + suffix + " ORDER BY p.score_modelo DESC, p.qtd_sinais_mesa DESC, p.valor_financiado DESC LIMIT ?",
        [*params, limit],
    ).fetchall()


def proposal_detail(connection: sqlite3.Connection, id_proposta: str):
    proposal = connection.execute(
        "SELECT * FROM propostas WHERE id_proposta = ?", (id_proposta,)
    ).fetchone()
    events = connection.execute(
        """
        SELECT dt_evento, tipo_evento, origem_registro, valor_financiado
        FROM eventos_risco
        WHERE id_proposta = ?
        ORDER BY dt_evento
        """,
        (id_proposta,),
    ).fetchall()
    return proposal, events
