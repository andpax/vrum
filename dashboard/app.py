"""Streamlit MVP for the VRUM fraud investigation dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import queries


ROOT = Path(__file__).resolve().parents[1]
DATABASE = Path(__file__).resolve().parent / "data" / "vrum_dashboard.sqlite"


st.set_page_config(
    page_title="VRUM · riscos de fraude",
    page_icon="⚠",
    layout="wide",
)


def brl(value: float) -> str:
    formatted = f"R$ {value:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def percent(value: float) -> str:
    return f"{value:.3f}%".replace(".", ",")


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


@st.cache_data(ttl=300)
def get_options(column: str) -> list[str]:
    return queries.options(get_connection(), column)


@st.cache_data(ttl=300)
def get_kpis(filters_tuple: tuple[tuple[str, str | None], ...]):
    return queries.kpis(get_connection(), dict(filters_tuple))


@st.cache_data(ttl=300)
def get_monthly(filters_tuple: tuple[tuple[str, str | None], ...]):
    return [tuple(row) for row in queries.monthly(get_connection(), dict(filters_tuple))]


@st.cache_data(ttl=300)
def get_ranking(dimension: str, filters_tuple: tuple[tuple[str, str | None], ...]):
    return [tuple(row) for row in queries.ranking(get_connection(), dimension, dict(filters_tuple))]


@st.cache_data(ttl=300)
def get_queue(filters_tuple: tuple[tuple[str, str | None], ...]):
    return [dict(row) for row in queries.investigation_queue(get_connection(), dict(filters_tuple))]


if not DATABASE.exists():
    st.error("Banco SQLite não encontrado.")
    st.code("python dashboard/build_database.py")
    st.stop()


st.title("Riscos de fraude")
st.caption(
    "Investigação histórica de fraude confirmada. A taxa usa as propostas como denominador; "
    "a fila mostra propostas nas zonas INVESTIGAR e BLOQUEAR."
)

with st.sidebar:
    st.header("Filtros")
    selected_filters: dict[str, str | None] = {}
    for column, label in (
        ("canal", "Canal"),
        ("uf_proposta", "UF da proposta"),
        ("safra", "Safra"),
        ("zona_decisao", "Zona de decisão"),
    ):
        values = ["Todos", *get_options(column)]
        selected = st.selectbox(label, values)
        selected_filters[column] = None if selected == "Todos" else selected

    filters_tuple = tuple(sorted(selected_filters.items()))
    st.divider()
    st.caption("Fonte: vrum_propostas_flags.csv + vrum_timeline_cronologica.csv")


kpi = get_kpis(filters_tuple)
cards = st.columns(5)
cards[0].metric("Propostas", integer(kpi["propostas"]))
cards[1].metric("Fraudes confirmadas", integer(kpi["fraudes"]))
cards[2].metric("Incidência", percent(kpi["taxa"]))
cards[3].metric("Exposição", brl(kpi["exposicao"]))
cards[4].metric("Fila manual", integer(kpi["fila"]))

st.subheader("Evolução da fraude confirmada")
monthly_rows = get_monthly(filters_tuple)
monthly_df = pd.DataFrame(
    [tuple(row) for row in monthly_rows],
    columns=["Mês", "Casos", "Exposição"],
)
if monthly_df.empty:
    st.info("Nenhum caso de fraude para os filtros selecionados.")
else:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.caption("Casos por mês")
        st.line_chart(monthly_df.set_index("Mês")["Casos"])
    with chart_right:
        st.caption("Exposição financeira por mês")
        st.bar_chart(monthly_df.set_index("Mês")["Exposição"])

st.subheader("Onde estão os casos?")
ranking_columns = st.columns(3)
for container, dimension, title in zip(
    ranking_columns,
    ("canal", "uf_proposta", "marca"),
    ("Por canal", "Por UF", "Por marca"),
):
    rows = get_ranking(dimension, filters_tuple)
    frame = pd.DataFrame([tuple(row) for row in rows], columns=["Dimensão", "Casos", "Exposição"])
    if not frame.empty:
        frame["Exposição"] = frame["Exposição"].map(brl)
    with container:
        st.caption(title)
        st.dataframe(frame, hide_index=True, use_container_width=True)

st.subheader("Fila para investigação manual")
queue_rows = get_queue(filters_tuple)
queue_df = pd.DataFrame([dict(row) for row in queue_rows])
if queue_df.empty:
    st.info("Nenhuma proposta nas zonas INVESTIGAR ou BLOQUEAR para os filtros selecionados.")
else:
    queue_df = queue_df.rename(
        columns={
            "id_proposta": "Proposta",
            "data_hora_proposta": "Data",
            "chassi_id_sintetico": "Chassi",
            "canal": "Canal",
            "uf_proposta": "UF",
            "tipo_proponente": "Perfil",
            "score_modelo": "Score",
            "qtd_sinais_mesa": "Sinais",
            "valor_financiado": "Valor financiado",
            "ltv_fipe": "LTV",
            "zona_decisao": "Zona",
        }
    )
    queue_df["Valor financiado"] = queue_df["Valor financiado"].map(brl)
    queue_df["Score"] = queue_df["Score"].round(3)
    queue_df["LTV"] = queue_df["LTV"].round(2)
    st.dataframe(queue_df, hide_index=True, use_container_width=True)

    selected_proposal = st.selectbox(
        "Abrir histórico da proposta",
        ["Selecione uma proposta", *queue_df["Proposta"].tolist()],
    )
    if selected_proposal != "Selecione uma proposta":
        proposal, events = queries.proposal_detail(get_connection(), selected_proposal)
        st.caption(f"Detalhes de {selected_proposal}")
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.json(dict(proposal))
        with detail_right:
            st.dataframe(pd.DataFrame([dict(event) for event in events]), hide_index=True, use_container_width=True)
