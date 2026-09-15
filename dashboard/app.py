"""Mesa VRUM — dashboard operacional de investigação de propostas.

Uso: `streamlit run dashboard/app.py` (requer output/vrum_propostas_flags.csv,
output/limiares_flags_mesa.csv e output/limiares_politica_vrum.csv gerados
pelo pipeline — ver docs/manual_uso_vrum.md).

Princípios de design (docs/design_mesa_vrum.md):
  - flags são EVIDÊNCIA para priorização, nunca confirmação de fraude;
  - score é critério de encaminhamento (top 5%), não probabilidade de fraude;
  - BLOQUEAR desabilitado (AUC OOT ~ 0,5);
  - linguagem neutra: "sinal de atenção", "indício", "exposição elevada".
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAGS_PATH = REPO_ROOT / "output" / "vrum_propostas_flags.csv"
CADASTRO_PATH = REPO_ROOT / "docs" / "cadastro_chassi_mock.csv"
LIMIARES_FLAGS_PATH = REPO_ROOT / "output" / "limiares_flags_mesa.csv"
LIMIARES_POLITICA_PATH = REPO_ROOT / "output" / "limiares_politica_vrum.csv"

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
FLAGS_FINANCEIRAS = [
    "flag_exposicao_alta",
    "flag_ltv_alto",
    "flag_exposicao_e_ltv_altos",
]

# flag -> (rótulo, coluna do valor observado, limiar, janela, por que investigar)
EVIDENCIAS = {
    "flag_indice_rotatividade_alto": (
        "Rotatividade alta", "indice_rotatividade_vrum", "P95 treino: {p95_indice_rotatividade}",
        "45 dias", "Transferências recentes em relação ao tempo de posse histórico do chassi.",
    ),
    "flag_frequencia_alta": (
        "Frequência alta", "transferencias_ultimos_45d", "P95 treino: {p95_transferencias_45d} propostas",
        "45 dias", "Várias propostas para o mesmo chassi em poucos dias.",
    ),
    "flag_intervalo_curto": (
        "Intervalo curto", "dias_desde_ultima_proposta", "<= 5 dias",
        "consecutivo", "Proposta atual muito próxima da anterior do mesmo chassi.",
    ),
    "flag_posse_historica_curta": (
        "Posse histórica curta", "tempo_posse_mediano_acumulado", "entre 0 e 15 dias",
        "acumulado", "Tempo mediano de posse entre propostas anteriores muito curto.",
    ),
    "flag_alternancia_pf_pj": (
        "Alternância PF/PJ", None, "troca vs proposta anterior",
        "consecutivo", "Mudança de tipo de proponente entre propostas consecutivas do chassi.",
    ),
    "flag_muitos_proponentes": (
        "Muitos proponentes", None, "P95 treino: {p95_proponentes_45d} (45d) / {p95_proponentes_90d} (90d)",
        "45/90 dias", "Proponentes distintos no mesmo chassi em janela curta.",
    ),
    "flag_muitas_ifs": (
        "Muitas IFs", None, "P95 treino: {p95_ifs_45d} (45d) / {p95_ifs_90d} (90d)",
        "45/90 dias", "Mesmo chassi circulando por várias instituições financeiras em janela curta.",
    ),
    "flag_mudanca_canal": (
        "Mudança de canal", None, "diferente da proposta anterior",
        "consecutivo", "Canal de origem mudou entre propostas consecutivas.",
    ),
    "flag_mudanca_uf": (
        "Mudança de UF", None, "diferente da proposta anterior",
        "consecutivo", "UF da proposta mudou entre propostas consecutivas.",
    ),
    "flag_historico_insuficiente": (
        "Histórico insuficiente", "qtd_propostas_historicas", "< 2 propostas anteriores",
        "acumulado", "Pouco histórico para confiar nos indicadores do chassi — cautela, não bloqueio.",
    ),
    "flag_exposicao_alta": (
        "Exposição alta", "valor_financiado", "P90 treino: {p90_exposicao}",
        "—", "Valor financiado entre os maiores da carteira — prioriza impacto financeiro.",
    ),
    "flag_ltv_alto": (
        "LTV alto", "ltv_fipe", "P90 treino: {p90_ltv}",
        "—", "Financiamento elevado em relação à referência FIPE do veículo.",
    ),
    "flag_exposicao_e_ltv_altos": (
        "Exposição + LTV altos", None, "exposição alta E LTV alto",
        "—", "Combinação financeira de maior prioridade para a mesa.",
    ),
}
TODAS_FLAGS = FLAGS_COMPORTAMENTAIS + ["flag_historico_insuficiente"] + FLAGS_FINANCEIRAS

st.set_page_config(page_title="Mesa VRUM", layout="wide", page_icon="🔍")

st.markdown(
    """
    <style>
    .stApp { background: #171717; color: #f4f4f4; }
    [data-testid="stHeader"] { background: #171717; }
    [data-testid="stSidebar"] { background: #202020; }
    [data-testid="stMetric"] {
        background: #303030; border-left: 3px solid #f5df24;
        border-radius: 4px; padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #d4d4d4; }
    [data-testid="stMetricValue"] { color: #ffffff; }
    .dossie-title { font-size: 2.45rem; font-weight: 700; letter-spacing: -1px; }
    .section-title {
        border-bottom: 2px solid #f5df24; padding-bottom: 5px;
        font-size: 1.05rem; font-weight: 700;
    }
    .flag-detected {
        background: #3c3c2b; border-left: 3px solid #f5df24;
        padding: 7px 10px; margin: 5px 0; border-radius: 2px;
    }
    .reason-box {
        background: #3c3c2b; border: 1px solid #f5df24;
        border-radius: 4px; padding: 11px 14px; margin: 8px 0 14px;
    }
    .muted { color: #bcbcbc; font-size: .82rem; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { color: #d8d8d8; }
    button[aria-selected="true"] { color: #f5df24 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def carregar() -> tuple[pl.DataFrame, pl.DataFrame, dict, float, float]:
    """Base da mesa, cadastro do veículo e limiares da política."""
    base = pl.read_csv(FLAGS_PATH, separator=";")
    cadastro = pl.read_csv(CADASTRO_PATH, separator=";")
    limiares = {
        r["limiar"]: r["valor"]
        for r in pl.read_csv(LIMIARES_FLAGS_PATH, separator=";").to_dicts()
    }
    limiar_inv = float(
        pl.read_csv(LIMIARES_POLITICA_PATH, separator=";")
        .filter(pl.col("regra") == "investigar")["limiar_score"][0]
    )
    exposicao_p90 = float(limiares["p90_exposicao"])
    return base, cadastro, limiares, limiar_inv, exposicao_p90


def motivo_da_proposta(score: float, hist_insuf: bool, expo_alta: bool, limiar: float) -> str:
    """Motivo do encaminhamento — espelha a regra de zona de src/flags_mesa.py."""
    por_score = score >= limiar
    por_hist = hist_insuf and expo_alta
    if por_score and por_hist:
        return "Ambos"
    if por_score:
        return "Score ≥ limiar"
    return "Hist. insuf. + exposição alta"


def montar_fila(base: pl.DataFrame) -> pd.DataFrame:
    """Fila INVESTIGAR na ordenação oficial: sinais, expo+LTV, valor, score."""
    return (
        base.filter(pl.col("zona_decisao") == "INVESTIGAR")
        .sort(
            ["qtd_sinais_mesa", "flag_exposicao_e_ltv_altos", "valor_financiado", "score_modelo"],
            descending=[True, True, True, True],
        )
        .to_pandas()
    )


def aplicar_filtros(fila: pd.DataFrame, f: dict) -> pd.DataFrame:
    m = pd.Series(True, index=fila.index)
    if f["safra"] != "Todas":
        m &= fila["safra"] == f["safra"]
    for col, val in [("if_id_sintetico", "if"), ("uf_proposta", "uf"), ("canal", "canal"), ("tipo_proponente", "tipo")]:
        if f[val] != "Todas":
            m &= fila[col] == f[val]
    m &= fila["valor_financiado"].between(f["valor"][0], f["valor"][1])
    if f["so_ltv_alto"]:
        m &= fila["flag_ltv_alto"]
    if f["so_hist_insuf"]:
        m &= fila["flag_historico_insuficiente"]
    for flag in f["flags"]:
        m &= fila[flag]
    if f["busca"]:
        termo = f["busca"].lower()
        m &= (
            fila["id_proposta"].str.lower().str.contains(termo, na=False)
            | fila["chassi_id_sintetico"].str.lower().str.contains(termo, na=False)
            | fila["cpf_cnpj_proponente_sintetico"].str.lower().str.contains(termo, na=False)
        )
    return fila[m]


def enriquecer_fila(fila: pd.DataFrame, limiar_inv: float) -> pd.DataFrame:
    """Adiciona prioridade, motivo e posição após filtros da mesa."""
    fila = fila.reset_index(drop=True).copy()
    fila["#"] = fila.index + 1
    fila["tier"] = np.where(
        fila["flag_exposicao_e_ltv_altos"], "P1",
        np.where(fila["qtd_sinais_mesa"] >= 4, "P2", "P3"),
    )
    fila["motivo"] = [
        motivo_da_proposta(score, historico, exposicao, limiar_inv)
        for score, historico, exposicao in zip(
            fila["score_modelo"],
            fila["flag_historico_insuficiente"],
            fila["flag_exposicao_alta"],
        )
    ]
    return fila


def brl(valor: float, casas: int = 0) -> str:
    """Formata valor monetário no padrão visual da mesa."""
    return f"R$ {valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def observacao_flag(linha: pd.Series, flag: str, limiares: dict) -> tuple[str, str, str]:
    """Retorna valor observado, limiar e janela para leitura rápida do sinal."""
    _, coluna, limiar, janela, _ = EVIDENCIAS[flag]
    if flag == "flag_muitos_proponentes":
        observado = (
            f"45d: {int(linha['proponentes_distintos_45d'])} · "
            f"90d: {int(linha['proponentes_distintos_90d'])}"
        )
    elif flag == "flag_muitas_ifs":
        observado = (
            f"45d: {int(linha['ifs_distintas_45d'])} · "
            f"90d: {int(linha['ifs_distintas_90d'])}"
        )
    elif flag == "flag_exposicao_e_ltv_altos":
        observado = f"valor: {brl(float(linha['valor_financiado']))} · LTV: {float(linha['ltv_fipe']):.2f}"
    elif coluna and pd.notna(linha[coluna]):
        valor = linha[coluna]
        observado = brl(float(valor)) if coluna == "valor_financiado" else f"{float(valor):.2f}"
    else:
        observado = "mudança detectada entre propostas consecutivas"
    return observado, limiar.format(**limiares), janela


def tela_dossie(
    linha: pd.Series,
    base: pl.DataFrame,
    cadastro: pl.DataFrame,
    limiares: dict,
    limiar_inv: float,
) -> None:
    """Visão proposta a proposta, inspirada no dossiê do chassi."""
    st.markdown(
        f"<div class='dossie-title'>Proposta de financiamento</div>"
        f"<div class='muted'>{linha['id_proposta']} · {linha['chassi_id_sintetico']} · "
        f"{linha['data_hora_proposta']} · {linha['safra']}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    motivo = motivo_da_proposta(
        linha["score_modelo"], linha["flag_historico_insuficiente"],
        linha["flag_exposicao_alta"], limiar_inv,
    )
    st.markdown(
        f"<div class='reason-box'><b>Motivo do encaminhamento:</b> {motivo}"
        f"<br><span class='muted'>A proposta foi priorizada para investigação; isso não representa confirmação de fraude.</span></div>",
        unsafe_allow_html=True,
    )

    veiculo = cadastro.filter(
        pl.col("chassi_id_sintetico") == linha["chassi_id_sintetico"]
    ).to_dicts()
    dados_veiculo = veiculo[0] if veiculo else {}
    st.markdown("<div class='section-title'>Dados da proposta e do veículo</div>", unsafe_allow_html=True)
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric(
        "Veículo",
        f"{dados_veiculo.get('marca', 'Indisponível')} · "
        f"{dados_veiculo.get('ano_modelo', 'ano indisponível')}",
    )
    v2.metric("Placa", dados_veiculo.get("placa_id_sintetica", "Indisponível"))
    v3.metric("UF registro", dados_veiculo.get("uf_registro", "Indisponível"))
    v4.metric("UF proposta", linha["uf_proposta"])
    v5.metric("Canal", linha["canal"])

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Proponente", linha["tipo_proponente"])
    p2.metric("Instituição financeira", linha["if_id_sintetico"])
    p3.metric("Entrada", brl(float(linha["valor_entrada"])))
    p4.metric("Prazo", f"{float(linha['prazo_meses']):.0f} meses")

    historico = base.filter(
        pl.col("chassi_id_sintetico") == linha["chassi_id_sintetico"]
    )
    total_flags = sum(bool(linha[flag]) for flag in TODAS_FLAGS)
    detectadas = [flag for flag in TODAS_FLAGS if bool(linha[flag])]

    esquerda, centro, direita = st.columns([1.05, 1.45, 2.25], gap="medium")
    with esquerda:
        st.metric("Valor financiado", brl(float(linha["valor_financiado"])))
        st.metric("FIPE de referência", brl(float(linha["valor_fipe_referencia"])))
        st.metric("LTV", f"{float(linha['ltv_fipe']):.2f}")
        st.caption(f"P90 LTV: {float(limiares['p90_ltv']):.2f}")

    with centro:
        st.markdown("<div class='section-title'>Score e sinais</div>", unsafe_allow_html=True)
        st.metric("Score do modelo", f"{float(linha['score_modelo']):.3f}")
        st.progress(
            min(int(float(linha["score_modelo"]) * 100), 100),
            text=f"Limiar: {limiar_inv:.3f} · critério de encaminhamento",
        )
        st.metric("Total de flags detectadas", f"{total_flags}/13")
        st.caption("Flags são evidências de atenção, não confirmação de fraude.")

    with direita:
        st.markdown("<div class='section-title'>Flags detectadas</div>", unsafe_allow_html=True)
        if detectadas:
            for flag in detectadas:
                rotulo = EVIDENCIAS[flag][0]
                _, _, _, _, porque = EVIDENCIAS[flag]
                observado, limiar, janela = observacao_flag(linha, flag, limiares)
                destaque = " · prioridade financeira" if flag == "flag_exposicao_e_ltv_altos" else ""
                st.markdown(
                    f"<div class='flag-detected'><b>{rotulo}{destaque}</b>"
                    f"<br><span class='muted'>Observado: {observado} · Limiar: {limiar} · Janela: {janela}</span>"
                    f"<br><span class='muted'>{porque}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Nenhuma flag acionada. Proposta encaminhada pelo score ou pela regra de histórico/exposição.")

        with st.expander("Como interpretar as flags"):
            st.caption(
                "Flag indica sinal de atenção. Não confirma fraude nem substitui a investigação. "
                "Os limiares vêm somente do treino."
            )
            for flag in TODAS_FLAGS:
                rotulo, _, _, janela, porque = EVIDENCIAS[flag]
                status = "acionada" if bool(linha[flag]) else "não acionada"
                st.markdown(f"**{rotulo}** — {status}. {porque} Janela: {janela}.")

    st.markdown("<div class='section-title'>Status dos últimos 45 dias</div>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Transferências", int(linha["transferencias_ultimos_45d"]))
    s2.metric("Proponentes distintos", int(linha["proponentes_distintos_45d"]))
    s3.metric("IFs distintas", int(linha["ifs_distintas_45d"]))
    s4.metric("Propostas históricas", int(linha["qtd_propostas_historicas"]))
    s5.metric("Zona", linha["zona_decisao"])

    st.divider()
    historico_pd = (
        historico.sort("data_hora_proposta")
        .select([
            "data_hora_proposta", "id_proposta", "tipo_proponente", "canal",
            "uf_proposta", "if_id_sintetico", "valor_financiado", "valor_fipe_referencia",
            "prazo_meses", "score_modelo", "qtd_sinais_mesa", "zona_decisao",
        ])
        .to_pandas()
    )
    historico_pd["data_hora_proposta"] = pd.to_datetime(historico_pd["data_hora_proposta"])
    historico_pd["selecionada"] = np.where(
        historico_pd["id_proposta"] == linha["id_proposta"], "← proposta atual", ""
    )

    grafico_col, tabela_col = st.columns([1.55, 1.45], gap="medium")
    with grafico_col:
        st.markdown("<div class='section-title'>Financiado x FIPE no histórico do chassi</div>", unsafe_allow_html=True)
        chart_data = historico_pd.melt(
            id_vars=["data_hora_proposta"],
            value_vars=["valor_financiado", "valor_fipe_referencia"],
            var_name="indicador", value_name="valor",
        )
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("data_hora_proposta:T", title="data"),
            y=alt.Y("valor:Q", title="R$"),
            color=alt.Color("indicador:N", title=""),
            tooltip=["data_hora_proposta:T", "indicador:N", "valor:Q"],
        ).properties(height=280)
        st.altair_chart(chart, width="stretch")
        st.caption("Histórico observado de propostas; não representa transferência de propriedade confirmada.")

    with tabela_col:
        st.markdown("<div class='section-title'>Propostas do mesmo chassi</div>", unsafe_allow_html=True)
        exibicao = historico_pd[[
            "data_hora_proposta", "id_proposta", "uf_proposta", "tipo_proponente",
            "qtd_sinais_mesa", "selecionada",
        ]].copy()
        exibicao["data_hora_proposta"] = exibicao["data_hora_proposta"].dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(exibicao, width="stretch", hide_index=True, height=280)

    with st.expander("Todas as evidências: valores observados, limiares e contexto"):
        regs = []
        for flag in TODAS_FLAGS:
            rotulo, col_valor, limiar_txt, janela, porque = EVIDENCIAS[flag]
            observado = (
                f"{linha[col_valor]:,.2f}".replace(",", ".")
                if col_valor and pd.notna(linha[col_valor]) else "—"
            )
            regs.append({
                "evidência": rotulo,
                "acionada": "sim" if bool(linha[flag]) else "não",
                "observado": observado,
                "limiar": limiar_txt.format(**limiares) if limiar_txt else "—",
                "janela": janela,
                "por que investigar": porque,
            })
        st.dataframe(pd.DataFrame(regs), width="stretch", hide_index=True)

    with st.expander("Checklist de investigação"):
        chave = str(linha["id_proposta"])
        st.checkbox("Validar documentos e dados cadastrais", key=f"docs_{chave}")
        st.checkbox("Confirmar coerência cronológica do chassi", key=f"timeline_{chave}")
        st.checkbox("Verificar mudanças de UF, canal e IF", key=f"movimentos_{chave}")
        st.checkbox("Conferir valor financiado, FIPE, entrada e prazo", key=f"financeiro_{chave}")
        st.caption("Checklist local da sessão. Ainda não há persistência da decisão ou das anotações do analista.")


def main() -> None:
    st.title("Proposta de financiamento")
    st.caption(
        "Mesa VRUM · investigação proposta a proposta. Os sinais abaixo são indicadores de atenção para priorização da análise. "
        "**Não representam confirmação de fraude.** O modelo tem discriminação OOT limitada; "
        "a decisão final depende da investigação do analista."
    )

    try:
        base, cadastro, limiares, limiar_inv, exposicao_p90 = carregar()
    except FileNotFoundError as e:
        st.error(f"Arquivo da mesa não encontrado: {e}. Rode o pipeline (ver docs/manual_uso_vrum.md).")
        st.stop()

    fila_total = montar_fila(base)

    with st.sidebar:
        st.header("Filtros")
        f = {
            "safra": st.selectbox("Safra", ["Todas", "treino", "validacao", "oot"]),
            "if": st.selectbox("IF", ["Todas", *sorted(fila_total["if_id_sintetico"].unique())]),
            "uf": st.selectbox("UF", ["Todas", *sorted(fila_total["uf_proposta"].unique())]),
            "canal": st.selectbox("Canal", ["Todas", *sorted(fila_total["canal"].unique())]),
            "tipo": st.selectbox("Proponente", ["Todas", "PF", "PJ"]),
            "valor": st.slider(
                "Valor financiado (R$)", 0.0, float(fila_total["valor_financiado"].max()),
                (0.0, float(fila_total["valor_financiado"].max())), step=1000.0, format="R$ %.0f",
            ),
            "so_ltv_alto": st.toggle("Somente LTV ≥ P90"),
            "so_hist_insuf": st.toggle("Somente histórico insuficiente"),
            "flags": st.multiselect("Somente com evidência", TODAS_FLAGS),
            "busca": st.text_input("Buscar (proposta / chassi / proponente)"),
        }

    fila = enriquecer_fila(aplicar_filtros(fila_total, f), limiar_inv)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Propostas na mesa", f"{len(fila):,}".replace(",", "."))
    k2.metric("% da carteira", f"{100 * len(fila) / base.height:.1f}%")
    k3.metric("Exposição da fila", f"R$ {fila['valor_financiado'].sum() / 1e9:,.2f} bi".replace(",", "."))
    k4.metric("Exposição + LTV altos", int(fila["flag_exposicao_e_ltv_altos"].sum()))
    k5.metric("Histórico insuficiente", int(fila["flag_historico_insuficiente"].sum()))
    st.caption(
        "Fila em INVESTIGAR = score ≥ limiar da validação OU histórico insuficiente + exposição alta. "
        "BLOQUEAR desabilitado por política (AUC OOT ≈ 0,5)."
    )

    if fila.empty:
        st.info("Nenhuma proposta na mesa com os filtros atuais.")
        st.stop()

    st.markdown("### Abrir proposta")
    busca_dossie = st.text_input(
        "Busca direta por proposta ou chassi",
        placeholder="Digite ID da proposta ou chassi",
        key="busca_dossie",
    ).strip().lower()
    if busca_dossie:
        correspondencias = fila[
            fila["id_proposta"].str.lower().str.contains(busca_dossie, na=False)
            | fila["chassi_id_sintetico"].str.lower().str.contains(busca_dossie, na=False)
        ]
        selecao_opcoes = correspondencias.head(500)["id_proposta"].tolist()
        st.caption(f"{len(correspondencias):,} proposta(s) encontrada(s); exibindo até 500.")
    else:
        selecao_opcoes = fila.head(500)["id_proposta"].tolist()
        st.caption("Sem busca: exibindo 500 primeiras propostas na ordenação oficial.")

    if not selecao_opcoes:
        st.warning("Nenhuma proposta encontrada para abrir.")
        st.stop()

    if st.session_state.get("dossie_select") not in selecao_opcoes:
        st.session_state["dossie_select"] = selecao_opcoes[0]
    indice_atual = selecao_opcoes.index(st.session_state["dossie_select"])
    anterior, contador, proxima = st.columns([1, 2, 1])
    with anterior:
        if st.button("← Anterior", width="stretch", disabled=len(selecao_opcoes) < 2):
            st.session_state["dossie_select"] = selecao_opcoes[(indice_atual - 1) % len(selecao_opcoes)]
            st.rerun()
    with contador:
        st.caption(f"Proposta {indice_atual + 1} de {len(selecao_opcoes)} selecionável(is)")
    with proxima:
        if st.button("Próxima →", width="stretch", disabled=len(selecao_opcoes) < 2):
            st.session_state["dossie_select"] = selecao_opcoes[(indice_atual + 1) % len(selecao_opcoes)]
            st.rerun()

    id_selecionado = st.selectbox(
        "Proposta selecionada",
        selecao_opcoes,
        format_func=lambda proposta: f"{proposta} · chassi {fila.loc[fila['id_proposta'] == proposta, 'chassi_id_sintetico'].iloc[0]}",
        key="dossie_select",
    )
    selecionada = fila.loc[fila["id_proposta"] == id_selecionado].iloc[0]

    tab_dossie, tab_fila, tab_graficos = st.tabs(
        ["Dossiê Chassi", "Triagem", "Panorama"]
    )

    with tab_dossie:
        tela_dossie(selecionada, base, cadastro, limiares, limiar_inv)

    with tab_fila:
        display = fila[[
            "#", "tier", "id_proposta", "chassi_id_sintetico", "data_hora_proposta",
            "motivo", "score_modelo", "valor_financiado", "ltv_fipe",
            "qtd_sinais_mesa", "flag_exposicao_e_ltv_altos",
        ]].rename(columns={
            "tier": "prioridade", "qtd_sinais_mesa": "sinais (n/9)",
            "flag_exposicao_e_ltv_altos": "expo+LTV",
        })
        display["sinais (n/9)"] = display["sinais (n/9)"].map(lambda v: f"{int(v)}/9")
        display["valor_financiado"] = display["valor_financiado"].map(lambda v: brl(v))
        display["expo+LTV"] = display["expo+LTV"].map(lambda v: "sim" if v else "")
        display["score_modelo"] = display["score_modelo"].map(lambda v: f"{v:.3f}")
        display["ltv_fipe"] = display["ltv_fipe"].map(lambda v: f"{v:.2f}")

        st.caption(
            "Ordenação oficial: sinais ↓ · exposição+LTV · valor ↓ · score ↓. "
            "Use o seletor no topo para abrir o dossiê proposta a proposta."
        )
        st.dataframe(
            display, on_select="rerun", selection_mode="single-row",
            width="stretch", hide_index=True, height=520,
        )

    with tab_graficos:
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Sinais mais acionados na fila**")
            acionadas = {
                EVIDENCIAS[f][0]: int(fila[f].sum()) for f in FLAGS_COMPORTAMENTAIS
            }
            st.bar_chart(pd.Series(acionadas).sort_values())
        with cB:
            st.markdown("**Score da fila** (limiar de encaminhamento marcado)")
            hist = np.histogram(fila["score_modelo"], bins=40)
            df_hist = pd.DataFrame({"score": hist[1][:-1], "propostas": hist[0]})
            chart = (
                alt.Chart(df_hist).mark_bar().encode(x="score:Q", y="propostas:Q")
                + alt.Chart(pd.DataFrame({"limiar": [limiar_inv]}))
                .mark_rule(color="darkorange", strokeWidth=2)
                .encode(x="limiar:Q")
            )
            st.altair_chart(chart, width='stretch')
        st.markdown("**Pareto da exposição** — fração da exposição da fila acumulada pela ordem de prioridade")
        valores = fila["valor_financiado"].to_numpy()
        if len(valores):
            cum = np.cumsum(valores) / valores.sum()
            passo = max(1, len(cum) // 400)
            df_pareto = pd.DataFrame({
                "propostas analisadas": np.arange(1, len(cum) + 1)[::passo],
                "exposição acumulada (%)": cum[::passo] * 100,
            })
            st.line_chart(df_pareto, x="propostas analisadas", y="exposição acumulada (%)")

    st.divider()
    with st.expander("ⓘ Nota metodológica e limitações"):
        st.markdown(
            "- Sinais/evidências priorizam a análise; **não comprovam fraude** e não geram bloqueio automático.\n"
            "- Limiares (P90/P95) calculados **somente no treino**; encaminhamento por score usa o quantil 95 da validação.\n"
            "- Modelo com **AUC OOT ≈ 0,5** nesta base sintética: sem discriminação comprovada fora do tempo.\n"
            "- A regra original `qtd_sinais >= 2` encaminharia ~75% da carteira sem discriminação — por isso sinais ordenam a fila e não definem a zona.\n"
            "- Campos de resultado futuro (`target_risco_90d` etc.) existem no dataset apenas para avaliação retrospectiva e não são evidência de decisão."
        )
        st.caption(f"Dataset: output/vrum_propostas_flags.csv · {base.height:,} propostas".replace(",", "."))


if __name__ == "__main__":
    main()
