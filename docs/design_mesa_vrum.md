# Design da Mesa de Análise VRUM — Proposta de Visualização Operacional

> Documento de design do dashboard da mesa de investigação (implementado em
> `dashboard/app.py`, Streamlit). Base de dados:
> `output/vrum_propostas_flags.csv` (gerada por `src/flags_mesa.py`).

A tela abre em **Proposta de financiamento**, com seleção proposta a proposta. Abas
secundárias preservam **Triagem** (fila completa) e **Panorama** (indicadores
agregados), sem alterar regras de encaminhamento ou priorização.

O dossiê oferece busca direta por proposta/chassi, navegação anterior/próxima e
checklist local de investigação. O checklist não persiste decisão enquanto a
base não tiver mecanismo de registro da análise.

---

## 1. Objetivo da tela

Permitir que um analista **priorize e investigue** as propostas encaminhadas
(score top 5% OU histórico insuficiente + exposição alta ≈ 8,9% da carteira),
entendendo em segundos **por que** cada proposta está ali e **quais
evidências** a cercam — sem sugerir confirmação de fraude. A tela organiza o
trabalho; a decisão é do analista.

## 2. Perfil do usuário

Analista de risco/fraude júnior a pleno. Volume diário alto, tempo por
proposta de 2–5 min. Não conhece o modelo internamente — precisa de tradução
das evidências, não de estatística. Usa a mesa no início do turno (visão
geral), durante o dia (fila + detalhe) e ao final (acompanhamento).

## 3. Fluxo de uso

```
Fila → Priorização → Motivo → Investigação → Decisão
```

| Etapa | O que o analista precisa | O que é enfeite |
|---|---|---|
| Fila | Quantos casos, exposição total, onde começar | Gráficos decorativos, score médio |
| Priorização | Ordem oficial já pronta, tiers visíveis | Reordenar manualmente (permitir, mas não default) |
| Motivo (30 s) | Por que está na mesa: score OU hist+exposição; chips de flags | Todas as features numéricas de uma vez |
| Investigação (2–5 min) | Timeline do chassi, valores vs limiares | Rede de propostas, animações |
| Decisão | Registrar desfecho | — |

**Lacuna conhecida:** a etapa Decisão não tem suporte na base atual. Seria
útil registrar a decisão do analista (aprovar/investigar mais/escalar) e o
tempo gasto — **não disponível na base atual**; melhoria futura essencial
(fecha o loop de avaliação da mesa).

## 4. Wireframe textual

```
┌────────────────────────────────────────────────────────────────────────┐
│ MESA VRUM · Investigação de propostas        [safra▾][IF▾][UF▾][⊕filtros] │
│ "Sinais são indicadores de atenção para priorização — não confirmação   │
│  de fraude. Modelo com discriminação OOT limitada."        [ⓘ detalhes] │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Na mesa  │ % carteira│ Exposição│ Expo+LTV │ Hist. insuf.│ Sinais na fila│
├──────────┴──────────┴──────────┴──────────┴──────────┴─────────────────┤
│ FILA — ordenação oficial: sinais ↓ · expo+LTV · valor ↓ · score ↓       │
│  # · Proposta/Chassi · Motivo · Score · Valor · LTV · Sinais · Flags    │
│  [selecionar linha = painel de evidências + timeline do chassi]         │
├─────────────────────────────────────────────────────────────────────────┤
│ [ⓘ nota metodológica · limiares do treino · data de geração do dataset] │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5. KPIs / cards (6, todos defensáveis)

1. **Propostas na mesa** (zona INVESTIGAR)
2. **% da carteira encaminhada**
3. **Exposição financeira da fila** — soma de `valor_financiado`
4. **Exposição + LTV altos** — count de `flag_exposicao_e_ltv_altos`
5. **Histórico insuficiente** — count de `flag_historico_insuficiente`
6. **Top sinais na fila** — barras: 5 flags mais acionadas dentro da fila

Não usar: score médio da fila (não decide nada), "% de fraude" (não existe),
gauge de risco (falsa precisão). Positivos conhecidos (`target_risco_90d`)
só em visão retrospectiva/acadêmica — no uso operacional o target ainda não
existe.

## 6. Tabela da fila

Colunas fixas (10) — o resto fica no drill-down: prioridade (`#` + tier),
proposta/chassi, data/hora, **motivo** (Score ≥ limiar · Hist. insuf. +
exposição alta · Ambos — derivado da regra de zona), score (barra), valor
financiado, LTV, sinais "n/9", badge expo+LTV, chips de flags.

Regra prática: **sem scroll horizontal**; 13 flags como colunas tornam a
tabela ilegível.

## 7. Filtros e navegação

Safra (treino/validação/oot; em produção: data), IF, UF, canal, tipo de
proponente, faixa de valor, LTV acima do P90, histórico insuficiente,
flags específicas (multiselect), busca livre (id/chassi/proponente).
Além dos filtros da fila, o dossiê permite busca direta por proposta/chassi e
navegação anterior/próxima entre os resultados. Ordenação default = regra
oficial da mesa (fixa).

## 8. Detalhamento da proposta (drill-down)

- **A. Encaminhamento** — motivo, score com limiar, zona
- **B. Identificação da proposta e veículo** — marca, ano-modelo, placa,
  UF de registro, UF da proposta, canal, PF/PJ, IF, entrada e prazo
- **C. Evidências por categoria** — cada flag: acionada? · valor observado ·
  limiar (do treino) · janela temporal · 1 frase de por quê investigar
- **D. Exposição financeira** — valor (vs P90 R$ 182.233), entrada, prazo,
  FIPE, LTV (vs P90 3,18)
- **E. Timeline do chassi** — propostas anteriores: data, proponente,
  canal, UF, IF, valor (construível da própria base via self-join por
  `chassi_id_sintetico`)
- **F. Checklist local** — validar documentos, coerência cronológica,
  mudanças de UF/canal/IF e dados financeiros; não substitui decisão do
  analista e não é persistido atualmente

Ausente na base (melhoria futura): CNAE do PJ, modalidade de aquisição (à
vista vs financiada), histórico de transferência real, decisão/tempo do
analista. Marca/ano existem no cadastro (join trivial futuro).

## 9. Representação visual das flags

Chips/pills agrupados por categoria (comportamentais 9, auxiliar,
financeiras 3). No dossiê, cada flag acionada também mostra valor observado,
limiar, janela e motivo de atenção; uma legenda explica todas as flags. Não
acionadas ficam em seção secundária. `flag_exposicao_e_ltv_altos` com
destaque. `qtd_sinais_mesa` como "6/9". Nunca: matriz de calor na tabela
principal, ⚠️ por flag.

## 10. Score, exposição e LTV

- **Score**: barra 0–1 com marcador no limiar (0,5499). Rótulo: "score do
  modelo (critério de encaminhamento: top 5%)". Nunca "probabilidade de
  fraude".
- **Exposição**: R$ formatado, destaque quando ≥ P90
- **LTV**: razão com marcador quando ≥ 3,18; tooltip LTV = valor/FIPE

## 11. Destaque de prioridade

- `#` = posição na ordenação oficial (sinais ↓, expo+LTV, valor ↓, score ↓)
- Tiers derivados da própria regra: **P1** = `flag_exposicao_e_ltv_altos`;
  **P2** = `qtd_sinais_mesa` ≥ 4; **P3** = demais
- Linha P1 com fundo levemente destacado; sem vermelho piscante

## 12. Gráficos que valem a pena (3)

1. Barras horizontais — sinais mais acionados na fila
2. Histograma de score da fila com limiar marcado
3. Pareto de exposição — exposição acumulada pela ordem de prioridade

## 13. Gráficos NÃO recomendados

Gauge/termômetro (falsa precisão), radar de flags (perfis inexistentes),
mapa de calor por UF (`flag_mudanca_uf` aciona 65% — mancha sem significado),
rede de propostas ligadas (insinua organização criminosa), pie charts,
scatter de 189 mil pontos.

## 14. Hierarquia visual

KPIs (5 s) → Fila (60% da tela) → Detalhe expandido (sob demanda) → Nota
metodológica (rodapé, 1 linha) → Filtros (colapsáveis).

## 15. Cores e semântica

- **Vermelho nunca = fraude** (reservado a erro de sistema)
- Neutros: cinza-escuro (prioridade alta/P1), âmbar (atenção/INVESTIGAR),
  azul (informativo/score), verde só em agregados de APROVAR
- INVESTIGAR = âmbar; BLOQUEAR não renderiza (desabilitado)
- Chips em tom único; números em fonte tabular

## 16. Exemplo de proposta (valores reais da base)

```
#1 · P1 · PROP_CHASSI_0001532613 · CHASSI_SYN_000506382 · 19/03/2026 · OOT
Motivo: Score ≥ limiar
Score 0,685 (limiar 0,550) · Valor R$ 218.954 · LTV 4,25 (P90 3,18)
Sinais 9/9 · expo+LTV — prioridade financeira máxima
Timeline: 6 propostas em ~80 dias, 6 IFs, 3 UFs, PF→PJ→PF
```

## 17. Ferramenta

**Streamlit** — mesma linguagem do pipeline, lê o CSV direto, tabela com
seleção de linha, filtros nativos, sem custo/licença (coerente com TCC).
Power BI: alternativa corporativa, drill-down por linha mais trabalhoso.
Looker Studio: não (fraco em detalhe por linha).

---

## O que é construível hoje vs. futuro

**Construível com a base atual:** tudo acima, via
`output/vrum_propostas_flags.csv` + `output/limiares_flags_mesa.csv` +
timeline por chassi (self-join).

**Melhoria futura da base:** CNAE do PJ, modalidade de aquisição,
decisão/tempo do analista, marca/ano do veículo (join com cadastro),
target maturado em dados reais.

**Hipótese (não renderizar):** qualquer narrativa de "padrão de fraude" — a
base sintética não sustenta (AUC OOT ~ 0,5).
