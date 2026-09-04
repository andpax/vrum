# Manual de Uso — VRUM

**Verificação de Risco e Uso de Motores**
Projeto acadêmico de pós-graduação (Residência em Análise de Dados e IA).

---

## 1. Visão geral

O VRUM é um sistema de apoio à decisão antifraude para financiamento veicular.
Ele reconstrói a linha do tempo de propriedade de cada veículo a partir do
CHASSI (propostas de financiamento + eventos de risco), transforma esse
histórico em indicadores explicáveis (velocidade de trocas, permanência de
posse, alternância PF/PJ, entre outros) e alimenta um modelo XGBoost com split
temporal, cujo score sustenta uma política operacional de três zonas:
**APROVAR**, **INVESTIGAR** (mesa de análise manual) e **BLOQUEAR**
(desabilitado por padrão). A prioridade da mesa é orientada por flags de regras
definidas em `docs/flags_para_mesa.txt`, sempre calculadas sem usar dados
futuros (anti-leakage).

## 2. Pré-requisitos e instalação

**Requisitos:**

- Python 3.12 ou superior
- Os 5 CSVs brutos em `docs/` (não versionados no git; ver seção 6)
  - `financiamentos_chassi_2026_01.csv`, `_02.csv`, `_03.csv` (propostas)
  - `eventos_target_chassi_90d.csv` (eventos de risco / target)
  - `cadastro_chassi_mock.csv` (metadados dos chassis)
- ~2 GB de RAM livre e ~1,5 GB de disco (dados + artefatos de `output/`)

**Instalação:**

```bash
# opção conda (usada no desenvolvimento)
conda create -n vrum python=3.12 -y
conda run -n vrum pip install -r requirements.txt
conda activate vrum

# opção venv + pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Variáveis de ambiente:** nenhuma obrigatória. A camada LLM/RAG (trabalho
futuro) usará `.env` para chaves de API — nunca comitar esse arquivo.

## 3. Estrutura de pastas e responsabilidade de cada módulo

```
vrum/
├── src/
│   ├── vrum_io.py                      # Carga compartilhada dos 5 CSVs (docs/) e
│   │                                   # padronização de colunas/chaves de join
│   ├── montagem_inicial.py             # PIPELINE 1/3: timeline unificada + features
│   │                                   # temporais + dataset de modelagem por chassi
│   ├── modelo_xgboost.py               # PIPELINE 2/3: features só com passado,
│   │                                   # split temporal 30d, treino XGBoost, métricas
│   ├── politica_operacional_vrum.py    # PIPELINE 3/3: score por safra, zonas
│   │                                   # APROVAR/INVESTIGAR/BLOQUEAR, estabilidade por IF
│   ├── ordenacao_cronologica_chassi.py # APOIO: entregável canônico da timeline
│   │                                   # (long CSV + JSON por chassi + sumário)
│   ├── pipeline_eventos.py             # APOIO: EDA de sanidade (duplicatas, órfãos)
│   ├── vrum_split_temporal.py          # APOIO: split temporal demonstrativo (mock)
│   └── vrum_decision_engine.py         # APOIO: motor de flags/zona em pandas
│                                       # (contrato de entrada na docstring)
├── tests/
│   └── test_vrum_pipeline.py           # unittest: target, features, split, flags
├── notebooks/                          # EDA e experimentos (espelham os scripts src)
├── docs/                               # dados brutos (gitignored) + documentação
│   ├── flags_para_mesa.txt             # regras de negócio da mesa de análise
│   └── manual_uso_vrum.md              # este manual
├── output/                             # artefatos gerados (gitignored)
├── dashboard/                          # reservado (trabalho futuro)
├── data/                               # reservado (vazio)
└── presentation/                       # reservado (materiais acadêmicos)
```

## 4. Ordem de execução do pipeline (passo a passo)

O pipeline principal tem 3 etapas obrigatórias, nesta ordem — cada uma consome
a saída da anterior em `output/`:

```bash
# 0. (opcional) EDA de sanidade das bases
python src/pipeline_eventos.py

# 1. Ingestão: timeline unificada + features temporais
#    gera output/vrum_timeline_completa.csv e output/vrum_dataset_modelagem.csv
python src/montagem_inicial.py

# 2. Modelo: features definitivas + split temporal + XGBoost
#    gera output/modelo_xgboost_vrum.json, metricas/importancia/split .csv
python src/modelo_xgboost.py

# 3. Política operacional: score por safra + zonas de decisão
#    gera output/politica_zonas_vrum.csv, estabilidade_safra_if_vrum.csv,
#           output/limiares_politica_vrum.csv
python src/politica_operacional_vrum.py

# Testes de regressão (não grava artefatos)
python -m unittest discover -s tests
```

Scripts de apoio (executáveis isoladamente, fora do pipeline principal):

```bash
python src/ordenacao_cronologica_chassi.py   # timeline canônica + JSON + sumário
python src/vrum_split_temporal.py            # demonstração do split (dados simulados)
```

**Principais artefatos de saída (`output/`):**

| Arquivo | Conteúdo |
|---|---|
| `vrum_timeline_completa.csv` | Timeline long com features (entrada da etapa 2) |
| `vrum_dataset_modelagem.csv` | Base agregada por chassi (EDA/dashboard) |
| `modelo_xgboost_vrum.json` | Modelo treinado |
| `metricas_xgboost_vrum.csv` | AUC/KS/PR-AUC/Precision/Recall por split |
| `split_temporal_30d_vrum.csv` | Limites de datas e volumes de cada split |
| `politica_zonas_vrum.csv` | Volumetria, captura de risco e exposição por zona/safra |
| `estabilidade_safra_if_vrum.csv` | AUC/PR-AUC e taxas de zona por safra e IF |
| `limiares_politica_vrum.csv` | Limiares de INVESTIGAR/BLOQUEAR (quantis da validação) |

## 5. Exemplos de uso (entrada e saída esperada)

**Execução típica da etapa 2** (`python src/modelo_xgboost.py`) — saída no
console (valores da última execução; a base é sintética e o desempenho OOT é
próximo do aleatório, conforme registrado em `docs/flags_para_mesa.txt`):

```
Split temporal de 30 dias:
treino      | 725.609 linhas | 10.659 positivos | 01/01 a 30/01/2026
validacao   | 794.643 linhas | 11.507 positivos | 31/01 a 02/03/2026
oot         | 608.961 linhas |  9.028 positivos | 02/03 a 31/03/2026

Métricas (oot): AUC 0,495 | KS 0,004 | PR-AUC 0,015 | Recall 0,652
Limiar escolhido na validação: 0,311551
```

**Política resultante** (`output/limiares_politica_vrum.csv`):

```
regra,quantil_validacao,limiar_score
investigar,0.95,0.5499
bloquear,0.99,0.5962
```

Proposta com score 0,80 no OOT: zona **BLOQUEAR** pela política; como a AUC OOT
está ~0,5, a regra vigente da mesa é **não aplicar bloqueio automático** — o
score serve para triagem e coleta de evidência, não como prova de fraude.

**Inspecionar um chassi específico** (timeline canônica):

```bash
python src/ordenacao_cronologica_chassi.py
# JSON por chassi (amostra de 500) em output/vrum_linha_do_tempo_chassi_amostra.json
```

```json
{
  "CHASSI_SYN_000035285": {
    "eventos": [
      {"ordem_evento": 1, "dt_evento": "2026-01-10T05:36:55",
       "tipo_evento": "PROPOSTA_FINANCIAMENTO", "id_proposta": "PROP_CHASSI_0000000001",
       "status_data": "VALIDA", "tipo_proponente": "PF", "evento_risco_chassi_90d": 0}
    ]
  }
}
```

## 6. Troubleshooting básico

| Problema | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError: .../docs/financiamentos_chassi_2026_01.csv` | CSVs brutos ausentes em `docs/` (não vão no git) | Obter os 5 CSVs e copiá-los para `docs/` |
| `MemoryError` / lentidão extrema | Bases somam ~2,1M propostas + timeline de ~515MB | Fechar outros processos; rodar etapa por etapa; usar máquina com ≥8GB RAM |
| `FileNotFoundError: modelo_xgboost_vrum.json` ao rodar a política | Etapa 2 ainda não executada | Executar `python src/modelo_xgboost.py` primeiro |
| Métricas/política com AUC OOT ~0,5 | Comportamento esperado nesta base sintética | Não habilitar BLOQUEAR; usar flags para triagem (ver `docs/flags_para_mesa.txt`) |
| Últimos dígitos de somas (`exposicao_total`, `saving_*`) variam entre execuções | Soma float paralela no `group_by` do polars (ULP) | Esperado; não é bug |
| `ModuleNotFoundError: numpy` | Python do sistema (3.14) sem deps | Usar o env `vrum` (conda) ou o `.venv` criado na seção 2 |
| Colunas "ausentes" ao usar `vrum_decision_engine.py` | Contrato exige `ltv_fipe`, `tipo_proponente_anterior` e coluna de split | Ver docstring do módulo: montar o DataFrame via notebook `processo_completo_vrum.ipynb` |
