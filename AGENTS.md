# AGENTS.md

Instruções para sessões OpenCode no repositório `vrum`.

## Contexto do projeto
- **vrum** = "Verificação de Risco e Uso de Motores". Projeto acadêmico (PT-BR).
- Objetivo: reconstruir a linha do tempo de propriedade de veículos financiados
  pelo CHASSI e gerar indicadores explicáveis para detectar tentativas de fraude
  em transações. Stack real: Python + pandas/polars + XGBoost + política de
  regras. Camada LLM/RAG (LangChain/ChromaDB) é trabalho futuro, não implementada.
- **Status: funcional.** Pipeline, testes e artefatos existem.
- Fonte da verdade para escopo/modelagem: `README.md` (janela de performance de
  90 dias, split temporal para evitar leakage, métricas AUC/KS/Precision/Recall/PR-AUC)
  e `docs/flags_para_mesa.txt` (regras da mesa de análise).

## Pipeline (ordem obrigatória)
```
docs/*.csv → src/montagem_inicial.py → src/modelo_xgboost.py → src/politica_operacional_vrum.py
```
- Carga compartilhada dos 5 CSVs brutos: `src/vrum_io.py` (DATA_DIR = `docs/`).
- Apoio (fora do pipeline principal): `ordenacao_cronologica_chassi.py`
  (entregável cronológico + JSON), `pipeline_eventos.py` (EDA de sanidade),
  `vrum_split_temporal.py` (demonstrativo), `vrum_decision_engine.py`
  (motor de flags em pandas; contrato de entrada na docstring).
- Manual de uso: `docs/manual_uso_vrum.md`.

## Layout
- `src/` — código-fonte (pipeline, features, modelo, política).
- `notebooks/` — EDA e experimentos (Jupyter; espelham/refinam os scripts src).
- `output/` — artefatos gerados (gitignored; ~500MB).
- `docs/` — documentação + dados brutos (5 CSVs, ~400MB, gitignored).
- `data/`, `dashboard/`, `presentation/` — reservados (vazios).

## Ambiente e tooling
- Ambiente: **conda env `vrum`** (Python 3.12). Binário:
  `/var/home/andpax/anaconda3/envs/vrum/bin/python`.
- Testes: `python -m unittest discover -s tests` (a partir da raiz).
- Execução: `python src/<script>.py` (cada etapa tem `main()`).
- Lint/typecheck: não adotados ainda. Ao adotar, registrar aqui os comandos.
- Secrets/LLM: variáveis em `.env` (gitignored). Nunca comitar chaves de API.
- Não comitar dados brutos ou artefatos grandes (`docs/*.csv`, `output/*`,
  `notebooks/docs/*`, `notebooks/output/*` já cobertos pelo `.gitignore`).

## Convenções
- Linguagem do projeto: **PT-BR** (README, docs, mensagens). Mantenha esse padrão.
- Comentários de código: pontuais (regras de negócio, decisões não óbvias).
- Nondeterminismo conhecido: somas float em `group_by` polars variam no último
  dígito (ULP) entre execuções — esperado, não é bug.
- Sem regras de branch/PR/commits definidas ainda.
