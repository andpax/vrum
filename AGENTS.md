# AGENTS.md

Instruções para sessões OpenCode no repositório `vrum`.

## Contexto do projeto
- **vrum** = "Verificação de Risco e Uso de Motores". Projeto acadêmico (PT-BR).
- Objetivo: reconstruir a linha do tempo de propriedade de veículos financiados
  pelo CHASSI e gerar indicadores explicáveis para detectar tentativas de fraude
  em transações. Stack declarada no README: Python + LLM/RAG.
- **Status: planejamento.** Ainda não há código, manifests, dependências,
  testes, lint, typecheck, CI ou `opencode.json`. **Não invente comandos** de
  build/test/lint — crie o scaffolding apenas quando solicitado.
- Fonte da verdade para escopo/modelagem: `README.md` (janela de performance de
  90 dias, split temporal para evitar leakage, métricas AUC/KS/Precision/Recall/PR-AUC).

## Layout planejado (diretórios vazios por enquanto)
- `src/` — código-fonte (pipeline, features, modelo).
- `notebooks/` — análise exploratória e experimentos (Jupyter; `.ipynb_checkpoints` ignorado).
- `data/` — dados. **Não comitar dados brutos/large**; mantenha apenas scripts de
  carga/amostras pequenas. (`.gitignore` cobre `.env`, mas `data/` em si não é ignorado.)
- `dashboard/` — dashboard de "Perfil de Propriedade" (suspeitos x legítimos).
  Framework ainda indefinido (`.gitignore` sugere Streamlit e/ou Marimo).
- `docs/` — documentação.
- `presentation/` — materiais de apresentação (acadêmico).

## Ambiente e tooling (a definir; registrar quando existir)
- Gerenciador de ambiente: **venv + pip** (`requirements.txt`, `.venv/` ignorado).
- Tooling sugerido pelo `.gitignore` (não comprometido): `ruff`, `pytest`, `mypy`,
  Jupyter. Ao adotar, registrar aqui os comandos exatos.
- Secrets/LLM: variáveis em `.env` (gitignored). Nunca comitar chaves de API.

## Convenções
- Linguagem do projeto: **PT-BR** (README, docs, mensagens). Mantenha esse padrão.
- Sem regras de branch/PR/commits definidas ainda.
