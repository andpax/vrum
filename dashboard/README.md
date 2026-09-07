# Dashboard VRUM

MVP em Streamlit para investigar riscos de fraude usando SQLite como camada de consulta.

## Preparação

O banco é derivado dos artefatos gerados pelo pipeline:

- `output/vrum_propostas_flags.csv`
- `output/vrum_timeline_cronologica.csv`

Recrie o banco a qualquer momento:

```bash
python dashboard/build_database.py
```

O arquivo local será criado em `dashboard/data/vrum_dashboard.sqlite` e não deve ser versionado.

## Execução

Instale as dependências do dashboard no ambiente `vrum`:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Escopo do MVP

- Filtros por canal, UF, safra e zona de decisão.
- KPIs de propostas, fraudes confirmadas, incidência e exposição.
- Evolução mensal de casos e exposição.
- Rankings por canal, UF e marca.
- Fila de propostas em `INVESTIGAR` e `BLOQUEAR`.
- Histórico de eventos de uma proposta selecionada.

O SQLite é uma camada derivada e local. O CSV continua sendo a fonte de verdade.
