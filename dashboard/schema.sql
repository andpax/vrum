PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fraude_mensal;
DROP TABLE IF EXISTS fraude_canal;
DROP TABLE IF EXISTS fraude_uf;
DROP TABLE IF EXISTS fraude_marca;
DROP TABLE IF EXISTS eventos_risco;
DROP TABLE IF EXISTS propostas;
DROP TABLE IF EXISTS carga_metadata;

CREATE TABLE propostas (
    id_proposta TEXT PRIMARY KEY,
    data_hora_proposta TEXT NOT NULL,
    chassi_id_sintetico TEXT NOT NULL,
    if_id_sintetico TEXT,
    cpf_cnpj_proponente_sintetico TEXT,
    tipo_proponente TEXT,
    canal TEXT,
    uf_proposta TEXT,
    safra TEXT,
    score_modelo REAL,
    target_observado INTEGER,
    target_risco_90d INTEGER,
    valor_financiado REAL,
    valor_entrada REAL,
    prazo_meses INTEGER,
    valor_fipe_referencia REAL,
    ltv_fipe REAL,
    dias_desde_ultima_proposta REAL,
    qtd_propostas_historicas INTEGER,
    qtd_sinais_mesa INTEGER,
    zona_decisao TEXT
);

CREATE TABLE eventos_risco (
    id_evento INTEGER PRIMARY KEY AUTOINCREMENT,
    id_proposta TEXT NOT NULL,
    chassi_id_sintetico TEXT NOT NULL,
    dt_evento TEXT NOT NULL,
    tipo_evento TEXT NOT NULL,
    origem_registro TEXT,
    valor_financiado REAL,
    canal TEXT,
    uf_proposta TEXT,
    marca TEXT,
    flag_evento_risco INTEGER
);

CREATE TABLE carga_metadata (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);

CREATE INDEX idx_propostas_chassi ON propostas(chassi_id_sintetico);
CREATE INDEX idx_propostas_filtros ON propostas(canal, uf_proposta, zona_decisao, safra);
CREATE INDEX idx_propostas_data ON propostas(data_hora_proposta);
CREATE INDEX idx_eventos_proposta ON eventos_risco(id_proposta);
CREATE INDEX idx_eventos_tipo_data ON eventos_risco(tipo_evento, dt_evento);
CREATE INDEX idx_eventos_chassi ON eventos_risco(chassi_id_sintetico);

CREATE TABLE fraude_mensal AS
SELECT
    substr(e.dt_evento, 1, 7) AS mes,
    COUNT(DISTINCT e.id_proposta) AS casos,
    COALESCE(SUM(e.valor_financiado), 0) AS exposicao
FROM eventos_risco e
WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
GROUP BY substr(e.dt_evento, 1, 7);

CREATE TABLE fraude_canal AS
SELECT
    COALESCE(e.canal, p.canal) AS canal,
    COUNT(DISTINCT e.id_proposta) AS casos,
    COALESCE(SUM(e.valor_financiado), 0) AS exposicao
FROM eventos_risco e
LEFT JOIN propostas p ON p.id_proposta = e.id_proposta
WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
GROUP BY COALESCE(e.canal, p.canal);

CREATE TABLE fraude_uf AS
SELECT
    COALESCE(e.uf_proposta, p.uf_proposta) AS uf_proposta,
    COUNT(DISTINCT e.id_proposta) AS casos,
    COALESCE(SUM(e.valor_financiado), 0) AS exposicao
FROM eventos_risco e
LEFT JOIN propostas p ON p.id_proposta = e.id_proposta
WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
GROUP BY COALESCE(e.uf_proposta, p.uf_proposta);

CREATE TABLE fraude_marca AS
SELECT
    e.marca AS marca,
    COUNT(DISTINCT e.id_proposta) AS casos,
    COALESCE(SUM(e.valor_financiado), 0) AS exposicao
FROM eventos_risco e
WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
GROUP BY e.marca;
