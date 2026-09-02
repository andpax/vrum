import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "docs"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def ler_csv(caminho):
    return pd.read_csv(
        caminho,
        encoding="utf-8-sig",
        sep=";"
    )


df_propostas = pd.concat(
    [
        ler_csv(DATA_DIR / f"financiamentos_chassi_2026_{i:02d}.csv")
        for i in (1, 2, 3)
    ],
    ignore_index=True,
)


df_eventos = ler_csv(
    DATA_DIR / "eventos_target_chassi_90d.csv"
)

df_chassis = ler_csv(
    DATA_DIR / "cadastro_chassi_mock.csv"
)


def padronizar_colunas(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    return df


df_propostas = padronizar_colunas(df_propostas)
df_eventos = padronizar_colunas(df_eventos)
df_chassis = padronizar_colunas(df_chassis)


def limpar_chave(df, coluna):
    df = df.copy()

    df[coluna] = (
        df[coluna]
        .astype("string")
        .str.strip()
    )

    return df


df_propostas = limpar_chave(df_propostas, "id_proposta")
df_propostas = limpar_chave(df_propostas, "chassi_id_sintetico")

df_eventos = limpar_chave(df_eventos, "id_proposta")

df_chassis = limpar_chave(df_chassis, "chassi_id_sintetico")


duplicadas_propostas = df_propostas[
    df_propostas["id_proposta"].duplicated(keep=False)
]

print("Propostas duplicadas:", len(duplicadas_propostas))

print("\n=== PROPOSTAS ===")
print("Linhas:", len(df_propostas))
print("Colunas:", list(df_propostas.columns))

print("\n=== EVENTOS ===")
print("Linhas:", len(df_eventos))
print("Colunas:", list(df_eventos.columns))

print("\n=== CHASSIS ===")
print("Linhas:", len(df_chassis))
print("Colunas:", list(df_chassis.columns))


print("\n=== CARDINALIDADE ===")

print(
    "ID propostas únicos:",
    df_propostas["id_proposta"].nunique()
)

print(
    "ID eventos únicos:",
    df_eventos["id_proposta"].nunique()
)

print(
    "Chassis únicos nas propostas:",
    df_propostas["chassi_id_sintetico"].nunique()
)

print(
    "Chassis únicos no cadastro:",
    df_chassis["chassi_id_sintetico"].nunique()
)

print("\n=== EVENTOS POR PROPOSTA ===")

eventos_por_proposta = (
    df_eventos
    .groupby("id_proposta")
    .size()
    .value_counts()
    .sort_index()
)

print(eventos_por_proposta)

# 1. Checagem de Propostas x Eventos
propostas_ids = set(df_propostas['id_proposta'])
eventos_ids = set(df_eventos['id_proposta'])

eventos_sem_proposta = len(eventos_ids - propostas_ids)
propostas_sem_evento = len(propostas_ids - eventos_ids)

print(f"Eventos sem proposta cadastrada: {eventos_sem_proposta} ({eventos_sem_proposta/len(eventos_ids):.1%})")
print(f"Propostas sem nenhum evento: {propostas_sem_evento} ({propostas_sem_evento/len(propostas_ids):.1%})")

# 2. Checagem de Chassis x Cadastro
chassis_propostas = set(df_propostas['chassi_id_sintetico'])
chassis_cadastro = set(df_chassis['chassi_id_sintetico'])

chassis_orfaos = len(chassis_propostas - chassis_cadastro)
print(f"Chassis em propostas sem cadastro em CHASSIS: {chassis_orfaos}")



# 1. Agregação no Nível de Propostas por Chassi
df_prop_agg = df_propostas.groupby('chassi_id_sintetico').agg(
    qtd_propostas=('id_proposta', 'count'),
    qtd_ifs_distintas=('if_id_sintetico', 'nunique'),
    qtd_proponentes_distintos=('cpf_cnpj_proponente_sintetico', 'nunique'),
    valor_financiado_max=('valor_financiado', 'max'),
    prazo_meses_medio=('prazo_meses', 'mean')
).reset_index()

# 2. Agregação no Nível de Eventos por Chassi
# Primeiro trazemos o chassi para a tabela de eventos via propostas
df_eventos_chassi = df_eventos.merge(
    df_propostas[['id_proposta', 'chassi_id_sintetico']],
    on='id_proposta',
    how='inner'
)

df_eventos_agg = df_eventos_chassi.groupby('chassi_id_sintetico').agg(
    target_fraude_90d=('evento_risco_chassi_90d', 'max'), # 1 se teve qualquer evento de risco
    qtd_eventos_total=('tipo_evento', 'count'),
    dias_ate_evento_min=('dias_ate_evento', 'min')
).reset_index()

# 3. Consolidação Final no Cadastro de Chassis
df_modelagem = df_chassis.merge(df_prop_agg, on='chassi_id_sintetico', how='left')
df_modelagem = df_modelagem.merge(df_eventos_agg, on='chassi_id_sintetico', how='left')

# Preenchimento de ausentes para chassis sem propostas/eventos
df_modelagem['qtd_propostas'] = df_modelagem['qtd_propostas'].fillna(0)
df_modelagem['target_fraude_90d'] = df_modelagem['target_fraude_90d'].fillna(0)

# Feature Engineering
df_modelagem['ltv_max'] = df_modelagem['valor_financiado_max'] / df_modelagem['valor_fipe_referencia']