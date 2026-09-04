"""Split temporal demonstrativo do VRUM (dados simulados).

Protótipo didático: gera base mock, otimiza memória e demonstra o split em
blocos consecutivos de 30 dias. O split usado de fato no pipeline vive em
src/modelo_xgboost.py::split_temporal_30_dias (mesma janela, dados reais).

Execução: `python src/vrum_split_temporal.py`
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(num_rows=100000):
    """
    Gera uma base estrutural fictícia para testar o split e o consumo de memória.
    O target aleatório não serve para avaliar desempenho antifraude.
    """
    print(f"Gerando {num_rows:,} linhas de dados simulados...")
    np.random.seed(42)
    
    # Criar datas entre 01/01/2026 e 31/03/2026
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 3, 31)
    delta_days = (end_date - start_date).days + 1
    
    random_days = np.random.randint(0, delta_days, size=num_rows)
    random_seconds = np.random.randint(0, 86400, size=num_rows)
    
    dates = [start_date + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]
    
    # Gerar Chassis e Proponentes repetidos para simular comportamento real
    chassis = [f"CHASSI_{np.random.randint(10000, 20000)}" for _ in range(num_rows)]
    proponentes = [f"PROP_{np.random.randint(50000, 150000)}" for _ in range(num_rows)]
    valores = np.random.uniform(20000, 150000, size=num_rows)
    
    # Target aleatório apenas para testar o pipeline estrutural.
    is_fraud = np.random.choice([0, 1], size=num_rows, p=[0.98, 0.02])
    
    df = pd.DataFrame({
        'id_proposta': range(1, num_rows + 1),
        'timestamp_proposta': pd.to_datetime(dates),
        'chassi_id': chassis,
        'proponente_id': proponentes,
        'valor_financiamento': valores.round(2),
        'fraude_detectada_90d': is_fraud # Target bruto fornecido
    })
    
    return df

def optimize_memory(df):
    """
    Reduz o uso de memória RAM convertendo os tipos de dados (Downcasting).
    Essencial para processar 3 milhões de linhas em computadores pessoais.
    """
    print("Otimizando consumo de memória...")
    initial_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and not isinstance(col_type, pd.DatetimeTZDtype) and col != 'timestamp_proposta':
            if 'int' in str(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif 'float' in str(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == object:
            # Converter ID/Categorias estáticas para category economiza muita memória
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
                
    final_mem = df.memory_usage().sum() / 1024**2
    print(f"Memória reduzida de {initial_mem:.2f} MB para {final_mem:.2f} MB ({((initial_mem-final_mem)/initial_mem)*100:.1f}% de economia)")
    return df

def apply_temporal_split(df, date_col='timestamp_proposta'):
    """
    Executa split temporal estrito em blocos consecutivos de 30 dias.
    Evita vazamento de dados (Data Leakage) simulando o ambiente real de produção.
    
    Base de dados total: 01/01/2026 a 31/03/2026 (~90 dias).
    - Treino: [início, início + 30 dias).
    - Validação: [início + 30 dias, início + 60 dias).
    - Out-of-time: [início + 60 dias, fim].
    """
    print("\nAplicando divisão temporal de 30 dias...")
    
    # Garantir ordenação cronológica
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Definir os marcos temporais
    inicio_treino = df[date_col].min()
    marco_30 = inicio_treino + timedelta(days=30)
    marco_60 = inicio_treino + timedelta(days=60)
    
    print(f"Período total dos dados: {inicio_treino.strftime('%d/%m/%Y')} até {df[date_col].max().strftime('%d/%m/%Y')}")
    
    # Criar as máscaras temporais
    mask_treino = (df[date_col] >= inicio_treino) & (df[date_col] < marco_30)
    mask_validacao = (df[date_col] >= marco_30) & (df[date_col] < marco_60)
    mask_oot = (df[date_col] >= marco_60)
    
    # Split das bases
    train_df = df[mask_treino].copy()
    val_df = df[mask_validacao].copy()
    oot_df = df[mask_oot].copy()
    
    # Criar coluna indicadora para auditoria do split
    df['split_group'] = 'OOT_Production'
    df.loc[mask_treino, 'split_group'] = 'Train_Set'
    df.loc[mask_validacao, 'split_group'] = 'Validation_Set'
    
    print("\nResultados do Split Temporal:")
    print(f" - [TREINO]      {inicio_treino.strftime('%d/%m/%Y')} a {(marco_30 - timedelta(seconds=1)).strftime('%d/%m/%Y')} | Linhas: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f" - [VALIDAÇÃO]   {marco_30.strftime('%d/%m/%Y')} a {(marco_60 - timedelta(seconds=1)).strftime('%d/%m/%Y')} | Linhas: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f" - [OUT-OF-TIME] {marco_60.strftime('%d/%m/%Y')} a {df[date_col].max().strftime('%d/%m/%Y')} | Linhas: {len(oot_df):,} ({len(oot_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, oot_df, df

# Exemplo de execução simulando o pipeline completo
if __name__ == "__main__":
    # 1. Simular base de propostas de janeiro a março (usando 500k linhas para teste local)
    raw_data = generate_mock_data(num_rows=500000)
    
    # 2. Otimizar uso de RAM
    optimized_data = optimize_memory(raw_data)
    
    # 3. Aplicar o split temporal estrito de 30 dias
    train, validation, oot, df_completo = apply_temporal_split(optimized_data)
    
    print("\n[VRUM] Estrutura do split pronta com sucesso!")
