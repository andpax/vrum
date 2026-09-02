import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(num_rows=100000):
    """
    Gera uma base de dados fictícia para simular a estrutura do desafio VRUM.
    Útil para testar o pipeline sem estourar a memória RAM.
    """
    print(f"Gerando {num_rows:,} linhas de dados simulados...")
    np.random.seed(42)
    
    # Criar datas entre 01/01/2026 e 31/03/2026
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 3, 31)
    delta_days = (end_date - start_date).days
    
    random_days = np.random.randint(0, delta_days, size=num_rows)
    random_seconds = np.random.randint(0, 86400, size=num_rows)
    
    dates = [start_date + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]
    
    # Gerar Chassis e Proponentes repetidos para simular comportamento real
    chassis = [f"CHASSI_{np.random.randint(10000, 20000)}" for _ in range(num_rows)]
    proponentes = [f"PROP_{np.random.randint(50000, 150000)}" for _ in range(num_rows)]
    valores = np.random.uniform(20000, 150000, size=num_rows)
    
    # Target simulado (ex: taxa de fraude de 2%)
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
    Executa o Split Temporal Estrito de 45 dias conforme sugerido pelo mentor Victor Acioli.
    Evita vazamento de dados (Data Leakage) simulando o ambiente real de produção.
    
    Base de dados total: 01/01/2026 a 31/03/2026 (~90 dias)
    - Bloco 1 (Modelagem): Propostas do dia 01/01 a 14/02 (~45 dias). 
      -> Possui janela de 45 dias completa até o fim da base (31/03) para observar o desfecho.
    - Bloco 2 (Produção/Out-of-Time): Propostas de 15/02 a 31/03 (~45 dias).
    """
    print("\nAplicando divisão temporal de 45 dias...")
    
    # Garantir ordenação cronológica
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Definir os marcos temporais
    inicio_treino = df[date_col].min()
    fim_treino = inicio_treino + timedelta(days=30)  # Janeiro inteiro (~31 dias)
    fim_bloco_modelagem = inicio_treino + timedelta(days=45) # 45 dias totais (01/01 a 14/02)
    
    print(f"Período total dos dados: {inicio_treino.strftime('%d/%m/%Y')} até {df[date_col].max().strftime('%d/%m/%Y')}")
    
    # Criar as máscaras temporais
    mask_treino = (df[date_col] >= inicio_treino) & (df[date_col] <= fim_treino)
    mask_validacao = (df[date_col] > fim_treino) & (df[date_col] <= fim_bloco_modelagem)
    mask_oot = (df[date_col] > fim_bloco_modelagem)
    
    # Split das bases
    train_df = df[mask_treino].copy()
    val_df = df[mask_validacao].copy()
    oot_df = df[mask_oot].copy()
    
    # Criar coluna indicadora para auditoria do split
    df['split_group'] = 'OOT_Production'
    df.loc[mask_treino, 'split_group'] = 'Train_Set'
    df.loc[mask_validacao, 'split_group'] = 'Validation_Set'
    
    print("\nResultados do Split Temporal:")
    print(f" - [TREINO]      {inicio_treino.strftime('%d/%m/%Y')} a {fim_treino.strftime('%d/%m/%Y')} | Linhas: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f" - [VALIDAÇÃO]   {(fim_treino + timedelta(days=1)).strftime('%d/%m/%Y')} a {fim_bloco_modelagem.strftime('%d/%m/%Y')} | Linhas: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f" - [OUT-OF-TIME] {(fim_bloco_modelagem + timedelta(days=1)).strftime('%d/%m/%Y')} a {df[date_col].max().strftime('%d/%m/%Y')} | Linhas: {len(oot_df):,} ({len(oot_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, oot_df, df

# Exemplo de execução simulando o pipeline completo
if __name__ == "__main__":
    # 1. Simular base de propostas de janeiro a março (usando 500k linhas para teste local)
    raw_data = generate_mock_data(num_rows=500000)
    
    # 2. Otimizar uso de RAM
    optimized_data = optimize_memory(raw_data)
    
    # 3. Aplicar o Split Temporal Estrito de 45 dias
    train, validation, oot, df_completo = apply_temporal_split(optimized_data)
    
    print("\n[VRUM] Estrutura do split pronta com sucesso!")
