import pandas as pd
import numpy as np  # Para geração de códigos aleatórios

# Etapa 1: Carregar a base operacional
try:
    df = pd.read_csv('base_operacional.csv', delimiter=';', encoding='utf-8')
    print("Base operacional carregada com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Exibir informações iniciais sobre o DataFrame
print("\nInformações da Base Operacional:")
print(df.info())

# Etapa 2: Remover colunas desnecessárias
columns_to_remove = [
    'Contrato', 'cod_equipe', 'des_equipe', 'responsavel', 'Supervisor',
    'eletricistas', 'cod_turno', 'obs_turno', 'OT', 'abertura_turno',
    'fechamento_turno', 'tempo_intervalo', 'OS/OT', 'UC', 'solicitante', 'bairro',
    'endereco', 'centro_servico', 'retorno_campo', 'observacao', 'des_tipo_grupo',
    'des_grupo', 'cod_atividade'
]
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
print("\nColunas removidas com sucesso!")

# Etapa 3: Ajustar valores de latitude e longitude
def format_lat_lon(value):
    try:
        if pd.isna(value) or value == '':  # Verificar valores vazios
            return ''  # Retornar vazio
        value = float(str(value).replace(',', '.'))  # Substituir vírgula por ponto
        return f"{value:.6f}"  # Garantir 6 casas decimais
    except:
        return ''  # Retornar vazio caso não seja possível converter

if 'latitude' in df.columns:
    df['latitude'] = df['latitude'].apply(format_lat_lon)
if 'longitude' in df.columns:
    df['longitude'] = df['longitude'].apply(format_lat_lon)
print("\nFormato de latitude e longitude ajustado.")

# Preencher latitude e longitude faltantes com base na localidade
if 'localidade' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
    reference_coords = df[df['latitude'] != ''][['localidade', 'latitude', 'longitude']].drop_duplicates()
    coord_dict = reference_coords.set_index('localidade')[['latitude', 'longitude']].T.to_dict()

    def fill_missing_lat_lon(row):
        if row['latitude'] == '' or row['longitude'] == '':
            coords = coord_dict.get(row['localidade'], {'latitude': '', 'longitude': ''})
            if row['latitude'] == '':
                row['latitude'] = coords.get('latitude', '')
            if row['longitude'] == '':
                row['longitude'] = coords.get('longitude', '')
        return row

    df = df.apply(fill_missing_lat_lon, axis=1)
    print("\nLatitude e longitude faltantes preenchidas com base na localidade.")

# Etapa 4: Converter valores reais para usar ponto como separador decimal
real_columns = ['valor_unitario', 'valor_total']
for col in real_columns:
    if col in df.columns:
        try:
            df[col] = (
                df[col].astype(str)
                .str.replace('.', '', regex=False)  # Remover separador de milhar
                .str.replace(',', '.', regex=False)  # Substituir vírgula por ponto decimal
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Converter para numérico
        except Exception as e:
            print(f"Erro ao tratar coluna {col}: {e}")
print("\nColunas de valores reais corrigidas.")

# Etapa 5: Garantir o preenchimento das colunas de data e hora
if 'data_servico' in df.columns:
    df['data_servico'] = pd.to_datetime(df['data_servico'], errors='coerce', dayfirst=True)

    # Filtro de intervalo de datas
    data_inicio = pd.Timestamp('2021-01-01')
    data_fim = pd.Timestamp('2024-08-31')
    df = df[(df['data_servico'] >= data_inicio) & (df['data_servico'] <= data_fim)]
    print("\nFiltro de intervalo de datas aplicado com sucesso!")

    # Preencher `data_deslocamento` com a data de `data_servico` e hora padrão
    if 'data_deslocamento' in df.columns:
        df['data_deslocamento'] = pd.to_datetime(df['data_deslocamento'], errors='coerce', dayfirst=True)
        df['data_deslocamento'] = df['data_deslocamento'].fillna(
            pd.to_datetime(df['data_servico'].dt.strftime('%Y-%m-%d') + ' 08:00:00')
        )

    # Preencher `data_inicio` com a data de `data_servico` e hora padrão
    if 'data_inicio' in df.columns:
        df['data_inicio'] = pd.to_datetime(df['data_inicio'], errors='coerce', dayfirst=True)
        df['data_inicio'] = df['data_inicio'].fillna(
            pd.to_datetime(df['data_servico'].dt.strftime('%Y-%m-%d') + ' 09:00:00')
        )

    # Preencher `data_fim` com a data de `data_servico` e hora padrão
    if 'data_fim' in df.columns:
        df['data_fim'] = pd.to_datetime(df['data_fim'], errors='coerce', dayfirst=True)
        df['data_fim'] = df['data_fim'].fillna(
            pd.to_datetime(df['data_servico'].dt.strftime('%Y-%m-%d') + ' 17:00:00')
        )

print("\nDatas ausentes preenchidas com base na data_servico e horários padrão.")

# Etapa 6: Padronizar a coluna unidade_medida
if 'unidade_medida' in df.columns:
    df['unidade_medida'] = df['unidade_medida'].replace({'UND': 'UN'})
print("\nValores da coluna unidade_medida padronizados para 'UN'.")

# Etapa 7: Criar códigos aleatórios para tipo_servico e des_atividade
def generate_random_codes(df, column_name):
    unique_values = df[column_name].dropna().unique()  # Obter valores únicos
    code_map = {value: np.random.randint(1000, 9999) for value in unique_values}  # Mapear códigos aleatórios
    return df[column_name].map(code_map)

if 'tipo_servico' in df.columns:
    df['tipo_servico_code'] = generate_random_codes(df, 'tipo_servico')
if 'des_atividade' in df.columns:
    df['des_atividade_code'] = generate_random_codes(df, 'des_atividade')

columns_order = list(df.columns)
if 'tipo_servico_code' in columns_order and 'tipo_servico' in columns_order:
    tipo_servico_idx = columns_order.index('tipo_servico')
    columns_order.insert(tipo_servico_idx, columns_order.pop(columns_order.index('tipo_servico_code')))
if 'des_atividade_code' in columns_order and 'des_atividade' in columns_order:
    des_atividade_idx = columns_order.index('des_atividade')
    columns_order.insert(des_atividade_idx, columns_order.pop(columns_order.index('des_atividade_code')))
df = df[columns_order]
print("\nCódigos aleatórios gerados e reorganizados.")

# Etapa 8: Salvar o arquivo tratado
df.to_csv('base_operacional_tratada.csv', sep=';', index=False)
print("\nArquivo tratado salvo como 'base_operacional_tratada.csv'.")
