# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # Necessário para HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# Etapa 1: Carregar os dados climáticos
# Tenta carregar o arquivo CSV contendo os dados climáticos
try:
    dados_climaticos = pd.read_csv('base_clima.csv', delimiter=';', encoding='utf-8')
    print("Base climática carregada com sucesso!")
except FileNotFoundError as e:
    # Mensagem de erro caso o arquivo não seja encontrado
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Exibir informações iniciais sobre o DataFrame
print("\nInformações da Base Climática:")
print(dados_climaticos.info())

# Etapa 2: Limpeza dos dados
# Remove duplicatas para garantir que os dados sejam únicos
dados_climaticos.drop_duplicates(inplace=True)
print("\nDuplicatas removidas com sucesso.")

# Preenchimento de valores ausentes para a variável 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
# Preenche com 0 pois a ausência de valor implica ausência de precipitação
if 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)' in dados_climaticos.columns:
    dados_climaticos['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].fillna(0, inplace=True)
print("\nValores ausentes em 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)' preenchidos com 0.")

# Aplicação de interpolação linear para preencher valores ausentes em outras colunas
# Esta técnica é útil para dados temporais, assumindo variação linear entre pontos conhecidos
dados_climaticos.interpolate(method='linear', inplace=True)
print("\nValores ausentes tratados com interpolação linear.")

# Listagem de variáveis contínuas que requerem limpeza e ajustes de formatação
variaveis_continuas = [
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'RADIACAO GLOBAL (Kj/m²)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)'
]

# Conversão de vírgulas para pontos e valores para float nas variáveis contínuas
for coluna in variaveis_continuas:
    if coluna in dados_climaticos.columns:
        dados_climaticos[coluna] = dados_climaticos[coluna].astype(str).str.replace(',', '.').astype(float)

# Ajuste de formato para LATITUDE e LONGITUDE
for coluna in ['LATITUDE', 'LONGITUDE']:
    if coluna in dados_climaticos.columns:
        # Substitui vírgulas por pontos
        dados_climaticos[coluna] = dados_climaticos[coluna].str.replace(',', '.')
        
        # Função para garantir que os valores sejam convertidos para o formato -##.###### com 6 casas decimais
        def ajustar_lat_lon(valor):
            try:
                valor_float = float(valor)
                return f"{valor_float:.6f}"
            except:
                return None
        dados_climaticos[coluna] = dados_climaticos[coluna].apply(ajustar_lat_lon)

# Remove valores fora do intervalo aceitável para latitude (-90 a 90) e longitude (-180 a 180)
dados_climaticos = dados_climaticos[
    (dados_climaticos['LATITUDE'].astype(float) >= -90) & (dados_climaticos['LATITUDE'].astype(float) <= 90) &
    (dados_climaticos['LONGITUDE'].astype(float) >= -180) & (dados_climaticos['LONGITUDE'].astype(float) <= 180)
]

# Conversão da coluna 'Data' para o formato datetime
# Permite manipulação e análises temporais
dados_climaticos['Data'] = pd.to_datetime(dados_climaticos['Data'], format='%d/%m/%Y', errors='coerce')

# Etapa de filtro: Manter apenas os dados entre 01/01/2021 e 31/08/2024
data_inicio = pd.Timestamp('2021-01-01')
data_fim = pd.Timestamp('2024-08-31')
dados_climaticos = dados_climaticos[(dados_climaticos['Data'] >= data_inicio) & (dados_climaticos['Data'] <= data_fim)]

print("\nFiltro por intervalo de datas aplicado com sucesso.")

# Ajuste da coluna 'Hora UTC' para o formato hh:mm:ss
if 'Hora UTC' in dados_climaticos.columns:
    dados_climaticos['Hora UTC'] = dados_climaticos['Hora UTC'].str.extract(r'(\d{4})')[0]
    dados_climaticos['Hora UTC'] = dados_climaticos['Hora UTC'].apply(
        lambda x: f"{x[:2]}:{x[2:]}:00" if pd.notnull(x) else None
    )

# Criação de uma nova coluna 'Data_Hora' combinando data e hora
dados_climaticos['Data_Hora'] = pd.to_datetime(
    dados_climaticos['Data'].astype(str) + ' ' + dados_climaticos['Hora UTC'],
    errors='coerce'
)
print("\nConversão de datas e horas realizada com sucesso.")

# Etapa 5: Preenchimento usando modelo preditivo
# Substitui valores ausentes com predições baseadas em regressão para cada variável contínua
for coluna in variaveis_continuas:
    if coluna in dados_climaticos.columns:
        if dados_climaticos[coluna].isnull().any():
            dados_completos = dados_climaticos.dropna(subset=[coluna])
            dados_incompletos = dados_climaticos[dados_climaticos[coluna].isnull()]

            if not dados_completos.empty and not dados_incompletos.empty:
                X_treino = dados_completos[variaveis_continuas].drop(columns=[coluna]).dropna()
                y_treino = dados_completos[coluna]

                # Garante consistência entre X_treino e y_treino
                X_treino, y_treino = X_treino.align(y_treino, join='inner', axis=0)

                X_teste = dados_incompletos[variaveis_continuas].drop(columns=[coluna]).fillna(0)

                modelo = HistGradientBoostingRegressor()
                modelo.fit(X_treino, y_treino)

                previsoes = modelo.predict(X_teste)

                dados_climaticos.loc[dados_climaticos[coluna].isnull(), coluna] = previsoes

# Remove qualquer linha restante com valores nulos após todas as etapas
dados_climaticos.dropna(inplace=True)
print(f"\nNúmero de linhas após remoção de valores nulos: {len(dados_climaticos)}")

# Etapa 6: Ajustar valores reais para 1 casa decimal
# Aplica para as variáveis contínuas, mantendo consistência nos dados
colunas_reais = [
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
    'RADIACAO GLOBAL (Kj/m²)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
    'VENTO, RAJADA MAXIMA (m/s)',
    'VENTO, VELOCIDADE HORARIA (m/s)'
]

for coluna in colunas_reais:
    if coluna in dados_climaticos.columns:
        dados_climaticos[coluna] = (
            dados_climaticos[coluna]
            .astype(str)
            .str.replace(',', '.', regex=False)  # Garante ponto como separador decimal
            .astype(float)
            .round(1)
        )

# Etapa 7: Normalização de variáveis climáticas
# As variáveis contínuas serão normalizadas para o intervalo [0, 1] usando Min-Max Scaling
scaler = MinMaxScaler()

# Lista das variáveis contínuas que serão normalizadas
variaveis_normalizadas = [
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'RADIACAO GLOBAL (Kj/m²)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)'
]

# Verifica se as variáveis contínuas existem e normaliza cada uma delas
for coluna in variaveis_normalizadas:
    if coluna in dados_climaticos.columns:
        # Cria uma nova coluna para armazenar os valores normalizados
        dados_climaticos[f"{coluna}_normalizada"] = scaler.fit_transform(
            dados_climaticos[[coluna]]
        ).round(6)  # Arredonda os valores normalizados para 4 casas decimais

print("\nNormalização das variáveis climáticas realizada com sucesso.")
print(dados_climaticos[[f"{coluna}_normalizada" for coluna in variaveis_normalizadas]].head())


# Exportação dos dados tratados para um novo arquivo CSV
dados_climaticos.to_csv('base_climatica_tratada.csv', index=False, sep=';', encoding='utf-8')
print("\nBase climática tratada salva como 'base_climatica_tratada.csv'.")
