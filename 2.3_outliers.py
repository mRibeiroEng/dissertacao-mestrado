import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuração global para estilo dos gráficos
sns.set(style="whitegrid")

# Etapa 1: Carregar as bases tratadas
try:
    df_operacional = pd.read_csv('base_operacional_tratada.csv', delimiter=';', encoding='utf-8')
    df_clima = pd.read_csv('base_climatica_tratada.csv', delimiter=';', encoding='utf-8')
    print("Bases tratadas carregadas com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar as bases: {e}")
    exit()

# Etapa 2: Converter colunas de data para datetime
df_clima['Data'] = pd.to_datetime(df_clima['Data'], errors='coerce', dayfirst=True)
df_operacional['data_servico'] = pd.to_datetime(df_operacional['data_servico'], errors='coerce', dayfirst=True)

# Remover valores nulos nas datas
df_clima.dropna(subset=['Data'], inplace=True)
df_operacional.dropna(subset=['data_servico'], inplace=True)

# Etapa 3: Relacionar eventos extremos com ocorrências operacionais
eventos_extremos = df_clima[
    (df_clima['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] > 50) |
    (df_clima['VENTO, RAJADA MAXIMA (m/s)'] > 15) |
    (df_clima['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] > 40)
]
eventos_extremos['AnoMes'] = eventos_extremos['Data'].dt.to_period('M')
df_operacional['AnoMes'] = df_operacional['data_servico'].dt.to_period('M')

eventos_por_mes = eventos_extremos.groupby('AnoMes').size()
ocorrencias_por_mes = df_operacional.groupby('AnoMes').size()

dados_consolidados = pd.DataFrame({
    'Eventos Extremos': eventos_por_mes,
    'Ocorrências Operacionais': ocorrencias_por_mes
}).fillna(0)

# Exibir resumo estatístico
print("Resumo Estatístico dos Dados Consolidados:")
print(dados_consolidados.describe())

# Etapa 4: Identificação de outliers
X = dados_consolidados[['Eventos Extremos']]
y = dados_consolidados['Ocorrências Operacionais']

# Regressão Linear
modelo = LinearRegression()
modelo.fit(X, y)

# Previsões e Resíduos
y_pred = modelo.predict(X)
residuos = y - y_pred

# Limites ajustados para identificar outliers
limite_superior = residuos.mean() + 2 * residuos.std()
limite_inferior = residuos.mean() - 2 * residuos.std()

# Identificar os outliers
outliers = dados_consolidados[(residuos > limite_superior) | (residuos < limite_inferior)]

# Salvar resíduos para análise detalhada
dados_consolidados['Residuos'] = residuos
dados_consolidados.to_csv('residuos_detalhados.csv', index=False)

# Salvar outliers em um arquivo CSV
if not outliers.empty:
    outliers.reset_index().rename(columns={'index': 'AnoMes'}).to_csv('outliers_detalhados.csv', index=False)
    print("Outliers detectados e salvos em 'outliers_detalhados.csv'.")
else:
    print("Nenhum outlier detectado com os critérios atuais.")

# Etapa 5: Gráfico de dispersão com outliers destacados
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=dados_consolidados['Eventos Extremos'],
    y=dados_consolidados['Ocorrências Operacionais'],
    color='blue', label='Dados'
)
plt.scatter(
    outliers['Eventos Extremos'], 
    outliers['Ocorrências Operacionais'], 
    color='red', label='Outliers', edgecolor='black', s=100
)
plt.plot(
    dados_consolidados['Eventos Extremos'], 
    y_pred, 
    color='black', linestyle='--', label='Linha de Tendência'
)
plt.title('Gráfico de Dispersão com Outliers Destacados')
plt.xlabel('Eventos Climáticos Extremos')
plt.ylabel('Ocorrências Operacionais')
plt.legend()
plt.savefig('grafico_dispersao_outliers_detalhados.png')
plt.show()
