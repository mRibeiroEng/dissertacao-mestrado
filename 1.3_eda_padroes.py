import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Etapa 3: Histogramas
# Histograma de eventos extremos por variável climática
fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
sns.histplot(df_clima['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'], bins=20, ax=axes[0], color='skyblue')
axes[0].set_title("Distribuição de Precipitação")
axes[0].set_xlabel("Precipitação (mm)")

sns.histplot(df_clima['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'], bins=20, ax=axes[1], color='orange')
axes[1].set_title("Distribuição de Temperatura Máxima")
axes[1].set_xlabel("Temperatura Máxima (°C)")

sns.histplot(df_clima['VENTO, RAJADA MAXIMA (m/s)'], bins=20, ax=axes[2], color='green')
axes[2].set_title("Distribuição de Rajadas de Vento")
axes[2].set_xlabel("Velocidade do Vento (m/s)")

plt.savefig('histogramas_climaticos.png')
plt.show()

# Etapa 4: Gráficos de Dispersão
# Relacionar eventos extremos com ocorrências operacionais
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

# Gráfico de Dispersão
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=dados_consolidados['Eventos Extremos'],
    y=dados_consolidados['Ocorrências Operacionais'],
    color='purple'
)
plt.title('Correlação entre Eventos Climáticos Extremos e Ocorrências Operacionais')
plt.xlabel('Eventos Climáticos Extremos')
plt.ylabel('Ocorrências Operacionais')
plt.savefig('grafico_dispersao_eventos_vs_ocorrencias.png')
plt.show()

# Etapa 5: Mapas de Calor
# Calcular a correlação entre variáveis climáticas
correlacoes_clima = df_clima[
    ['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)', 'VENTO, RAJADA MAXIMA (m/s)']
].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlacoes_clima, annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title('Mapa de Calor - Correlação entre Variáveis Climáticas')
plt.savefig('mapa_calor_climatico.png')
plt.show()

# Etapa 6: Séries Temporais
# Série temporal de eventos extremos
plt.figure(figsize=(12, 6))
plt.plot(dados_consolidados.index.astype(str), dados_consolidados['Eventos Extremos'], label='Eventos Extremos', marker='o', color='green')
plt.title('Eventos Climáticos Extremos ao Longo do Tempo')
plt.xlabel('Mês/Ano')
plt.ylabel('Número de Eventos Extremos')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('serie_temporal_eventos_extremos.png')
plt.show()

# Série temporal de ocorrências operacionais
plt.figure(figsize=(12, 6))
plt.plot(dados_consolidados.index.astype(str), dados_consolidados['Ocorrências Operacionais'], label='Ocorrências Operacionais', marker='o', color='blue')
plt.title('Ocorrências Operacionais ao Longo do Tempo')
plt.xlabel('Mês/Ano')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('serie_temporal_ocorrencias.png')
plt.show()

