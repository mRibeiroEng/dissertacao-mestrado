import pandas as pd
import matplotlib.pyplot as plt

# Etapa 1: Carregar as bases tratadas
try:
    df_operacional = pd.read_csv('base_operacional_tratada.csv', delimiter=';', encoding='utf-8')
    df_clima = pd.read_csv('base_climatica_tratada.csv', delimiter=';', encoding='utf-8')
    print("Bases tratadas carregadas com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar as bases: {e}")
    exit()

# Filtrar apenas ocorrências onde "tipo_servico" contém a palavra "emergencial"
df_operacional_filtrado = df_operacional[
    df_operacional['tipo_servico'].str.contains('emergencial', case=False, na=False)
]

# Converter colunas de data para datetime
df_clima['Data'] = pd.to_datetime(df_clima['Data'], errors='coerce', dayfirst=True)
df_operacional_filtrado['data_servico'] = pd.to_datetime(df_operacional_filtrado['data_servico'], errors='coerce', dayfirst=True)

# Remover valores nulos nas datas
df_clima.dropna(subset=['Data'], inplace=True)
df_operacional_filtrado.dropna(subset=['data_servico'], inplace=True)

# Filtrar eventos climáticos extremos
eventos_extremos = df_clima[
    (df_clima['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] > 50) |
    (df_clima['VENTO, RAJADA MAXIMA (m/s)'] > 15) |
    (df_clima['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] > 40)
]

# Agrupar por Ano/Mês
eventos_extremos['AnoMes'] = eventos_extremos['Data'].dt.to_period('M')
eventos_por_mes = eventos_extremos.groupby('AnoMes').size()

df_operacional_filtrado['AnoMes'] = df_operacional_filtrado['data_servico'].dt.to_period('M')
ocorrencias_por_mes = df_operacional_filtrado.groupby('AnoMes').size()

# Consolidar os dados
dados_consolidados = pd.DataFrame({
    'Eventos Extremos': eventos_por_mes,
    'Ocorrências Operacionais': ocorrencias_por_mes
}).fillna(0)

# Certificar-se de alinhar todos os meses
todos_os_meses = pd.period_range(
    start=min(eventos_por_mes.index.min(), ocorrencias_por_mes.index.min()),
    end=max(eventos_por_mes.index.max(), ocorrencias_por_mes.index.max()),
    freq='M'
)
dados_consolidados = dados_consolidados.reindex(todos_os_meses, fill_value=0)

# Etapa 2: Criar o gráfico
fig, ax1 = plt.subplots(figsize=(12, 6))

# Gráfico de barras para eventos extremos
ax1.bar(
    dados_consolidados.index.astype(str),
    dados_consolidados['Eventos Extremos'],
    color='skyblue',
    label='Eventos Extremos'
)
ax1.set_xlabel('Mês/Ano', fontsize=12)
ax1.set_ylabel('Número de Eventos Extremos', fontsize=12, color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_xticks(range(0, len(dados_consolidados.index), 6))  # Mostrar um rótulo a cada 6 meses
ax1.set_xticklabels(dados_consolidados.index.astype(str)[::6], rotation=45, ha='right')

# Eixo secundário para ocorrências operacionais
ax2 = ax1.twinx()
ax2.plot(
    dados_consolidados.index.astype(str),
    dados_consolidados['Ocorrências Operacionais'],
    color='red',
    label='Ocorrências Operacionais',
    marker='o'
)
ax2.set_ylabel('Número de Ocorrências Operacionais', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Adicionar legendas
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=10)

# Título do gráfico
plt.title('Relação entre Eventos Extremos e Ocorrências Operacionais Emergenciais', fontsize=14)

# Ajustar layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('eventos_vs_ocorrencias_emergenciais.png')
plt.show()

print("Gráfico salvo como 'eventos_vs_ocorrencias_emergenciais.png'.")
