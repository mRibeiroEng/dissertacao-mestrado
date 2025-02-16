import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar a base tratada
try:
    df_tratada = pd.read_csv('base_climatica_tratada.csv', delimiter=';', encoding='utf-8')
    print("Base climática tratada carregada com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar a base: {e}")
    exit()

# Converter a coluna de data para o formato datetime
df_tratada['Data'] = pd.to_datetime(df_tratada['Data'], errors='coerce')

# Filtrar apenas as colunas relevantes para eventos extremos
eventos_extremos = df_tratada[
    (df_tratada['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] > 50) |  # Exemplo: Chuva maior que 50mm
    (df_tratada['VENTO, RAJADA MAXIMA (m/s)'] > 15) |        # Exemplo: Rajadas de vento > 15 m/s
    (df_tratada['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'] > 40)  # Exemplo: Temperatura maior que 40ºC
]

# Agrupar por data e estação para contar eventos extremos
eventos_extremos_agrupados = (
    eventos_extremos
    .groupby(['Data', 'ESTACAO'])
    .size()
    .reset_index(name='Ocorrencias')
)

# Adicionar uma coluna com mês e ano no formato desejado (ex: Jan/2020 em português)
eventos_extremos_agrupados['MesAno'] = eventos_extremos_agrupados['Data'].dt.strftime('%b/%Y').str.capitalize()

# Garantir que todos os meses/anos estejam presentes para todas as estações
todos_mes_ano = pd.date_range(
    start=eventos_extremos_agrupados['Data'].min(),
    end=eventos_extremos_agrupados['Data'].max(),
    freq='MS'
).strftime('%b/%Y').str.capitalize()

todos_mes_ano = pd.Index(sorted(todos_mes_ano, key=lambda x: pd.to_datetime(x, format='%b/%Y')))
estacoes = eventos_extremos_agrupados['ESTACAO'].unique()

# Criar um DataFrame alinhado para evitar desalinhamento
dados_alinhados = pd.DataFrame(0, index=todos_mes_ano, columns=estacoes)

# Preencher o DataFrame alinhado com os valores
for estacao in estacoes:
    estacao_data = eventos_extremos_agrupados[eventos_extremos_agrupados['ESTACAO'] == estacao]
    estacao_data = estacao_data.groupby('MesAno')['Ocorrencias'].sum()
    
    # Alinhar os índices usando reindex
    estacao_data = estacao_data.reindex(todos_mes_ano, fill_value=0)
    dados_alinhados[estacao] = estacao_data

# Configurar os dados para o gráfico
x = np.arange(len(dados_alinhados.index))  # Posições no eixo X
width = 0.15  # Largura das barras
colors = ['blue', 'orange', 'green', 'red', 'purple']  # Cores para as barras

# Criar o gráfico
fig, ax = plt.subplots(figsize=(20, 10))

for i, estacao in enumerate(estacoes):
    ax.bar(
        x + i * width,  # Deslocar as barras horizontalmente
        dados_alinhados[estacao],
        width=width,
        label=estacao,
        color=colors[i % len(colors)]
    )

# Configurar os rótulos do eixo X
ax.set_xticks(x + width * (len(estacoes) - 1) / 2)
ax.set_xticklabels(dados_alinhados.index, rotation=45, fontsize=10)

# Adicionar título e legendas
ax.set_title('Ocorrência de Eventos Climáticos Extremos ao Longo dos Meses', fontsize=16)
ax.set_xlabel('Mês/Ano', fontsize=12)
ax.set_ylabel('Número de Eventos Extremos', fontsize=12)

# Adicionar legenda para as estações
legenda = ax.legend(title='Estação Automática', fontsize=10, title_fontsize=12, loc='upper right', ncol=1)

# Adicionar a caixa de explicação ao lado esquerdo da legenda
caixa_texto = (
    "Critérios para Eventos Climáticos Extremos:\n"
    "- Precipitação > 50 mm\n"
    "- Rajadas de vento > 15 m/s\n"
    "- Temperatura máxima > 40ºC"
)
props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.7, edgecolor='black')

# Posição da caixa ajustada manualmente ao lado da legenda
ax.text(
    0.6, 0.95, caixa_texto, transform=ax.transAxes,
    fontsize=10, verticalalignment='top', horizontalalignment='right',
    bbox=props
)

# Adicionar grid para melhor visualização
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')

# Salvar o gráfico
plt.tight_layout()
plt.savefig('eventos_extremos_barras_com_legenda_e_caixa.png')
plt.show()

print("Gráfico gerado e salvo como 'eventos_extremos_barras_com_legenda_e_caixa.png'.")
