import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# ✨ Carregar o modelo treinado e o scaler
print("\U0001F4E5 Carregando modelo treinado...")
modelo = load("modelo_random_forest.joblib")  # Certifique-se de que o modelo treinado está salvo corretamente
scaler = load("scaler.joblib")  # Carregar o normalizador usado no treinamento

# ✨ Criar subconjunto de dados diretamente da base original, sem depender de um CSV externo
print("\U0001F4C2 Criando subconjunto de dados para previsão...")
df_base = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")

features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)",
    "valor_unitario"  # Adicionando a coluna necessária
]

df_novo = df_base[df_base["qtd_atividade"] > 0].sample(n=100, random_state=42)

# ✨ Remover valores ausentes antes da normalização
df_novo.dropna(subset=["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "valor_unitario"], inplace=True)
print(f"✅ Subconjunto criado com {df_novo.shape[0]} registros.")

# ✨ Normalizar os dados
X_novo = scaler.transform(df_novo[[
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)"
]])

# ✨ Realizar previsões
df_novo["Previsao_Ocorrencia"] = modelo.predict(X_novo)

df_novo["Custo_Estimado"] = df_novo["Previsao_Ocorrencia"] * df_novo["valor_unitario"]

# ✨ Salvar previsões em CSV
nome_arquivo_resultado = "3.8_previsoes_resultados.csv"
print(f"\U0001F4BE Salvando resultados em `{nome_arquivo_resultado}`...")
df_novo.to_csv(nome_arquivo_resultado, index=False, sep=";")
print(f"✅ Previsões salvas com {df_novo.shape[0]} registros.")

# ✨ Gerar gráficos
plt.figure(figsize=(10, 5))
sns.countplot(x="Previsao_Ocorrencia", data=df_novo, palette="viridis")
plt.title("Distribuição das Previsões de Ocorrências")
plt.xlabel("Ocorrência Prevista")
plt.ylabel("Contagem")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_novo["Custo_Estimado"].dropna(), bins=50, kde=True, color="blue")
plt.title("Distribuição dos Custos Estimados")
plt.xlabel("Custo Estimado (R$)")
plt.ylabel("Frequência")
plt.show()

print("✅ Aplicação do modelo concluída! Resultados disponíveis para análise.")
