import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from joblib import load

# 📌 1️⃣ Carregar o modelo treinado e os dados para previsão
print("📥 Carregando modelo treinado...")
modelo = load("modelo_random_forest.joblib")  # Certifique-se de que o modelo treinado está salvo corretamente
scaler = load("scaler.joblib")  # Carregar o normalizador usado no treinamento

print("📂 Carregando `07_aplicacao_do_modelo_interpretacao.csv`...")
df_novo = pd.read_csv("07_aplicacao_do_modelo_interpretacao.csv", delimiter=";", encoding="utf-8")
print(f"✅ Base de previsão carregada com {df_novo.shape[0]} registros e {df_novo.shape[1]} colunas.")

# 📌 2️⃣ Selecionar as colunas de entrada (features)
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)"
]

# 📌 3️⃣ Removendo valores ausentes antes da normalização
print("✅ Removendo valores ausentes...")
df_novo.dropna(subset=features, inplace=True)
print(f"✅ Após remoção de valores ausentes, restam {df_novo.shape[0]} registros.")

# 📌 4️⃣ Aplicar a normalização
print("🔄 Normalizando os dados...")
X_novo = scaler.transform(df_novo[features])

# 📌 5️⃣ Realizar previsões
print("🔮 Gerando previsões...")
df_novo["Previsao_Ocorrencia"] = modelo.predict(X_novo)

# 📌 6️⃣ Analisando impactos financeiros (estimativa de custos)
df_novo["Custo_Estimado"] = df_novo["Previsao_Ocorrencia"] * df_novo["valor_unitario"]

# 📌 7️⃣ Salvar previsões em CSV
print("💾 Salvando resultados em `07_previsoes_resultados.csv`...")
df_novo.to_csv("07_previsoes_resultados.csv", index=False, sep=";")
print(f"✅ Previsões salvas com {df_novo.shape[0]} registros.")

# 📌 8️⃣ Gerar gráficos de análise
print("📈 Gerando gráficos de análise...")
plt.figure(figsize=(10, 5))
sns.countplot(x="Previsao_Ocorrencia", data=df_novo, palette="viridis")
plt.title("Distribuição das Previsões de Ocorrências")
plt.xlabel("Ocorrência Prevista")
plt.ylabel("Contagem")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_novo["Custo_Estimado"], bins=50, kde=True, color="blue")
plt.title("Distribuição dos Custos Estimados")
plt.xlabel("Custo Estimado (R$)")
plt.ylabel("Frequência")
plt.show()

print("✅ Aplicação do modelo concluída! Resultados disponíveis para análise.")
