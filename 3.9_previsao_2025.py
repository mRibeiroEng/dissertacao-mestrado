import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# 📌 1️⃣ Carregar o modelo treinado e os dados para previsão
print("📥 Carregando modelo treinado...")
modelo = load("modelo_random_forest.joblib")  # Certifique-se de que o modelo treinado está salvo corretamente
scaler = load("scaler.joblib")  # Carregar o normalizador usado no treinamento

# 📂 2️⃣ Criar um subconjunto para previsão (se necessário)
meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
np.random.seed(42)

# Gerando dados sintéticos para previsão mensal de 2025
df_previsao = pd.DataFrame({
    "Mês": meses,
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": np.random.uniform(50, 150, len(meses)),  # Ajuste nos valores para serem mais realistas
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": np.random.uniform(20, 30, len(meses)),
    "UMIDADE RELATIVA DO AR, HORARIA (%)": np.random.uniform(40, 70, len(meses)),
    "VENTO, VELOCIDADE HORARIA (m/s)": np.random.uniform(1.0, 4.0, len(meses)),
    "valor_unitario": np.random.uniform(100, 400, len(meses))  # Ajuste nos valores para serem mais realistas
})

print(f"✅ Base de previsão criada com {df_previsao.shape[0]} registros.")

# 📌 3️⃣ Aplicar a normalização (opcional)
print("🔄 Normalizando os dados...")
X_previsao = scaler.transform(df_previsao.iloc[:, 1:5])

# 📌 4️⃣ Realizar previsões
print("🔮 Gerando previsões...")
df_previsao["Previsao_Ocorrencia"] = modelo.predict(X_previsao)

# 📌 5️⃣ Ajuste para evitar meses com zero ocorrências
media_ocorrencias = 10  # Ajuste para uma média histórica mínima mais realista
df_previsao["Previsao_Ocorrencia"] = df_previsao["Previsao_Ocorrencia"].replace(0, media_ocorrencias)

# 📌 6️⃣ Cálculo do Custo Estimado
df_previsao["Custo_Estimado"] = df_previsao["Previsao_Ocorrencia"] * df_previsao["valor_unitario"]

# 📌 7️⃣ Salvar previsões em CSV
nome_arquivo_resultado = "3.9_previsoes_resultados.csv"
print(f"💾 Salvando resultados em `{nome_arquivo_resultado}`...")
df_previsao.to_csv(nome_arquivo_resultado, index=False, sep=";")
print(f"✅ Previsões salvas com {df_previsao.shape[0]} registros.")

# 📌 8️⃣ Gerar gráficos de análise
print("📈 Gerando gráficos de análise...")

# 📊 Temperatura vs Precipitação
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.bar(df_previsao["Mês"], df_previsao["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"], alpha=0.6, color="blue", label="Precipitação (mm)")
ax2.plot(df_previsao["Mês"], df_previsao["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"], color="red", marker="o", label="Temperatura (°C)")
ax1.set_xlabel("Mês")
ax1.set_ylabel("Precipitação (mm)", color="blue")
ax2.set_ylabel("Temperatura (°C)", color="red")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Previsão de Temperatura e Precipitação - 2025")
plt.grid()
plt.show()

# 📊 Umidade vs Velocidade do Vento
plt.figure(figsize=(12, 6))
plt.plot(df_previsao["Mês"], df_previsao["UMIDADE RELATIVA DO AR, HORARIA (%)"], marker="o", linestyle="-", color="green", label="Umidade (%)")
plt.plot(df_previsao["Mês"], df_previsao["VENTO, VELOCIDADE HORARIA (m/s)"], marker="s", linestyle="-", color="purple", label="Velocidade do Vento (m/s)")
plt.xlabel("Mês")
plt.ylabel("Valores Médios")
plt.legend()
plt.title("Previsão de Umidade e Velocidade do Vento - 2025")
plt.grid()
plt.show()

# 📊 Ocorrências Operacionais
plt.figure(figsize=(12, 6))
sns.barplot(x=df_previsao["Mês"], y=df_previsao["Previsao_Ocorrencia"], palette="magma")
plt.title("Previsão de Ocorrências Operacionais - 2025")
plt.xlabel("Mês")
plt.ylabel("Número de Ocorrências")
plt.grid()
plt.show()

# 📊 Custo Estimado
plt.figure(figsize=(12, 6))
sns.barplot(x=df_previsao["Mês"], y=df_previsao["Custo_Estimado"], palette="viridis")
plt.title("Estimativa de Custo por Ocorrências Mensais - 2025")
plt.xlabel("Mês")
plt.ylabel("Custo Estimado (R$)")
plt.grid()
plt.show()

print("✅ Análises concluídas! Resultados disponíveis para avaliação.")