import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 📂 Carregar a base para garantir que usamos os mesmos dados do treinamento
print("\n📂 Carregando `base_fusionada.csv` apenas para extrair a normalização...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")

# 📊 Seleção das mesmas features usadas no treinamento
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", 
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", 
    "UMIDADE RELATIVA DO AR, HORARIA (%)", 
    "VENTO, VELOCIDADE HORARIA (m/s)"
]

X = df[features].dropna()  # Removendo valores ausentes para evitar erro na normalização

# 🔄 Criar e treinar o scaler novamente
print("🔄 Criando e treinando o scaler novamente...")
scaler = StandardScaler()
scaler.fit(X)  # Apenas ajustamos o scaler aos dados, sem necessidade de reequilibrar

# 📥 Salvar o scaler para uso no arquivo 07
dump(scaler, "scaler.joblib")
print("✅ Scaler salvo como `scaler.joblib`!")
