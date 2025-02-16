import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 📂 Carregar base de dados
print("📂 Carregando `base_fusionada.csv`...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")
print(f"✅ Base carregada com {df.shape[0]} registros e {df.shape[1]} colunas.")

# 🔄 Remover valores ausentes
print("✅ Removendo valores ausentes...")
df.dropna(inplace=True)
print(f"✅ Após remoção de valores ausentes, restam {df.shape[0]} registros.")

# 📌 Definir Features e Target
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)"
]
target = "qtd_atividade_bin"

# ✂️ Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, stratify=df[target], random_state=42)

# 🔄 Aplicando SMOTE + Undersampling
print("🔄 Aplicando SMOTE e undersampling para balanceamento...")
over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)  # Aumenta a classe minoritária até 50% da majoritária
under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Reduz a classe majoritária
pipeline = Pipeline(steps=[('o', over_sampler), ('u', under_sampler)])
X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
print(f"✅ Base balanceada: {X_train_res.shape[0]} registros após SMOTE e undersampling.")

# 🔄 Normalização
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 📌 Treinar Modelos
print("\n🔬 Testando Modelos de Machine Learning...\n")
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
}

resultados = {}
for nome, modelo in modelos.items():
    print(f"🚀 Treinando Modelo: {nome}")
    modelo.fit(X_train_res, y_train_res)
    y_pred = modelo.predict(X_test)
    
    acuracia = accuracy_score(y_test, y_pred)
    print(f"📊 Acurácia: {acuracia:.4f}\n")
    print("📊 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    resultados[nome] = {"Acurácia": acuracia}

# 📊 Comparação Final
resultados_df = pd.DataFrame(resultados).T
plt.figure(figsize=(8, 4))
sns.barplot(x=resultados_df.index, y=resultados_df["Acurácia"], palette="viridis")
plt.title("Comparação de Modelos - Acurácia")
plt.ylabel("Acurácia")
plt.xticks(rotation=45)
plt.show()

print("✅ Ajustes concluídos! O modelo está pronto para análise final.")
