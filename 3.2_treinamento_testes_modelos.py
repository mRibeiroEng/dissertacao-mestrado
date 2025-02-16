# 📥 Importando Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# 📌 1️⃣ Carregar base de dados já fusionada
print("📂 Carregando `base_fusionada.csv`...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")

print(f"✅ Base carregada com {df.shape[0]} registros e {df.shape[1]} colunas.")

# 📌 2️⃣ Seleção de variáveis
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",  
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",  
    "UMIDADE RELATIVA DO AR, HORARIA (%)",  
    "VENTO, VELOCIDADE HORARIA (m/s)",  
    "valor_unitario"  
]
target = "qtd_atividade_bin"

# 🏗️ 3️⃣ Tratamento de valores ausentes (Preenchendo com a média)
print("🔄 Tratando valores ausentes...")

imputer = SimpleImputer(strategy="mean")  # Usa a média para preencher NaN
df[features] = imputer.fit_transform(df[features])  # Aplica a substituição

# 📌 4️⃣ Remover linhas onde `qtd_atividade_bin` está ausente
df = df.dropna(subset=[target])
print(f"✅ Após limpeza, restam {df.shape[0]} registros.")

# ✂️ 5️⃣ Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 📊 6️⃣ Aplicação de SMOTE para balanceamento
print("🔄 Aplicando SMOTE para balanceamento da base...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"✅ Base balanceada: {X_train_res.shape[0]} registros após SMOTE.")

# 🔄 7️⃣ Normalização dos dados
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 📌 8️⃣ Dicionário de Modelos a testar
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "Redes Neurais": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# 🎯 🔬 9️⃣ Treinamento e Avaliação dos Modelos
resultados = {}

print("\n🔬 Testando Modelos de Machine Learning...\n")
for nome, modelo in modelos.items():
    print(f"\n🚀 Treinando Modelo: {nome}")
    modelo.fit(X_train_res, y_train_res)
    y_pred = modelo.predict(X_test)
    
    acuracia = accuracy_score(y_test, y_pred)
    f1 = cross_val_score(modelo, X_train_res, y_train_res, cv=3, scoring='f1').mean()
    
    print(f"📊 Acurácia: {acuracia:.4f}")
    print(f"⚖️ F1-Score: {f1:.4f}")
    print("\n📊 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    resultados[nome] = {
        "Acurácia": acuracia,
        "F1-Score": f1
    }

# 📊 🔄 Comparação Final entre Modelos
resultados_df = pd.DataFrame(resultados).T
print("\n📊 Comparação Final entre Modelos:")
print(resultados_df)

# 📈 🔄 Gráfico de Comparação
plt.figure(figsize=(10, 5))
sns.barplot(x=resultados_df.index, y=resultados_df["F1-Score"], palette="viridis")
plt.title("Comparação de Modelos - F1-Score")
plt.ylabel("F1-Score")
plt.xticks(rotation=45)
plt.show()
