import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from joblib import dump

# 📂 Carregar base de dados
print("\n📂 Carregando `base_fusionada.csv`...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")
print(f"✅ Base carregada com {df.shape[0]} registros e {df.shape[1]} colunas.")

# 🔄 Remover valores ausentes
print("✅ Removendo valores ausentes...")
df.dropna(inplace=True)
print(f"✅ Após remoção de valores ausentes, restam {df.shape[0]} registros.")

# 📊 Seleção de Variáveis
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", 
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", 
    "UMIDADE RELATIVA DO AR, HORARIA (%)", 
    "VENTO, VELOCIDADE HORARIA (m/s)"
]
target = "qtd_atividade_bin"

X = df[features]
y = df[target]

# ✂️ Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"✅ Base separada: {X_train.shape[0]} treino / {X_test.shape[0]} teste")

# 🔄 Normalização
target_scaler = StandardScaler()
X_train = target_scaler.fit_transform(X_train)
X_test = target_scaler.transform(X_test)

# 🔄 Aplicação de Técnicas de Balanceamento Avançadas
print("🔄 Aplicando SMOTETomek para balanceamento avançado...")
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
print(f"✅ Base balanceada com SMOTETomek: {X_train_res.shape[0]} registros")

# 🚀 Treinar o Modelo Random Forest
print("\n🚀 Treinando Modelo: Random Forest")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_res, y_train_res)
y_pred = modelo_rf.predict(X_test)

# 📥 Salvando o modelo treinado para uso posterior
dump(modelo_rf, "modelo_random_forest.joblib")
print("✅ Modelo Random Forest salvo como `modelo_random_forest.joblib`!")

# 📊 Avaliação do Modelo
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"📊 Acurácia: {acc:.4f}")
print(f"⚖️ Balanced Accuracy: {bal_acc:.4f}")
print(f"📈 AUC-ROC: {roc_auc:.4f}")
print("\n📊 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# 📊 Plotando a Comparação
resultados = pd.DataFrame({"Métrica": ["Acurácia", "Balanced Accuracy", "AUC-ROC"],
                           "Valor": [acc, bal_acc, roc_auc]})
plt.figure(figsize=(10,5))
sns.barplot(x="Métrica", y="Valor", data=resultados, palette="viridis")
plt.title("Comparação de Métricas do Modelo")
plt.ylim(0, 1)
plt.show()

print("✅ Ajustes concluídos! O modelo está pronto para análise final.")
