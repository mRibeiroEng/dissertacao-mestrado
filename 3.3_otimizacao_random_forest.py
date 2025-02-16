import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📂 Carregar base já processada
print("📂 Carregando `base_fusionada.csv`...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")
print(f"✅ Base carregada com {df.shape[0]} registros e {df.shape[1]} colunas.")

# 📊 Seleção de Variáveis
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", 
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)"
]
target = "qtd_atividade_bin"

# 🚨 Remover valores ausentes antes do treinamento
df = df.dropna(subset=features + [target])

# ✂️ Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 🔄 Normalização
print("🔄 Aplicando normalização nos dados...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔍 Otimização dos hiperparâmetros
print("🛠️ Iniciando otimização do modelo Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 🚀 Melhor Modelo
best_model = grid_search.best_estimator_
print(f"✅ Melhor Modelo: {grid_search.best_params_}")

# 📊 Avaliação do modelo otimizado
y_pred = best_model.predict(X_test)
print("📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("📊 Matriz de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 📈 Visualização da Importância das Features
print("📈 Plotando importância das variáveis...")
feature_importances = pd.DataFrame({'Variável': features, 'Importância': best_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importância', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Variável', data=feature_importances, palette='viridis')
plt.title("Importância das Variáveis no Modelo Random Forest")
plt.show()

print("✅ Otimização concluída! O modelo ajustado está pronto para previsão e análise.")
