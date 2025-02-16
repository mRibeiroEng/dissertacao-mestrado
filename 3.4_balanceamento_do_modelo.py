import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Carregar os dados fusionados
print("📂 Carregando `base_fusionada.csv`...")
df = pd.read_csv("base_fusionada.csv", delimiter=";", encoding="utf-8")
print(f"✅ Base carregada com {df.shape[0]} registros e {df.shape[1]} colunas.")

# 📊 Seleção de Features (adicionando novas variáveis)
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", 
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "VENTO, VELOCIDADE HORARIA (m/s)",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
    "RADIACAO GLOBAL (Kj/m²)"
]
target = "qtd_atividade_bin"

# 🚨 Verificar valores ausentes
df.dropna(subset=features + [target], inplace=True)
print(f"✅ Após remoção de valores ausentes, restam {df.shape[0]} registros.")

# 🏗️ Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, stratify=df[target], random_state=42)

# 🔄 Balanceamento de Classes
print("🔄 Aplicando SMOTE e undersampling para balanceamento...")
smote = SMOTE(sampling_strategy=0.5, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)
print(f"✅ Base balanceada: {X_train_res.shape[0]} registros após SMOTE e undersampling.")

# 🔄 Normalização\scaler = StandardScaler()
scaler = StandardScaler()  # 🔹 Definindo o scaler antes de usá-lo
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 📌 Modelos a testar
modelos = {
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=len(y_train_res) / sum(y_train_res == 1), random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 🎯 Treinamento e Avaliação
df_resultados = {}
print("\n🔬 Testando Modelos de Machine Learning...\n")
for nome, modelo in modelos.items():
    print(f"🚀 Treinando Modelo: {nome}")
    modelo.fit(X_train_res, y_train_res)
    y_pred = modelo.predict(X_test)
    
    acuracia = np.mean(y_pred == y_test)
    print(f"📊 Acurácia: {acuracia:.4f}")
    print(f"\n📊 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    df_resultados[nome] = {
        "Acurácia": acuracia
    }

# 📊 Comparação Final
resultados_df = pd.DataFrame(df_resultados).T
print("\n📊 Comparação Final entre Modelos:")
print(resultados_df)

# 📈 Gráfico de Comparação
plt.figure(figsize=(10, 5))
sns.barplot(x=resultados_df.index, y=resultados_df["Acurácia"], palette="viridis")
plt.title("Comparação de Modelos - Acurácia")
plt.ylabel("Acurácia")
plt.xticks(rotation=45)
plt.show()

print("✅ Ajustes concluídos! O modelo está pronto para análise final.")
