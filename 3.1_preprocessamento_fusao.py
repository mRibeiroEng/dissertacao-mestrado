import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# 📌 1️⃣ Carregar bases de dados
print("📥 Carregando bases de dados...")
df_climatica = pd.read_csv("base_climatica_tratada.csv", delimiter=";", encoding="utf-8")
df_operacional = pd.read_csv("base_operacional_tratada.csv", delimiter=";", encoding="utf-8")

print(f"✅ Base climática carregada com {df_climatica.shape[0]} registros e {df_climatica.shape[1]} colunas.")
print(f"✅ Base operacional carregada com {df_operacional.shape[0]} registros e {df_operacional.shape[1]} colunas.")

# 📌 2️⃣ Converter colunas de data para datetime
print("📆 Convertendo colunas de data para o formato datetime...")
df_climatica["Data"] = pd.to_datetime(df_climatica["Data"], errors='coerce')
df_operacional["data_servico"] = pd.to_datetime(df_operacional["data_servico"], errors='coerce')

# 📌 3️⃣ Determinar a estação meteorológica mais próxima para cada ocorrência (Usando KDTree)
print("📍 Associando cada ocorrência à estação meteorológica mais próxima...")

# Criando matriz de coordenadas das estações meteorológicas
coord_estacoes = df_climatica[["LATITUDE", "LONGITUDE"]].to_numpy()
tree = cKDTree(coord_estacoes)  # Criando KDTree para busca rápida

# Criando matriz de coordenadas das ocorrências operacionais
coord_ocorrencias = df_operacional[["latitude", "longitude"]].dropna().to_numpy()

# Encontrando a estação mais próxima para cada ocorrência
distancias, indices = tree.query(coord_ocorrencias)

# Mapeando índices das estações às ocorrências
df_operacional.loc[df_operacional[["latitude", "longitude"]].dropna().index, "estacao_proxima"] = indices

# Juntando os dados climáticos às ocorrências operacionais
df_climatica = df_climatica.reset_index()
df_operacional = df_operacional.merge(df_climatica, left_on="estacao_proxima", right_on="index", how="left").drop(columns=["index", "estacao_proxima"])

print(f"✅ Estações associadas! {df_operacional.shape[0]} registros processados.")

# 📌 4️⃣ Criar variável alvo binária `qtd_atividade_bin`
print("🎯 Criando variável alvo binária `qtd_atividade_bin`...")

df_operacional["qtd_atividade"] = df_operacional["qtd_atividade"].astype(str).str.replace(r"\.", "", regex=True)  # Remove pontos (milhar)
df_operacional["qtd_atividade"] = df_operacional["qtd_atividade"].str.replace(",", ".", regex=True)  # Converte vírgula para ponto decimal
df_operacional["qtd_atividade"] = pd.to_numeric(df_operacional["qtd_atividade"], errors="coerce")  # Converte para float
df_operacional["qtd_atividade_bin"] = (df_operacional["qtd_atividade"] > 0).astype(int)

# 📌 5️⃣ Salvar base final fusionada
print("💾 Salvando base processada em `base_fusionada.csv`...")
df_operacional.to_csv("base_fusionada.csv", index=False, sep=";")

print(f"✅ Processamento concluído! A base fusionada contém {df_operacional.shape[0]} registros e {df_operacional.shape[1]} colunas.")
