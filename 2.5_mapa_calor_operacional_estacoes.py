import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap

# Etapa 1: Carregar a base tratada
try:
    df_tratada = pd.read_csv('base_operacional_tratada.csv', delimiter=';', encoding='utf-8')
    print("Base tratada carregada com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Etapa 2: Carregar o shapefile do estado de Goiás
try:
    goias_shape = gpd.read_file('shapefile/goias_shapefile.shp')  # Substitua pelo caminho do seu shapefile
    print("Shapefile do estado de Goiás carregado com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o shapefile: {e}")
    exit()

# Definir o CRS original se não estiver definido
if goias_shape.crs is None:
    goias_shape.set_crs(epsg=4674, inplace=True)  # Definir o CRS original, ajuste se necessário

# Transformar para WGS84 (EPSG:4326) para compatibilidade com Folium
goias_shape = goias_shape.to_crs(epsg=4326)

# Etapa 3: Filtrar coordenadas válidas para o mapa de calor
df_tratada['latitude'] = pd.to_numeric(df_tratada['latitude'], errors='coerce')
df_tratada['longitude'] = pd.to_numeric(df_tratada['longitude'], errors='coerce')
geo_data = df_tratada.dropna(subset=['latitude', 'longitude'])

# Criar o mapa base
m = folium.Map(location=[-16.3333, -49.6667], zoom_start=7)  # Coordenadas centrais de Goiás

# Adicionar o contorno do estado
geojson_data = goias_shape.to_json()
folium.GeoJson(geojson_data, name="Goiás").add_to(m)

# Adicionar mapa de calor
heat_data = list(zip(geo_data['latitude'], geo_data['longitude']))
HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)

# Adicionar nomes visíveis dos municípios de interesse (Goiânia, São Luís de Montes Belos e Goianésia)
municipios_interesse = {
    "GOIANIA": [-16.642778, -49.270278],  # Movido significativamente para a esquerda
    "SÃO LUÍS DE MONTES BELOS": [-16.521, -50.3725],
    "GOIANÉSIA": [-15.317, -49.1167]
}

for municipio, coord in municipios_interesse.items():
    folium.Marker(
        location=coord,
        icon=None,
        popup=None,
        tooltip=municipio
    ).add_to(m)

# Adicionar marcadores das Estações Automáticas de Clima com ícone de nuvem e cor vermelha
estacoes = [
    {"nome": "GOIANESIA", "latitude": -15.220278, "longitude": -48.99, "situacao": "Operante"},
    {"nome": "GOIANIA", "latitude": -16.642778, "longitude": -49.220278, "situacao": "Operante"},
    {"nome": "GOIAS", "latitude": -15.939722, "longitude": -50.141389, "situacao": "Operante"},
    {"nome": "IPORA", "latitude": -16.423056, "longitude": -51.148889, "situacao": "Operante"},
    {"nome": "PARAUNA", "latitude": -16.9625, "longitude": -50.425556, "situacao": "Operante"}
]

for estacao in estacoes:
    folium.Marker(
        location=[estacao["latitude"], estacao["longitude"]],
        icon=folium.Icon(color="red", icon="cloud", prefix="fa"),  # Ícone de nuvem com cor vermelha
        popup=f"<b>Estação:</b> {estacao['nome']}<br><b>Situação:</b> {estacao['situacao']}",
        tooltip=estacao['nome']
    ).add_to(m)

# Adicionar controle de camadas
folium.LayerControl().add_to(m)

# Salvar o mapa em um arquivo HTML
m.save('mapa_calor_goias_com_estacoes_e_ajuste_regional.html')
print("Mapa de calor gerado e salvo como 'mapa_calor_goias_com_estacoes_e_ajuste_regional.html'.")
