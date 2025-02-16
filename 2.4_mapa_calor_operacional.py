import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from folium.features import DivIcon  # Importação correta do DivIcon

# Etapa 1: Carregar a base tratada
try:
    df_tratada = pd.read_csv('base_operacional_tratada.csv', delimiter=';', encoding='utf-8')
    print("Base tratada carregada com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Etapa 2: Carregar o shapefile do estado de Goiás
try:
    goias_shape = gpd.read_file('shapefile/goias_shapefile.shp', encoding='utf-8')  # Garantir codificação correta
    print("Shapefile do estado de Goiás carregado com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o shapefile: {e}")
    exit()

# Definir o CRS original se não estiver definido
if goias_shape.crs is None:
    goias_shape.set_crs(epsg=4674, inplace=True)  # Definir o CRS original, ajuste se necessário

# Transformar para WGS84 (EPSG:4326) para compatibilidade com Folium
goias_shape = goias_shape.to_crs(epsg=4326)

# Etapa 3: Filtrar coordenadas válidas
df_tratada['latitude'] = pd.to_numeric(df_tratada['latitude'], errors='coerce')
df_tratada['longitude'] = pd.to_numeric(df_tratada['longitude'], errors='coerce')
geo_data = df_tratada.dropna(subset=['latitude', 'longitude'])

# Etapa 4: Criar o mapa de calor
# Converter shapefile para GeoJSON para sobreposição no folium
geojson_data = goias_shape.to_json()

# Criar o mapa base
m = folium.Map(location=[-16.3333, -49.6667], zoom_start=7)  # Coordenadas centrais de Goiás

# Adicionar o contorno do estado
folium.GeoJson(geojson_data, name="Goiás").add_to(m)

# Adicionar mapa de calor
heat_data = list(zip(geo_data['latitude'], geo_data['longitude']))
HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)

# Etapa 5: Adicionar nomes e marcadores dos municípios de interesse
municipios_interesse = ["Goiânia", "São Luís de Montes Belos", "Goianésia"]
municipios_shape = goias_shape[goias_shape['NM_MUN'].isin(municipios_interesse)]

# Calcular os centroides dos municípios selecionados
municipios_shape['centroid'] = municipios_shape.geometry.centroid
for _, row in municipios_shape.iterrows():
    try:
        centroid = row['centroid']
        municipio = row['NM_MUN']

        # Adicionar marcador para cada município
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.Icon(color="blue", icon="info-sign"),
            popup=f'<b>{municipio}</b>',
            tooltip=f'Clique para mais informações: {municipio}'
        ).add_to(m)

        # Adicionar o nome do município diretamente no mapa
        folium.map.Marker(
            location=[centroid.y, centroid.x],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12px; color: black;"><b>{municipio}</b></div>',
            ),
        ).add_to(m)
    except AttributeError as e:
        print(f"Erro ao adicionar nome ou marcador do município: {e}")

# Adicionar controle de camadas
folium.LayerControl().add_to(m)

# Salvar o mapa em um arquivo HTML
m.save('mapa_calor_goias_com_municipios_e_marcadores.html')
print("Mapa de calor gerado e salvo como 'mapa_calor_goias_com_municipios_e_marcadores.html'.")
