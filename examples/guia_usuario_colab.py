# WBGAPI360: Guia Completa de Usuario
# Cliente Profesional para Datos del Banco Mundial
# Autor: Maykol Medrano
# Version: 0.3.0
# ============================================
# Este notebook demuestra todas las funcionalidades de la libreria wbgapi360
# desde operaciones basicas hasta analisis avanzados y visualizaciones.

# %% [markdown]
# # WBGAPI360: Guia Completa de Usuario
# 
# **Cliente Profesional para Datos del Banco Mundial (Data360 API)**
# 
# Esta guia demuestra todas las funcionalidades de la libreria, organizadas de menor a mayor complejidad:
# 
# 1. **Instalacion y Configuracion**
# 2. **Busqueda de Indicadores**
# 3. **Descarga de Datos**
# 4. **Visualizaciones Basicas**
# 5. **Visualizaciones Avanzadas (Mapas)**
# 6. **Analisis Estadistico**
# 7. **Comparaciones entre Paises**
# 8. **Rankings Globales y Regionales**
# 9. **Casos de Uso Practicos**
# 
# ---

# %% [markdown]
# ## 1. Instalacion y Configuracion

# %%
# Instalacion desde PyPI
# Para funcionalidades basicas (busqueda y datos):
!pip install wbgapi360 -q

# Para visualizaciones (graficos y mapas):
!pip install wbgapi360[visual] -q

# %%
# Importar la libreria
import wbgapi360 as wb
import pandas as pd

# Verificar version instalada
print(f"Version instalada: {wb.__version__}")
print(f"Autor: {wb.__author__}")

# %% [markdown]
# ---
# ## 2. Busqueda de Indicadores
# 
# La funcion `search()` permite encontrar indicadores usando lenguaje natural.
# Utiliza un algoritmo de ranking inteligente que considera:
# - Coincidencia exacta de tokens
# - Similitud fuzzy (tolerante a errores tipograficos)
# - Prioridad por base de datos (WDI por defecto)

# %%
# 2.1 Busqueda basica: encontrar indicadores de PIB
resultados = wb.search("GDP")
print("Indicadores encontrados para 'GDP':\n")
for r in resultados[:5]:
    print(f"  - {r['code']}: {r['name']}")

# %%
# 2.2 Busqueda en espanol: inflacion
resultados_inflacion = wb.search("inflation", limit=5)
print("\nIndicadores de Inflacion:")
for r in resultados_inflacion:
    print(f"  - {r['code']}: {r['name']}")

# %%
# 2.3 Busqueda con errores tipograficos (el algoritmo fuzzy lo corrige)
# Escribimos "educcation" en lugar de "education"
resultados_edu = wb.search("educcation spending", limit=5)
print("\nIndicadores de Educacion (con typo corregido):")
for r in resultados_edu:
    print(f"  - {r['code']}: {r['name']}")

# %%
# 2.4 Busqueda por tema especifico
temas = ["poverty", "inequality", "unemployment", "life expectancy", "CO2 emissions"]

for tema in temas:
    res = wb.search(tema, limit=1)
    if res:
        print(f"{tema.upper():20} -> {res[0]['code']}")

# %% [markdown]
# ---
# ## 3. Descarga de Datos
# 
# La funcion `get_data()` descarga datos del Banco Mundial con auto-correccion inteligente.
# 
# **Parametros principales:**
# - `indicator`: Codigo del indicador (ej: "NY.GDP.MKTP.CD") o alias (ej: "GDP")
# - `economies`: Codigo(s) de pais ISO3 (ej: "CHL", ["PER", "MEX"])
# - `years`: Numero de anos historicos (defecto: 5)
# - `labels`: Si True, convierte codigos a nombres legibles
# - `as_json`: Si True, devuelve JSON en lugar de DataFrame

# %%
# 3.1 Descarga basica: PIB de Chile (ultimos 5 anos)
df_chile = wb.get_data(
    indicator="NY.GDP.MKTP.CD",
    economies="CHL"
)
print("PIB de Chile (USD corrientes):")
display(df_chile)

# %%
# 3.2 Descarga con multiples paises
# Alianza del Pacifico: Chile, Peru, Mexico, Colombia
df_alianza = wb.get_data(
    indicator="NY.GDP.PCAP.CD",
    economies=["CHL", "PER", "MEX", "COL"],
    years=10
)
print("PIB per capita - Alianza del Pacifico (10 anos):")
display(df_alianza)

# %%
# 3.3 Descarga con etiquetas legibles
df_labels = wb.get_data(
    indicator="SP.POP.TOTL",
    economies=["BRA", "ARG", "COL"],
    years=5,
    labels=True
)
print("Poblacion con nombres de paises:")
display(df_labels)

# %%
# 3.4 Descarga de multiples indicadores
# PIB + Poblacion + Esperanza de vida
indicadores = ["NY.GDP.MKTP.CD", "SP.POP.TOTL", "SP.DYN.LE00.IN"]
df_multi = wb.get_data(
    indicator=indicadores,
    economies=["CHL", "PER"],
    years=5
)
print("Multiples indicadores:")
display(df_multi)

# %%
# 3.5 Exportar datos como JSON (para APIs o JavaScript)
json_data = wb.get_data(
    indicator="FP.CPI.TOTL.ZG",
    economies="ARG",
    years=10,
    as_json=True
)
print("Inflacion Argentina (formato JSON):")
print(json_data[:500] + "...")

# %% [markdown]
# ---
# ## 4. Visualizaciones Basicas
# 
# La funcion `plot()` genera graficos con estetica estilo Financial Times.
# 
# **Tipos de graficos disponibles:**
# - `trend`: Lineas temporales
# - `bar`: Barras horizontales
# - `column`: Barras verticales
# - `scatter`: Dispersion
# - `dumbbell`: Comparacion de dos puntos
# - `stacked`: Barras apiladas
# - `area`: Area apilada
# - `bump`: Cambios de ranking
# - `donut`: Parte del todo (circular)
# - `treemap`: Jerarquia
# - `heatmap`: Mapa de calor

# %%
# 4.1 Grafico de tendencia (lineas)
df_trend = wb.get_data(
    indicator="NY.GDP.MKTP.KD.ZG",  # Crecimiento del PIB
    economies=["CHL", "PER", "COL", "MEX"],
    years=15,
    labels=True
)

path = wb.plot(
    chart_type="trend",
    data=df_trend,
    title="Crecimiento Economico",
    subtitle="Tasa de crecimiento del PIB real (%), 2009-2024"
)
print(f"Grafico guardado en: {path}")

# %%
# 4.2 Grafico de barras horizontales
df_bar = wb.get_data(
    indicator="NY.GDP.PCAP.CD",
    economies=["USA", "DEU", "JPN", "GBR", "FRA", "CHL", "MEX", "BRA"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="bar",
    data=df_bar,
    title="PIB per Capita 2023",
    subtitle="USD corrientes, paises seleccionados"
)
print(f"Grafico guardado en: {path}")

# %%
# 4.3 Grafico de dispersion (scatter)
# Relacion entre PIB per capita y esperanza de vida
df_scatter = wb.get_data(
    indicator=["NY.GDP.PCAP.CD", "SP.DYN.LE00.IN"],
    economies=["CHL", "ARG", "BRA", "MEX", "COL", "PER", "URY", "ECU", "BOL", "PRY"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="scatter",
    data=df_scatter,
    title="Desarrollo vs Longevidad",
    subtitle="Relacion entre PIB per capita y esperanza de vida, 2023"
)

# %%
# 4.4 Grafico Dumbbell (comparacion inicio vs fin)
# Comparar valores de 2010 vs 2023
df_2010 = wb.get_data("NY.GDP.PCAP.CD", ["CHL", "PER", "MEX", "COL"], years=1)
df_2010["year"] = 2010
df_2023 = wb.get_data("NY.GDP.PCAP.CD", ["CHL", "PER", "MEX", "COL"], years=1)
df_2023["year"] = 2023

# Combinar para dumbbell
df_dumbbell = pd.concat([df_2010, df_2023])

path = wb.plot(
    chart_type="dumbbell",
    data=df_dumbbell,
    title="Progreso en PIB per Capita",
    subtitle="Comparacion 2010 vs 2023 - Alianza del Pacifico"
)

# %%
# 4.5 Grafico de Area Apilada
df_area = wb.get_data(
    indicator="EG.USE.ELEC.KH.PC",
    economies=["CHL", "PER", "COL"],
    years=15,
    labels=True
)

path = wb.plot(
    chart_type="area",
    data=df_area,
    title="Consumo Electrico",
    subtitle="kWh per capita, evolucion historica"
)

# %%
# 4.6 Grafico Donut (part-to-whole)
# Composicion de poblacion regional
df_donut = wb.get_data(
    indicator="SP.POP.TOTL",
    economies=["BRA", "MEX", "ARG", "COL", "PER", "VEN", "CHL"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="donut",
    data=df_donut,
    title="Poblacion Latinoamericana",
    subtitle="Distribucion por pais, 2023"
)

# %%
# 4.7 Treemap (jerarquia visual)
path = wb.plot(
    chart_type="treemap",
    data=df_donut,
    title="Proporcion de Poblacion",
    subtitle="Principales economias de America Latina"
)

# %% [markdown]
# ---
# ## 5. Visualizaciones Avanzadas: Mapas
# 
# La libreria incluye 4 tipos de mapas coropleticos:
# - `map`: Mapa de calor geografico (gradiente)
# - `map_bubble`: Circulos proporcionales
# - `map_diverging`: Escala divergente (rojo-blanco-azul)
# - `map_categorical`: Categorias discretas
# 
# **Parametros especiales:**
# - `region`: Filtro geografico (latam, africa, asia, europe, mena, etc.)
# - `bbox`: Bounding box personalizado [lon_min, lat_min, lon_max, lat_max]
# - `bins`: Umbrales para categorias (map_categorical)
# - `category_labels`: Etiquetas para cada categoria

# %%
# 5.1 Mapa coropletico global
df_map_global = wb.get_data(
    indicator="NY.GDP.PCAP.CD",
    economies=["WLD"],  # Todos los paises
    years=1
)

path = wb.plot(
    chart_type="map",
    data=df_map_global,
    title="PIB per Capita Mundial",
    subtitle="USD corrientes, 2023"
)

# %%
# 5.2 Mapa regional: America Latina
df_latam = wb.get_data(
    indicator="SI.POV.NAHC",  # Poverty headcount
    economies=["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "URY", "VEN",
               "MEX", "GTM", "HND", "SLV", "NIC", "CRI", "PAN"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="map",
    data=df_latam,
    title="Pobreza en America Latina",
    subtitle="Porcentaje de poblacion bajo linea nacional de pobreza",
    region="latam"
)

# %%
# 5.3 Mapa de burbujas (circulos proporcionales)
df_pop = wb.get_data(
    indicator="SP.POP.TOTL",
    economies=["CHL", "ARG", "BRA", "PER", "COL", "MEX", "VEN", "ECU", "BOL", "URY"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="map_bubble",
    data=df_pop,
    title="Poblacion en Sudamerica",
    subtitle="Tamano proporcional a habitantes, 2023",
    region="southamerica"
)

# %%
# 5.4 Mapa divergente (crecimiento positivo/negativo)
df_growth = wb.get_data(
    indicator="NY.GDP.MKTP.KD.ZG",
    economies=["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "URY", "VEN"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="map_diverging",
    data=df_growth,
    title="Crecimiento del PIB",
    subtitle="Tasa anual (%), 2023 - Rojo=Negativo, Azul=Positivo",
    region="southamerica"
)

# %%
# 5.5 Mapa categorico (clasificacion por umbrales)
df_income = wb.get_data(
    indicator="NY.GDP.PCAP.CD",
    economies=["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "URY", "VEN",
               "MEX", "GTM", "HND", "SLV", "NIC", "CRI", "PAN"],
    years=1
)

# Clasificacion por nivel de ingreso (umbrales del Banco Mundial 2024)
path = wb.plot(
    chart_type="map_categorical",
    data=df_income,
    title="Clasificacion por Ingreso",
    subtitle="Segun umbrales del Banco Mundial, 2024",
    region="latam",
    bins=[0, 1135, 4465, 13845, 100000],
    category_labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"]
)

# %% [markdown]
# ---
# ## 6. Analisis Estadistico Avanzado
# 
# La libreria incluye funciones de analisis para series temporales.
# Estas funciones estan disponibles a traves del modulo MCP para uso con agentes AI,
# pero tambien pueden ser invocadas directamente.

# %%
# 6.1 Importar funciones de analisis desde el servidor MCP
from wbgapi360.mcp.server import _analyze_trend, _rank_countries, _compare_countries

# %%
# 6.2 Analisis de tendencia con estadisticas
# Crecimiento del PIB de Peru en los ultimos 20 anos
import asyncio
import json

async def analizar_tendencia():
    resultado = await _analyze_trend(
        indicator="NY.GDP.MKTP.KD.ZG",
        economy="PER",
        years=20,
        include_stats=True
    )
    return json.loads(resultado)

trend_peru = asyncio.get_event_loop().run_until_complete(analizar_tendencia())

print("Analisis de Tendencia: Crecimiento PIB Peru")
print("=" * 50)
print(f"Promedio anual:    {trend_peru['statistics']['mean']:.2f}%")
print(f"Volatilidad (std): {trend_peru['statistics']['std']:.2f}%")
print(f"Crecimiento total: {trend_peru['statistics']['total_growth']:.2f}%")
print(f"CAGR:              {trend_peru['statistics']['cagr']:.2f}%")
print(f"Tendencia:         {trend_peru['statistics']['trend']}")

# %%
# 6.3 Ranking de paises por indicador
async def obtener_ranking():
    resultado = await _rank_countries(
        indicator="NY.GDP.PCAP.CD",
        region="latam",
        top_n=10
    )
    return json.loads(resultado)

ranking_latam = asyncio.get_event_loop().run_until_complete(obtener_ranking())

print("\nTop 10 PIB per Capita en America Latina")
print("=" * 50)
for i, pais in enumerate(ranking_latam['ranking'], 1):
    print(f"{i:2}. {pais['economy']:15} ${pais['value']:,.0f}")

# %%
# 6.4 Comparacion normalizada entre paises
async def comparar_paises():
    resultado = await _compare_countries(
        economies=["CHL", "PER", "COL", "MEX"],
        indicators="NY.GDP.PCAP.CD",
        years=15,
        normalize=True  # Base 100 en el primer ano
    )
    return json.loads(resultado)

comparacion = asyncio.get_event_loop().run_until_complete(comparar_paises())

print("\nEvolucion Normalizada del PIB per Capita (Base 100)")
print("=" * 50)
df_comp = pd.DataFrame(comparacion['data'])
display(df_comp.tail())

# %% [markdown]
# ---
# ## 7. Casos de Uso Practicos
# 
# Ejemplos de analisis completos combinando busqueda, datos y visualizacion.

# %%
# Caso 1: Dashboard Economico de un Pais
# =====================================

pais = "CHL"
nombre_pais = "Chile"

# Indicadores clave
indicadores_clave = {
    "PIB (USD)": "NY.GDP.MKTP.CD",
    "PIB per capita": "NY.GDP.PCAP.CD",
    "Crecimiento PIB": "NY.GDP.MKTP.KD.ZG",
    "Inflacion": "FP.CPI.TOTL.ZG",
    "Desempleo": "SL.UEM.TOTL.ZS",
    "Poblacion": "SP.POP.TOTL"
}

print(f"Dashboard Economico: {nombre_pais}")
print("=" * 50)

for nombre, codigo in indicadores_clave.items():
    try:
        df = wb.get_data(codigo, pais, years=1)
        valor = df["OBS_VALUE"].iloc[-1] if not df.empty else "N/A"
        print(f"{nombre:20}: {valor:>15,.2f}" if isinstance(valor, (int, float)) else f"{nombre:20}: {valor}")
    except Exception as e:
        print(f"{nombre:20}: Error - {str(e)[:30]}")

# %%
# Caso 2: Comparativa Regional con Visualizacion
# ===============================================

# Desigualdad en America Latina (Indice de Gini)
df_gini = wb.get_data(
    indicator="SI.POV.GINI",
    economies=["CHL", "ARG", "BRA", "COL", "PER", "MEX", "CRI", "URY"],
    years=1,
    labels=True
)

if not df_gini.empty:
    path = wb.plot(
        chart_type="bar",
        data=df_gini,
        title="Desigualdad en America Latina",
        subtitle="Indice de Gini - Mayor valor indica mayor desigualdad"
    )
    print(f"Grafico generado: {path}")
else:
    print("No hay datos disponibles para el indicador Gini")

# %%
# Caso 3: Analisis de Comercio Internacional
# ==========================================

# Exportaciones como % del PIB
df_exports = wb.get_data(
    indicator="NE.EXP.GNFS.ZS",
    economies=["CHL", "PER", "MEX", "COL", "BRA", "ARG"],
    years=15,
    labels=True
)

path = wb.plot(
    chart_type="trend",
    data=df_exports,
    title="Apertura Comercial",
    subtitle="Exportaciones de bienes y servicios (% del PIB)"
)

# %%
# Caso 4: Indicadores de Desarrollo Humano
# ========================================

# Esperanza de vida al nacer
df_life = wb.get_data(
    indicator="SP.DYN.LE00.IN",
    economies=["CHL", "CRI", "URY", "ARG", "PAN", "MEX", "COL", "PER", "BRA", "ECU"],
    years=1,
    labels=True
)

path = wb.plot(
    chart_type="bar",
    data=df_life.sort_values("OBS_VALUE", ascending=True),
    title="Esperanza de Vida al Nacer",
    subtitle="Anos, principales economias latinoamericanas, 2022"
)

# %%
# Caso 5: Mapa de Cambio Climatico
# ================================

# Emisiones de CO2 per capita
df_co2 = wb.get_data(
    indicator="EN.ATM.CO2E.PC",
    economies=["USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER", "VEN", "ECU"],
    years=1
)

path = wb.plot(
    chart_type="map_diverging",
    data=df_co2,
    title="Emisiones de CO2 per Capita",
    subtitle="Toneladas metricas por persona, 2021",
    region="northamerica"
)

# %% [markdown]
# ---
# ## 8. Referencia Rapida de la API
# 
# ### Funciones Principales
# 
# | Funcion | Descripcion | Ejemplo |
# |---------|-------------|---------|
# | `search(query, limit)` | Buscar indicadores | `wb.search("GDP", 5)` |
# | `get_data(indicator, economies, years, labels, as_json)` | Descargar datos | `wb.get_data("NY.GDP.MKTP.CD", "CHL", 10)` |
# | `plot(chart_type, data, title, subtitle, **kwargs)` | Generar graficos | `wb.plot("trend", df, "Titulo")` |
# 
# ### Tipos de Graficos
# 
# | Tipo | Uso Recomendado |
# |------|-----------------|
# | `trend` | Series temporales, comparacion de lineas |
# | `bar` | Rankings, comparaciones horizontales |
# | `column` | Comparaciones verticales |
# | `scatter` | Correlaciones entre variables |
# | `dumbbell` | Comparacion de dos puntos en el tiempo |
# | `stacked` | Composicion (barras apiladas) |
# | `area` | Evolucion de composicion |
# | `bump` | Cambios de ranking en el tiempo |
# | `donut` | Proporcion del todo |
# | `treemap` | Jerarquia visual |
# | `heatmap` | Correlaciones/densidad |
# | `map` | Mapas coropleticos (gradiente) |
# | `map_bubble` | Mapas con circulos proporcionales |
# | `map_diverging` | Mapas con escala divergente |
# | `map_categorical` | Mapas con categorias discretas |
# 
# ### Regiones Predefinidas para Mapas
# 
# `latam`, `europe`, `africa`, `asia`, `mena`, `subsaharan`, `eastasia`, `oceania`, `northamerica`, `southamerica`, `caribbean`, `centralamerica`
# 
# ---
# 
# ## 9. Informacion Adicional
# 
# **Repositorio:** https://github.com/MaykolMedrano/mcp_wbgapi360
# 
# **PyPI:** https://pypi.org/project/wbgapi360/
# 
# **Licencia:** MIT
# 
# **Autor:** Maykol Medrano (mmedrano2@uc.cl)
