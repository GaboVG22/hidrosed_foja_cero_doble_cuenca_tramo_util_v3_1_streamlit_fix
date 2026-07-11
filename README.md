# HidroSed · Foja Cero v3: doble cuenca + tramo útil de cauce

Aplicación Streamlit para delimitar dos cuencas a partir de DEM y generar curvas de nivel solo en el tramo hidráulico útil.

## Entradas obligatorias

1. **PC cuenca soporte/general** en KMZ/KML con punto.
2. **PC hidrológico/cálculo** en KMZ/KML con punto.
3. **Eje del cauce** en KMZ/KML con línea.
4. **DEM**, mediante una de estas opciones:
   - descarga automática COP30/NASADEM/SRTM desde OpenTopography con API Key;
   - carga manual de DEM GeoTIFF.

## Entrada opcional

- Perfil topográfico longitudinal de respaldo en CSV/TXT/XLSX/XLS, con columnas de distancia/progresiva y cota/elevación.

## Cambio principal v3

La app ya no genera curvas en todo el eje del cauce. Ahora recorta internamente el eje y usa solo el tramo comprendido entre:

- PC cuenca soporte/general;
- PC hidrológico/cálculo.

Las curvas detalladas y las curvas topográficas de apoyo se generan solo en:

`intersección de ambas cuencas ∩ buffer del tramo útil PC soporte–PC hidrológico`

Esto reduce la sobrecarga, evita curvas fuera del sector de análisis y mejora la coherencia del KMZ para cuencas grandes.

## Salidas

- KMZ unificado con:
  - puntos de control;
  - puntos ajustados al drenaje;
  - cuenca soporte;
  - subcuenca hidrológica;
  - intersección de cuencas;
  - eje completo;
  - tramo útil PC soporte–PC hidrológico;
  - corredor de curvas;
  - curvas detalladas;
  - curvas de apoyo.
- JSON resumen técnico.
- Perfil longitudinal DEM del tramo útil.
- Plantilla CSV tipo HEC-RAS para cauce rectangular/trapecial.

## Streamlit Cloud

Main file path:

```text
app.py
```

Python:

```text
3.11
```

## Parámetros recomendados

Para una primera prueba estable:

```text
Resolución interna: 60 m
Margen DEM: 25 km
DEM parciales: 4
Buffer curvas detalladas: 500 m
Equidistancia curvas detalladas: 10 m
Curvas de apoyo: activadas
Buffer apoyo: 1500 m
Equidistancia apoyo: 50 m
```

Para cuencas mayores a 1000 km²:

```text
Resolución interna: 90 m o 120 m
Margen DEM: 40 a 80 km
DEM parciales: 4 a 8
Buffer curvas detalladas: 1000 a 2000 m
Equidistancia curvas detalladas: 25 a 50 m
Curvas de apoyo: 50 a 100 m
```
