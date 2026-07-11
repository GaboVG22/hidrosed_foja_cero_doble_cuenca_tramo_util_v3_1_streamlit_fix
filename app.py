# -*- coding: utf-8 -*-
"""
HidroSed · Foja Cero v3: doble cuenca + tramo útil de cauce
Aplicación Streamlit liviana, sin pysheds/geopandas/scipy, con eje obligatorio, tramo útil entre puntos de control y control de calidad para cuencas grandes.

Objetivo:
- Ingresar PC cuenca soporte, PC hidrológico/cálculo y eje de cauce obligatorio.
- Perfil topográfico longitudinal opcional como respaldo de drenaje.
- Descargar o cargar DEM GeoTIFF.
- Delimitar dos polígonos de cuenca por D8 interno, restringiendo el ajuste de los puntos al corredor del eje.
- Validar especialmente cuencas grandes: borde DEM, eje contenido, puntos contenidos, acumulación y forma del polígono.
- Generar curvas de nivel solo en la intersección de ambas cuencas y en el corredor del tramo de eje comprendido entre PC cuenca soporte y PC hidrológico.
- Exportar KMZ unificado, perfil longitudinal DEM y plantilla CSV tipo HEC-RAS.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import zipfile
import heapq
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import streamlit as st

import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.features import shapes, geometry_mask
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, GeometryCollection, shape, mapping, box
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid

from pyproj import CRS, Transformer, Geod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuración Streamlit
# -----------------------------
st.set_page_config(
    page_title="HidroSed · Doble cuenca + tramo cauce",
    page_icon="💧",
    layout="wide",
)

GEOD = Geod(ellps="WGS84")

D8_OFFSETS = np.array([
    [0, 1],    # E
    [1, 1],    # SE
    [1, 0],    # S
    [1, -1],   # SW
    [0, -1],   # W
    [-1, -1],  # NW
    [-1, 0],   # N
    [-1, 1],   # NE
], dtype=np.int16)
D8_DIST_MULT = np.array([1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2)], dtype=np.float32)

# -----------------------------
# Estructuras
# -----------------------------
@dataclass
class BasinResult:
    name: str
    original_lonlat: Tuple[float, float]
    snapped_lonlat: Tuple[float, float]
    snapped_rowcol: Tuple[int, int]
    snap_distance_m: float
    snap_radius_m: float
    area_km2: float
    perimeter_km: float
    touches_dem_edge: bool
    shape_index: float
    confidence: float
    polygon_utm: object
    polygon_wgs84: object
    outlet_accumulation_cells: int
    warnings: List[str]

@dataclass
class DemInfo:
    path: str
    crs: str
    width: int
    height: int
    cell_size_m: float
    bounds_utm: Tuple[float, float, float, float]
    nodata: Optional[float]

# -----------------------------
# Utilidades KML/KMZ
# -----------------------------
def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def read_kml_or_kmz(uploaded) -> str:
    data = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("El KMZ no contiene archivo KML interno.")
            return zf.read(kml_names[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def _parse_coord_text(text: str) -> List[Tuple[float, float, float]]:
    coords = []
    if not text:
        return coords
    for part in text.replace("\n", " ").replace("\t", " ").split():
        vals = part.split(",")
        if len(vals) >= 2:
            try:
                lon = float(vals[0]); lat = float(vals[1]); alt = float(vals[2]) if len(vals) > 2 and vals[2] != "" else 0.0
                coords.append((lon, lat, alt))
            except Exception:
                continue
    return coords


def extract_kml_geometries(uploaded) -> Dict[str, List[Tuple[str, object]]]:
    """Devuelve geometrías WGS84: points, lines, polygons."""
    kml = read_kml_or_kmz(uploaded)
    root = ET.fromstring(kml.encode("utf-8"))
    geoms = {"points": [], "lines": [], "polygons": []}

    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue
        name = "sin_nombre"
        for child in pm:
            if _strip_ns(child.tag) == "name" and child.text:
                name = child.text.strip()
                break

        for elem in pm.iter():
            tag = _strip_ns(elem.tag)
            if tag == "Point":
                ctext = None
                for c in elem.iter():
                    if _strip_ns(c.tag) == "coordinates":
                        ctext = c.text; break
                coords = _parse_coord_text(ctext or "")
                if coords:
                    lon, lat, _ = coords[0]
                    geoms["points"].append((name, Point(lon, lat)))
            elif tag == "LineString":
                ctext = None
                for c in elem.iter():
                    if _strip_ns(c.tag) == "coordinates":
                        ctext = c.text; break
                coords = _parse_coord_text(ctext or "")
                if len(coords) >= 2:
                    geoms["lines"].append((name, LineString([(lon, lat) for lon, lat, _ in coords])))
            elif tag == "Polygon":
                rings = []
                for lr in elem.iter():
                    if _strip_ns(lr.tag) == "LinearRing":
                        ctext = None
                        for c in lr.iter():
                            if _strip_ns(c.tag) == "coordinates":
                                ctext = c.text; break
                        coords = _parse_coord_text(ctext or "")
                        if len(coords) >= 4:
                            rings.append([(lon, lat) for lon, lat, _ in coords])
                if rings:
                    try:
                        poly = Polygon(rings[0], holes=rings[1:] if len(rings) > 1 else None)
                        if poly.is_valid:
                            geoms["polygons"].append((name, poly))
                        else:
                            geoms["polygons"].append((name, make_valid(poly)))
                    except Exception:
                        pass
    return geoms


def first_point_from_upload(uploaded, label: str) -> Tuple[str, Point]:
    geoms = extract_kml_geometries(uploaded)
    if not geoms["points"]:
        raise ValueError(f"El archivo de {label} no contiene un punto válido.")
    return geoms["points"][0]


def first_line_from_upload(uploaded, label: str) -> Tuple[str, LineString]:
    geoms = extract_kml_geometries(uploaded)
    if not geoms["lines"]:
        raise ValueError(f"El archivo de {label} no contiene una línea válida.")
    return geoms["lines"][0]

# -----------------------------
# Proyección y geometría
# -----------------------------
def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def transformer_to(src: Union[str, CRS], dst: Union[str, CRS]) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def project_geom(geom, src: Union[str, CRS], dst: Union[str, CRS]):
    t = transformer_to(src, dst)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), geom)


def geom_area_km2_utm(geom) -> float:
    return float(geom.area) / 1_000_000.0 if geom and not geom.is_empty else 0.0


def geom_perim_km_utm(geom) -> float:
    return float(geom.length) / 1000.0 if geom and not geom.is_empty else 0.0


def shape_index(geom) -> float:
    if geom is None or geom.is_empty or geom.area <= 0:
        return 999.0
    return float((geom.length ** 2) / (4 * math.pi * geom.area))


def geodesic_bbox_area_km2(bounds: Tuple[float, float, float, float]) -> float:
    west, south, east, north = bounds
    lons = [west, east, east, west, west]
    lats = [south, south, north, north, south]
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(area) / 1_000_000.0

# -----------------------------
# DEM: descarga, mosaico y reproyección
# -----------------------------
def split_bbox(bounds: Tuple[float, float, float, float], n_tiles: int) -> List[Tuple[float, float, float, float]]:
    west, south, east, north = bounds
    n_tiles = max(1, int(n_tiles))
    cols = int(math.ceil(math.sqrt(n_tiles)))
    rows = int(math.ceil(n_tiles / cols))
    tiles = []
    dx = (east - west) / cols
    dy = (north - south) / rows
    for r in range(rows):
        for c in range(cols):
            if len(tiles) >= n_tiles:
                break
            w = west + c * dx; e = west + (c + 1) * dx
            s = south + r * dy; n = south + (r + 1) * dy
            # pequeña superposición para evitar grietas
            pad_x = dx * 0.002
            pad_y = dy * 0.002
            tiles.append((w - pad_x, s - pad_y, e + pad_x, n + pad_y))
    return tiles


def download_opentopo_tile(demtype: str, bounds: Tuple[float, float, float, float], api_key: str, out_path: str, timeout: int = 180) -> None:
    west, south, east, north = bounds
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": f"{south:.8f}",
        "north": f"{north:.8f}",
        "west": f"{west:.8f}",
        "east": f"{east:.8f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        msg = r.text[:500] if r.text else "sin detalle"
        raise RuntimeError(f"OpenTopography respondió {r.status_code}: {msg}")
    content_type = r.headers.get("content-type", "").lower()
    if "tif" not in content_type and "octet" not in content_type and len(r.content) < 1024:
        raise RuntimeError(f"Respuesta de OpenTopography no parece ser GeoTIFF: {r.text[:500]}")
    with open(out_path, "wb") as f:
        f.write(r.content)


def merge_tiles(tile_paths: List[str], out_path: str) -> str:
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": True,
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()
    return out_path


def create_bbox_from_inputs(pc1: Point, pc2: Point, axis: LineString, margin_km: float) -> Tuple[float, float, float, float]:
    lon0, lat0 = pc1.x, pc1.y
    crs_utm = utm_crs_from_lonlat(lon0, lat0)
    geoms = [pc1, pc2, axis]
    union_wgs = unary_union(geoms)
    union_utm = project_geom(union_wgs, "EPSG:4326", crs_utm)
    buffered_utm = union_utm.buffer(float(margin_km) * 1000.0)
    bbox_wgs = project_geom(buffered_utm.envelope, crs_utm, "EPSG:4326")
    return bbox_wgs.bounds  # west, south, east, north


def load_manual_dem(uploaded, tmpdir: str) -> str:
    out = os.path.join(tmpdir, "dem_manual.tif")
    with open(out, "wb") as f:
        f.write(uploaded.getvalue())
    return out


def reproject_dem_to_utm(src_path: str, dst_crs: CRS, resolution_m: float, tmpdir: str) -> Tuple[np.ndarray, Affine, CRS, Optional[float], str]:
    dst_path = os.path.join(tmpdir, "dem_utm_resample.tif")
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        left, bottom, right, top = src.bounds
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, left, bottom, right, top,
            resolution=resolution_m
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32",
            "count": 1,
            "nodata": -9999.0,
            "compress": "deflate",
            "driver": "GTiff",
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=-9999.0,
                resampling=Resampling.bilinear,
            )
    with rasterio.open(dst_path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        arr[arr == nodata] = np.nan
        return arr, ds.transform, ds.crs, nodata, dst_path

# -----------------------------
# Hidrología D8 liviana
# -----------------------------
def priority_flood_fill(dem: np.ndarray, nodata_mask: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Relleno de depresiones con gradiente epsilon. Apropiado para DEM remuestreado."""
    nrows, ncols = dem.shape
    filled = dem.copy().astype("float32")
    visited = np.zeros((nrows, ncols), dtype=bool)
    heap: List[Tuple[float, int, int]] = []

    valid = ~nodata_mask & np.isfinite(filled)
    if not np.any(valid):
        raise ValueError("DEM sin celdas válidas.")

    # Borde exterior y borde contra nodata
    candidates = np.zeros_like(valid, dtype=bool)
    candidates[0, :] = True; candidates[-1, :] = True; candidates[:, 0] = True; candidates[:, -1] = True
    # celdas válidas adyacentes a nodata actúan como borde hidrológico
    invalid = ~valid
    for dr, dc in D8_OFFSETS:
        src_r0 = max(0, -dr); src_r1 = nrows - max(0, dr)
        src_c0 = max(0, -dc); src_c1 = ncols - max(0, dc)
        dst_r0 = max(0, dr); dst_r1 = nrows - max(0, -dr)
        dst_c0 = max(0, dc); dst_c1 = ncols - max(0, -dc)
        candidates[src_r0:src_r1, src_c0:src_c1] |= invalid[dst_r0:dst_r1, dst_c0:dst_c1]

    edge_cells = np.argwhere(candidates & valid)
    for r, c in edge_cells:
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in D8_OFFSETS:
            rr = r + int(dr); cc = c + int(dc)
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            if visited[rr, cc] or not valid[rr, cc]:
                continue
            visited[rr, cc] = True
            val = float(filled[rr, cc])
            if val <= elev:
                val = elev + eps
                filled[rr, cc] = val
            heapq.heappush(heap, (val, rr, cc))

    filled[~valid] = np.nan
    return filled


def compute_d8_flow(filled: np.ndarray, transform: Affine) -> Tuple[np.ndarray, np.ndarray]:
    nrows, ncols = filled.shape
    valid = np.isfinite(filled)
    flow = np.full((nrows, ncols), -1, dtype=np.int8)
    best = np.zeros((nrows, ncols), dtype=np.float32)
    cell = abs(float(transform.a))

    for d, (dr, dc) in enumerate(D8_OFFSETS):
        dr = int(dr); dc = int(dc)
        r0 = max(0, -dr); r1 = nrows - max(0, dr)
        c0 = max(0, -dc); c1 = ncols - max(0, dc)
        rr0 = max(0, dr); rr1 = nrows - max(0, -dr)
        cc0 = max(0, dc); cc1 = ncols - max(0, -dc)
        center = filled[r0:r1, c0:c1]
        neigh = filled[rr0:rr1, cc0:cc1]
        ok = valid[r0:r1, c0:c1] & valid[rr0:rr1, cc0:cc1]
        slope = np.where(ok, (center - neigh) / (cell * D8_DIST_MULT[d]), -np.inf)
        sub_best = best[r0:r1, c0:c1]
        sub_flow = flow[r0:r1, c0:c1]
        mask = slope > sub_best
        sub_best[mask] = slope[mask]
        sub_flow[mask] = d
        best[r0:r1, c0:c1] = sub_best
        flow[r0:r1, c0:c1] = sub_flow
    flow[~valid] = -1
    return flow, valid


def receiver_flat_indices(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    nrows, ncols = flow.shape
    n = nrows * ncols
    rec = np.full(n, -1, dtype=np.int64)
    rows, cols = np.where((flow >= 0) & valid)
    dirs = flow[rows, cols].astype(int)
    r2 = rows + D8_OFFSETS[dirs, 0]
    c2 = cols + D8_OFFSETS[dirs, 1]
    inside = (r2 >= 0) & (r2 < nrows) & (c2 >= 0) & (c2 < ncols) & valid[r2, c2]
    src_idx = rows[inside] * ncols + cols[inside]
    dst_idx = r2[inside] * ncols + c2[inside]
    rec[src_idx] = dst_idx
    return rec


def compute_accumulation(rec: np.ndarray, valid: np.ndarray) -> np.ndarray:
    nrows, ncols = valid.shape
    n = nrows * ncols
    valid_flat = valid.ravel()
    indeg = np.zeros(n, dtype=np.int32)
    srcs = np.where((rec >= 0) & valid_flat)[0]
    dsts = rec[srcs]
    np.add.at(indeg, dsts, 1)
    acc = np.zeros(n, dtype=np.float64)
    acc[valid_flat] = 1.0
    # queue cells with no donors
    from collections import deque
    q = deque(np.where(valid_flat & (indeg == 0))[0].tolist())
    processed = 0
    while q:
        i = q.popleft(); processed += 1
        j = rec[i]
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(int(j))
    # If cycles remain, acc is still usable but warn indirectly by sinks.
    return acc.reshape((nrows, ncols)).astype(np.float32)


def world_to_rowcol(transform: Affine, x: float, y: float) -> Tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))


def rowcol_to_world(transform: Affine, row: int, col: int) -> Tuple[float, float]:
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def snap_to_accumulation(point_utm: Point, acc: np.ndarray, valid: np.ndarray, transform: Affine, radius_m: float) -> Tuple[int, int, float, int]:
    row, col = world_to_rowcol(transform, point_utm.x, point_utm.y)
    nrows, ncols = acc.shape
    cell = abs(float(transform.a))
    rad = max(1, int(math.ceil(radius_m / cell)))
    r0 = max(0, row - rad); r1 = min(nrows, row + rad + 1)
    c0 = max(0, col - rad); c1 = min(ncols, col + rad + 1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError("El punto de control queda fuera del DEM reproyectado.")
    sub_acc = acc[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    rr, cc = np.indices(sub_acc.shape)
    dist_cells = np.sqrt((rr + r0 - row) ** 2 + (cc + c0 - col) ** 2)
    within = (dist_cells * cell <= radius_m) & sub_valid
    if not np.any(within):
        raise ValueError("No hay celdas válidas dentro del radio de ajuste al cauce.")
    sub_acc[~within] = -np.inf
    idx = int(np.nanargmax(sub_acc))
    sr, sc = np.unravel_index(idx, sub_acc.shape)
    out_r = r0 + int(sr); out_c = c0 + int(sc)
    sx, sy = rowcol_to_world(transform, out_r, out_c)
    dist = point_utm.distance(Point(sx, sy))
    return out_r, out_c, float(dist), int(acc[out_r, out_c])




def snap_to_accumulation_near_axis(
    point_utm: Point,
    acc: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    radius_m: float,
    axis_utm: LineString,
    axis_buffer_m: float,
) -> Tuple[int, int, float, int, float]:
    """Ajusta el punto al drenaje, pero solo acepta celdas dentro de un corredor alrededor del eje.

    Esto evita uno de los errores más frecuentes: que el punto sea atraído por una quebrada vecina
    de mayor acumulación, generando un polígono angosto o una cuenca equivocada.
    """
    row, col = world_to_rowcol(transform, point_utm.x, point_utm.y)
    nrows, ncols = acc.shape
    cell = abs(float(transform.a))
    rad = max(1, int(math.ceil(radius_m / cell)))
    r0 = max(0, row - rad); r1 = min(nrows, row + rad + 1)
    c0 = max(0, col - rad); c1 = min(ncols, col + rad + 1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError("El punto de control queda fuera del DEM reproyectado.")

    sub_acc = acc[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    rr, cc = np.indices(sub_acc.shape)
    dist_cells = np.sqrt((rr + r0 - row) ** 2 + (cc + c0 - col) ** 2)
    within_radius = (dist_cells * cell <= radius_m) & sub_valid

    sub_transform = transform * Affine.translation(c0, r0)
    axis_corridor = axis_utm.buffer(float(axis_buffer_m))
    within_axis = geometry_mask([mapping(axis_corridor)], out_shape=sub_acc.shape, transform=sub_transform, invert=True, all_touched=True)
    candidate_mask = within_radius & within_axis
    if not np.any(candidate_mask):
        raise ValueError(
            f"No se encontró celda de drenaje válida dentro de {radius_m:.0f} m del punto y {axis_buffer_m:.0f} m del eje. "
            "Aumenta el radio de ajuste o revisa que el eje del cauce coincida con el drenaje real."
        )

    # Score: acumulación domina; distancia al punto resuelve empates y evita saltos innecesarios.
    score = np.full(sub_acc.shape, -np.inf, dtype=np.float64)
    score[candidate_mask] = np.log1p(np.maximum(sub_acc[candidate_mask], 0.0)) - 0.002 * (dist_cells[candidate_mask] * cell / max(cell, 1.0))
    idx = int(np.nanargmax(score))
    sr, sc = np.unravel_index(idx, score.shape)
    out_r = r0 + int(sr); out_c = c0 + int(sc)
    sx, sy = rowcol_to_world(transform, out_r, out_c)
    pt_snap = Point(sx, sy)
    dist_pc = point_utm.distance(pt_snap)
    dist_axis = axis_utm.distance(pt_snap)
    return out_r, out_c, float(dist_pc), int(acc[out_r, out_c]), float(dist_axis)


def line_inside_fraction(line_utm: LineString, polygon_utm) -> float:
    """Fracción de la longitud de una línea contenida dentro de un polígono."""
    if line_utm is None or line_utm.is_empty or line_utm.length <= 0 or polygon_utm is None or polygon_utm.is_empty:
        return 0.0
    try:
        inter = line_utm.intersection(polygon_utm)
        return float(inter.length / line_utm.length)
    except Exception:
        return 0.0


def point_margin_to_polygon_boundary(point_utm: Point, polygon_utm) -> float:
    if polygon_utm is None or polygon_utm.is_empty:
        return -1.0
    if not polygon_utm.contains(point_utm) and not polygon_utm.touches(point_utm):
        return -float(point_utm.distance(polygon_utm))
    return float(point_utm.distance(polygon_utm.boundary))



def _append_unique_coord(coords_out: List[Tuple[float, float]], pt: Point, tol: float = 1e-8) -> None:
    """Agrega coordenada evitando duplicados consecutivos."""
    xy = (float(pt.x), float(pt.y))
    if not coords_out:
        coords_out.append(xy)
        return
    if abs(coords_out[-1][0] - xy[0]) > tol or abs(coords_out[-1][1] - xy[1]) > tol:
        coords_out.append(xy)


def substring_linestring_by_distance(line: LineString, start_m: float, end_m: float) -> LineString:
    """Extrae un tramo de LineString entre dos distancias acumuladas.

    Se implementa internamente para no depender de shapely.ops.substring.
    """
    if line is None or line.is_empty or line.length <= 0:
        return LineString()
    total = float(line.length)
    start = max(0.0, min(float(start_m), total))
    end = max(0.0, min(float(end_m), total))
    if end < start:
        start, end = end, start
    if end - start <= 1e-6:
        return LineString()

    coords = [(float(x), float(y)) for x, y, *rest in line.coords]
    out: List[Tuple[float, float]] = []
    _append_unique_coord(out, line.interpolate(start))

    cum0 = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        cum1 = cum0 + seg_len
        if seg_len <= 0:
            cum0 = cum1
            continue
        if cum1 <= start:
            cum0 = cum1
            continue
        if cum0 >= end:
            break
        # Si el vértice final del segmento cae dentro del rango, se conserva.
        if start < cum1 < end:
            _append_unique_coord(out, Point(b[0], b[1]))
        cum0 = cum1

    _append_unique_coord(out, line.interpolate(end))
    if len(out) < 2:
        return LineString()
    return LineString(out)


def axis_segment_between_control_points(axis_utm: LineString, pc_cuenca_utm: Point, pc_hidro_utm: Point, min_length_m: float = 50.0) -> Tuple[LineString, Dict[str, float]]:
    """Obtiene el tramo útil del eje comprendido entre PC cuenca soporte y PC hidrológico.

    El tramo se define por la proyección ortogonal de ambos puntos de control sobre el eje.
    Esto permite que las curvas de nivel se generen solo en el tramo hidráulico real a modelar.
    """
    if axis_utm is None or axis_utm.is_empty or axis_utm.length <= 0:
        raise ValueError("El eje del cauce está vacío o no tiene longitud válida.")
    d_cuenca = float(axis_utm.project(pc_cuenca_utm))
    d_hidro = float(axis_utm.project(pc_hidro_utm))
    start = min(d_cuenca, d_hidro)
    end = max(d_cuenca, d_hidro)
    seg = substring_linestring_by_distance(axis_utm, start, end)
    if seg.is_empty or seg.length < float(min_length_m):
        raise ValueError(
            "El tramo del eje entre PC cuenca soporte y PC hidrológico es demasiado corto. "
            "Revisa si ambos puntos caen sobre el mismo extremo del eje o si el eje no corresponde al cauce analizado."
        )
    info = {
        "dist_pc_cuenca_al_eje_m": float(pc_cuenca_utm.distance(axis_utm)),
        "dist_pc_hidro_al_eje_m": float(pc_hidro_utm.distance(axis_utm)),
        "progresiva_pc_cuenca_m": d_cuenca,
        "progresiva_pc_hidro_m": d_hidro,
        "inicio_tramo_m": start,
        "fin_tramo_m": end,
        "longitud_eje_total_m": float(axis_utm.length),
        "longitud_tramo_util_m": float(seg.length),
        "porcentaje_eje_usado_curvas": float(100.0 * seg.length / axis_utm.length) if axis_utm.length > 0 else 0.0,
    }
    return seg, info

def sample_dem_profile_along_axis(axis_utm: LineString, dem: np.ndarray, transform: Affine, step_m: float) -> pd.DataFrame:
    """Muestrea el DEM a lo largo del eje obligatorio. Devuelve distancia acumulada y cota DEM."""
    if axis_utm is None or axis_utm.is_empty or axis_utm.length <= 0:
        return pd.DataFrame(columns=["dist_m", "dist_km", "elev_dem_m"])
    step_m = max(float(step_m), abs(float(transform.a)))
    distances = np.arange(0.0, axis_utm.length + step_m * 0.5, step_m)
    rows = []
    nrows, ncols = dem.shape
    for d in distances:
        pt = axis_utm.interpolate(min(float(d), axis_utm.length))
        r, c = world_to_rowcol(transform, pt.x, pt.y)
        z = np.nan
        if 0 <= r < nrows and 0 <= c < ncols:
            z = float(dem[r, c]) if np.isfinite(dem[r, c]) else np.nan
        rows.append({"dist_m": float(d), "dist_km": float(d) / 1000.0, "x_utm": pt.x, "y_utm": pt.y, "elev_dem_m": z})
    return pd.DataFrame(rows)


def read_reference_longitudinal_profile(uploaded) -> Optional[pd.DataFrame]:
    """Lee perfil longitudinal opcional desde CSV/TXT/XLSX.
    Busca columnas de distancia y cota. Si no reconoce nombres, usa las dos primeras columnas numéricas.
    """
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(data))
        else:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception:
        # fallback para CSV con punto y coma o tabulación
        try:
            df = pd.read_csv(io.BytesIO(data), sep=";")
        except Exception as e:
            raise ValueError(f"No se pudo leer perfil longitudinal de respaldo: {e}")
    if df.empty:
        raise ValueError("El perfil longitudinal de respaldo está vacío.")
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}
    dist_col = None; elev_col = None
    for c, lc in lower.items():
        if any(k in lc for k in ["dist", "km", "progres", "absc", "station", "chain", "long"]):
            dist_col = c; break
    for c, lc in lower.items():
        if any(k in lc for k in ["cota", "elev", "alt", "z", "msnm", "m.s.n.m"]):
            elev_col = c; break
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    if dist_col is None or elev_col is None:
        nums = []
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() >= 2:
                nums.append(c)
        if len(nums) >= 2:
            dist_col = dist_col or nums[0]
            elev_col = elev_col or nums[1]
        else:
            raise ValueError("No se identificaron columnas numéricas de distancia y cota en el perfil.")
    out = pd.DataFrame({
        "dist_raw": pd.to_numeric(df[dist_col], errors="coerce"),
        "elev_ref_m": pd.to_numeric(df[elev_col], errors="coerce"),
    }).dropna()
    if out.empty:
        raise ValueError("El perfil no contiene pares distancia-cota válidos.")
    # Si la columna parece estar en km, convertir a m. Detecta por nombre o magnitud pequeña.
    lc = lower.get(dist_col, "")
    if "km" in lc and "m" not in lc.replace("km", ""):
        out["dist_m"] = out["dist_raw"] * 1000.0
    else:
        out["dist_m"] = out["dist_raw"].astype(float)
        # Si todas las distancias son muy pequeñas y crecientes, probablemente están en km.
        if out["dist_m"].max() < 300 and out["dist_m"].max() > 0:
            out["dist_m"] = out["dist_m"] * 1000.0
    out = out.sort_values("dist_m").drop_duplicates("dist_m")
    return out[["dist_m", "elev_ref_m"]]


def compare_dem_profile_with_reference(dem_profile: pd.DataFrame, ref_profile: Optional[pd.DataFrame]) -> Dict[str, object]:
    if ref_profile is None or dem_profile.empty:
        return {"available": False}
    dem_ok = dem_profile.dropna(subset=["elev_dem_m", "dist_m"])
    ref_ok = ref_profile.dropna(subset=["elev_ref_m", "dist_m"])
    if len(dem_ok) < 3 or len(ref_ok) < 3:
        return {"available": False, "warning": "Perfil con muy pocos puntos válidos para comparar."}
    d = dem_ok["dist_m"].to_numpy(dtype=float)
    z = dem_ok["elev_dem_m"].to_numpy(dtype=float)
    rd = ref_ok["dist_m"].to_numpy(dtype=float)
    rz = ref_ok["elev_ref_m"].to_numpy(dtype=float)
    max_common = min(float(np.nanmax(d)), float(np.nanmax(rd)))
    mask = (d >= float(np.nanmin(rd))) & (d <= max_common)
    if mask.sum() < 3:
        return {"available": False, "warning": "El rango de distancias del perfil de respaldo no coincide con el eje."}
    zi = z[mask]
    di = d[mask]
    ref_forward = np.interp(di, rd, rz)
    # También compara perfil invertido, por si el eje fue dibujado en sentido contrario.
    rd_rev = float(np.nanmax(rd)) - rd
    order = np.argsort(rd_rev)
    ref_reverse = np.interp(di, rd_rev[order], rz[order])
    rmse_f = float(np.sqrt(np.nanmean((zi - ref_forward) ** 2)))
    rmse_r = float(np.sqrt(np.nanmean((zi - ref_reverse) ** 2)))
    if rmse_r < rmse_f:
        bias = float(np.nanmean(zi - ref_reverse)); orientation = "perfil de respaldo parece invertido respecto del eje"; rmse = rmse_r
    else:
        bias = float(np.nanmean(zi - ref_forward)); orientation = "perfil de respaldo en mismo sentido aparente que el eje"; rmse = rmse_f
    return {"available": True, "n_compare": int(mask.sum()), "rmse_m": rmse, "bias_m": bias, "orientation_note": orientation}

def upstream_mask_from_outlet(rec: np.ndarray, valid: np.ndarray, outlet_row: int, outlet_col: int) -> np.ndarray:
    nrows, ncols = valid.shape
    outlet = outlet_row * ncols + outlet_col
    valid_flat = valid.ravel()
    srcs = np.where((rec >= 0) & valid_flat)[0]
    dsts = rec[srcs]
    order = np.argsort(dsts, kind="mergesort")
    dst_sorted = dsts[order]
    src_sorted = srcs[order]
    mask_flat = np.zeros(nrows * ncols, dtype=bool)
    stack = [int(outlet)]
    mask_flat[outlet] = True
    while stack:
        node = stack.pop()
        lo = np.searchsorted(dst_sorted, node, side="left")
        hi = np.searchsorted(dst_sorted, node, side="right")
        for donor in src_sorted[lo:hi]:
            donor = int(donor)
            if not mask_flat[donor]:
                mask_flat[donor] = True
                stack.append(donor)
    return mask_flat.reshape((nrows, ncols))


def polygonize_mask(mask: np.ndarray, transform: Affine) -> object:
    arr = mask.astype(np.uint8)
    geoms = []
    for geom, val in shapes(arr, mask=mask, transform=transform):
        if int(val) == 1:
            try:
                geoms.append(shape(geom))
            except Exception:
                pass
    if not geoms:
        return Polygon()
    poly = unary_union(geoms)
    if not poly.is_valid:
        poly = make_valid(poly)
    # eliminar polígonos muy pequeños si aparece multipolígono
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def mask_touches_edge(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return True
    return bool(np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))


def evaluate_confidence(area_km2: float, snap_distance_m: float, snap_radius_m: float, touches_edge: bool, shape_idx: float, acc_cells: int, point_inside: bool) -> Tuple[float, List[str]]:
    score = 100.0
    warnings = []
    if touches_edge:
        score -= 35; warnings.append("La cuenca toca el borde del DEM: ampliar área de descarga.")
    if not point_inside:
        score -= 30; warnings.append("El punto original no queda dentro del polígono delimitado.")
    if area_km2 < 0.2:
        score -= 30; warnings.append("Área muy pequeña; posible ajuste a drenaje equivocado.")
    if snap_radius_m > 0:
        rel = snap_distance_m / snap_radius_m
        if rel > 0.85:
            score -= 20; warnings.append("El punto ajustado quedó muy cerca del límite del radio de búsqueda.")
        elif rel > 0.50:
            score -= 10; warnings.append("El punto ajustado se alejó de forma relevante del punto original.")
    if shape_idx > 20:
        score -= 30; warnings.append("Polígono extremadamente alargado/fraccionado; revisar delimitación.")
    elif shape_idx > 8:
        score -= 12; warnings.append("Polígono muy alargado; revisar coherencia con el cauce.")
    if acc_cells < 100:
        score -= 20; warnings.append("Acumulación en punto de salida muy baja; posible punto fuera del cauce.")
    return max(0.0, min(100.0, score)), warnings


def delineate_basin_candidates(name: str, pc_wgs84: Point, dem: np.ndarray, transform: Affine, crs_utm: CRS, rec: np.ndarray, acc: np.ndarray, valid: np.ndarray, snap_radii: List[float], min_area_km2: float = 0.0, axis_utm: Optional[LineString] = None, axis_buffer_m: float = 500.0, hard_axis_snap: bool = True) -> BasinResult:
    pc_utm = project_geom(pc_wgs84, "EPSG:4326", crs_utm)
    candidates: List[BasinResult] = []
    to_wgs = transformer_to(crs_utm, "EPSG:4326")
    for radius in snap_radii:
        try:
            axis_dist = None
            if axis_utm is not None:
                try:
                    sr, sc, dist, acc_cells, axis_dist = snap_to_accumulation_near_axis(pc_utm, acc, valid, transform, float(radius), axis_utm, axis_buffer_m)
                except Exception:
                    if hard_axis_snap:
                        raise
                    sr, sc, dist, acc_cells = snap_to_accumulation(pc_utm, acc, valid, transform, float(radius))
            else:
                sr, sc, dist, acc_cells = snap_to_accumulation(pc_utm, acc, valid, transform, float(radius))
            mask = upstream_mask_from_outlet(rec, valid, sr, sc)
            poly_utm = polygonize_mask(mask, transform)
            if poly_utm.is_empty:
                continue
            area = geom_area_km2_utm(poly_utm)
            if area < min_area_km2:
                pass
            perim = geom_perim_km_utm(poly_utm)
            si = shape_index(poly_utm)
            touches = mask_touches_edge(mask)
            sx, sy = rowcol_to_world(transform, sr, sc)
            snapped_wgs = to_wgs.transform(sx, sy)
            poly_wgs = project_geom(poly_utm, crs_utm, "EPSG:4326")
            point_inside = poly_utm.buffer(2 * abs(transform.a)).contains(pc_utm) or poly_utm.touches(pc_utm)
            conf, warns = evaluate_confidence(area, dist, float(radius), touches, si, acc_cells, point_inside)
            if axis_dist is not None:
                if axis_dist > axis_buffer_m:
                    conf -= 25
                    warns.append(f"Punto ajustado quedó a {axis_dist:.0f} m del eje, fuera de tolerancia.")
                else:
                    warns.append(f"Ajuste controlado por eje: punto ajustado a {axis_dist:.0f} m del eje.")
            conf = max(0.0, min(100.0, float(conf)))
            candidates.append(BasinResult(
                name=name,
                original_lonlat=(pc_wgs84.x, pc_wgs84.y),
                snapped_lonlat=(float(snapped_wgs[0]), float(snapped_wgs[1])),
                snapped_rowcol=(int(sr), int(sc)),
                snap_distance_m=float(dist),
                snap_radius_m=float(radius),
                area_km2=float(area),
                perimeter_km=float(perim),
                touches_dem_edge=bool(touches),
                shape_index=float(si),
                confidence=float(conf),
                polygon_utm=poly_utm,
                polygon_wgs84=poly_wgs,
                outlet_accumulation_cells=int(acc_cells),
                warnings=warns,
            ))
        except Exception as e:
            continue
    if not candidates:
        raise RuntimeError(f"No se pudo delimitar una cuenca candidata para {name}.")
    # Se privilegia alta confianza, luego área razonable y acumulación.
    candidates.sort(key=lambda c: (c.confidence, math.log1p(c.outlet_accumulation_cells), c.area_km2), reverse=True)
    return candidates[0]

# -----------------------------
# Curvas de nivel en corredor/intersección
# -----------------------------
def crop_array_to_geom(dem: np.ndarray, transform: Affine, geom_utm, pad_m: float = 100.0):
    if geom_utm.is_empty:
        raise ValueError("Geometría de recorte vacía.")
    minx, miny, maxx, maxy = geom_utm.bounds
    win = from_bounds(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, transform)
    nrows, ncols = dem.shape
    row_off = max(0, int(math.floor(win.row_off)))
    col_off = max(0, int(math.floor(win.col_off)))
    height = min(nrows - row_off, int(math.ceil(win.height)) + 2)
    width = min(ncols - col_off, int(math.ceil(win.width)) + 2)
    if height <= 2 or width <= 2:
        raise ValueError("El recorte del DEM queda sin celdas suficientes.")
    sub = dem[row_off:row_off + height, col_off:col_off + width]
    sub_transform = transform * Affine.translation(col_off, row_off)
    mask_inside = geometry_mask([mapping(geom_utm)], out_shape=sub.shape, transform=sub_transform, invert=True, all_touched=True)
    return sub.copy(), sub_transform, mask_inside


def generate_contours(dem: np.ndarray, transform: Affine, clip_geom_utm, interval_m: float, simplification_m: float, crs_utm: CRS, max_levels: int = 2000) -> List[Tuple[float, LineString]]:
    sub, sub_transform, mask_inside = crop_array_to_geom(dem, transform, clip_geom_utm, pad_m=max(100.0, interval_m * 10))
    arr = sub.astype("float32")
    arr[~mask_inside] = np.nan
    if np.all(~np.isfinite(arr)):
        return []
    zmin = float(np.nanmin(arr)); zmax = float(np.nanmax(arr))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return []
    start = math.ceil(zmin / interval_m) * interval_m
    end = math.floor(zmax / interval_m) * interval_m
    levels = np.arange(start, end + 0.1 * interval_m, interval_m, dtype=float)
    if len(levels) > max_levels:
        step = max(1, int(math.ceil(len(levels) / max_levels)))
        levels = levels[::step]
    if len(levels) < 1:
        return []
    nrows, ncols = arr.shape
    xs = sub_transform.c + (np.arange(ncols) + 0.5) * sub_transform.a
    ys = sub_transform.f + (np.arange(nrows) + 0.5) * sub_transform.e
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111)
    try:
        cs = ax.contour(X, Y, arr, levels=levels)
        contours = []
        # Matplotlib 3.8+ collections are available; allsegs is simpler.
        for level, segs in zip(cs.levels, cs.allsegs):
            for seg in segs:
                if seg is None or len(seg) < 2:
                    continue
                line = LineString(seg)
                if line.length < max(2 * abs(transform.a), simplification_m):
                    continue
                # Recortar estrictamente dentro del corredor/intersección
                inter = line.intersection(clip_geom_utm)
                if inter.is_empty:
                    continue
                parts = []
                if isinstance(inter, LineString):
                    parts = [inter]
                elif isinstance(inter, MultiLineString):
                    parts = list(inter.geoms)
                elif isinstance(inter, GeometryCollection):
                    parts = [g for g in inter.geoms if isinstance(g, LineString)]
                for p in parts:
                    if p.length >= max(2 * abs(transform.a), simplification_m):
                        if simplification_m > 0:
                            p = p.simplify(simplification_m, preserve_topology=False)
                        contours.append((float(level), p))
        return contours
    finally:
        plt.close(fig)

# -----------------------------
# HEC-RAS-like CSV
# -----------------------------
def generate_channel_template_csv(channel_type: str, bottom_width_m: float, depth_m: float, side_slope_hv: float, overbank_m: float, manning_n: float, axis_utm: Optional[LineString], section_interval_m: float) -> pd.DataFrame:
    if channel_type == "Rectangular":
        eps = max(0.05, bottom_width_m * 0.002)
        stations = [0.0, overbank_m, overbank_m + eps, overbank_m + eps + bottom_width_m, overbank_m + 2 * eps + bottom_width_m, overbank_m + 2 * eps + bottom_width_m + overbank_m]
        elevs = [depth_m, depth_m, 0.0, 0.0, depth_m, depth_m]
    else:
        z = max(0.0, side_slope_hv)
        left_toe = overbank_m + z * depth_m
        right_toe = left_toe + bottom_width_m
        right_bank = right_toe + z * depth_m
        stations = [0.0, overbank_m, left_toe, right_toe, right_bank, right_bank + overbank_m]
        elevs = [depth_m, depth_m, 0.0, 0.0, depth_m, depth_m]

    # Generar varias secciones si hay eje y distancia positiva; de lo contrario una sección tipo.
    rows = []
    xs_locations = [0.0]
    if axis_utm is not None and axis_utm.length > 0 and section_interval_m > 0:
        n = max(1, int(math.floor(axis_utm.length / section_interval_m)))
        xs_locations = [i * section_interval_m for i in range(n + 1)]
        if xs_locations[-1] < axis_utm.length:
            xs_locations.append(axis_utm.length)
    for i, dist in enumerate(xs_locations):
        xs_id = f"XS_{i+1:04d}"
        river_station_km = dist / 1000.0
        for st, el in zip(stations, elevs):
            rows.append({
                "XS_ID": xs_id,
                "River_Station_km": round(river_station_km, 4),
                "Station_m": round(float(st), 3),
                "Elevation_relative_m": round(float(el), 3),
                "Manning_n": float(manning_n),
                "Channel_Type": channel_type,
                "Comment": "Plantilla relativa tipo HEC-RAS; ajustar cotas absolutas en etapa hidráulica.",
            })
    return pd.DataFrame(rows)

# -----------------------------
# Export KML/KMZ
# -----------------------------
def coord_str_from_geom(geom, precision: int = 8) -> str:
    return " ".join([f"{x:.{precision}f},{y:.{precision}f},0" for x, y in geom.coords])


def kml_polygon(name: str, geom, style: str, desc: str = "") -> str:
    if geom is None or geom.is_empty:
        return ""
    if isinstance(geom, MultiPolygon):
        return "\n".join(kml_polygon(name, g, style, desc) for g in geom.geoms)
    if not isinstance(geom, Polygon):
        return ""
    outer = coord_str_from_geom(LineString(list(geom.exterior.coords)))
    inner_kml = ""
    for ring in geom.interiors:
        inner_kml += f"<innerBoundaryIs><LinearRing><coordinates>{coord_str_from_geom(LineString(list(ring.coords)))}</coordinates></LinearRing></innerBoundaryIs>"
    return f"""
<Placemark><name>{html.escape(name)}</name><styleUrl>#{style}</styleUrl><description><![CDATA[{desc}]]></description>
<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>{inner_kml}</Polygon></Placemark>
"""


def kml_line(name: str, geom, style: str, desc: str = "") -> str:
    if geom is None or geom.is_empty:
        return ""
    if isinstance(geom, MultiLineString):
        return "\n".join(kml_line(name, g, style, desc) for g in geom.geoms)
    if not isinstance(geom, LineString) or len(geom.coords) < 2:
        return ""
    coords = coord_str_from_geom(geom)
    return f"""
<Placemark><name>{html.escape(name)}</name><styleUrl>#{style}</styleUrl><description><![CDATA[{desc}]]></description>
<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>
"""


def kml_point(name: str, pt: Point, style: str, desc: str = "") -> str:
    return f"""
<Placemark><name>{html.escape(name)}</name><styleUrl>#{style}</styleUrl><description><![CDATA[{desc}]]></description>
<Point><coordinates>{pt.x:.8f},{pt.y:.8f},0</coordinates></Point></Placemark>
"""


def build_kmz(results: Dict, contours_detail_wgs: List[Tuple[float, LineString]], contours_support_wgs: List[Tuple[float, LineString]], out_name: str = "hidrosed_doble_cuenca.kmz") -> bytes:
    styles = """
<Style id="pc_cuenca"><IconStyle><color>ff00ffff</color><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>
<Style id="pc_hidro"><IconStyle><color>ff0000ff</color><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>
<Style id="pc_snap"><IconStyle><color>ff00aaff</color><scale>0.9</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon></IconStyle></Style>
<Style id="basin_support"><LineStyle><color>ff00aa00</color><width>3</width></LineStyle><PolyStyle><color>2600ff00</color></PolyStyle></Style>
<Style id="basin_hydro"><LineStyle><color>ffff0000</color><width>3</width></LineStyle><PolyStyle><color>26ff0000</color></PolyStyle></Style>
<Style id="intersection"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><color>22ffff00</color></PolyStyle></Style>
<Style id="corridor"><LineStyle><color>ffcc00cc</color><width>2</width></LineStyle><PolyStyle><color>18cc00cc</color></PolyStyle></Style>
<Style id="axis"><LineStyle><color>ffff5500</color><width>3</width></LineStyle></Style>
<Style id="axis_segment"><LineStyle><color>ff0000ff</color><width>5</width></LineStyle></Style>
<Style id="contour_detail"><LineStyle><color>ff664422</color><width>1.4</width></LineStyle></Style>
<Style id="contour_support"><LineStyle><color>ff999999</color><width>1.0</width></LineStyle></Style>
"""
    body = []
    pc_cuenca = results["pc_cuenca_wgs84"]
    pc_hidro = results["pc_hidro_wgs84"]
    body.append("<Folder><name>Puntos de control</name>")
    body.append(kml_point("PC cuenca soporte", pc_cuenca, "pc_cuenca", "Punto para delimitar cuenca topográfica de soporte."))
    body.append(kml_point("PC hidrológico/cálculo", pc_hidro, "pc_hidro", "Punto para delimitar subcuenca hidrológica de cálculo."))
    if results.get("snap_cuenca_wgs84"):
        body.append(kml_point("PC cuenca ajustado al drenaje", results["snap_cuenca_wgs84"], "pc_snap"))
    if results.get("snap_hidro_wgs84"):
        body.append(kml_point("PC hidrológico ajustado al drenaje", results["snap_hidro_wgs84"], "pc_snap"))
    body.append("</Folder>")
    body.append("<Folder><name>Cuencas</name>")
    body.append(kml_polygon("Cuenca topográfica de soporte", results["basin_support_wgs84"], "basin_support", results.get("desc_support", "")))
    body.append(kml_polygon("Subcuenca hidrológica de cálculo", results["basin_hydro_wgs84"], "basin_hydro", results.get("desc_hydro", "")))
    if results.get("intersection_wgs84") is not None and not results["intersection_wgs84"].is_empty:
        body.append(kml_polygon("Intersección de cuencas", results["intersection_wgs84"], "intersection", "Zona común entre cuenca soporte y subcuenca hidrológica."))
    body.append("</Folder>")
    body.append("<Folder><name>Eje y corredor</name>")
    body.append(kml_line("Eje del cauce completo", results["axis_wgs84"], "axis"))
    if results.get("axis_segment_wgs84") is not None and not results["axis_segment_wgs84"].is_empty:
        body.append(kml_line("Tramo útil PC soporte a PC hidrológico", results["axis_segment_wgs84"], "axis_segment", "Este es el tramo usado para generar curvas de nivel y corredor hidráulico."))
    if results.get("corridor_wgs84") is not None and not results["corridor_wgs84"].is_empty:
        body.append(kml_polygon("Corredor de curvas alrededor del eje", results["corridor_wgs84"], "corridor"))
    body.append("</Folder>")
    body.append("<Folder><name>Curvas de nivel detalladas en corredor</name>")
    for level, line in contours_detail_wgs:
        body.append(kml_line(f"Curva {level:.0f} m", line, "contour_detail"))
    body.append("</Folder>")
    if contours_support_wgs:
        body.append("<Folder><name>Curvas topográficas de apoyo</name>")
        for level, line in contours_support_wgs:
            body.append(kml_line(f"Apoyo {level:.0f} m", line, "contour_support"))
        body.append("</Folder>")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>HidroSed doble cuenca</name>{styles}{''.join(body)}</Document></kml>
"""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml.encode("utf-8"))
    return mem.getvalue()


def contours_to_wgs(contours_utm: List[Tuple[float, LineString]], crs_utm: CRS) -> List[Tuple[float, LineString]]:
    return [(lvl, project_geom(line, crs_utm, "EPSG:4326")) for lvl, line in contours_utm]

# -----------------------------
# UI
# -----------------------------
def main():
    st.title("HidroSed · Foja Cero v3: doble cuenca + tramo útil de cauce")
    st.caption("Delimita cuenca soporte y subcuenca hidrológica; genera curvas solo en el tramo del eje entre PC cuenca soporte y PC hidrológico.")

    with st.expander("Criterio experto incorporado", expanded=False):
        st.markdown("""
**Lecciones aplicadas:**

- No se generan curvas densas en toda la cuenca; solo en el corredor del eje del cauce.
- Para DEM grandes se usa resolución interna controlada y advertencias por borde del DEM.
- Se generan dos polígonos: cuenca topográfica de soporte y subcuenca hidrológica de cálculo.
- El **eje del cauce es obligatorio** y se usa como control de drenaje para evitar ajustes a quebradas vecinas.
- Se puede cargar un perfil longitudinal topográfico de respaldo para contrastarlo con el DEM.
- La zona de curvas se define como: **intersección de ambas cuencas ∩ buffer del tramo del eje entre PC cuenca soporte y PC hidrológico**.
- Si una cuenca toca el borde del DEM, el resultado no debe considerarse definitivo.
- Se evita `pysheds`, `geopandas`, `scipy` y Earth Engine para reducir errores de instalación.
        """)

    st.sidebar.header("1. Entradas")
    pc_cuenca_file = st.sidebar.file_uploader("PC cuenca soporte/general (KMZ/KML con punto)", type=["kmz", "kml"], key="pc_cuenca")
    pc_hidro_file = st.sidebar.file_uploader("PC hidrológico/cálculo (KMZ/KML con punto)", type=["kmz", "kml"], key="pc_hidro")
    axis_file = st.sidebar.file_uploader("Eje del cauce OBLIGATORIO (KMZ/KML con línea)", type=["kmz", "kml"], key="axis")
    profile_file = st.sidebar.file_uploader("Perfil topográfico longitudinal de respaldo opcional (CSV/XLSX)", type=["csv", "txt", "xlsx", "xls"], key="perfil_respaldo")

    st.sidebar.header("2. DEM")
    dem_mode = st.sidebar.radio("Fuente DEM", ["Descargar COP30 desde OpenTopography", "Cargar DEM GeoTIFF manual"], index=0)
    api_key = ""
    dem_upload = None
    demtype = "COP30"
    if dem_mode.startswith("Descargar"):
        demtype = st.sidebar.selectbox("DEM OpenTopography", ["COP30", "NASADEM", "SRTMGL1", "SRTMGL3"], index=0)
        api_key = st.sidebar.text_input("API Key OpenTopography", type="password")
    else:
        dem_upload = st.sidebar.file_uploader("DEM GeoTIFF", type=["tif", "tiff"], key="dem_manual")

    st.sidebar.header("3. Parámetros de área")
    margin_km = st.sidebar.number_input("Margen DEM alrededor de puntos/eje (km)", min_value=1.0, max_value=150.0, value=25.0, step=1.0)
    tiles = st.sidebar.slider("DEM parciales para descarga", min_value=1, max_value=12, value=4, step=1)
    internal_res = st.sidebar.selectbox("Resolución interna de cálculo", [30, 60, 90, 120, 150], index=1, help="Para cuencas >1000 km² se recomienda 90 m o 120 m.")
    snap_radii_text = st.sidebar.text_input("Radios de ajuste al cauce (m)", value="150,300,600,1000")
    st.sidebar.header("3b. Control de calidad de delimitación")
    axis_snap_buffer = st.sidebar.number_input("Tolerancia eje-drenaje para ajustar PC (m)", min_value=50.0, max_value=3000.0, value=500.0, step=50.0, help="El punto ajustado al drenaje debe caer cerca del eje obligatorio; evita capturar otra quebrada.")
    min_axis_support_pct = st.sidebar.slider("Mínimo del eje dentro de cuenca soporte (%)", min_value=50, max_value=100, value=85, step=5)
    min_axis_intersection_pct = st.sidebar.slider("Mínimo del eje dentro de la intersección (%)", min_value=10, max_value=100, value=40, step=5, help="Si la intersección es menor, la app entrega advertencia y el KMZ debe considerarse preliminar.")

    st.sidebar.header("4. Curvas por tramo de eje")
    buffer_detail = st.sidebar.number_input("Buffer curvas detalladas alrededor del tramo PC soporte–PC hidrológico (m por lado)", min_value=20.0, max_value=5000.0, value=500.0, step=50.0)
    interval_detail = st.sidebar.number_input("Equidistancia curvas detalladas (m)", min_value=1.0, max_value=200.0, value=10.0, step=1.0)
    simpl_detail = st.sidebar.number_input("Simplificación curvas detalladas (m)", min_value=0.0, max_value=200.0, value=10.0, step=5.0)
    generate_support = st.sidebar.checkbox("Generar curvas topográficas de apoyo en corredor amplio", value=True)
    buffer_support = st.sidebar.number_input("Buffer curvas de apoyo sobre tramo útil (m por lado)", min_value=100.0, max_value=10000.0, value=1500.0, step=100.0, disabled=not generate_support)
    interval_support = st.sidebar.number_input("Equidistancia curvas de apoyo (m)", min_value=5.0, max_value=500.0, value=50.0, step=5.0, disabled=not generate_support)

    run = st.button("Ejecutar delimitación + curvas + KMZ", type="primary")

    tab1, tab2, tab3, tab4 = st.tabs(["Resultado técnico", "Perfil longitudinal", "Plantilla cauce HEC-RAS", "Criterios por superficie"])

    with tab4:
        st.markdown("""
### Parámetros recomendados por superficie estimada de cuenca

| Superficie | Resolución interna | Buffer curvas | Curvas detalladas | Curvas apoyo | DEM parciales |
|---:|---:|---:|---:|---:|---:|
| ≤ 50 km² | 30 m | 100–250 m | 1–5 m | 25 m | 1 |
| 50–300 km² | 30–60 m | 250–500 m | 5–10 m | 25–50 m | 1–2 |
| 300–1000 km² | 60–90 m | 500–1000 m | 10–25 m | 50 m | 2–4 |
| >1000 km² | 90–120 m | 1000–2000 m | 25–50 m | 50–100 m | 4–8 |

**Regla principal:** para cuencas grandes no se generan curvas finas en toda la cuenca ni en todo el eje si no corresponde. Las curvas detalladas se limitan al tramo entre PC cuenca soporte y PC hidrológico.
        """)

    with tab2:
        st.subheader("Perfil longitudinal del eje obligatorio")
        st.caption("Luego de ejecutar, la app genera un perfil DEM sobre el eje y, si cargas perfil de respaldo, compara cotas y sentido del drenaje.")
        st.info("El perfil de respaldo es opcional. Debe tener columnas de distancia/progresiva y cota/elevación, en CSV o Excel.")

    with tab3:
        st.subheader("Plantilla de cauce rectangular/trapecial estilo HEC-RAS")
        colA, colB, colC = st.columns(3)
        with colA:
            channel_type = st.selectbox("Tipo de cauce", ["Trapecial", "Rectangular"])
            bottom_width = st.number_input("Ancho de fondo B (m)", min_value=0.1, value=5.0, step=0.5)
            depth = st.number_input("Altura/tirante geométrico H (m)", min_value=0.1, value=2.0, step=0.1)
        with colB:
            side_slope = st.number_input("Talud lateral z = H:V", min_value=0.0, value=1.5, step=0.25, disabled=(channel_type == "Rectangular"))
            overbank = st.number_input("Bermas/extensión lateral (m)", min_value=0.0, value=5.0, step=0.5)
            manning_n = st.number_input("Manning n", min_value=0.010, max_value=0.200, value=0.035, step=0.001, format="%.3f")
        with colC:
            section_interval = st.number_input("Intervalo de secciones sobre eje (m)", min_value=0.0, value=100.0, step=25.0)
            st.caption("Si el eje ya fue cargado, se generan secciones repetidas por kilometraje. Si no, se descarga una sección tipo.")
        axis_utm_for_template = None
        try:
            if axis_file and pc_cuenca_file and pc_hidro_file:
                _, pc_tmp = first_point_from_upload(pc_cuenca_file, "PC cuenca")
                _, pc_hidro_tmp = first_point_from_upload(pc_hidro_file, "PC hidrológico")
                _, axis_tmp = first_line_from_upload(axis_file, "eje")
                crs_tmp = utm_crs_from_lonlat(pc_tmp.x, pc_tmp.y)
                axis_tmp_utm = project_geom(axis_tmp, "EPSG:4326", crs_tmp)
                pc_tmp_utm = project_geom(pc_tmp, "EPSG:4326", crs_tmp)
                pc_hidro_tmp_utm = project_geom(pc_hidro_tmp, "EPSG:4326", crs_tmp)
                axis_utm_for_template, _ = axis_segment_between_control_points(axis_tmp_utm, pc_tmp_utm, pc_hidro_tmp_utm, min_length_m=10.0)
        except Exception:
            axis_utm_for_template = None
        df_template = generate_channel_template_csv(channel_type, bottom_width, depth, side_slope, overbank, manning_n, axis_utm_for_template, section_interval)
        st.dataframe(df_template.head(30), use_container_width=True)
        st.download_button("Descargar plantilla CSV", df_template.to_csv(index=False).encode("utf-8"), "plantilla_cauce_hecras.csv", "text/csv")

    if not run:
        return

    with tab1:
        if not pc_cuenca_file or not pc_hidro_file or not axis_file:
            st.error("Debes cargar PC cuenca, PC hidrológico y eje del cauce.")
            return
        if dem_mode.startswith("Descargar") and not api_key:
            st.error("Debes ingresar API Key de OpenTopography o usar DEM manual.")
            return
        if dem_mode.startswith("Cargar") and not dem_upload:
            st.error("Debes cargar un DEM GeoTIFF manual.")
            return

        try:
            pc_cuenca_name, pc_cuenca = first_point_from_upload(pc_cuenca_file, "PC cuenca")
            pc_hidro_name, pc_hidro = first_point_from_upload(pc_hidro_file, "PC hidrológico")
            axis_name, axis_wgs84 = first_line_from_upload(axis_file, "eje de cauce")
            ref_profile_df = read_reference_longitudinal_profile(profile_file) if profile_file else None
            crs_utm = utm_crs_from_lonlat(pc_cuenca.x, pc_cuenca.y)
            axis_utm = project_geom(axis_wgs84, "EPSG:4326", crs_utm)
            pc_cuenca_utm = project_geom(pc_cuenca, "EPSG:4326", crs_utm)
            pc_hidro_utm = project_geom(pc_hidro, "EPSG:4326", crs_utm)
            axis_segment_utm, axis_segment_info = axis_segment_between_control_points(
                axis_utm, pc_cuenca_utm, pc_hidro_utm, min_length_m=100.0
            )
            axis_segment_wgs84 = project_geom(axis_segment_utm, crs_utm, "EPSG:4326")
            # Control preliminar: el eje debe tener una relación geométrica clara con los puntos de control.
            # En v3 se mide distancia perpendicular al eje, no a los extremos, porque el eje puede venir más largo que el tramo útil.
            dist_axis_to_pc_cuenca = axis_segment_info["dist_pc_cuenca_al_eje_m"]
            dist_axis_to_pc_hidro = axis_segment_info["dist_pc_hidro_al_eje_m"]
            snap_radii = [float(x.strip()) for x in snap_radii_text.split(",") if x.strip()]
            if not snap_radii:
                snap_radii = [150, 300, 600, 1000]
        except Exception as e:
            st.error(f"Error leyendo entradas KMZ/KML: {e}")
            return

        st.info(f"CRS de trabajo: {crs_utm.to_string()}")
        tmp = tempfile.TemporaryDirectory()
        tmpdir = tmp.name
        try:
            if dem_mode.startswith("Descargar"):
                bbox = create_bbox_from_inputs(pc_cuenca, pc_hidro, axis_wgs84, margin_km)
                bbox_area = geodesic_bbox_area_km2(bbox)
                st.write("**BBox DEM solicitado:**", {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3], "area_bbox_km2_aprox": round(bbox_area, 1)})
                if bbox_area > 12000:
                    st.error("El área de descarga supera 12.000 km². Reduce el margen o usa un DEM manual preparado por mosaicos.")
                    return
                if bbox_area > 5000:
                    st.warning("Área DEM grande. Se recomienda resolución interna ≥ 90 m y curvas detalladas ≥ 25 m.")
                tile_bounds = split_bbox(bbox, tiles)
                tile_paths = []
                progress = st.progress(0, text="Descargando DEM por partes...")
                for i, tb in enumerate(tile_bounds, start=1):
                    out_tile = os.path.join(tmpdir, f"tile_{i:02d}.tif")
                    download_opentopo_tile(demtype, tb, api_key, out_tile)
                    tile_paths.append(out_tile)
                    progress.progress(i / len(tile_bounds), text=f"Descargado DEM parcial {i}/{len(tile_bounds)}")
                if len(tile_paths) == 1:
                    dem_src_path = tile_paths[0]
                else:
                    dem_src_path = os.path.join(tmpdir, "dem_mosaico.tif")
                    merge_tiles(tile_paths, dem_src_path)
            else:
                dem_src_path = load_manual_dem(dem_upload, tmpdir)

            st.write("Reproyectando/remuestreando DEM...")
            dem, dem_transform, dem_crs, nodata, dem_utm_path = reproject_dem_to_utm(dem_src_path, crs_utm, float(internal_res), tmpdir)
            n_cells = int(np.isfinite(dem).sum())
            st.write(f"DEM interno: {dem.shape[0]} filas × {dem.shape[1]} columnas · {n_cells:,} celdas válidas · resolución {internal_res} m")
            if n_cells > 4_000_000:
                st.error("DEM interno demasiado grande para Streamlit Cloud. Usa resolución 120–150 m o reduce el margen.")
                return

            dem_axis_profile = sample_dem_profile_along_axis(axis_segment_utm, dem, dem_transform, max(float(internal_res), 30.0))
            profile_compare = compare_dem_profile_with_reference(dem_axis_profile, ref_profile_df)
            with tab2:
                st.write("**Perfil longitudinal DEM sobre tramo útil del eje PC soporte–PC hidrológico**")
                st.dataframe(dem_axis_profile.head(100), use_container_width=True)
                st.download_button("Descargar perfil longitudinal DEM CSV", dem_axis_profile.to_csv(index=False).encode("utf-8"), "perfil_longitudinal_dem_eje.csv", "text/csv")
                if profile_compare.get("available"):
                    st.success(f"Perfil de respaldo comparado: RMSE={profile_compare['rmse_m']:.2f} m · sesgo={profile_compare['bias_m']:.2f} m · {profile_compare['orientation_note']}.")
                elif ref_profile_df is not None:
                    st.warning(profile_compare.get("warning", "No se pudo comparar el perfil de respaldo con el perfil DEM."))
                else:
                    st.info("No se cargó perfil de respaldo. Se entrega solo el perfil DEM generado desde el eje.")

            # Confirmar puntos dentro del DEM
            for label, pt in [("PC cuenca", pc_cuenca_utm), ("PC hidrológico", pc_hidro_utm)]:
                r, c = world_to_rowcol(dem_transform, pt.x, pt.y)
                if r < 0 or r >= dem.shape[0] or c < 0 or c >= dem.shape[1]:
                    st.error(f"{label} queda fuera del DEM. Aumenta margen o revisa coordenadas.")
                    return

            st.write("Rellenando depresiones DEM...")
            nodata_mask = ~np.isfinite(dem)
            filled = priority_flood_fill(dem, nodata_mask)
            st.write("Calculando dirección y acumulación D8...")
            flow, valid = compute_d8_flow(filled, dem_transform)
            rec = receiver_flat_indices(flow, valid)
            acc = compute_accumulation(rec, valid)

            st.write("Delimitando cuenca soporte...")
            basin_support = delineate_basin_candidates("Cuenca soporte", pc_cuenca, dem, dem_transform, crs_utm, rec, acc, valid, snap_radii, axis_utm=axis_utm, axis_buffer_m=float(axis_snap_buffer), hard_axis_snap=True)
            st.write("Delimitando subcuenca hidrológica...")
            basin_hydro = delineate_basin_candidates("Subcuenca hidrológica", pc_hidro, dem, dem_transform, crs_utm, rec, acc, valid, snap_radii, axis_utm=axis_utm, axis_buffer_m=float(axis_snap_buffer), hard_axis_snap=True)

            inter_utm = basin_support.polygon_utm.intersection(basin_hydro.polygon_utm)
            if inter_utm.is_empty:
                st.error("La intersección entre ambas cuencas es vacía. Revisa puntos de control o DEM.")
                return
            inter_utm = make_valid(inter_utm)
            if isinstance(inter_utm, GeometryCollection):
                polys = [g for g in inter_utm.geoms if isinstance(g, (Polygon, MultiPolygon))]
                inter_utm = unary_union(polys) if polys else Polygon()
            inter_wgs = project_geom(inter_utm, crs_utm, "EPSG:4326")

            corridor_detail_utm = axis_segment_utm.buffer(float(buffer_detail)).intersection(inter_utm)
            if corridor_detail_utm.is_empty:
                st.error("El corredor del eje no intersecta la zona común de ambas cuencas.")
                return
            corridor_detail_wgs = project_geom(corridor_detail_utm, crs_utm, "EPSG:4326")

            support_clip_utm = None
            if generate_support:
                support_clip_utm = axis_segment_utm.buffer(float(buffer_support)).intersection(inter_utm)

            # Auditoría geométrica obligatoria con eje y puntos de control
            axis_support_frac = line_inside_fraction(axis_utm, basin_support.polygon_utm)
            axis_hydro_frac = line_inside_fraction(axis_utm, basin_hydro.polygon_utm)
            axis_inter_frac = line_inside_fraction(axis_utm, inter_utm)
            segment_support_frac = line_inside_fraction(axis_segment_utm, basin_support.polygon_utm)
            segment_hydro_frac = line_inside_fraction(axis_segment_utm, basin_hydro.polygon_utm)
            segment_inter_frac = line_inside_fraction(axis_segment_utm, inter_utm)
            pc_hidro_margin_in_support = point_margin_to_polygon_boundary(pc_hidro_utm, basin_support.polygon_utm)
            pc_cuenca_margin_in_hydro = point_margin_to_polygon_boundary(pc_cuenca_utm, basin_hydro.polygon_utm)
            quality_flags = []
            if axis_support_frac * 100 < float(min_axis_support_pct):
                quality_flags.append(f"Eje dentro de cuenca soporte solo {axis_support_frac*100:.1f}%; mínimo requerido {min_axis_support_pct}%.")
            if axis_inter_frac * 100 < float(min_axis_intersection_pct):
                quality_flags.append(f"Eje completo dentro de intersección solo {axis_inter_frac*100:.1f}%; puede ser normal si el eje incluye tramos fuera del intervalo de análisis.")
            if segment_inter_frac * 100 < float(min_axis_intersection_pct):
                quality_flags.append(f"Tramo útil PC soporte–PC hidrológico dentro de intersección solo {segment_inter_frac*100:.1f}%; revisar puntos, eje o delimitación.")
            if pc_hidro_margin_in_support < 0:
                quality_flags.append("El PC hidrológico no queda dentro de la cuenca soporte; la jerarquía de cuencas puede estar invertida o mal delimitada.")
            if basin_support.area_km2 > 1000 or basin_hydro.area_km2 > 1000:
                if internal_res < 90:
                    quality_flags.append("Cuenca >1000 km² con resolución interna menor a 90 m: se recomienda repetir con 90–120 m para estabilidad.")
                if basin_support.touches_dem_edge or basin_hydro.touches_dem_edge:
                    quality_flags.append("Cuenca grande toca borde DEM: resultado NO definitivo; ampliar margen o preparar DEM manual más amplio.")
            if dist_axis_to_pc_cuenca > max(500.0, float(axis_snap_buffer)):
                quality_flags.append(f"El PC cuenca soporte está a {dist_axis_to_pc_cuenca:.0f} m del eje; revisar coherencia PC/eje.")
            if dist_axis_to_pc_hidro > max(500.0, float(axis_snap_buffer)):
                quality_flags.append(f"El PC hidrológico está a {dist_axis_to_pc_hidro:.0f} m del eje; revisar coherencia PC/eje.")

            st.write("Generando curvas de nivel detalladas dentro del corredor...")
            contours_detail_utm = generate_contours(dem, dem_transform, corridor_detail_utm, float(interval_detail), float(simpl_detail), crs_utm)
            contours_support_utm = []
            if generate_support and support_clip_utm is not None and not support_clip_utm.is_empty:
                st.write("Generando curvas topográficas de apoyo...")
                contours_support_utm = generate_contours(dem, dem_transform, support_clip_utm, float(interval_support), max(float(simpl_detail), 20.0), crs_utm, max_levels=1000)

            contours_detail_wgs = contours_to_wgs(contours_detail_utm, crs_utm)
            contours_support_wgs = contours_to_wgs(contours_support_utm, crs_utm)

            # Reporte
            def basin_table_row(b: BasinResult):
                return {
                    "Elemento": b.name,
                    "Área km²": round(b.area_km2, 3),
                    "Perímetro km": round(b.perimeter_km, 3),
                    "Confianza preliminar %": round(b.confidence, 1),
                    "Toca borde DEM": b.touches_dem_edge,
                    "Distancia ajuste m": round(b.snap_distance_m, 1),
                    "Radio ajuste m": round(b.snap_radius_m, 1),
                    "Acumulación celdas": b.outlet_accumulation_cells,
                    "Índice forma": round(b.shape_index, 2),
                }
            df_summary = pd.DataFrame([basin_table_row(basin_support), basin_table_row(basin_hydro)])
            st.dataframe(df_summary, use_container_width=True)

            inter_area = geom_area_km2_utm(inter_utm)
            corridor_area = geom_area_km2_utm(corridor_detail_utm)
            st.metric("Área intersección de cuencas", f"{inter_area:.2f} km²")
            st.metric("Área corredor curvas detalladas", f"{corridor_area:.2f} km²")
            st.metric("Longitud tramo eje usado para curvas", f"{axis_segment_info['longitud_tramo_util_m']/1000.0:.2f} km")
            st.metric("Curvas detalladas generadas", f"{len(contours_detail_wgs)}")
            st.metric("Curvas de apoyo generadas", f"{len(contours_support_wgs)}")
            st.write("**Control eje / cuencas**")
            st.dataframe(pd.DataFrame([{
                "Eje dentro cuenca soporte %": round(axis_support_frac * 100, 1),
                "Eje dentro subcuenca hidro %": round(axis_hydro_frac * 100, 1),
                "Eje completo dentro intersección %": round(axis_inter_frac * 100, 1),
                "Tramo útil dentro intersección %": round(segment_inter_frac * 100, 1),
                "Longitud eje total km": round(axis_segment_info["longitud_eje_total_m"] / 1000.0, 3),
                "Longitud tramo útil km": round(axis_segment_info["longitud_tramo_util_m"] / 1000.0, 3),
                "PC soporte a eje m": round(axis_segment_info["dist_pc_cuenca_al_eje_m"], 1),
                "PC hidro a eje m": round(axis_segment_info["dist_pc_hidro_al_eje_m"], 1),
                "PC hidrológico dentro soporte, margen m": round(pc_hidro_margin_in_support, 1),
                "PC cuenca dentro subcuenca, margen m": round(pc_cuenca_margin_in_hydro, 1),
                "Dist. PC cuenca al eje m": round(dist_axis_to_pc_cuenca, 1),
                "Dist. PC hidrológico al eje m": round(dist_axis_to_pc_hidro, 1),
            }]), use_container_width=True)

            all_warnings = []
            all_warnings.extend(quality_flags)
            for b in [basin_support, basin_hydro]:
                for w in b.warnings:
                    all_warnings.append(f"{b.name}: {w}")
            if basin_support.touches_dem_edge or basin_hydro.touches_dem_edge:
                all_warnings.append("Advertencia crítica: una o ambas cuencas tocan el borde del DEM. Resultado no definitivo.")
            if all_warnings:
                st.warning("\n".join([f"- {w}" for w in all_warnings]))
            else:
                st.success("No se detectaron advertencias críticas preliminares.")

            results = {
                "pc_cuenca_wgs84": pc_cuenca,
                "pc_hidro_wgs84": pc_hidro,
                "snap_cuenca_wgs84": Point(*basin_support.snapped_lonlat),
                "snap_hidro_wgs84": Point(*basin_hydro.snapped_lonlat),
                "basin_support_wgs84": basin_support.polygon_wgs84,
                "basin_hydro_wgs84": basin_hydro.polygon_wgs84,
                "intersection_wgs84": inter_wgs,
                "corridor_wgs84": corridor_detail_wgs,
                "axis_wgs84": axis_wgs84,
                "axis_segment_wgs84": axis_segment_wgs84,
                "desc_support": json.dumps({k: v for k, v in asdict(basin_support).items() if k not in ["polygon_utm", "polygon_wgs84"]}, ensure_ascii=False, indent=2),
                "desc_hydro": json.dumps({k: v for k, v in asdict(basin_hydro).items() if k not in ["polygon_utm", "polygon_wgs84"]}, ensure_ascii=False, indent=2),
            }
            kmz_bytes = build_kmz(results, contours_detail_wgs, contours_support_wgs)
            st.download_button("Descargar KMZ unificado", kmz_bytes, "hidrosed_doble_cuenca_tramo_util_v3.kmz", "application/vnd.google-earth.kmz")

            summary = {
                "fecha": datetime.utcnow().isoformat() + "Z",
                "crs_trabajo": crs_utm.to_string(),
                "dem_mode": dem_mode,
                "demtype": demtype if dem_mode.startswith("Descargar") else "manual",
                "resolucion_interna_m": internal_res,
                "cuenca_soporte": {k: v for k, v in asdict(basin_support).items() if k not in ["polygon_utm", "polygon_wgs84"]},
                "subcuenca_hidrologica": {k: v for k, v in asdict(basin_hydro).items() if k not in ["polygon_utm", "polygon_wgs84"]},
                "area_interseccion_km2": inter_area,
                "area_corredor_curvas_km2": corridor_area,
                "curvas_detalladas": len(contours_detail_wgs),
                "curvas_apoyo": len(contours_support_wgs),
                "control_eje": {
                    "axis_support_fraction_pct": axis_support_frac * 100,
                    "axis_hydro_fraction_pct": axis_hydro_frac * 100,
                    "axis_intersection_fraction_pct": axis_inter_frac * 100,
                    "segment_support_fraction_pct": segment_support_frac * 100,
                    "segment_hydro_fraction_pct": segment_hydro_frac * 100,
                    "segment_intersection_fraction_pct": segment_inter_frac * 100,
                    "axis_segment_info": axis_segment_info,
                    "pc_hidro_margin_in_support_m": pc_hidro_margin_in_support,
                    "dist_pc_cuenca_to_axis_m": dist_axis_to_pc_cuenca,
                    "dist_pc_hidro_to_axis_m": dist_axis_to_pc_hidro,
                    "axis_snap_buffer_m": axis_snap_buffer,
                },
                "perfil_longitudinal": profile_compare,
                "advertencias": all_warnings,
            }
            st.download_button("Descargar resumen JSON", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"), "resumen_hidrosed_doble_cuenca.json", "application/json")

        except Exception as e:
            st.error(f"Error de proceso: {e}")
            st.exception(e)
        finally:
            tmp.cleanup()

if __name__ == "__main__":
    main()
