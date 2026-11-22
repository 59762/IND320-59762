"""Shapefile -> GeoJSON conversion and area geometry (geopandas / shapely).

Nothing is pre-built: the two NVE shapefiles in data/shapefiles are read, reprojected to
WGS84 and converted to GeoJSON the first time they are needed, then cached by Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import streamlit as st

from utils.config import (AREA_TYPES, EL_NAMES, NO_AREA_NR, VASS_NAMES, VASS_NORTH, VASS_WEST)

SHAPE_DIR = Path(__file__).resolve().parents[1] / "data" / "shapefiles"
EL_DIR = SHAPE_DIR / "electricity_areas"     # Budområder.shp  (column 'budomr' = "NO 1" ... "NO 5")
REGION_DIR = SHAPE_DIR / "NVE_Regions"       # NVE_Region.shp  (column 'REGION': RV, RM, RN, RØ, RS)
SIMPLIFY_M = 500                             # outline simplification in metres (EPSG:25833)


def _first_shp(folder: Path) -> Path:
    # glob instead of a fixed name: "Budområder.shp" is stored with a mangled file name in some zips
    files = sorted(folder.glob("*.shp"))
    if not files:
        raise FileNotFoundError(f"No .shp file found in {folder}")
    return files[0]


@st.cache_resource(show_spinner=False)
def _area_polygons(area_type: str) -> dict[int, "shapely.Geometry"]:
    """{area_nr: polygon} in EPSG:25833 (metres) for one area type, straight from the shapefiles."""
    if area_type in ("EL", "NO"):
        gdf = gpd.read_file(_first_shp(EL_DIR))
        nr = gdf["budomr"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        if area_type == "EL":
            return {int(n): shapely.union_all(gdf.geometry[nr == n].values) for n in sorted(nr.unique())}
        return {NO_AREA_NR: shapely.union_all(gdf.geometry.values)}
    if area_type == "VASS":
        gdf = gpd.read_file(_first_shp(REGION_DIR), encoding="utf-8")
        code = gdf["REGION"]
        masks = {1: ~code.isin(VASS_WEST + VASS_NORTH),   # East by exclusion (Ø-encoding safe)
                 2: code.isin(VASS_WEST),
                 3: code.isin(VASS_NORTH)}
        return {n: shapely.union_all(gdf.geometry[m].values) for n, m in masks.items() if m.any()}
    raise ValueError(f"Unknown area type {area_type!r}")


def area_names(area_type: str) -> dict[int, str]:
    if area_type == "EL":
        return dict(EL_NAMES)
    if area_type == "VASS":
        return dict(VASS_NAMES)
    return {NO_AREA_NR: "Norway"}


def area_label(area_type: str, area_nr: int) -> str:
    """Human-readable label, e.g. 'NO3 – Central Norway'."""
    return area_names(area_type).get(int(area_nr), f"{area_type} {area_nr}")


@st.cache_data(show_spinner=False)
def load_geojson(area_type: str) -> dict:
    """GeoJSON FeatureCollection (WGS84) for one area type, converted from the shapefiles.

    Each feature has properties ``area_type``, ``area_nr`` (int) and ``name``.
    """
    polys = _area_polygons(area_type)
    names = area_names(area_type)
    gdf = gpd.GeoDataFrame(
        {"area_type": area_type, "area_nr": list(polys), "name": [names.get(n, str(n)) for n in polys]},
        geometry=list(polys.values()), crs="EPSG:25833",
    )
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    return json.loads(gdf.to_crs("EPSG:4326").to_json(drop_id=True))


def area_numbers(area_type: str) -> list[int]:
    return sorted(int(n) for n in _area_polygons(area_type))


def sample_points(area_type: str, area_nr: int, n: int) -> list[tuple[float, float]]:
    """n reproducible (lat, lon) points that lie INSIDE the area polygon (rejection sampling)."""
    geom = _area_polygons(area_type)[int(area_nr)]
    rng = np.random.default_rng(int(area_nr) * 1009 + len(area_type) * 7919 + n)
    minx, miny, maxx, maxy = geom.bounds
    xs: list[float] = []
    ys: list[float] = []
    while len(xs) < n:
        x, y = rng.uniform(minx, maxx, 500), rng.uniform(miny, maxy, 500)
        ok = shapely.contains_xy(geom, x, y)
        xs += x[ok].tolist()
        ys += y[ok].tolist()
    pts = gpd.GeoSeries(gpd.points_from_xy(xs[:n], ys[:n]), crs="EPSG:25833").to_crs("EPSG:4326")
    return [(float(p.y), float(p.x)) for p in pts]


__all__ = ["AREA_TYPES", "area_label", "area_names", "area_numbers", "load_geojson", "sample_points"]
