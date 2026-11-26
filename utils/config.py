"""Constants shared by the map, correlation and forecast pages (no Streamlit code here)."""
from __future__ import annotations

# ----------------------------------------------------------------------------- areas
# Magasinstatistikk has three area types: EL (price areas), VASS (water areas), NO (Norway).
AREA_TYPES = {
    "EL": "Price areas (EL)",
    "VASS": "Water areas (VASS)",
    "NO": "Norway (NO)",
}

# The "Colour areas by" selector on the map page. The value is the area type whose polygons are
# drawn; None means "pick the area type separately" (weather can be shown on any of the three).
COLOR_MODES = {
    "Price area": "EL",
    "Water area": "VASS",
    "Weather property": None,
}

EL_NAMES = {
    1: "NO1 – Eastern Norway",
    2: "NO2 – South-western Norway",
    3: "NO3 – Central Norway",
    4: "NO4 – Northern Norway",
    5: "NO5 – Western Norway",
}

# VASS areas are built from the NVE regions shapefile (same assumption as the Weather page):
#   1 = East (RS + RØ), 2 = West (RV), 3 = North (RM + RN).
# East is matched "by exclusion" so the Ø in the shapefile's encoding never matters.
# Change here if your VASS numbers mean something else.
VASS_WEST = ["RV"]
VASS_NORTH = ["RM", "RN"]
VASS_NAMES = {
    1: "VASS 1 – East (RS + RØ)",
    2: "VASS 2 – West (RV)",
    3: "VASS 3 – North (RM + RN)",
}

NO_AREA_NR = 0           # the NO polygon is a single feature; its number is only a map key

# ------------------------------------------------------------------ Magasinstatistikk
RESERVOIR_VARS = {
    "fill_percentage": {"label": "Reservoir filling", "unit": "%", "clip": (0.0, 100.0),
                        "scale": "Blues", "zmid": None},
    "fill_TWh":        {"label": "Stored energy", "unit": "TWh", "clip": (0.0, None),
                        "scale": "Blues", "zmid": None},
    "capacity_TWh":    {"label": "Capacity", "unit": "TWh", "clip": (0.0, None),
                        "scale": "Blues", "zmid": None},
    "change_fill_pct": {"label": "Weekly change in filling", "unit": "pp", "clip": (None, None),
                        "scale": "RdBu", "zmid": 0.0},
}

# ------------------------------------------------------------------ Open-Meteo (ERA5)
# agg = how DAILY values are combined into weeks / the chosen period
WEATHER_VARS = {
    "temperature_2m_mean":     {"label": "Temperature", "unit": "°C", "agg": "mean",
                                "scale": "RdYlBu_r", "zmid": 0.0},
    "precipitation_sum":       {"label": "Precipitation", "unit": "mm", "agg": "sum",
                                "scale": "Blues", "zmid": None},
    "snowfall_sum":            {"label": "Snowfall", "unit": "cm", "agg": "sum",
                                "scale": "Purples", "zmid": None},
    "wind_speed_10m_max":      {"label": "Wind speed (daily max)", "unit": "km/h", "agg": "mean",
                                "scale": "Viridis", "zmid": None},
    "shortwave_radiation_sum": {"label": "Solar radiation", "unit": "MJ/m²", "agg": "sum",
                                "scale": "YlOrRd", "zmid": None},
}

ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"
NVE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

DEFAULT_SAMPLE_POINTS = 4        # ERA5 points per area (fewer = faster, more = better area mean)
MAX_WEATHER_YEARS = 12           # longest weather window fetched in one go (API quota)
ERA5_LAG_DAYS = 6                # ERA5 is published with a delay of about five days

MAP_STYLE = "carto-positron"     # token-free basemap
SERIES_COLORS = ["#1f77b4", "#e67e22", "#2ca02c", "#8e44ad", "#d62728"]
SELECT_COLOR = "#e4572e"         # outline of the clicked area
