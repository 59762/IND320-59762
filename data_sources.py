from __future__ import annotations

import numpy as np
import pandas as pd
import requests

NVE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/era5"

# Daily variables requested from Open-Meteo (ERA5). Shown as options in the app.
WEATHER_VARS = {
    "temperature_2m_mean": "Temperature (°C, mean)",
    "precipitation_sum": "Precipitation (mm, sum)",
    "snowfall_sum": "Snowfall (cm, sum)",
    "wind_speed_10m_max": "Wind speed (km/h, max)",
    "shortwave_radiation_sum": "Solar radiation (MJ/m², sum)",
}

# Representative (lon, lat) per price area - same points as the Weather page.
AREA_CENTRE = {
    1: (10.75, 59.9), 2: (6.5, 58.7), 3: (10.5, 63.4), 4: (17.0, 68.4), 5: (6.0, 61.2),
}

NVE_RENAME = {
    "dato_Id": "date", "omrType": "area_type", "omrnr": "area_nr",
    "fyllingsgrad": "fill_percentage", "kapasitet_TWh": "capacity_TWh",
    "fylling_TWh": "fill_TWh",
}


class DataSourceError(RuntimeError):
    """Raised with a human-readable message when an external source fails."""


# --------------------------------------------------------------------------- NVE
def load_nve_data(timeout: int = 30) -> pd.DataFrame:
    """Download the full Magasinstatistikk table and normalise column names."""
    try:
        r = requests.get(NVE_URL, timeout=timeout)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:  # network, HTTP or JSON problems -> one clear message
        raise DataSourceError(f"Could not download Magasinstatistikk from NVE: {e}") from e
    df = df.rename(columns={k: v for k, v in NVE_RENAME.items() if k in df.columns})
    missing = {"date", "area_type", "area_nr", "fill_percentage"} - set(df.columns)
    if missing:
        raise DataSourceError(f"NVE response is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["area_nr"] = pd.to_numeric(df["area_nr"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def get_fill_series(df: pd.DataFrame, area_type: str, area_nr: int) -> pd.Series:
    """Fill percentage (weekly, as published) for ONE region as a datetime-indexed Series."""
    m = (df["area_type"] == area_type) & (df["area_nr"] == area_nr)
    s = df.loc[m].set_index("date")["fill_percentage"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.dropna().empty:
        raise DataSourceError(f"No fill data for {area_type}{area_nr}.")
    s.name = f"{area_type}{area_nr}"
    return s


# ------------------------------------------------------------------- Open-Meteo
def _fetch_openmeteo(lon, lat, start, end, variables, timeout=60) -> pd.DataFrame:
    r = requests.get(
        OPENMETEO_ARCHIVE, timeout=timeout,
        params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "daily": ",".join(variables), "timezone": "Europe/Oslo"},
    )
    r.raise_for_status()
    d = r.json().get("daily")
    if not d or "time" not in d:
        raise DataSourceError("Open-Meteo returned no daily data.")
    out = pd.DataFrame(d)
    out["time"] = pd.to_datetime(out["time"])
    return out.set_index("time")


def load_weather_daily(area_type: str, area_nr: int, start: str, end: str,
                       variables=tuple(WEATHER_VARS)) -> pd.DataFrame:
    """
    Daily ERA5 weather for a region. EL areas use their representative point; any other
    area type (NO total, VASS) falls back to the mean over the five EL points.
    DAILY resolution is requested on purpose: 24x less data than hourly and it is
    aggregated to weeks/months anyway.
    """
    points = [AREA_CENTRE[area_nr]] if area_type == "EL" and area_nr in AREA_CENTRE \
        else list(AREA_CENTRE.values())
    frames = []
    try:
        for lon, lat in points:
            frames.append(_fetch_openmeteo(lon, lat, start, end, list(variables)))
    except DataSourceError:
        raise
    except Exception as e:
        raise DataSourceError(f"Could not download weather from Open-Meteo: {e}") from e
    return pd.concat(frames).groupby(level=0).mean()


# ------------------------------------------------- offline / demo data (no network)
def make_synthetic_weather(start="2000-01-01", end="2025-12-31", seed=0) -> pd.DataFrame:
    """Plausible Norwegian daily weather (seasonal temperature, gamma-distributed precip)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    doy = idx.dayofyear.to_numpy()
    seasonal = 4.0 - 11.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):                      # AR(1) weather noise
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 2.0)
    temp = seasonal + noise
    wet = rng.random(len(idx)) < 0.45
    precip = np.where(wet, rng.gamma(1.2, 5.0, len(idx)), 0.0) * (1 + 0.3 * np.sin(2 * np.pi * (doy - 200) / 365.25))
    snow = np.where(temp < 0, precip * 0.8, 0.0)       # cm, crude
    wind = 20 + 8 * np.cos(2 * np.pi * doy / 365.25) + rng.gamma(2, 3, len(idx))
    solar = np.clip(11 - 10 * np.cos(2 * np.pi * (doy - 172 + 182) / 365.25), 0.3, None) * rng.uniform(0.5, 1.0, len(idx))
    return pd.DataFrame({"temperature_2m_mean": temp, "precipitation_sum": precip,
                         "snowfall_sum": snow, "wind_speed_10m_max": wind,
                         "shortwave_radiation_sum": solar}, index=idx)


def make_synthetic_nve(weather: pd.DataFrame | None = None, area_nr: int = 3, seed=1) -> pd.DataFrame:
    """
    NVE-shaped table whose fill level is driven by the synthetic weather through a tiny
    reservoir model (snow accumulates below 0 °C, melts above, demand peaks in winter).
    Gives the exogenous variables genuine -- but noisy -- predictive value for testing.
    """
    rng = np.random.default_rng(seed)
    w = weather if weather is not None else make_synthetic_weather()
    snowpack, storage, rows = 0.0, 60.0, []
    for date, r in w.iterrows():
        t, p = r["temperature_2m_mean"], r["precipitation_sum"]
        if t < 0:
            snowpack += p
            rain = 0.0
        else:
            rain = p
        melt = min(snowpack, 3.0 * max(t, 0.0))
        snowpack -= melt
        inflow = 0.055 * (rain + melt)
        demand = 0.14 + 0.15 * np.cos(2 * np.pi * (date.dayofyear - 15) / 365.25)
        storage = float(np.clip(storage + inflow - demand + rng.normal(0, 0.03), 0, 100))
        if date.weekday() == 6:                         # NVE publishes weekly (Sundays)
            rows.append((date, storage))
    out = pd.DataFrame(rows, columns=["date", "fill_percentage"])
    out["area_type"], out["area_nr"] = "EL", area_nr      # same columns as load_nve_data()
    return out[["date", "area_type", "area_nr", "fill_percentage"]]
