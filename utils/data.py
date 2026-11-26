"""Data loading for the part-4 pages: NVE Magasinstatistikk and Open-Meteo ERA5 (daily)."""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from utils import geo
from utils.config import (ERA5_LAG_DAYS, ERA5_URL, NVE_URL, RESERVOIR_VARS, WEATHER_VARS)


class DataError(RuntimeError):
    """Problem with an external source, with a message that can be shown to the user."""


# ------------------------------------------------------------------------------ NVE
NVE_RENAME = {
    "dato_Id": "date", "omrType": "area_type", "omrnr": "area_nr",
    "fyllingsgrad": "fill_percentage", "kapasitet_TWh": "capacity_TWh", "fylling_TWh": "fill_TWh",
    "fyllingsgrad_forrige_uke": "prev_week_fill_pct", "endring_fyllingsgrad": "change_fill_pct",
}


@st.cache_data(ttl=3600, show_spinner="Loading NVE Magasinstatistikk…")
def load_nve() -> pd.DataFrame:
    """Full Magasinstatistikk table with English column names (filling in percent)."""
    try:
        r = requests.get(NVE_URL, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:
        raise DataError(f"Could not download Magasinstatistikk from NVE: {e}") from e
    df = df.rename(columns={k: v for k, v in NVE_RENAME.items() if k in df.columns})
    missing = {"date", "area_type", "area_nr", "fill_percentage"} - set(df.columns)
    if missing:
        raise DataError(f"NVE response is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["area_nr"] = pd.to_numeric(df["area_nr"], errors="coerce")
    df = df.dropna(subset=["date", "area_nr"]).sort_values("date").reset_index(drop=True)
    df["area_nr"] = df["area_nr"].astype(int)
    # NVE may deliver the filling as a fraction (0-1); the app always works in percent.
    if df["fill_percentage"].max() <= 1.5:
        for c in ("fill_percentage", "prev_week_fill_pct", "change_fill_pct"):
            if c in df:
                df[c] = df[c] * 100.0
    if "change_fill_pct" not in df:
        df["change_fill_pct"] = df.groupby(["area_type", "area_nr"])["fill_percentage"].diff()
    return df


def data_range(df: pd.DataFrame) -> tuple[date, date]:
    return df["date"].min().date(), df["date"].max().date()


def area_numbers_in_data(df: pd.DataFrame, area_type: str) -> list[int]:
    return sorted(df.loc[df["area_type"] == area_type, "area_nr"].unique().tolist())


def area_series(df: pd.DataFrame, area_type: str, area_nr: int, var: str,
                start=None, end=None) -> pd.Series:
    """Weekly (week ending Sunday) series of one Magasinstatistikk variable for one area."""
    m = df["area_type"] == area_type
    if area_type != "NO":                       # NO is one series whatever number NVE gives it
        m &= df["area_nr"] == int(area_nr)
    s = df.loc[m].set_index("date")[var].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")].resample("W-SUN").last()
    if start is not None or end is not None:
        s = s.loc[pd.Timestamp(start) if start is not None else None:
                  pd.Timestamp(end) if end is not None else None]
    s.name = var
    return s


def area_value(df: pd.DataFrame, area_type: str, area_nr: int, var: str, start, end, stat: str) -> float:
    s = area_series(df, area_type, area_nr, var, start, end).dropna()
    if s.empty:
        return np.nan
    if stat == "Latest in period":
        return float(s.iloc[-1])
    if stat == "Change over period":
        return float(s.iloc[-1] - s.iloc[0])
    return float(s.mean())


# ------------------------------------------------------------------------- weather
def _get_json(params: dict, tries: int = 4):
    for i in range(tries):
        try:
            r = requests.get(ERA5_URL, params=params, timeout=60)
        except requests.RequestException as e:
            raise DataError(f"Could not reach Open-Meteo: {e}") from e
        if r.status_code == 429:                # rate limited -> wait and retry
            time.sleep(2 * (i + 1))
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise DataError(f"Open-Meteo error: {e}") from e
        return r.json()
    raise DataError("Open-Meteo rate limit reached - wait a minute, or shorten the time frame / "
                    "reduce the number of sample points.")


def latest_weather_date() -> date:
    return date.today() - timedelta(days=ERA5_LAG_DAYS)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _weather_year(area_type: str, area_nr: int, year: int, n_points: int) -> pd.DataFrame:
    """Daily ERA5 for one calendar year: mean over n_points random points INSIDE the area polygon.

    One call per (area, year) so that moving the time slider inside a year never re-downloads.
    """
    start, end = date(year, 1, 1), min(date(year, 12, 31), latest_weather_date())
    if end < start:
        return pd.DataFrame(columns=list(WEATHER_VARS), index=pd.DatetimeIndex([], name="time"), dtype=float)
    pts = geo.sample_points(area_type, area_nr, n_points)
    js = _get_json({
        "latitude": ",".join(f"{la:.4f}" for la, _ in pts),
        "longitude": ",".join(f"{lo:.4f}" for _, lo in pts),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(WEATHER_VARS), "timezone": "Europe/Oslo",
    })
    frames = []
    for block in (js if isinstance(js, list) else [js]):    # several points -> a list of blocks
        d = block.get("daily")
        if d and "time" in d:
            f = pd.DataFrame(d)
            f["time"] = pd.to_datetime(f["time"])
            frames.append(f.set_index("time"))
    if not frames:
        raise DataError("Open-Meteo returned no daily data.")
    return pd.concat(frames).groupby(level=0).mean()[list(WEATHER_VARS)]


def weather_daily(area_type: str, area_nr: int, start, end, n_points: int) -> pd.DataFrame:
    """Daily weather (area mean) between start and end (inclusive)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = [_weather_year(area_type, int(area_nr), y, int(n_points))
             for y in range(start.year, end.year + 1)]
    out = pd.concat(parts).sort_index()
    return out.loc[start:end]


def weather_weekly(area_type: str, area_nr: int, start, end, n_points: int) -> pd.DataFrame:
    """Weekly (week ending Sunday) weather: sums for precipitation/snow/radiation, means otherwise.
    Weeks with fewer than 5 observed days are left empty."""
    daily = weather_daily(area_type, area_nr, start, end, n_points)
    rules = {c: WEATHER_VARS[c]["agg"] for c in WEATHER_VARS}
    weekly = daily.resample("W-SUN").agg(rules)
    n_obs = daily.notna().resample("W-SUN").sum()
    return weekly.where(n_obs >= 5)


def weather_period_value(area_type: str, area_nr: int, start, end, var: str, n_points: int) -> float:
    """One number per area for the map: mean daily value over the period; for precipitation, snow
    and radiation the mean WEEKLY total (daily mean x 7), so periods of any length are comparable."""
    daily = weather_daily(area_type, area_nr, start, end, n_points)[var].dropna()
    if daily.empty:
        return np.nan
    v = float(daily.mean())
    return v * 7.0 if WEATHER_VARS[var]["agg"] == "sum" else v


def weather_unit(var: str) -> str:
    u = WEATHER_VARS[var]["unit"]
    return f"{u}/week" if WEATHER_VARS[var]["agg"] == "sum" else u


def var_label(var: str) -> str:
    meta = RESERVOIR_VARS.get(var) or WEATHER_VARS.get(var)
    return f"{meta['label']} ({meta['unit']})"
