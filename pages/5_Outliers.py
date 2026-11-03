"""
pages/5_Outliers.py
IND320 – Part 3 – Data Quality
Tab 1: Precipitation outlier detection (DCT high-pass + SPC control bounds)
Tab 2: Temperature anomaly detection (Local Outlier Factor)

Weather data (hourly ERA5) is downloaded on-demand from Open-Meteo for the
selected electricity area.  Results are cached for 6 hours.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Outlier & Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 Outlier & Anomaly Detection")
st.caption(
    "Precipitation outlier detection using DCT + SPC, and temperature anomaly "
    "detection using Local Outlier Factor (LOF).  "
    "Weather data: Open-Meteo ERA5 reanalysis."
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/era5"

AREA_NAMES = {
    1: "NO1 – Eastern Norway",
    2: "NO2 – South-western Norway",
    3: "NO3 – Central Norway",
    4: "NO4 – Northern Norway",
    5: "NO5 – Western Norway",
}

# Approximate bounding boxes for rejection-sampling fallback
# {area_nr: (lon_min, lon_max, lat_min, lat_max)}
AREA_BBOX = {
    1: ( 9.5, 13.0, 58.5, 61.0),
    2: ( 4.8,  9.0, 57.9, 60.0),
    3: ( 8.0, 17.5, 62.0, 65.5),
    4: (13.5, 30.0, 65.5, 71.2),
    5: ( 4.9,  8.8, 60.0, 62.5),
}

# Representative centre point per area (used when API is called in demo mode)
AREA_CENTRE = {
    1: ( 10.75,  59.9),
    2: (  6.5,   58.7),
    3: ( 10.5,   63.4),
    4: ( 17.0,   68.4),
    5: (  6.0,   61.2),
}


# ─────────────────────────────────────────────────────────────────────────────
# Weather data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner="Downloading ERA5 weather data from Open-Meteo…")
def load_weather(area_nr: int, year: int, n_locations: int = 20) -> pd.DataFrame:
    """Download hourly ERA5 precipitation and temperature for an electricity area.

    For interactive use we limit to `n_locations` sample points so the call
    returns quickly (< 30 s for 20 locations).

    Parameters
    ----------
    area_nr : int   NVE electricity area number (1–5).
    year : int      Calendar year to fetch.
    n_locations : int  Number of random locations to average.  Default 20.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns: precipitation, temperature_2m
    """
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    rng = np.random.default_rng(seed=area_nr * 31 + year)
    lon_min, lon_max, lat_min, lat_max = AREA_BBOX[area_nr]
    lons = rng.uniform(lon_min, lon_max, n_locations)
    lats = rng.uniform(lat_min, lat_max, n_locations)

    all_dfs = []
    for lon, lat in zip(lons, lats):
        try:
            r = requests.get(
                OPENMETEO_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "start_date": start_date,
                    "end_date":   end_date,
                    "hourly":     "precipitation,temperature_2m",
                    "models":     "era5",
                    "timezone":   "Europe/Oslo",
                    "timeformat": "iso8601",
                },
                timeout=30,
            )
            r.raise_for_status()
            h = r.json().get("hourly", {})
            if not h or "time" not in h:
                continue
            df_loc = pd.DataFrame(h)
            df_loc["time"] = pd.to_datetime(df_loc["time"])
            df_loc.set_index("time", inplace=True)
            all_dfs.append(df_loc[["precipitation", "temperature_2m"]])
        except Exception:
            pass

    if not all_dfs:
        raise RuntimeError(
            "No weather data could be retrieved from Open-Meteo.  "
            "Check your internet connection."
        )

    df_mean = pd.concat(all_dfs).groupby(level=0).mean()
    df_mean.index.name = "datetime"
    return df_mean


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def dct_spc_outliers(
    df: pd.DataFrame,
    col: str = "precipitation",
    freq_cutoff: float = 0.03,
    n_std: float = 3.0,
) -> tuple[plt.Figure, pd.DataFrame]:
    """DCT high-pass filter + robust SPC outlier detection."""
    series = df[col].resample("D").mean().dropna()
    values = series.values.astype(float)
    n = len(values)

    # DCT-II → zero out low-frequency baseline → IDCT = SATV
    coeffs = dct(values, norm="ortho")
    cutoff_idx = max(1, int(freq_cutoff * n))
    coeffs_hp = coeffs.copy()
    coeffs_hp[:cutoff_idx] = 0.0
    satv = idct(coeffs_hp, norm="ortho")

    # Robust SPC: MAD × 1.4826 ≈ σ for Gaussian
    med = np.median(satv)
    mad_scaled = np.median(np.abs(satv - med)) * 1.4826
    ucl = med + n_std * mad_scaled
    lcl = med - n_std * mad_scaled
    is_outlier = (satv > ucl) | (satv < lcl)
    baseline = values - satv

    summary = pd.DataFrame({
        "date":  series.index[is_outlier],
        "value": values[is_outlier],
        "satv":  satv[is_outlier],
    })

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, values,
            color="steelblue", linewidth=0.8, label="Daily mean", zorder=2)
    ax.scatter(
        series.index[is_outlier], values[is_outlier],
        color="crimson", s=55, zorder=4,
        label=f"Outlier  (n={is_outlier.sum()}, {100*is_outlier.mean():.1f}%)"
    )
    ax.fill_between(series.index, baseline + lcl, baseline + ucl,
                    color="orange", alpha=0.15, label=f"SPC ±{n_std}σ band")
    ax.plot(series.index, baseline + ucl,
            color="darkorange", linewidth=0.8, linestyle="--")
    ax.plot(series.index, baseline + lcl,
            color="darkorange", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Daily Precipitation – DCT High-Pass + SPC Outlier Detection\n"
        f"freq_cutoff={freq_cutoff}, n_std={n_std}",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation (mm/h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, summary


def lof_temperature_anomalies(
    df: pd.DataFrame,
    col: str = "temperature_2m",
    outlier_fraction: float = 0.01,
    n_neighbors: int = 20,
) -> tuple[plt.Figure, pd.DataFrame]:
    """LOF-based temperature anomaly detection."""
    series = df[col].resample("D").mean().dropna()
    X = series.values.reshape(-1, 1)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=outlier_fraction,
        novelty=False,
    )
    labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_
    is_anomaly = labels == -1

    summary = pd.DataFrame({
        "date":        series.index[is_anomaly],
        "temperature": series.values[is_anomaly],
        "lof_score":   lof_scores[is_anomaly],
    }).sort_values("lof_score", ascending=False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(series.index, series.values,
             color="steelblue", linewidth=0.8, label="Daily mean temp")
    ax1.scatter(
        series.index[is_anomaly], series.values[is_anomaly],
        color="crimson", s=60, zorder=4,
        label=f"Anomaly (n={is_anomaly.sum()}, {100*is_anomaly.mean():.1f}%)"
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Daily Temperature – LOF Anomaly Detection\n"
        f"outlier_fraction={outlier_fraction}, n_neighbors={n_neighbors}",
        fontsize=12,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(series.index, lof_scores,
            color="lightsteelblue", width=1.5, label="LOF score")
    ax2.scatter(series.index[is_anomaly], lof_scores[is_anomaly],
                color="crimson", s=30, zorder=4)
    ax2.axhline(1.0, color="darkorange", linestyle="--", linewidth=0.9,
                label="LOF = 1 (baseline)")
    ax2.set_ylabel("LOF score")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    return fig, summary


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

area_nr = st.sidebar.selectbox(
    "Electricity area",
    options=list(AREA_NAMES.keys()),
    format_func=lambda x: AREA_NAMES[x],
    index=2,   # NO3 default
)

year = st.sidebar.number_input(
    "Year", min_value=1990, max_value=2025, value=2024, step=1
)

n_locations = st.sidebar.slider(
    "Locations to average (ERA5)", min_value=1, max_value=100, value=20,
    help="More locations = better regional estimate but slower download."
)

st.sidebar.markdown("---")
st.sidebar.subheader("SPC / DCT parameters")
freq_cutoff = st.sidebar.slider(
    "DCT freq. cutoff", min_value=0.01, max_value=0.15, value=0.03, step=0.01,
    help="Fraction of low-frequency DCT coefficients removed (seasonal baseline)."
)
n_std = st.sidebar.slider(
    "SPC width (σ)", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("LOF parameters")
outlier_fraction = st.sidebar.slider(
    "Outlier fraction", min_value=0.005, max_value=0.10, value=0.01, step=0.005,
    format="%.3f",
    help="Expected proportion of anomalous days."
)
n_neighbors = st.sidebar.slider(
    "LOF n_neighbors", min_value=5, max_value=50, value=20, step=5
)

# ─────────────────────────────────────────────────────────────────────────────
# Load weather data
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_weather = load_weather(area_nr, year, n_locations=n_locations)
    st.success(
        f"**{AREA_NAMES[area_nr]}**, {year} — "
        f"{len(df_weather)} hourly observations loaded "
        f"(mean of {n_locations} ERA5 locations)."
    )
except Exception as e:
    st.error(f"Weather data unavailable: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_spc, tab_lof = st.tabs(["🌧 Precipitation Outliers (SPC)", "🌡 Temperature Anomalies (LOF)"])


# ── Tab 1: Precipitation SPC ──────────────────────────────────────────────────
with tab_spc:
    st.subheader("Precipitation – DCT High-Pass Filter + SPC Control Limits")
    st.markdown(
        """
        **Method:**
        1. Resample hourly precipitation to daily mean.
        2. Apply **Discrete Cosine Transform (DCT-II)**.
           Zero out the lowest `freq_cutoff` fraction of coefficients (seasonal baseline).
           Invert (IDCT) → *Seasonally Adjusted Variation* (**SATV**).
        3. Compute **robust SPC limits**: `median ± n_std × MAD_scaled`
           where `MAD_scaled = MAD × 1.4826 ≈ σ` (unaffected by outliers themselves).
        4. Flag days outside the control band as **outliers** (shown in red).
        """
    )

    with st.spinner("Running DCT + SPC analysis…"):
        try:
            fig_spc, outliers_df = dct_spc_outliers(
                df_weather,
                freq_cutoff=freq_cutoff,
                n_std=n_std,
            )
            st.pyplot(fig_spc)
            plt.close(fig_spc)

            st.metric(
                "Outlier days",
                f"{len(outliers_df)}",
                f"{100 * len(outliers_df) / df_weather.resample('D').mean().dropna().shape[0]:.1f}%"
            )

            if not outliers_df.empty:
                with st.expander("Show outlier table"):
                    st.dataframe(
                        outliers_df.reset_index(drop=True)
                        .assign(date=lambda d: d["date"].dt.date),
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Analysis failed: {e}")


# ── Tab 2: Temperature LOF ────────────────────────────────────────────────────
with tab_lof:
    st.subheader("Temperature – Local Outlier Factor (LOF) Anomaly Detection")
    st.markdown(
        """
        **Method:**
        LOF compares the local density of each data point to the densities of its
        *k* nearest neighbours.  Points surrounded by sparse neighbourhoods (LOF >> 1)
        are flagged as anomalies.

        - **Upper panel:** daily mean temperature with anomaly days in red.
        - **Lower panel:** LOF score for each day (higher = more anomalous).
        """
    )

    with st.spinner("Running LOF anomaly detection…"):
        try:
            fig_lof, anomalies_df = lof_temperature_anomalies(
                df_weather,
                outlier_fraction=outlier_fraction,
                n_neighbors=n_neighbors,
            )
            st.pyplot(fig_lof)
            plt.close(fig_lof)

            st.metric(
                "Anomaly days",
                f"{len(anomalies_df)}",
                f"{100 * len(anomalies_df) / df_weather.resample('D').mean().dropna().shape[0]:.1f}%"
            )

            if not anomalies_df.empty:
                with st.expander("Show anomaly table"):
                    st.dataframe(
                        anomalies_df.reset_index(drop=True)
                        .assign(date=lambda d: d["date"].dt.date)
                        [["date", "temperature", "lof_score"]],
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Data source
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📄 Data source"):
    st.markdown(
        f"""
        **Open-Meteo Historical Weather API** – ERA5 reanalysis model
        Endpoint: `{OPENMETEO_URL}`
        Variables: `precipitation`, `temperature_2m`
        Timezone: `Europe/Oslo`

        **Methods:**
        - DCT: `scipy.fft.dct` / `scipy.fft.idct`
        - LOF: `sklearn.neighbors.LocalOutlierFactor`
        """
    )
