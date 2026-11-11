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

# Page config
st.set_page_config(page_title="Outlier & Anomaly Detection", page_icon="🔍", layout="wide",)
st.title("🔍 Outlier & Anomaly Detection")
st.caption(
    "Precipitation outlier detection using DCT + SPC, and temperature anomaly "
    "detection using Local Outlier Factor (LOF).  "
    "Weather data: Open-Meteo ERA5 reanalysis."
)

NVE_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

# Constants
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

# Weather data loading
@st.cache_data(ttl=21600, show_spinner="Downloading ERA5 weather data from Open-Meteo…")
def load_weather(area_nr: int, year: int, n_locations: int = 20) -> pd.DataFrame:

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


@st.cache_data(ttl=3600, show_spinner="Loading NVE area list…")
def load_nve_areas() -> pd.DataFrame:
    """Return the distinct (area_type, area_nr) pairs available in the NVE data."""
    r = requests.get(f"{NVE_BASE}/HentOffentligData", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={"omrType": "area_type", "omrnr": "area_nr"})
    return df[["area_type", "area_nr"]].dropna().drop_duplicates().reset_index(drop=True)

try:
    areas = load_nve_areas()
except Exception as e:
    st.warning(f"Could not load NVE data: {e}")
    st.stop()

# Area selection 
area_types = sorted(areas["area_type"].dropna().unique().tolist())
col_type, col_nr = st.columns([2, 1])
with col_type:
    area_type = st.radio(
        "Area type",
        area_types,
        #index=area_types.index("EL") if "EL" in area_types else 0,
        horizontal=True,
        help="EL = electricity bidding zones, NO = Norway total, VASS = water catchment areas",
    )
areas = areas[areas["area_type"] == area_type].copy()
area_numbers = sorted(
    areas.loc[areas["area_type"] == area_type, "area_nr"].dropna().astype(int).unique().tolist()
)
with col_nr:
    area_nr = st.selectbox(
        "Area number",
        area_numbers,
        #index=area_numbers.index(3) if area_type == "EL" and 3 in area_numbers else 0,  
    )
areas = areas[areas["area_nr"] == area_nr].copy()
area_label = f"{area_type}{area_nr}"

year = st.selectbox(
    "Year", options=list(range(1990, 2026)), index=34  # 2024 default
)

#n_locations = st.slider(
#    "Locations to average (ERA5)", min_value=1, max_value=100, value=20,
#    help="More locations = better regional estimate but slower download."
#)

#n_points = st.select_slider(
#    "Random points in the area",
#    options=[50, 100, 200, 500, 1000],
#    value=100,
#    help="Number of random points to sample in the area for averaging. More points = better estimate but slower download.",
#)


# Download only when the button is pressed. The result is kept in st.session_state, so
# changing a slider in the tabs below does not trigger a new download.
if st.button("Load weather data"):
        try:
            df_loaded = load_weather(area_nr, year, )
            st.session_state["weather"] = {
                "df": df_loaded, "area_type": area_type, "area_nr": area_nr,
                "year": year,
            }
        except Exception as e:
            st.session_state.pop("weather", None)
            st.error(f"Weather data unavailable: {e}")

if "weather" not in st.session_state:
    st.info("Choose an area and year, "
            "then click **Load weather data**.")
    st.stop()

loaded = st.session_state["weather"]
df_weather = loaded["df"]
st.success(
    f"**{loaded['area_type']}{loaded['area_nr']}**, {loaded['year']} — "
    f"{len(df_weather)} hourly observations loaded "
)
# The loaded data keeps being analysed; tell the user if the inputs above no longer match it
if (loaded["area_type"], loaded["area_nr"]) != (area_type, area_nr) \
        or loaded["year"] != year:
    st.info("The selection above has changed since the data was loaded. "
            "Click **Load weather data** to update the analysis.")


# Load weather data
try:
    df_weather = load_weather(area_nr, year)
except Exception as e:
    st.error(f"Weather data unavailable: {e}")
    st.stop()


# Analysis functions

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


# Tabs
tab_spc, tab_lof = st.tabs(["🌧 Precipitation Outliers (SPC)", "🌡 Temperature Anomalies (LOF)"])


# Tab 1: Precipitation SPC 
with tab_spc:
    
    freq_cutoff = st.slider(
        "DCT freq. cutoff", min_value=0.01, max_value=0.15, value=0.03, step=0.01,
        help="Fraction of low-frequency DCT coefficients removed (seasonal baseline)."
    )
    n_std = st.slider(
        "SPC width (σ)", min_value=1.0, max_value=5.0, value=3.0, step=0.5
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


# Tab 2: Temperature LOF 
with tab_lof:

    outlier_fraction = st.slider(
        "Outlier fraction", min_value=0.005, max_value=0.10, value=0.01, step=0.005,
        format="%.3f",
        help="Expected proportion of anomalous days."
    )
    n_neighbors = st.slider(
        "LOF n_neighbors", min_value=5, max_value=50, value=20, step=5
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

# Data source
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
