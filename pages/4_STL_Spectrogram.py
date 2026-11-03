import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from scipy.signal import spectrogram as stft_spectrogram
from statsmodels.tsa.seasonal import STL

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STL & Spectrogram",
    page_icon="📈",
    layout="wide",
)
st.title("📈 STL Decomposition & Spectrogram")
st.caption(
    "Seasonal-Trend decomposition using LOESS (STL) and Short-Time "
    "Fourier Transform spectrogram of Norwegian reservoir fill-percentage data."
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (shared with other pages)
# ─────────────────────────────────────────────────────────────────────────────
NVE_API = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

AREA_NAMES = {
    1: "NO1 – Eastern Norway",
    2: "NO2 – South-western Norway",
    3: "NO3 – Central Norway",
    4: "NO4 – Northern Norway",
    5: "NO5 – Western Norway",
}

RENAME_MAP = {
    "dato_Id":                  "date",
    "omrType":                  "area_type",
    "omrnr":                    "area_nr",
    "fyllingsgrad":             "fill_percentage",
    "kapasitet_TWh":            "capacity_TWh",
    "fylling_TWh":              "fill_TWh",
    "fyllingsgrad_forrige_uke": "prev_week_fill_pct",
    "endring_fyllingsgrad":     "change_fill_pct",
}


@st.cache_data(ttl=3600, show_spinner="Loading NVE reservoir data…")
def load_nve_data() -> pd.DataFrame:
    """Fetch and cache the full NVE Magasinstatistikk dataset."""
    r = requests.get(f"{NVE_API}/HentOffentligData", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_fill_series(df: pd.DataFrame, area_nr: int) -> pd.Series:
    """Extract the weekly fill-percentage series for a single electricity area."""
    mask = (df["area_type"] == "EL") & (df["area_nr"] == area_nr)
    s = (
        df.loc[mask]
        .set_index("date")["fill_percentage"]
        .resample("W").mean()
        .interpolate(method="linear")
    )
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls (shared across both tabs)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

area_nr = st.sidebar.selectbox(
    "Electricity area",
    options=list(AREA_NAMES.keys()),
    format_func=lambda x: AREA_NAMES[x],
    index=2,   # default NO3
)

st.sidebar.markdown("---")
st.sidebar.subheader("STL parameters")
stl_period   = st.sidebar.number_input("Period (weeks)", min_value=4, max_value=104, value=52, step=1)
stl_seasonal = st.sidebar.slider("Seasonal smoother", min_value=3, max_value=51, value=13, step=2,
                                  help="Must be odd; larger = smoother seasonal component")
stl_robust   = st.sidebar.checkbox("Robust fitting", value=True,
                                    help="Reduces influence of outliers on trend estimation")

st.sidebar.markdown("---")
st.sidebar.subheader("Spectrogram parameters")
spec_window  = st.sidebar.slider("Window length (weeks)", min_value=26, max_value=260, value=104, step=2)
spec_overlap = st.sidebar.slider("Overlap (weeks)", min_value=0, max_value=spec_window - 1,
                                  value=int(spec_window * 0.75))

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_nve = load_nve_data()
    fill_series = get_fill_series(df_nve, area_nr)
except Exception as e:
    st.error(f"Could not load NVE data: {e}")
    st.stop()

if fill_series.empty or fill_series.dropna().empty:
    st.warning(f"No fill-percentage data found for area {area_nr}.")
    st.stop()

clean = fill_series.dropna()

st.info(
    f"**{AREA_NAMES[area_nr]}** – "
    f"{len(clean)} weekly observations, "
    f"{clean.index[0].date()} → {clean.index[-1].date()}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_stl, tab_spec = st.tabs(["📉 STL Decomposition", "🔊 Spectrogram"])


# ── Tab 1: STL ────────────────────────────────────────────────────────────────
with tab_stl:
    st.subheader("Seasonal-Trend Decomposition using LOESS (STL)")
    st.markdown(
        """
        STL splits the time series into three additive components:
        - **Trend** – long-run movement in reservoir levels
        - **Seasonal** – the annual filling-and-drawdown cycle
        - **Residual** – remaining noise and irregular fluctuations

        The seasonal component should capture the winter drawdown and spring/summer refill.
        A flat residual suggests the model explains most variation.
        """
    )

    with st.spinner("Running STL decomposition…"):
        try:
            seasonal = stl_seasonal if stl_seasonal % 2 == 1 else stl_seasonal + 1
            stl_model = STL(
                clean,
                period=stl_period,
                seasonal=seasonal,
                robust=stl_robust,
            )
            result = stl_model.fit()

            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            components = [
                (clean,            "Observed",  "steelblue"),
                (result.trend,     "Trend",     "#e67e22"),
                (result.seasonal,  "Seasonal",  "#27ae60"),
                (result.resid,     "Residual",  "#8e44ad"),
            ]
            for ax, (data, label, color) in zip(axes, components):
                ax.plot(clean.index, data, color=color, linewidth=0.9)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                if label == "Residual":
                    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            axes[0].set_title(
                f"STL Decomposition – {AREA_NAMES[area_nr]} Fill Percentage (%)\n"
                f"period={stl_period}, seasonal={seasonal}, robust={stl_robust}",
                fontsize=12,
            )
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axes[-1].set_xlabel("Date")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Seasonal σ", f"{result.seasonal.std():.2f} %")
            col2.metric("Trend σ",    f"{result.trend.std():.2f} %")
            col3.metric("Residual σ", f"{result.resid.std():.2f} %")

        except Exception as e:
            st.error(f"STL failed: {e}")


# ── Tab 2: Spectrogram ────────────────────────────────────────────────────────
with tab_spec:
    st.subheader("Power Spectrogram (STFT)")
    st.markdown(
        """
        A spectrogram shows how the frequency content of the signal evolves over time.
        - The **x-axis** is time (years from start of series).
        - The **y-axis** is frequency in cycles per year.
        - **Colour** (dark → bright) encodes power in dB.

        The strong horizontal band at **1 cycle/year** reflects the predictable annual
        filling cycle.  Bands at higher frequencies indicate shorter-period fluctuations.
        """
    )

    with st.spinner("Computing spectrogram…"):
        try:
            x = clean.values.astype(float)
            noverlap = min(spec_overlap, spec_window - 1)

            f_cpw, t_seg, Sxx = stft_spectrogram(
                x,
                fs=1.0,
                window="hann",
                nperseg=spec_window,
                noverlap=noverlap,
            )
            f_cpy  = f_cpw * 52.0           # cycles/week → cycles/year
            t_yr   = t_seg / 52.0           # weeks → years from start

            fig, ax = plt.subplots(figsize=(14, 5))
            pcm = ax.pcolormesh(
                t_yr,
                f_cpy,
                10 * np.log10(Sxx + 1e-12),
                cmap="inferno",
                shading="gouraud",
            )
            fig.colorbar(pcm, ax=ax, label="Power (dB)")
            ax.axhline(1.0, color="white", linewidth=0.9, linestyle="--",
                       label="Annual (1 cy/yr)")
            ax.axhline(2.0, color="cyan",  linewidth=0.9, linestyle="--",
                       label="Semi-annual (2 cy/yr)")
            ax.set_ylim(0, 10)
            ax.set_xlabel("Time from series start (years)")
            ax.set_ylabel("Frequency (cycles per year)")
            ax.set_title(
                f"Spectrogram – {AREA_NAMES[area_nr]} Fill Percentage\n"
                f"window={spec_window} weeks, overlap={noverlap} weeks",
                fontsize=12,
            )
            ax.legend(loc="upper right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Spectrogram failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Data source
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📄 Data source"):
    st.markdown(
        """
        **NVE Magasinstatistikk** – weekly reservoir fill-percentage data
        Endpoint: `https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData`

        **Methods:**
        - STL: `statsmodels.tsa.seasonal.STL`
        - Spectrogram: `scipy.signal.spectrogram` (Hann window, STFT)
        """
    )
