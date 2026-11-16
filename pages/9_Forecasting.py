import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this file from pages/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_sources as ds                                   # noqa: E402
import forecasting as fc                                    # noqa: E402
from forecast_plots import forecast_figure, horizon_error_figure   # noqa: E402

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 Reservoir filling forecast")
st.caption("Online machine learning with **River**: lag-based regressors or SNARIMAX, optional weather "
           "(exogenous) variables, prediction intervals from quantile loss.")

D = fc.load_defaults()          # tuned defaults (see tune_forecast.py)

# Streamlit >= 1.50 replaced use_container_width=True by width="stretch" (old name now deprecated).
_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}


# ------------------------------------------------------------------ cached data loading
@st.cache_data(ttl=3600, show_spinner="Downloading Magasinstatistikk from NVE…")
def cached_nve():
    return ds.load_nve_data()


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading ERA5 weather from Open-Meteo…")
def cached_weather(area_type, area_nr, start, end):
    return ds.load_weather_daily(area_type, area_nr, start, end)


@st.cache_data(show_spinner=False)
def cached_demo():
    w = ds.make_synthetic_weather()
    return ds.make_synthetic_nve(w), w


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_forecast(fill, weather, exog, train_start, train_end, params_json):
    """Cache key = data + every parameter. params are passed as JSON to keep them hashable."""
    p = json.loads(params_json)
    p["quantiles"] = tuple(p["quantiles"])
    return fc.forecast_fill(fill, weather, exog=list(exog), train_start=train_start,
                            train_end=train_end, **p)


# ---------------------------------------------------------------- data source (with fallback)
demo = st.sidebar.toggle("Use synthetic demo data", value=st.session_state.get("fc_demo", False),
                         help="Works offline. Useful when NVE / Open-Meteo cannot be reached.")
st.session_state["fc_demo"] = demo
try:
    if demo:
        nve, weather_demo = cached_demo()
    else:
        nve = cached_nve()
except ds.DataSourceError as e:
    st.error(f"{e}\n\nSwitch on **Use synthetic demo data** in the sidebar to explore the page offline.")
    st.stop()

# ---------------------------------------------------------------------------- region choice
types = sorted(nve["area_type"].dropna().unique())
shared = st.session_state.get("selected_area", {})
t_default = shared.get("type", "EL" if "EL" in types else types[0])
with st.sidebar.form("fc_form"):
    st.subheader("Region")
    c1, c2 = st.columns(2)
    area_type = c1.selectbox("Area type", types, index=types.index(t_default) if t_default in types else 0)
    nrs = sorted(nve.loc[nve["area_type"] == area_type, "area_nr"].dropna().astype(int).unique())
    n_default = shared.get("nr", 3)
    area_nr = c2.selectbox("Area", nrs, index=nrs.index(n_default) if n_default in nrs else 0)

    st.subheader("Time frame")
    try:
        fill_all = ds.get_fill_series(nve, area_type, area_nr)
    except ds.DataSourceError as e:
        st.error(str(e)); st.stop()
    d0, d1 = fill_all.index.min().date(), fill_all.index.max().date()
    map_t = st.session_state.get("map_time")               # synced with the map page
    default_range = (max(d0, map_t[0]), min(d1, map_t[1])) if map_t else (max(d0, d1 - pd.Timedelta(days=365 * D["history_years"])), d1)
    train_range = st.slider("Training data (end = forecast origin)", d0, d1, default_range,
                            help="Everything after the end date is hidden from the model and "
                                 "shown as 'Actual' so you can judge the forecast.")
    resolution = st.radio("Time resolution", list(fc.RESOLUTIONS), horizontal=True,
                          index=list(fc.RESOLUTIONS).index(D["resolution"]),
                          format_func=lambda r: fc.RESOLUTIONS[r]["label"],
                          help="Monthly is much faster; weekly keeps NVE's native resolution.")
    horizon = st.slider("Forecast horizon (steps)", 1, 52 if resolution == "W" else 24,
                        D["horizon"] if resolution == D["resolution"] else (26 if resolution == "W" else 6))

    st.subheader("Weather (exogenous)")
    exog = st.multiselect("Weather variables", list(ds.WEATHER_VARS), default=fc.load_default_exog(),
                          format_func=lambda c: ds.WEATHER_VARS[c],
                          help="Used as features at the forecast origin (lag_ml) or via climatology "
                               "for the future (SNARIMAX).")

    st.subheader("Model")
    method = st.radio("Method", ["lag_ml", "snarimax"], horizontal=True,
                      index=["lag_ml", "snarimax"].index(D["method"]),
                      format_func=lambda m: {"lag_ml": "Lag ML", "snarimax": "SNARIMAX"}[m])
    regressor = st.selectbox("Online regressor", ["linear", "pa", "tree"],
                             index=["linear", "pa", "tree"].index(D["regressor"]),
                             help="linear = SGD linear regression, pa = passive-aggressive, tree = Hoeffding adaptive tree")
    with st.expander("Hyper-parameters"):
        lr = st.select_slider("Learning rate (linear)", [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                              value=D["regressor_params"]["lr"], help="Too high diverges (0.1 does).")
        n_lags = st.slider("Fill lags", 1, 24, D["n_lags"])
        exog_lags = st.slider("Weather lags", 1, 8, D["exog_lags"])
        n_harm = st.slider("Season harmonics", 1, 5, D["n_harmonics"])
        delta = st.checkbox("Predict change from last value", D["predict_delta"])
    interval = st.selectbox("Prediction interval", ["quantile", "residual", "none"],
                            index=["quantile", "residual", "none"].index(D["interval"] or "none"),
                            help="quantile = River Quantile loss; residual = empirical online errors")
    animate = st.checkbox("Animate forecast", True)
    go_btn = st.form_submit_button("▶ Run forecast", type="primary", **STRETCH)

st.session_state["selected_area"] = {"type": area_type, "nr": int(area_nr)}   # share with other pages

# ---------------------------------------------------------------------------- weather data
weather, wx_note = None, None
if exog:
    try:
        if demo:
            weather = weather_demo
        else:
            weather = cached_weather(area_type, area_nr, "2000-01-01",
                                     (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
    except ds.DataSourceError as e:
        st.warning(f"{e} Forecasting **without** weather variables.")
        exog = []

# ------------------------------------------------------------------------------ forecast


params = dict(method=method, resolution=resolution, horizon=horizon, n_lags=n_lags,
              exog_lags=exog_lags, n_harmonics=n_harm, regressor=regressor,
              regressor_params={"lr": lr, "l2": 0.0}, predict_delta=delta,
              interval=None if interval == "none" else interval, quantiles=list(D["quantiles"]))
try:
    with st.spinner("Training River models online and forecasting…"):
        res = cached_forecast(fill_all, weather, tuple(exog), pd.Timestamp(train_range[0]),
                              pd.Timestamp(train_range[1]), json.dumps(params, sort_keys=True))
except fc.ForecastError as e:
    st.warning(f"⚠️ {e}")
    st.stop()
except Exception as e:                                       # never show a raw traceback to the user
    st.error(f"Forecast failed: {e}")
    st.stop()

for n in res.notes:
    st.info(n)

# ------------------------------------------------------------------------------ results
m = res.metrics
cols = st.columns(4)
cols[0].metric("Forecast origin", f"{res.origin:%Y-%m-%d}")
cols[1].metric("Training points", len(res.history))
if m:
    cols[2].metric("MAE vs actual", f"{m['mae']:.2f} pp",
                   delta=f"{m['mae'] - m['mae_seasonal_naive']:+.2f} vs seasonal-naive", delta_color="inverse")
    cols[3].metric("Interval coverage", f"{m['coverage']:.0%}" if "coverage" in m else "–")
else:
    cols[2].metric("MAE vs actual", "–", help="No observations after the forecast origin.")
    cols[3].metric("Run time", f"{res.elapsed_s:.2f} s")

st.plotly_chart(forecast_figure(res, animate=animate,
                                title=f"{area_type}{area_nr} – {fc.RESOLUTIONS[resolution]['label']} fill forecast"),
                **STRETCH)

left, right = st.columns([1, 1], gap="large")
with left:
    if res.horizon_mae.notna().any():
        st.plotly_chart(horizon_error_figure(res), **STRETCH)
with right:
    st.subheader("Forecast table")
    st.dataframe(res.forecast.round(2), **STRETCH)
    st.download_button("Download CSV", res.forecast.to_csv().encode(), f"forecast_{area_type}{area_nr}.csv")

with st.expander("Settings used / how it works"):
    st.json({**res.params, "exog_used": res.exog_used, "elapsed_s": round(res.elapsed_s, 2)})
    st.markdown(
        "Weekly NVE data is aggregated to the chosen resolution; weather (daily ERA5) is summed (precipitation, "
        "snow) or averaged (the rest) to the same step. The model is trained **online**, walking forward in time, "
        "so the forecast only uses data up to the origin. See `forecasting.py`.")
