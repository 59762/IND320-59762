import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from lib import forecast as F
from lib import nve, state, weather
from lib.config import RESERVOIR_VARS, WEATHER_VARS, label

st.set_page_config(page_title="Water Magazine Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 Water Magazine Forecasting")
st.caption("Forecast the weekly reservoir series of the area chosen on the map. Training window = the map's "
           "time period; forecast horizon = the sidebar setting shared with the map page.")

state.sidebar(show_horizon=True)
start, end = state.period()
horizon = int(state.get("horizon"))
atype, anr = state.get("area_type"), state.get("area_nr")
lo, hi = nve.data_range()

top1, top2, top3 = st.columns([1.3, 1.4, 1.3])
var = top1.selectbox("Variable to forecast", list(RESERVOIR_VARS), key="fc_var",
                     format_func=lambda v: RESERVOIR_VARS[v]["label"])
top2.markdown(f"**Area:** {state.selected_label()}  \n**Training window:** {start:%Y-%m-%d} → {end:%Y-%m-%d}  \n"
              f"**Horizon:** {horizon} weeks → {(end + pd.Timedelta(weeks=horizon)):%Y-%m-%d}")
with top3:
    if st.button("Back-test: end training `horizon` weeks before latest data",
                 help="Moves the period end back so the forecast can be compared with observations."):
        state.put("period", (start.date(), (hi - pd.Timedelta(weeks=horizon)).date()))
        st.rerun()
    if end < hi and st.button("Reset period end to latest data"):
        state.put("period", (start.date(), hi.date()))
        st.rerun()

y = nve.area_series(atype, anr, var, start, end)
n_weeks = len(y)
if n_weeks < 60:
    st.warning(f"The training window has only {n_weeks} weeks - choose at least ~2 years in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ parameters
family = st.radio("Model family", ["SARIMAX", "Machine learning"], horizontal=True)
with st.form("fc_form"):
    exog_vars: list[str] = []
    if family == "SARIMAX":
        a, b, c = st.columns(3)
        p = a.number_input("p (AR)", 0, 5, 1)
        d = a.number_input("d (differencing)", 0, 2, 0)
        q = a.number_input("q (MA)", 0, 5, 0)
        P = b.number_input("P (seasonal AR)", 0, 3, 1)
        D = b.number_input("D (seasonal diff.)", 0, 2, 1)
        Q = b.number_input("Q (seasonal MA)", 0, 3, 0)
        s = c.selectbox("Seasonal period s (weeks)", [52, 26, 13, 4, 0], help="52 = annual cycle; 0 = no seasonality")
        trend = c.selectbox("Trend term", [None, "c", "t", "ct"], format_func=lambda v: "none" if v is None else v)
        if s and n_weeks < 2 * s + 20:
            st.info(f"Window of {n_weeks} weeks is short for s = {s}; the fit may be unstable.")
    else:
        a, b, c = st.columns(3)
        mname = a.selectbox("Model", F.ML_MODELS, index=1)
        n_lags = b.slider("Autoregressive lags (weeks)", 1, 12, 4)
        annual = b.checkbox("Add 52-week lag", value=True, help="Used only with > 110 weeks of data.")
        target = c.radio("Target", ["level", "weekly change"], horizontal=True)
        mp = {}
        if mname == "Ridge regression":
            mp["alpha"] = a.slider("Ridge alpha", 0.01, 100.0, 1.0)
        else:
            mp["n_estimators"] = a.slider("Trees / boosting rounds", 50, 600, 200, step=50)
            mp["max_depth"] = b.slider("Max depth", 2, 16, 8 if mname == "Random forest" else 3)
            if mname == "Gradient boosting":
                mp["learning_rate"] = c.slider("Learning rate", 0.01, 0.3, 0.05)
    exog_vars = st.multiselect("Weather as exogenous input (future weather = climatology of the training window)",
                               list(WEATHER_VARS), format_func=lambda v: WEATHER_VARS[v]["label"])
    level = st.slider("Confidence level (%)", 50, 99, 95)
    run = st.form_submit_button("Run forecast", type="primary")

# ------------------------------------------------------------------ run
if run:
    clip = RESERVOIR_VARS[var]["clip"]
    exog = None
    try:
        with st.spinner("Fitting model..."):
            if exog_vars:
                exog = weather.weekly_weather(atype, anr, start, end)[exog_vars]
            alpha = 1 - level / 100
            if family == "SARIMAX":
                fc = F.forecast_sarimax(y, exog, horizon, order=(p, d, q),
                                        seasonal_order=(P, D, Q, s) if s else (0, 0, 0, 0),
                                        trend=trend, alpha=alpha, clip=clip)
                desc = f"SARIMAX({p},{d},{q})" + (f"({P},{D},{Q},{s})" if s else "")
            else:
                fc = F.forecast_ml(y, exog, horizon, mname, mp, n_lags=n_lags, use_annual_lag=annual,
                                   target=target, alpha=alpha, clip=clip)
                desc = f"{mname} ({target}, {n_lags} lags)"
        st.session_state["fc_result"] = dict(fc=fc, desc=desc, area=state.selected_label(), var=var,
                                             start=start, end=end, horizon=horizon, exog=exog_vars)
    except Exception as e:
        st.session_state.pop("fc_result", None)
        st.error(f"Forecast failed: {e}")

res = st.session_state.get("fc_result")
if not res:
    st.info("Set the parameters above and press **Run forecast**.")
    st.stop()

stale = (res["area"], res["var"], res["start"], res["end"], res["horizon"]) != (
    state.selected_label(), var, start, end, horizon)
if stale:
    st.warning("Area, variable, period or horizon changed since the last run - press **Run forecast** to update.")

fc: F.Forecast = res["fc"]
full = nve.area_series(atype, anr, res["var"])
actual_after = full[(full.index > res["end"])]
history = F.regularize(nve.area_series(atype, anr, res["var"], res["start"], res["end"]))
ev = F.evaluate(fc, actual_after)

st.subheader(f"{res['desc']} · {res['area']}")
cols = st.columns(4)
cols[0].metric("Forecast end value", f"{fc.mean.iloc[-1]:.1f} {RESERVOIR_VARS[res['var']]['unit']}")
cols[1].metric(f"{int((1 - fc.alpha) * 100)} % interval at end",
               f"{fc.lower.iloc[-1]:.1f} – {fc.upper.iloc[-1]:.1f}")
for (k, v), col in zip(fc.info.items(), cols[2:]):
    col.metric(k, f"{v:.2f}" if isinstance(v, float) else str(v))

hist_w = st.slider("History shown (weeks)", 26, len(history), min(len(history), max(104, 3 * res["horizon"])))
fig = F.animated_figure(history, fc, actual_after.iloc[:res["horizon"]] if len(actual_after) else None,
                        label(res["var"]), f"{res['desc']} - {res['horizon']}-week forecast", hist_w)
st.plotly_chart(fig)
st.caption("Press ▶ Play to watch the forecast and its confidence band unfold week by week, or drag the slider. "
           + ("Exogenous weather: " + ", ".join(WEATHER_VARS[v]["label"] for v in res["exog"])
              + " (future = climatology)." if res["exog"] else ""))

if ev:
    st.markdown("**Back-test against observations after the cut-off**")
    st.dataframe(pd.DataFrame([ev]).round(3), hide_index=True)
else:
    st.caption("No observations after the training end, so there is nothing to compare with. "
               "Use the back-test button to hold out the last weeks.")

out = pd.DataFrame({"forecast": fc.mean, "lower": fc.lower, "upper": fc.upper})
out.index.name = "week_ending"
with st.expander("Forecast table"):
    st.dataframe(out.round(3))
    st.download_button("Download CSV", out.to_csv().encode(), "forecast.csv", "text/csv")

with st.expander("📄 Method"):
    st.markdown(
        "- **SARIMAX** (`statsmodels`): intervals come from the model's Kalman-filter forecast variance.\n"
        "- **Machine learning** (`scikit-learn`): features are lagged values, sin/cos of the week of year and "
        "optional weather; forecasts are recursive. Intervals are quantiles of 300 simulated paths where each "
        "step adds a resampled one-step error measured on a held-out 20 % of the training window.\n"
        "- Forecasts are clipped to physical limits (fill level 0-100 %).\n"
        "- Training period and horizon are the same settings as on the map page."
    )
