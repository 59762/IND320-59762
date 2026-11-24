import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from utils import data, forecast as F, state
from utils.config import MAX_WEATHER_YEARS, RESERVOIR_VARS, WEATHER_VARS

st.set_page_config(page_title="Water Magazine Forecasting", page_icon="🔮", layout="wide")
state.init()

st.title("🔮 Water Magazine Forecasting")
st.caption("Forecast the weekly reservoir series of the area selected on the map. The training window is the "
           "map's time frame and the horizon is the same setting as on the map page.")

try:
    df = data.load_nve()
except data.DataError as e:
    st.error(str(e))
    st.stop()
lo, hi = data.data_range(df)
state.ensure_period(lo, hi)
hi_ts = pd.Timestamp(hi)

# area, variable, time frame
with st.container(border=True):
    state.area_picker()
    state.period_slider(lo, hi, "Training window (shared with the map)")
    c1, c2, c3 = st.columns([1, 1.3, 2])
    with c1:
        state.horizon_input()
    with c2:
        fvar = st.selectbox("Variable to forecast", list(RESERVOIR_VARS), key=state.bind("fc_var"),
                            format_func=lambda v: RESERVOIR_VARS[v]["label"], on_change=state.commit,
                            args=("fc_var",))
    start, end = state.period()
    atype, anr = state.selected()
    horizon = int(state.get("horizon"))
    with c3:
        st.markdown(f"**Area:** {state.selected_label()}  \n"
                    f"**Training:** {start:%Y-%m-%d} → {end:%Y-%m-%d} · "
                    f"**Forecast:** {horizon} weeks → {end + pd.Timedelta(weeks=horizon):%Y-%m-%d}")
        b1, b2 = st.columns(2)
        if b1.button("Hold out the last weeks (back-test)",
                     help="Moves the end of the training window back by the horizon, so the forecast can be "
                          "compared with what really happened."):
            state.put("period", (start.date(), (hi_ts - pd.Timedelta(weeks=horizon)).date()))
            st.rerun()
        if end < hi_ts - pd.Timedelta(days=7) and b2.button("Train up to the latest data"):
            state.put("period", (start.date(), hi))
            st.rerun()

raw = data.area_series(df, atype, anr, fvar, start, end)
try:
    y = F.regularize(raw)
except F.ForecastError as e:
    st.warning(str(e))
    st.stop()
if len(y) < 40:
    st.warning(f"The training window has only {len(y)} weeks; choose at least about two years.")
    st.stop()
clip = RESERVOIR_VARS[fvar]["clip"]

# model + parameters
family = st.radio("Model family", ["SARIMAX", "Machine learning"], horizontal=True)
with st.form("forecast_form"):
    if family == "SARIMAX":
        a, b, c = st.columns(3)
        p = a.number_input("p (AR)", 0, 5, 1)
        d = a.number_input("d (differencing)", 0, 2, 0)
        q = a.number_input("q (MA)", 0, 5, 0)
        P = b.number_input("P (seasonal AR)", 0, 3, 1)
        D = b.number_input("D (seasonal differencing)", 0, 2, 1)
        Q = b.number_input("Q (seasonal MA)", 0, 3, 0)
        s = c.selectbox("Seasonal period (weeks)", [52, 26, 13, 4, 0],
                        format_func=lambda v: "none" if v == 0 else str(v), help="52 = the annual cycle")
        trend = c.selectbox("Trend term", [None, "c", "t", "ct"],
                            format_func=lambda v: "none" if v is None else {"c": "constant", "t": "linear", "ct": "both"}[v])
    else:
        a, b, c = st.columns(3)
        mname = a.selectbox("Model", F.ML_MODELS, index=1)
        n_lags = b.slider("Autoregressive lags (weeks)", 1, 12, 4)
        annual = b.checkbox("Add the value 52 weeks back", value=True, help="Needs more than 110 weeks of training data.")
        target = c.radio("Target", ["level", "weekly change"], horizontal=True,
                         help="'weekly change' models the difference to the previous week (often steadier).")
        exog_lags = c.slider("Weather lags (weeks)", 0, 8, 2, help="Weather from this many weeks back is used as well.")
        mp = {}
        if mname == "Ridge regression":
            mp["alpha"] = a.slider("Ridge alpha", 0.01, 100.0, 1.0)
        else:
            mp["n_estimators"] = a.slider("Trees / boosting rounds", 50, 600, 200, step=50)
            mp["max_depth"] = a.slider("Max depth", 2, 16, 8 if mname == "Random forest" else 3)
            if mname == "Gradient boosting":
                mp["learning_rate"] = b.slider("Learning rate", 0.01, 0.3, 0.05)

    e1, e2, e3 = st.columns([2, 1.5, 1])
    exog_vars = e1.multiselect("Exogenous weather variables", list(WEATHER_VARS),
                               format_func=lambda v: WEATHER_VARS[v]["label"],
                               help="Weekly ERA5 weather for the selected area, used as extra model inputs.")
    fut_mode = e2.radio("Future weather", ["Climatology", "Observed (back-test only)"],
                        help="The forecast needs weather for the coming weeks. Climatology = the average of each "
                             "calendar week in the training window. Observed = what actually happened: only "
                             "possible when the horizon lies in the past, and optimistic.")
    level = e3.slider("Confidence level (%)", 50, 99, 95)
    run = st.form_submit_button("Run forecast", type="primary")

# run
if run:
    exog_hist = exog_fut = None
    n_clim = 0
    try:
        with st.spinner("Fitting the model …"):
            fut_idx = F.future_index(y, horizon)
            if exog_vars:
                if (y.index[-1] - y.index[0]).days > 365.25 * MAX_WEATHER_YEARS:
                    raise F.ForecastError(f"Weather can cover at most {MAX_WEATHER_YEARS} years: shorten the "
                                          "training window or remove the weather variables.")
                observed = fut_mode.startswith("Observed")
                w_end = min(fut_idx[-1], pd.Timestamp(data.latest_weather_date())) if observed else y.index[-1]
                wk = data.weather_weekly(atype, anr, y.index[0] - pd.Timedelta(days=6), w_end, int(state.get("n_points")))
                exog_hist, exog_fut, n_clim = F.build_exog(wk, exog_vars, y.index, fut_idx,
                                                           "observed" if observed else "climatology")
            alpha = 1 - level / 100
            if family == "SARIMAX":
                fc = F.forecast_sarimax(y, exog_hist, exog_fut, horizon, (p, d, q), (P, D, Q, s), trend, alpha, clip)
            else:
                fc = F.forecast_ml(y, exog_hist, exog_fut, horizon, mname, mp, n_lags, annual, exog_lags, target, alpha, clip)
        if n_clim:
            fc.notes.append(f"{n_clim} of {horizon} forecast weeks use climatology instead of observed weather "
                            "because ERA5 does not cover them yet.")
        st.session_state["fc_result"] = dict(fc=fc, history=y, var=fvar, exog=exog_vars, mode=fut_mode,
                                             key=(atype, anr, fvar, start, end, horizon), area=state.selected_label())
    except (F.ForecastError, data.DataError) as e:
        st.session_state.pop("fc_result", None)
        st.error(str(e))
    except Exception as e:
        st.session_state.pop("fc_result", None)
        st.error(f"The model could not be fitted: {e}")

res = st.session_state.get("fc_result")
if not res:
    st.info("Set the parameters above and press **Run forecast**.")
    st.stop()
if res["key"] != (atype, anr, fvar, start, end, horizon):
    st.warning("Area, variable, training window or horizon changed since the last run: press **Run forecast** to update.")

fc: F.Forecast = res["fc"]
history: pd.Series = res["history"]
full = data.area_series(df, res["key"][0], res["key"][1], res["var"])
actual = full[(full.index > history.index[-1]) & (full.index <= fc.mean.index[-1])]
ev = F.evaluate(fc, actual, float(history.iloc[-1]))
unit = RESERVOIR_VARS[res["var"]]["unit"]

# results
st.subheader(f"{fc.description} · {res['area']}")
for note in fc.notes:
    st.warning(note)
cols = st.columns(4)
cols[0].metric(f"Forecast, last week", f"{fc.mean.iloc[-1]:.1f} {unit}")
cols[1].metric(f"{int((1 - fc.alpha) * 100)} % interval, last week", f"{fc.lower.iloc[-1]:.1f} – {fc.upper.iloc[-1]:.1f}")
for (k, v), col in zip(fc.info.items(), cols[2:]):
    col.metric(k, f"{v:.2f}" if isinstance(v, float) else str(v))

hist_w = st.slider("History shown (weeks)", 8, max(len(history), 9), min(len(history), max(104, 3 * horizon)))
fig = F.animated_figure(history, fc, actual if len(actual) else None, data.var_label(res["var"]),
                        f"{fc.description}: {len(fc.mean)}-week forecast", hist_w)
st.plotly_chart(fig)
extra = (f" Weather inputs: {', '.join(WEATHER_VARS[v]['label'] for v in res['exog'])} "
         f"(future weather: {res['mode'].split(' (')[0].lower()}).") if res["exog"] else ""
st.caption("Press ▶ Play to watch the forecast and its confidence band grow week by week, or drag the slider." + extra)

if ev:
    st.markdown("**Comparison with what happened after the training window**")
    st.dataframe(pd.DataFrame([ev]).round(3), hide_index=True)
else:
    st.caption("No observations after the end of the training window, so there is nothing to compare with. "
               "Use *Hold out the last weeks* to run a back-test.")

out = pd.DataFrame({"forecast": fc.mean, "lower": fc.lower, "upper": fc.upper})
out.index.name = "week_ending"
with st.expander("Forecast table"):
    st.dataframe(out.round(3))
    st.download_button("Download CSV", out.to_csv().encode(), "forecast.csv", "text/csv")

with st.expander("📄 Method"):
    st.markdown(
        "- **Dynamic forecasting:** every forecast week is built from the model's own earlier forecasts, never from "
        "observed values after the training window.\n"
        "- **SARIMAX** (`statsmodels`): confidence intervals come from the model's own forecast variance; weather "
        "is standardised before fitting.\n"
        "- **Machine learning** (`scikit-learn`): features are lagged values, sin/cos of the week of year, an optional "
        "52-week lag and optional (lagged) weather. Intervals are quantiles of 300 simulated paths: at each week a "
        "random one-step error, drawn from a hold-out of the training window, is added and fed back. They are "
        "approximate and often too narrow for the *level* target.\n"
        "- **Future weather:** climatology repeats the average year, so it carries no information about the weather "
        "to come; weather variables help most when they explain the *past* state that persists into the forecast."
    )
