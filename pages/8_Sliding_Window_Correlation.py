import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import analysis, data, state
from utils.config import MAX_WEATHER_YEARS, RESERVOIR_VARS, SERIES_COLORS, WEATHER_VARS

st.set_page_config(page_title="Sliding Window Correlation", page_icon="📈", layout="wide")
state.init()

st.title("📈 Meteorology & Sliding Window Correlation")
st.caption("Does the weather move with the reservoir? Pearson correlation in a trailing window that slides "
           "through the time frame, with an adjustable lag between the two series.")

try:
    df = data.load_nve()
except data.DataError as e:
    st.error(str(e))
    st.stop()
lo, hi = data.data_range(df)
state.ensure_period(lo, hi)

# ------------------------------------------------------------------------- area + time frame (shared)
with st.container(border=True):
    state.area_picker()
    state.period_slider(lo, hi)
start, end = state.period()
atype, anr = state.selected()
st.markdown(f"**Area:** {state.selected_label()} · **Time frame:** {start:%Y-%m-%d} → {end:%Y-%m-%d} "
            "(same area and time frame as on the map page)")

eff_start = max(start, end - pd.DateOffset(years=MAX_WEATHER_YEARS))
if eff_start > start:
    st.info(f"Weather is limited to {MAX_WEATHER_YEARS} years: using {eff_start:%Y-%m-%d} → {end:%Y-%m-%d}.")

# ------------------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
wvar = c1.selectbox("Meteorological property", list(WEATHER_VARS), key=state.bind("weather_prop"),
                    format_func=lambda v: WEATHER_VARS[v]["label"], on_change=state.commit, args=("weather_prop",))
rvar = c2.selectbox("Magasinstatistikk variable", list(RESERVOIR_VARS), key=state.bind("corr_var"),
                    format_func=lambda v: RESERVOIR_VARS[v]["label"], on_change=state.commit, args=("corr_var",))
lag = c3.slider("Lag (weeks)", -26, 26, key=state.bind("corr_lag"), on_change=state.commit, args=("corr_lag",),
                help="> 0: the weather is taken that many weeks BEFORE the reservoir value (weather leads). "
                     "< 0: the reservoir leads the weather.")
window = c4.slider("Window length (weeks)", 8, 156, step=2, key=state.bind("corr_window"),
                   on_change=state.commit, args=("corr_window",))
how = st.radio("Transformation", analysis.TRANSFORMS, key=state.bind("corr_transform"), horizontal=True,
               on_change=state.commit, args=("corr_transform",),
               help="Both series follow the annual cycle, which inflates the raw correlation. "
                    "'Weekly change' correlates week-to-week differences; 'Seasonal anomaly' subtracts the "
                    "mean of each ISO week over the time frame (needs a few years).")

# ------------------------------------------------------------------------- data
reservoir = data.area_series(df, atype, anr, rvar, eff_start, end)
if reservoir.dropna().shape[0] < window + 10:
    st.warning(f"Only {reservoir.dropna().shape[0]} weeks of data in the time frame; widen it or shorten the window.")
    st.stop()
try:
    with st.spinner("Fetching ERA5 weather …"):
        wk = data.weather_weekly(atype, anr, eff_start, end, int(state.get("n_points")))
except data.DataError as e:
    st.error(f"Weather data unavailable: {e}")
    st.stop()
wk = wk.reindex(reservoir.index)
wser = wk[wvar]
if wser.notna().sum() < window + 10:
    st.warning("Too little weather data in the time frame.")
    st.stop()

r = analysis.sliding_corr(wser, reservoir, window, lag, how)
overall = analysis.overall_corr(wser, reservoir, lag, how)
bound = analysis.corr_bound(window)
rv = r.dropna()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Correlation over whole time frame", f"{overall:.2f}" if pd.notna(overall) else "–")
m2.metric("Mean sliding r", f"{rv.mean():.2f}" if len(rv) else "–")
m3.metric("Windows outside ±95 % bound", f"{(rv.abs() > bound).mean() * 100:.0f} %" if len(rv) else "–",
          help="Bound from Fisher's z for independent samples. Weekly series are autocorrelated, so the "
               "bound is optimistic: read it as a guide, not a test.")
if len(rv):
    best_t = rv.abs().idxmax()
    m4.metric("Strongest window", f"{rv.loc[best_t]:.2f}", f"ends {best_t:%Y-%m-%d}", delta_color="off")

tab_slide, tab_lag, tab_all = st.tabs(["Sliding window", "Lag profile", "All weather properties"])

# ------------------------------------------------------------------------- tab 1
with tab_slide:
    shown_w = wser.shift(lag)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.28, 0.28, 0.44],
                        subplot_titles=(data.var_label(rvar), f"{data.var_label(wvar)} – shifted by lag = {lag} wk", ""))
    fig.add_trace(go.Scatter(x=reservoir.index, y=reservoir.values, name=RESERVOIR_VARS[rvar]["label"], mode="lines",
                             line=dict(color=SERIES_COLORS[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=shown_w.index, y=shown_w.values, name=WEATHER_VARS[wvar]["label"], mode="lines",
                             line=dict(color=SERIES_COLORS[1], width=1.6)), row=2, col=1)
    x = list(r.index)
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[bound] * len(x) + [-bound] * len(x), fill="toself",
                             fillcolor="rgba(138,137,132,0.18)", line=dict(width=0), hoverinfo="skip",
                             name="±95 % bound (no correlation)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=r.index, y=r.values, name=f"Sliding r ({window} wk)", mode="lines",
                             line=dict(color=SERIES_COLORS[2], width=2.4),
                             hovertemplate="%{x|%Y-%m-%d}<br>r = %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="#8a8984", width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[-1, 1], title_text="Pearson r", row=3, col=1)
    fig.update_layout(height=720, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig)
    st.caption(f"r at week t uses the {window} weeks ending at t. The weather is taken {abs(lag)} week(s) "
               f"{'earlier' if lag >= 0 else 'later'} than the reservoir value. Transformation: {how}.")

# ------------------------------------------------------------------------- tab 2
with tab_lag:
    max_lag = st.slider("Lags shown (± weeks)", 4, 52, 26)
    prof = analysis.lag_profile(wser, reservoir, max_lag, how)
    colors = [SERIES_COLORS[1] if k == lag else SERIES_COLORS[0] for k in prof.index]
    fig_p = go.Figure(go.Bar(x=prof.index, y=prof.values, marker_color=colors,
                             hovertemplate="lag %{x} wk<br>r = %{y:.2f}<extra></extra>"))
    fig_p.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[-1, 1], title="r"),
                        xaxis_title="Lag (weeks; > 0 = weather leads)",
                        title="Correlation over the whole time frame for each lag")
    st.plotly_chart(fig_p)
    if prof.notna().any():
        best = int(prof.abs().idxmax())
        st.caption(f"Strongest lag: **{best} weeks** (r = {prof[best]:.2f}). The selected lag is in orange. "
                   "A clear peak at a positive lag means the weather leads the reservoir: that is what a forecast can use.")

# ------------------------------------------------------------------------- tab 3
with tab_all:
    rows = []
    for v in WEATHER_VARS:
        pr = analysis.lag_profile(wk[v], reservoir, 26, how)
        k = int(pr.abs().idxmax()) if pr.notna().any() else None
        rows.append({"Weather property": WEATHER_VARS[v]["label"],
                     f"r at lag {lag}": analysis.overall_corr(wk[v], reservoir, lag, how),
                     "Best lag (±26 wk)": k, "r at best lag": pr[k] if k is not None else np.nan})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.style.format({f"r at lag {lag}": "{:.2f}", "r at best lag": "{:.2f}"}), hide_index=True)
    st.caption(f"Correlation with {RESERVOIR_VARS[rvar]['label'].lower()} for {state.selected_label()}, "
               f"{how.lower()}. Change the transformation above and watch how much of the raw correlation is just the season.")

with st.expander("📄 Method"):
    st.markdown(
        "- Reservoir: NVE Magasinstatistikk, weekly. Weather: Open-Meteo ERA5 daily, mean over random points "
        "inside the area, aggregated to weeks ending Sunday (mean for temperature and wind, sum for "
        "precipitation, snow and radiation).\n"
        "- Sliding correlation: `pandas.Series.rolling(window).corr()` between the transformed series after "
        "shifting the weather by the lag.\n"
        "- Correlation is not causation: reservoir levels are driven by operation and prices as well as inflow. "
        "Look for a stable sign and size across windows rather than single peaks."
    )
