import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import analysis, nve, state, weather
from lib.config import DIVERGING, RESERVOIR_VARS, SERIES, WEATHER_VARS, label

st.set_page_config(page_title="Sliding Window Correlation", page_icon="📈", layout="wide")
st.title("📈 Meteorology & Sliding Window Correlation")
st.caption("Does the weather explain the reservoir? Pearson correlation in a trailing window that slides "
           "through the selected period, with an adjustable lag between the two series.")

state.sidebar(show_horizon=False)
start, end = state.period()
atype, anr = state.get("area_type"), state.get("area_nr")
st.markdown(f"**Area:** {state.selected_label()} · **Period:** {start:%Y-%m-%d} → {end:%Y-%m-%d} "
            "(both from the sidebar / map page)")

# ------------------------------------------------------------------ controls
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
wvar = c1.selectbox("Meteorological property", list(WEATHER_VARS), key=state.bind("weather_prop"),
                    format_func=lambda v: WEATHER_VARS[v]["label"], on_change=state.commit, args=("weather_prop",))
rvar = c2.selectbox("Magasinstatistikk variable", list(RESERVOIR_VARS), key=state.bind("corr_var"),
                    format_func=lambda v: RESERVOIR_VARS[v]["label"], on_change=state.commit, args=("corr_var",))
lag = c3.slider("Lag (weeks)", -26, 26, key=state.bind("corr_lag"), on_change=state.commit, args=("corr_lag",),
                help="> 0: weather is taken that many weeks BEFORE the reservoir value. "
                     "< 0: the reservoir leads the weather.")
window = c4.slider("Window length (weeks)", 8, 156, key=state.bind("corr_window"), step=2,
                   on_change=state.commit, args=("corr_window",))
how = st.radio("Transformation", analysis.TRANSFORMS, key=state.bind("corr_transform"), horizontal=True,
               on_change=state.commit, args=("corr_transform",),
               help="Raw series share the annual cycle, which inflates correlation. 'Seasonal anomaly' subtracts "
                    "the mean for each ISO week over the selected period; 'Weekly difference' correlates changes.")

# ------------------------------------------------------------------ data
reservoir = nve.area_series(atype, anr, rvar, start, end)
if len(reservoir) < window + 10:
    st.warning(f"Only {len(reservoir)} weeks in the period; choose a longer period or a shorter window.")
    st.stop()
try:
    with st.spinner("Fetching ERA5 weather..."):
        wk = weather.weekly_weather(atype, anr, start - pd.Timedelta(weeks=30), end)
except Exception as e:
    st.error(f"Weather data unavailable: {e}")
    st.stop()
wser = wk[wvar]

r = analysis.sliding_corr(wser, reservoir, window, lag, how)
pair = analysis.lagged_pair(analysis.transform(wser, how), analysis.transform(reservoir, how), lag).dropna()
overall = pair["weather"].corr(pair["reservoir"]) if len(pair) > 8 else np.nan
bound = analysis.corr_bounds(window)
rv = r.dropna()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Overall r in period", f"{overall:.2f}")
m2.metric("Mean sliding r", f"{rv.mean():.2f}" if len(rv) else "-")
m3.metric("Windows beyond ±95 % bound", f"{(rv.abs() > bound).mean() * 100:.0f} %" if len(rv) else "-",
          help="Bound from Fisher z for independent samples; weekly series are autocorrelated, "
               "so this is optimistic.")
if len(rv):
    m4.metric("Strongest window", f"{rv.loc[rv.abs().idxmax()]:.2f}", f"ends {rv.abs().idxmax():%Y-%m-%d}",
              delta_color="off")

tab1, tab2, tab3 = st.tabs(["Sliding window", "Lag explorer", "All variable pairs"])

# ------------------------------------------------------------------ tab 1
with tab1:
    shown_w = analysis.lagged_pair(wser, reservoir, lag)["weather"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.28, 0.28, 0.44],
                        subplot_titles=(label(rvar), f"{label(wvar)} - shifted by lag = {lag} wk", ""))
    fig.add_trace(go.Scatter(x=reservoir.index, y=reservoir.values, name=RESERVOIR_VARS[rvar]["label"],
                             mode="lines", line=dict(color=SERIES[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=shown_w.index, y=shown_w.values, name=WEATHER_VARS[wvar]["label"],
                             mode="lines", line=dict(color=SERIES[1], width=1.6)), row=2, col=1)
    x = r.index
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=[bound] * len(x) + [-bound] * len(x), fill="toself",
                             fillcolor="rgba(138,137,132,0.18)", line=dict(width=0), hoverinfo="skip",
                             name="±95 % bound (no correlation)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=r.values, name=f"Sliding r ({window} wk)", mode="lines",
                             line=dict(color=SERIES[2], width=2.4),
                             hovertemplate="%{x|%Y-%m-%d}<br>r = %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="#8a8984", width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[-1, 1], title_text="Pearson r", row=3, col=1)
    fig.update_layout(height=700, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig)
    st.caption(f"r at week *t* uses the trailing {window} weeks ending at *t*; weather is taken {lag} week(s) "
               f"{'earlier' if lag >= 0 else 'later'} than the reservoir value. Transformation: {how}.")

# ------------------------------------------------------------------ tab 2
with tab2:
    max_lag = st.slider("Lag range shown (± weeks)", 4, 52, 26)
    prof = analysis.lag_profile(wser, reservoir, max_lag, how)
    colors = [SERIES[1] if l == lag else SERIES[0] for l in prof.index]
    figp = go.Figure(go.Bar(x=prof.index, y=prof.values, marker_color=colors,
                            hovertemplate="lag %{x} wk<br>r = %{y:.2f}<extra></extra>"))
    figp.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[-1, 1], title="r"),
                       xaxis_title="Lag (weeks; > 0 = weather leads)", title="Correlation over the whole period vs lag")
    st.plotly_chart(figp)
    if prof.notna().any():
        best = prof.abs().idxmax()
        st.caption(f"Strongest lag: **{best} weeks** (r = {prof[best]:.2f}); selected lag in orange.")

    mat = analysis.lag_time_matrix(wser, reservoir, window, max_lag, how)
    fh = go.Figure(go.Heatmap(
        x=mat.columns, y=mat.index, z=mat.values, zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, DIVERGING[0]], [0.25, DIVERGING[1]], [0.5, DIVERGING[2]], [0.75, DIVERGING[3]],
                    [1, DIVERGING[4]]],
        colorbar=dict(title="r", thickness=12),
        hovertemplate="%{x|%Y-%m-%d}<br>lag %{y} wk<br>r = %{z:.2f}<extra></extra>"))
    fh.add_hline(y=lag, line=dict(color="#0b0b0b", width=1.5, dash="dot"))
    fh.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Lag (weeks)",
                     title=f"Sliding correlation ({window}-week window) for every lag")
    st.plotly_chart(fh)

# ------------------------------------------------------------------ tab 3
with tab3:
    rows = {}
    for rk in RESERVOIR_VARS:
        rs = nve.area_series(atype, anr, rk, start, end)
        rows[RESERVOIR_VARS[rk]["label"]] = {}
        for wk_ in WEATHER_VARS:
            p = analysis.lagged_pair(analysis.transform(wk[wk_], how), analysis.transform(rs, how), lag).dropna()
            rows[RESERVOIR_VARS[rk]["label"]][WEATHER_VARS[wk_]["label"]] = (
                p["weather"].corr(p["reservoir"]) if len(p) > 8 else np.nan)
    cm = pd.DataFrame(rows)
    fm = go.Figure(go.Heatmap(
        z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, zmid=0, text=np.round(cm.values, 2),
        texttemplate="%{text}", colorscale=[[0, DIVERGING[0]], [0.25, DIVERGING[1]], [0.5, DIVERGING[2]],
                                            [0.75, DIVERGING[3]], [1, DIVERGING[4]]],
        colorbar=dict(title="r", thickness=12)))
    fm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                     title=f"Correlation over the period, lag = {lag} wk, {how.lower()}")
    st.plotly_chart(fm)
    st.dataframe(cm.round(3))

with st.expander("📄 Method"):
    st.markdown(
        "- Reservoir: NVE Magasinstatistikk, weekly (Sunday). Weather: Open-Meteo ERA5 daily, mean of three "
        "points inside the area, aggregated to the same Monday-Sunday weeks (mean for temperature/wind, sum "
        "for precipitation/snow/radiation).\n"
        "- Sliding correlation: `pandas.Series.rolling(window).corr()` between the transformed series after "
        "shifting the weather by the lag.\n"
        "- Correlation is not causation: reservoir levels are driven mostly by operation and market prices. "
        "Look for stable sign and size across windows rather than single peaks."
    )
