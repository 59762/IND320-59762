"""
forecast_plots.py - Plotly figures for the forecasting page (no Streamlit imports,
so they can also be used from the Jupyter notebook).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from forecasting import ForecastResult

# One place to keep the colours so the figures follow the app theme. Override via ``theme=``.
THEME = {
    "history": "#1f77b4", "forecast": "#e4572e", "band": "rgba(228,87,46,0.18)",
    "actual": "#2b2d42", "template": "plotly_white",
}


def forecast_figure(res: ForecastResult, *, animate: bool = True, history_steps: int = 60,
                    title: str | None = None, theme: dict | None = None,
                    frame_ms: int = 400) -> go.Figure:
    """
    History + forecast + interval band (+ actual outcome when it is known).

    With ``animate=True`` the forecast is revealed one step at a time via Plotly frames
    (Play/Pause buttons and a slider): the user literally watches the model "forecast
    dynamically" from the origin forward and, when actual data exist, sees the miss grow.
    """
    th = {**THEME, **(theme or {})}
    hist = res.history.iloc[-history_steps:]
    f = res.forecast
    # Prepend the origin so line and band start exactly at the last observation.
    x_f = [res.origin, *f.index]
    mean = [res.history.iloc[-1], *f["mean"]]
    has_band = bool(f["lower"].notna().any())
    lo = [res.history.iloc[-1], *f["lower"]]
    hi = [res.history.iloc[-1], *f["upper"]]
    n = len(x_f)

    def upto(k):                      # first k points of the forecast traces
        return dict(x=x_f[:k], y=None)

    k0 = 1 if animate else n          # start frame: only the origin visible when animating
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", mode="lines",
                             line=dict(color=th["history"], width=2)))
    if len(res.actual):
        fig.add_trace(go.Scatter(x=[res.origin, *res.actual.index],
                                 y=[res.history.iloc[-1], *res.actual.values], name="Actual",
                                 mode="lines+markers", line=dict(color=th["actual"], dash="dot"),
                                 marker=dict(size=6)))
    # trace indices that the animation updates (band upper, band lower, mean)
    idx0 = len(fig.data)
    fig.add_trace(go.Scatter(x=x_f[:k0], y=hi[:k0], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", visible=has_band))
    fig.add_trace(go.Scatter(x=x_f[:k0], y=lo[:k0], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=th["band"], name="Prediction interval",
                             hoverinfo="skip", visible=has_band, showlegend=has_band))
    fig.add_trace(go.Scatter(x=x_f[:k0], y=mean[:k0], name="Forecast", mode="lines+markers",
                             line=dict(color=th["forecast"], width=3), marker=dict(size=7)))
    ymin = min(hist.min(), *(v for v in lo if v == v), *(res.actual.values if len(res.actual) else [100]))
    ymax = max(hist.max(), *(v for v in hi if v == v), *(res.actual.values if len(res.actual) else [0]))
    pad = 0.05 * (ymax - ymin + 1e-9)

    if animate:
        fig.frames = [go.Frame(name=str(k), traces=[idx0, idx0 + 1, idx0 + 2],
                               data=[go.Scatter(x=x_f[:k], y=hi[:k]), go.Scatter(x=x_f[:k], y=lo[:k]),
                                     go.Scatter(x=x_f[:k], y=mean[:k])]) for k in range(1, n + 1)]
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.18, showactive=False, buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True,
                                      transition=dict(duration=150))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ])],
            sliders=[dict(active=0, x=0.18, len=0.8, y=-0.12, currentvalue=dict(prefix="Steps ahead: ", font=dict(size=13)),
                          steps=[dict(label=str(k - 1), method="animate",
                                      args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                 for k in range(1, n + 1)])],
        )
    fig.update_layout(
        template=th["template"], title=title or "Reservoir filling forecast",
        yaxis=dict(title="Fill (%)", range=[max(0, ymin - pad), min(105, ymax + pad)]),
        xaxis=dict(title=None, range=[hist.index[0], f.index[-1] + (f.index[-1] - f.index[-2]) * 0.3
                                      if len(f) > 1 else f.index[-1]]),
        hovermode="x unified", margin=dict(l=10, r=10, t=80, b=90 if animate else 30),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    return fig


def horizon_error_figure(res: ForecastResult, theme: dict | None = None) -> go.Figure:
    """Bar chart of the progressive-validation MAE per forecast step."""
    th = {**THEME, **(theme or {})}
    s = res.horizon_mae.dropna()
    fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=th["forecast"],
                           hovertemplate="step %{x}: %{y:.2f} pp<extra></extra>"))
    fig.update_layout(template=th["template"], height=260, margin=dict(l=10, r=10, t=30, b=30),
                      title="Online (progressive) error by forecast step",
                      xaxis_title="Steps ahead", yaxis_title="MAE (pp)")
    return fig
