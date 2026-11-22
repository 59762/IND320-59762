"""Forecasting of the weekly reservoir series: SARIMAX and lag-based machine learning.

Both families produce a *dynamic* multi-step forecast: every step uses the model's own earlier
predictions (SARIMAX via the Kalman filter, ML via recursion), never the observed future values.
Exogenous weather needs future values; they are either the climatology of the training window
(honest) or the observed weather (only possible for a back-test).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.config import SERIES_COLORS

FREQ = "W-SUN"
ML_MODELS = ["Ridge regression", "Random forest", "Gradient boosting"]
N_SIM = 300                       # simulated paths for the ML prediction interval


class ForecastError(ValueError):
    """Problem the user can fix (too little data, bad parameter, missing weather ...)."""


@dataclass
class Forecast:
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    alpha: float
    description: str
    info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- data prep
def regularize(y: pd.Series, max_gap: int = 3) -> pd.Series:
    """Regular weekly grid, small gaps interpolated, leading/trailing gaps trimmed."""
    y = y.asfreq(FREQ).interpolate(limit=max_gap, limit_area="inside")
    first, last = y.first_valid_index(), y.last_valid_index()
    if first is None:
        raise ForecastError("No data in the selected time frame.")
    y = y.loc[first:last]
    if y.isna().any():
        raise ForecastError(f"The series has gaps longer than {max_gap} weeks inside the time frame; "
                            "choose another time frame.")
    return y


def future_index(y: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq=FREQ)


def _woy(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.isocalendar().week.clip(upper=52), dtype=int)


def build_exog(weather_weekly: pd.DataFrame, cols: list[str], hist_index: pd.DatetimeIndex,
               fut_index: pd.DatetimeIndex, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Exogenous matrices for the training weeks and for the forecast weeks, plus the number of
    forecast weeks that fell back to climatology although 'observed' was requested.

    mode 'climatology': future = mean of each ISO week over the training window.
    mode 'observed'   : future = the weather that actually occurred (back-test). Weeks that ERA5
                        does not cover yet (about a week of delay) use climatology instead.
    """
    hist = weather_weekly[cols].reindex(hist_index).interpolate(limit=3, limit_direction="both")
    if hist.isna().any().any():
        raise ForecastError("Weather data is missing for part of the training window.")
    clim = hist.groupby(_woy(hist.index)).mean().reindex(range(1, 53)).interpolate(limit_direction="both")
    fut = pd.DataFrame(clim.loc[_woy(fut_index)].to_numpy(), index=fut_index, columns=cols)
    n_filled = 0
    if mode == "observed":
        obs = weather_weekly[cols].reindex(fut_index)
        ok = obs.notna().all(axis=1)
        fut.loc[ok] = obs.loc[ok]
        n_filled = int((~ok).sum())
    return hist, fut, n_filled


def _clip(a, bounds):
    lo, hi = bounds
    return np.clip(a, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)


# ----------------------------------------------------------------------------- SARIMAX
def forecast_sarimax(y: pd.Series, exog_hist, exog_fut, horizon: int, order, seasonal_order,
                     trend, alpha: float, clip) -> Forecast:
    y = y.copy()
    y.index.freq = FREQ
    s = seasonal_order[3]
    if s and len(y) < 2 * s + 10:
        raise ForecastError(f"Seasonal period {s} needs at least {2 * s + 10} weeks of training data "
                            f"(have {len(y)}). Lengthen the time frame or choose a shorter period.")
    notes = []
    mu = sd = None
    if exog_hist is not None:                      # standardise: keeps the optimiser well-behaved
        mu, sd = exog_hist.mean(), exog_hist.std().replace(0, 1.0)
        exog_hist = (exog_hist - mu) / sd
        exog_fut = (exog_fut - mu) / sd
    mod = SARIMAX(y, exog=exog_hist, order=order, seasonal_order=seasonal_order, trend=trend,
                  enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False, maxiter=60, method="lbfgs")
        fc = res.get_forecast(steps=horizon, exog=exog_fut)
    ci = fc.conf_int(alpha=alpha)
    idx = future_index(y, horizon)
    mean = pd.Series(_clip(fc.predicted_mean.to_numpy(), clip), index=idx)
    lower = pd.Series(_clip(ci.iloc[:, 0].to_numpy(), clip), index=idx)
    upper = pd.Series(_clip(ci.iloc[:, 1].to_numpy(), clip), index=idx)
    if not np.isfinite(res.llf):
        notes.append("The fit did not converge cleanly - try fewer parameters or a longer time frame.")
    elif not res.mle_retvals.get("converged", True):
        notes.append("The optimiser stopped before converging; treat the forecast with care.")
    desc = f"SARIMAX{order}" + (f"x{tuple(seasonal_order)}" if s else "")
    return Forecast(mean, lower, upper, alpha, desc,
                    info={"AIC": float(res.aic), "BIC": float(res.bic), "Training weeks": len(y)}, notes=notes)


# ----------------------------------------------------------------------------- machine learning
def _model(name: str, params: dict):
    if name == "Ridge regression":
        return Ridge(alpha=params.get("alpha", 1.0))
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 200),
                                     max_depth=params.get("max_depth", 8), min_samples_leaf=2,
                                     n_jobs=-1, random_state=0)
    if name == "Gradient boosting":
        return GradientBoostingRegressor(n_estimators=params.get("n_estimators", 200),
                                         max_depth=params.get("max_depth", 3),
                                         learning_rate=params.get("learning_rate", 0.05), random_state=0)
    raise ForecastError(f"Unknown model {name!r}")


def _z(Y: np.ndarray, t: np.ndarray, k: int, target: str) -> np.ndarray:
    """Target-space value k steps before t: the level, or the weekly change."""
    return Y[:, t - k] if target == "level" else Y[:, t - k] - Y[:, t - k - 1]


def _features(Y: np.ndarray, t: np.ndarray, woy: np.ndarray, E: np.ndarray | None, n_lags: int,
              annual: bool, exog_lags: int, target: str) -> np.ndarray:
    """Feature matrix, shape (n_paths * len(t), F): lags of the target, week-of-year, weather."""
    P = Y.shape[0]
    cols = [_z(Y, t, k, target) for k in range(1, n_lags + 1)]
    if annual:
        cols.append(_z(Y, t, 52, target))
    ang = 2 * np.pi * woy[t] / 52.0
    cols += [np.tile(np.sin(ang), (P, 1)), np.tile(np.cos(ang), (P, 1))]
    if E is not None:
        for k in range(exog_lags + 1):
            for j in range(E.shape[1]):
                cols.append(np.tile(E[t - k, j], (P, 1)))
    return np.stack(cols, axis=-1).reshape(-1, len(cols))


def forecast_ml(y: pd.Series, exog_hist, exog_fut, horizon: int, name: str, params: dict, n_lags: int,
                annual: bool, exog_lags: int, target: str, alpha: float, clip) -> Forecast:
    n = len(y)
    annual_requested = annual
    annual = annual and n > 110
    first = max(n_lags, 52 if annual else 0, exog_lags if exog_hist is not None else 0) + 1
    if n - first < 30:
        raise ForecastError(f"Too little training data for these settings: {n - first} usable weeks "
                            "(need at least 30). Lengthen the time frame or use fewer lags.")
    fut_idx = future_index(y, horizon)
    idx_all = y.index.append(fut_idx)
    woy = _woy(idx_all)
    T = n + horizon
    Y = np.full((1, T), np.nan)
    Y[0, :n] = y.to_numpy()
    E = None
    if exog_hist is not None:
        E = np.vstack([exog_hist.to_numpy(), exog_fut.to_numpy()])
        E = (E - E[:n].mean(0)) / np.where(E[:n].std(0) == 0, 1.0, E[:n].std(0))

    ts = np.arange(first, n)
    X = _features(Y[:, :n], ts, woy, E, n_lags, annual, exog_lags, target)
    z = _z(Y[:, :n], ts, 0, target).ravel()

    # honest one-step residuals: fit on the first 80 %, score the last 20 %, then refit on everything
    cut = int(len(ts) * 0.8)
    probe = _model(name, params).fit(X[:cut], z[:cut])
    resid = z[cut:] - probe.predict(X[cut:])
    resid = resid - resid.mean()                    # centre: the band must not be shifted off the forecast
    model = _model(name, params).fit(X, z)

    rng = np.random.default_rng(0)
    P = 1 + N_SIM                                   # path 0 = noise-free forecast
    Yp = np.repeat(Y, P, axis=0)
    for t in range(n, T):
        tt = np.array([t])
        pred = model.predict(_features(Yp, tt, woy, E, n_lags, annual, exog_lags, target))
        noise = np.concatenate([[0.0], rng.choice(resid, N_SIM)])
        zt = pred + noise
        Yp[:, t] = (zt if target == "level" else Yp[:, t - 1] + zt)
        Yp[:, t] = _clip(Yp[:, t], clip)
    paths = Yp[:, n:]
    mean = paths[0]
    lo, hi = np.quantile(paths[1:], [alpha / 2, 1 - alpha / 2], axis=0)
    lo, hi = np.minimum(lo, mean), np.maximum(hi, mean)
    desc = f"{name} ({'weekly change' if target != 'level' else 'level'}, {n_lags} lags" + \
           (", +52 wk" if annual else "") + ")"
    notes = []
    if annual_requested and not annual:
        notes.append("The 52-week lag was skipped: it needs more than 110 weeks of training data.")
    return Forecast(pd.Series(mean, index=fut_idx), pd.Series(lo, index=fut_idx), pd.Series(hi, index=fut_idx),
                    alpha, desc, info={"Training weeks": n, "Residual σ": float(np.std(resid))}, notes=notes)


# ----------------------------------------------------------------------------- evaluation
def evaluate(fc: Forecast, actual: pd.Series, last_value: float) -> dict:
    """Compare with observations that exist after the training window (empty dict if none)."""
    a = actual.reindex(fc.mean.index).dropna()
    if a.empty:
        return {}
    e = fc.mean.reindex(a.index) - a
    persistence = (last_value - a).abs().mean()
    inside = ((a >= fc.lower.reindex(a.index)) & (a <= fc.upper.reindex(a.index))).mean()
    return {"Weeks compared": len(a), "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAE persistence (last value)": float(persistence),
            f"Observed inside {int((1 - fc.alpha) * 100)} % band": float(inside)}


# ----------------------------------------------------------------------------- animation
def animated_figure(history: pd.Series, fc: Forecast, actual: pd.Series | None, ylabel: str,
                    title: str, history_weeks: int) -> go.Figure:
    """History + forecast with a Play button: the forecast line and its confidence band grow week by week."""
    h = history.iloc[-history_weeks:]
    fx = [h.index[-1]] + list(fc.mean.index)                    # start at the last observation
    fm = [float(h.iloc[-1])] + fc.mean.tolist()
    fl = [float(h.iloc[-1])] + fc.lower.tolist()
    fu = [float(h.iloc[-1])] + fc.upper.tolist()
    level = int((1 - fc.alpha) * 100)
    c_hist, c_fc = SERIES_COLORS[0], SERIES_COLORS[1]

    def band(k):
        return dict(x=fx[:k + 1] + fx[:k + 1][::-1], y=fu[:k + 1] + fl[:k + 1][::-1])

    n = len(fx)
    fig = go.Figure(
        data=[
            go.Scatter(x=h.index, y=h.values, name="Observed (training)", mode="lines",
                       line=dict(color=c_hist, width=2)),
            go.Scatter(x=[], y=[], name="Observed after cut-off", mode="markers",
                       marker=dict(color="#222", size=7)) if actual is None or actual.dropna().empty else
            go.Scatter(x=actual.index, y=actual.values, name="Observed after cut-off", mode="markers",
                       marker=dict(color="#222", size=7)),
            go.Scatter(**band(n - 1), name=f"{level} % interval", fill="toself", mode="lines",
                       fillcolor="rgba(230,126,34,0.20)", line=dict(width=0), hoverinfo="skip"),
            go.Scatter(x=fx, y=fm, name="Forecast", mode="lines+markers",
                       line=dict(color=c_fc, width=2.5), marker=dict(size=5),
                       hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>Forecast</extra>"),
        ],
        frames=[go.Frame(name=str(k), traces=[2, 3],
                         data=[go.Scatter(**band(k)), go.Scatter(x=fx[:k + 1], y=fm[:k + 1])])
                for k in range(n)],
    )
    ymin = min(float(h.min()), min(fl))
    ymax = max(float(h.max()), max(fu))
    if actual is not None and not actual.dropna().empty:
        ymin, ymax = min(ymin, float(actual.min())), max(ymax, float(actual.max()))
    pad = 0.06 * (ymax - ymin or 1.0)
    # The figure opens on the full forecast (last frame), so Play must name the frames explicitly and
    # run them from the first one; 'fromcurrent' would start at the end and do nothing.
    play = dict(mode="immediate", frame=dict(duration=350, redraw=True), fromcurrent=False,
                transition=dict(duration=150))
    fig.update_layout(
        title=title, height=520, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(range=[h.index[0], fx[-1]]), yaxis=dict(title=ylabel, range=[ymin - pad, ymax + pad]),
        legend=dict(orientation="h", y=-0.28),
        shapes=[dict(type="line", x0=fx[0], x1=fx[0], y0=0, y1=1, yref="paper",
                     line=dict(color="#8a8984", dash="dot", width=1))],
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
                          buttons=[dict(label="▶ Play", method="animate", args=[[str(k) for k in range(n)], play]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))]),
                                   dict(label="↺ Full forecast", method="animate",
                                        args=[[str(n - 1)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                 transition=dict(duration=0))])])],
        sliders=[dict(active=n - 1, x=0.22, len=0.78, y=-0.12, yanchor="top",
                      currentvalue=dict(prefix="Week: ", font=dict(size=12)),
                      steps=[dict(label=("start" if k == 0 else f"{fx[k]:%Y-%m-%d}"), method="animate",
                                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for k in range(n)])],
    )
    return fig
