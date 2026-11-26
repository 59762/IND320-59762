"""River-based forecasting of reservoir fill (SNARIMAX and lag-based online ML).

This is the same code as section 4.3 of the notebook (cells for helpers, ForecastResult,
_run_snarimax, _run_lag_ml and forecast_fill), so the Streamlit page and the notebook give
identical forecasts. Do not edit one without the other.

Not to be confused with utils/forecast.py (SARIMAX / scikit-learn, used by the Water Forecasting page).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from river import forest, linear_model, optim, preprocessing, time_series

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = ROOT / "config" / "forecast_defaults.json"


SEASON_LENGTH = {"W": 52, "M": 12}                         # seasonal period per resolution
DEFAULT_LAGS = {"W": (1, 2, 3, 4, 52), "M": (1, 2, 3, 12)}
DEFAULT_FUTURE_RULE = {"weather": "climatology", "currency": "last"}   # per exog source; others: 'last'


def to_resolution(obj, resolution: str):
    # Weekly data is returned unchanged; 'M' gives monthly means (month-end labels)
    if resolution == "W":
        return obj
    if resolution == "M":
        return obj.resample("ME").mean()
    raise ValueError("resolution must be 'W' or 'M'")


def _season_key(index: pd.DatetimeIndex, resolution: str) -> np.ndarray:
    # Week of year (1..52) or month (1..12): used for climatology and Fourier terms
    if resolution == "W":
        return np.minimum(index.isocalendar().week.to_numpy().astype(int), 52)
    return index.month.to_numpy()


def _fourier(index: pd.DatetimeIndex, resolution: str) -> list:
    # sin/cos seasonal terms (2 harmonics) as feature dicts, one per timestamp
    pos = _season_key(index, resolution) / SEASON_LENGTH[resolution]
    return [{"s1": np.sin(2 * np.pi * p), "c1": np.cos(2 * np.pi * p),
             "s2": np.sin(4 * np.pi * p), "c2": np.cos(4 * np.pi * p)} for p in pos]


def _future_values(known: pd.DataFrame, fut_index: pd.DatetimeIndex, rule: str, resolution: str) -> pd.DataFrame:
    # Values of an exogenous source for dates after the last known observation.
    #   'last'         : repeat the last known value
    #   'climatology'  : mean of the same week/month in all earlier years
    if rule == "last":
        return pd.DataFrame(np.tile(known.iloc[-1].to_numpy(), (len(fut_index), 1)),
                            index=fut_index, columns=known.columns)
    if rule == "climatology":
        clim = known.groupby(_season_key(known.index, resolution)).mean()
        vals = clim.reindex(_season_key(fut_index, resolution))
        vals.index = fut_index
        return vals.fillna(known.mean())                    # safety net for a week/month never seen
    raise ValueError("future_rule must be 'last' or 'climatology'")


def build_exog_features(exog: dict, exog_vars, exog_lags, hist_index, fut_index, train_end,
                        resolution="W", future_rule=None) -> pd.DataFrame:
    # Feature table (index = hist_index + fut_index) with one column per variable and lag.
    # Column names: 'temperature' (lag 0), 'temperature_lag4', ...
    # Values after `train_end` are never taken from the data, only from the rule (no leakage).
    rules = {**DEFAULT_FUTURE_RULE, **(future_rule or {})}
    all_index = hist_index.append(fut_index)
    wanted = {}                                            # source name -> list of columns
    for var in exog_vars:
        src = next((n for n, d in exog.items() if var in d.columns), None)
        if src is None:
            raise ValueError(f"Exogenous variable {var!r} not found. Available: "
                             f"{ {n: list(d.columns) for n, d in exog.items()} }")
        wanted.setdefault(src, []).append(var)

    pieces = []
    for src, cols in wanted.items():
        d = to_resolution(exog[src][cols], resolution).sort_index()
        known = d.loc[:train_end].dropna(how="any")
        if known.empty:
            raise ValueError(f"Source {src!r} has no data before train_end={train_end}")
        last_ok = known.index.max()                        # last real observation we may use
        ext = d.reindex(all_index)
        late = ext.index > last_ok                         # future AND not-yet-published days
        ext.loc[late] = _future_values(known, ext.index[late], rules.get(src, "last"), resolution).to_numpy()
        pieces.append(ext.interpolate(limit=4, limit_area="inside"))   # fill short internal holes
    base = pd.concat(pieces, axis=1)[list(exog_vars)]

    feats = {}
    for col in base.columns:
        for k in exog_lags:
            feats[col if k == 0 else f"{col}_lag{k}"] = base[col].shift(k)
    return pd.DataFrame(feats)


def make_regressor(kind="linear", lr=0.01, l2=0.0, intercept_lr=0.1, n_trees=10, seed=42, loss=None):
    # River regressor used by both models
    if kind == "linear":
        return preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(lr), l2=l2, intercept_lr=intercept_lr,
            loss=loss if loss is not None else optim.losses.Squared())
    if kind == "forest":
        if loss is not None:
            raise ValueError("Quantile intervals need regressor='linear'")
        return forest.ARFRegressor(n_models=n_trees, seed=seed)
    raise ValueError("regressor must be 'linear' or 'forest'")

@dataclass
class ForecastResult:
    # Everything forecast_fill() returns
    forecast: pd.DataFrame          # index = forecast dates, columns forecast / lower / upper
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train: int
    model: str
    resolution: str
    settings: dict
    fit_seconds: float
    notes: list = field(default_factory=list)


def _run_snarimax(y_s, X_train, X_future, horizon, orders, regressor):
    # SNARIMAX (SARIMAX in River). One pass over the training data, then a dynamic multi-step forecast.
    model = time_series.SNARIMAX(regressor=regressor, **orders)
    for i, yi in enumerate(y_s):
        model.learn_one(yi, X_train[i] if X_train is not None else None)
    return np.asarray(model.forecast(horizon, xs=X_future), dtype=float)


def _run_lag_ml(y_s, X_train, X_future, fut_index, hist_index, resolution, lags, seasonal_terms,
                n_passes, regressor):
    # Lag-based ML: features = past values of y (+ exogenous + Fourier terms). Forecast is recursive:
    # each prediction is appended to the history and used as lag for the next step.
    max_lag = max(lags)
    four_h = _fourier(hist_index, resolution) if seasonal_terms else None
    four_f = _fourier(fut_index, resolution) if seasonal_terms else None

    def features(hist, i, x_exog, four):
        f = {f"y_lag{l}": hist[i - l] for l in lags}       # value l steps before position i
        if x_exog is not None:
            f.update(x_exog)
        if four is not None:
            f.update(four)
        return f

    for _ in range(n_passes):                               # several passes help the SGD models converge
        for i in range(max_lag, len(y_s)):
            regressor.learn_one(features(y_s, i, X_train[i] if X_train else None,
                                         four_h[i] if four_h else None), y_s[i])
    hist, out = list(y_s), []
    for h in range(len(fut_index)):
        pred = regressor.predict_one(features(hist, len(hist), X_future[h] if X_future else None,
                                              four_f[h] if four_f else None))
        hist.append(pred)
        out.append(pred)
    return np.asarray(out, dtype=float)

def forecast_fill(
    y: pd.Series,
    horizon: int = 26,
    history: int | None = 312,
    train_end=None,
    resolution: str = "W",
    model: str = "snarimax",
    # --- SNARIMAX orders: (p, d, q) non-seasonal, (sp, sd, sq) seasonal with period m
    # Default = SARIMA(3,0,1)(0,1,0)m: seasonal differencing (sd=1) removes the strong yearly cycle.
    # This is only a starting point; section 5 tests alternatives and the best one becomes the app default.
    p: int = 3, d: int = 0, q: int = 1, sp: int = 0, sd: int = 1, sq: int = 0, m: int | None = None,
    # --- lag-based ML
    lags: tuple | None = None, seasonal_terms: bool = True, n_passes: int = 3,
    # --- regressor settings (both models)
    regressor: str = "linear", lr: float = 0.01, l2: float = 0.0, intercept_lr: float = 0.1,
    n_trees: int = 10,
    # --- data sources
    exog: dict | None = None, exog_vars: tuple = (), exog_lags: tuple = (0,), future_rule: dict | None = None,
    # --- confidence interval (bonus): quantile regression
    interval: bool = False, level: float = 0.8,
    seed: int = 42,
) -> ForecastResult:
    # Forecast the next `horizon` steps of the fill of ONE area with River.
    #
    #  y          weekly fill series with a DatetimeIndex (see make_fill_series)
    #  horizon    number of steps ahead (weeks if resolution='W', months if 'M')
    #  history    number of most recent steps used for training (None = all)
    #  train_end  last date that may be used (default = end of y). Used for backtests.
    #  resolution 'W' (native) or 'M' (monthly means)
    #  model      'snarimax' | 'lag_ml'
    #  exog       {source: weekly DataFrame}; exog_vars picks columns, exog_lags the lags (in steps)
    #  future_rule  e.g. {'weather': 'climatology', 'currency': 'last'}
    #  interval   True -> also return `lower` and `upper` (quantile regression, level = 0.8 -> 10 %..90 %)
    t0 = time.perf_counter()
    notes = []

    # 0. Input checks 
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if model not in ("snarimax", "lag_ml"):
        raise ValueError("model must be 'snarimax' or 'lag_ml'")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if exog_vars and not exog:
        raise ValueError("exog_vars given but exog is empty")
    y = y.dropna().sort_index()
    if y.empty:
        raise ValueError("y has no data")
    m = m or SEASON_LENGTH[resolution]
    lags = tuple(lags) if lags else DEFAULT_LAGS[resolution]

    # 1. Training window 
    train_end = pd.Timestamp(train_end) if train_end is not None else y.index[-1]
    y_full = to_resolution(y.loc[:train_end], resolution)       # everything we are allowed to see
    y_tr = y_full.iloc[-history:] if history else y_full        # the part used for training
    freq = f"W-{y_tr.index[-1].day_name()[:3].upper()}" if resolution == "W" else "ME"
    fut_index = pd.date_range(y_tr.index[-1], periods=horizon + 1, freq=freq)[1:]

    # 2. Exogenous features (training rows and future rows) 
    X_all = None
    if exog_vars:
        # Features are built over the FULL history so that the first training rows still have their lags
        X_all = build_exog_features(exog, exog_vars, exog_lags, y_full.index, fut_index,
                                    y_full.index[-1], resolution, future_rule)
        ok = X_all.loc[y_tr.index].notna().all(axis=1)      # rows where every lagged feature exists
        if not ok.any():
            raise ValueError("No training rows with complete exogenous data. Check exog dates and lags.")
        first_ok = ok[ok].index[0]
        if first_ok > y_tr.index[0]:
            notes.append(f"training start moved to {first_ok.date()} (exogenous data starts later)")
            y_tr = y_tr.loc[first_ok:]
        if X_all.loc[y_tr.index].isna().any().any() or X_all.loc[fut_index].isna().any().any():
            raise ValueError("The exogenous data has gaps inside the training window. "
                             "Use a shorter history or fill the gaps first.")
        X_train = X_all.loc[y_tr.index].to_dict("records")
        X_future = X_all.loc[fut_index].to_dict("records")
    else:
        X_train = X_future = None

    needed = max(lags) + 20 if model == "lag_ml" else m * max(sp + sd, 1) + p + q + 10
    if len(y_tr) < needed:
        raise ValueError(f"Only {len(y_tr)} training steps, but at least {needed} are needed for these settings. "
                         "Use a longer history or smaller orders/lags.")

    #  3. Standardise the target 
    mu, sd_y = float(y_tr.mean()), float(y_tr.std()) or 1.0
    y_s = ((y_tr - mu) / sd_y).to_numpy()
    orders = dict(p=p, d=d, q=q, m=m, sp=sp, sd=sd, sq=sq)

    def run(loss=None):
        reg = make_regressor(regressor, lr, l2, intercept_lr, n_trees, seed, loss)
        if model == "snarimax":
            return _run_snarimax(y_s, X_train, X_future, horizon, orders, reg)
        return _run_lag_ml(y_s, X_train, X_future, fut_index, y_tr.index, resolution, lags,
                           seasonal_terms, n_passes, reg)

    #  4. Fit and forecast 
    point = run()
    out = pd.DataFrame({"forecast": point * sd_y + mu}, index=fut_index)
    if interval:
        a = (1 - level) / 2
        out["lower"] = run(optim.losses.Quantile(a)) * sd_y + mu
        out["upper"] = run(optim.losses.Quantile(1 - a)) * sd_y + mu
        out["lower"] = np.minimum(out["lower"], out["forecast"])   # keep lower <= forecast <= upper
        out["upper"] = np.maximum(out["upper"], out["forecast"])
    else:
        out["lower"] = out["upper"] = np.nan

    #  5. Keep values inside the physical range (fill is 0..1 or 0..100) 
    top = 1.0 if float(y.max()) <= 1.0 else 100.0
    out = out.clip(lower=0.0, upper=top)
    if not np.isfinite(out["forecast"]).all():
        notes.append("WARNING: the model diverged (NaN/inf). Lower the learning rate `lr`.")

    settings = dict(horizon=horizon, history=history, resolution=resolution, model=model, p=p, d=d, q=q, sp=sp,
                    sd=sd, sq=sq, m=m, lags=lags, seasonal_terms=seasonal_terms, n_passes=n_passes,
                    regressor=regressor, lr=lr, l2=l2, intercept_lr=intercept_lr, n_trees=n_trees,
                    exog_vars=tuple(exog_vars), exog_lags=tuple(exog_lags), interval=interval, level=level)
    return ForecastResult(out, y_tr.index[0], y_tr.index[-1], len(y_tr), model, resolution, settings,
                          time.perf_counter() - t0, notes)

def load_defaults(path=None) -> dict:
    """Settings the Streamlit page starts with, as keyword arguments for forecast_fill().

    Reads config/forecast_defaults.json (written by the notebook after the model comparison).
    """
    p = Path(path) if path else DEFAULTS_PATH
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    kwargs = {"resolution": cfg.get("resolution", "W"), **cfg.get("settings", {})}
    for key in ("exog_vars", "exog_lags", "lags"):          # JSON has lists, forecast_fill wants tuples
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])
    return kwargs
