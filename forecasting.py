from __future__ import annotations

import json
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from river import compose, linear_model, optim, preprocessing, time_series, tree

# --------------------------------------------------------------------------- settings
RESOLUTIONS = {
    #       pandas rule, season length (steps/year), label
    "W": {"rule": "W-SUN", "period": 52, "label": "weekly"},
    "MS": {"rule": "MS", "period": 12, "label": "monthly"},
}

# Built-in defaults = result of `python tune_forecast.py --demo` (synthetic data, because the
# sandbox had no network). Re-run the tuner on the real NVE/Open-Meteo data with --write to
# replace them with forecast_defaults.json. Overridden automatically when that file exists.
DEFAULTS: dict = {
    "method": "lag_ml",          # "lag_ml" | "snarimax"
    "resolution": "MS",          # "W" | "MS"
    "history_years": 15,         # length of the training history before the origin
    "horizon": 6,                # steps ahead (weeks or months, depending on resolution)
    "n_lags": 6,                 # fill lags used as features
    "exog_lags": 2,              # lags per weather variable
    "n_harmonics": 4,            # sin/cos season pairs
    "regressor": "linear",       # "linear" | "pa" | "tree"
    "regressor_params": {"lr": 0.003, "l2": 0.0},   # lr=0.1 diverged in tuning
    "predict_delta": True,       # model y[t+h]-y[t] instead of y[t+h]
    "interval": "quantile",      # "quantile" | "residual" | None
    "quantiles": (0.1, 0.9),     # 80 % prediction interval
}
_DEFAULTS_FILE = Path(__file__).with_name("forecast_defaults.json")


def load_defaults() -> dict:
    """Built-in defaults merged with tuned ones from forecast_defaults.json (if present)."""
    out = dict(DEFAULTS)
    if _DEFAULTS_FILE.exists():
        try:
            tuned = json.loads(_DEFAULTS_FILE.read_text())
            tuned = tuned.get("params", tuned)
            if "quantiles" in tuned:
                tuned["quantiles"] = tuple(tuned["quantiles"])
            out.update({k: v for k, v in tuned.items() if k in DEFAULTS})
        except Exception as e:  # a broken file must never break the app
            warnings.warn(f"Ignoring unreadable {_DEFAULTS_FILE.name}: {e}")
    return out


def load_default_exog() -> list[str]:
    """Weather variables pre-selected in the app (empty unless the tuner found them useful)."""
    if _DEFAULTS_FILE.exists():
        try:
            return list(json.loads(_DEFAULTS_FILE.read_text()).get("default_exog", []))
        except Exception:
            pass
    return []


class ForecastError(ValueError):
    """User-facing problem (too little data, bad parameter, ...)."""


# ------------------------------------------------------------------------- result types
@dataclass
class ForecastResult:
    forecast: pd.DataFrame            # index = future dates; columns mean, lower, upper
    history: pd.Series                # training data actually used (at chosen resolution)
    actual: pd.Series                 # observations after the origin (may be empty)
    origin: pd.Timestamp              # last observation used for training
    params: dict                      # fully resolved parameters
    horizon_mae: pd.Series            # progressive-validation MAE per horizon step
    metrics: dict = field(default_factory=dict)   # holdout MAE/RMSE, baselines, coverage
    exog_used: list = field(default_factory=list)
    notes: list = field(default_factory=list)     # warnings for the UI (dropped columns, ...)
    elapsed_s: float = 0.0


@dataclass
class BacktestResult:
    by_horizon: pd.DataFrame          # MAE model / seasonal-naive / persistence, coverage
    mae: float                        # mean MAE over horizons and origins
    mae_seasonal_naive: float
    mae_persistence: float
    coverage: float | None
    n_origins: int
    elapsed_s: float
    params: dict


# ---------------------------------------------------------------------- data preparation
def _naive(obj):
    """Drop timezone info after converting to Europe/Oslo (see data_sources.py policy)."""
    if getattr(obj.index, "tz", None) is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_convert("Europe/Oslo").tz_localize(None)
    return obj


def aggregate_weather(weather: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Aggregate hourly/daily weather to weekly/monthly. Precip/snow are summed, the rest averaged."""
    rule = RESOLUTIONS[resolution]["rule"]
    rules = {c: ("sum" if any(k in c.lower() for k in ("precip", "snow", "rain")) else "mean")
             for c in weather.columns}
    return weather.resample(rule).agg(rules)


def prepare_dataset(fill: pd.Series, weather: pd.DataFrame | None, resolution: str,
                    exog: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Regular-grid DataFrame: column 'y' = fill %, then one column per usable exogenous
    variable. Returns (df, used_exog, notes). Small gaps are interpolated (max 4 steps).
    """
    if resolution not in RESOLUTIONS:
        raise ForecastError(f"resolution must be one of {list(RESOLUTIONS)}")
    notes: list[str] = []
    rule = RESOLUTIONS[resolution]["rule"]
    fill = _naive(fill.astype(float))
    if fill.dropna().empty:
        raise ForecastError("The fill series contains no valid values.")
    y = fill.resample(rule).mean().interpolate(limit=4).dropna()
    df = y.to_frame("y")

    used: list[str] = []
    if exog:
        if weather is None or weather.empty:
            notes.append("No weather data available - forecasting without exogenous variables.")
        else:
            wx = aggregate_weather(_naive(weather), resolution).reindex(df.index)
            for c in exog:
                if c not in wx.columns:
                    notes.append(f"Weather variable '{c}' not found - skipped.")
                    continue
                col = wx[c].interpolate(limit=4).ffill().bfill()
                if col.isna().any() or wx[c].isna().mean() > 0.2:
                    notes.append(f"'{c}' has too many missing values - skipped.")
                    continue
                df[c] = col
                used.append(c)
    return df, used, notes


# ---------------------------------------------------------------------- regressor factory
def _make_regressor(kind: str, params: dict, loss=None):
    """River regressor from a name + hyper-parameters. ``loss`` only applies to 'linear'."""
    p = dict(params or {})
    if kind == "linear":
        return linear_model.LinearRegression(
            optimizer=optim.SGD(p.get("lr", 0.01)),
            loss=loss if loss is not None else optim.losses.Squared(),
            l2=p.get("l2", 0.0),
            intercept_lr=p.get("intercept_lr", 0.01),
        )
    if kind == "pa":     # Passive-Aggressive regressor
        return linear_model.PARegressor(C=p.get("C", 0.1), mode=p.get("mode", 2),
                                        eps=p.get("eps", 0.1))
    if kind == "tree":   # Hoeffding adaptive tree (non-linear, slower)
        return tree.HoeffdingAdaptiveTreeRegressor(
            grace_period=p.get("grace_period", 20), max_depth=p.get("max_depth", 6),
            seed=p.get("seed", 42))
    raise ForecastError(f"Unknown regressor '{kind}' (use linear, pa or tree).")


def _pipeline(kind, params, loss=None):
    return preprocessing.StandardScaler() | _make_regressor(kind, params, loss)


def _season_feats(ts: pd.Timestamp, prefix: str, k: int) -> dict:
    """Sin/cos pairs of the day-of-year phase (k harmonics)."""
    phase = 2 * np.pi * (ts.dayofyear - 1) / 365.25
    out = {}
    for j in range(1, k + 1):
        out[f"{prefix}_sin{j}"] = np.sin(j * phase)
        out[f"{prefix}_cos{j}"] = np.cos(j * phase)
    return out


# ------------------------------------------------------------- engine 1: direct lag model
class _DirectOnline:
    """One River model per horizon (and per quantile); see module docstring."""

    ERR_WINDOW = 260          # progressive errors kept per horizon (residual intervals)
    MIN_SEEN = 20             # samples a model must have seen before its errors count

    def __init__(self, df: pd.DataFrame, exog: list[str], horizon: int, p: dict):
        self.p, self.H, self.exog = p, horizon, exog
        self.y = df["y"].to_numpy(float)
        self.T = len(self.y)
        rule = RESOLUTIONS[p["resolution"]]["rule"]
        fut = pd.date_range(df.index[-1], periods=horizon + 1, freq=rule)[1:]
        self.dates = df.index.append(fut)                       # observed + future dates
        self.ex = {c: df[c].to_numpy(float) for c in exog}
        k = p["n_harmonics"]
        self.season = [_season_feats(d, "s", k) for d in self.dates]
        self.first = max(p["n_lags"], p["exog_lags"] if exog else 1) - 1   # first usable origin
        self.base = {}                                           # cache: features at origin o
        kind, rp = p["regressor"], p["regressor_params"]
        qs = tuple(p["quantiles"]) if p["interval"] == "quantile" and kind == "linear" else ()
        self.q = qs
        # Quantile regressors learn the RESIDUAL of the mean model (well scaled, converges fast)
        qp = {"lr": rp.get("q_lr", 0.05), "l2": rp.get("l2", 0.0),
              "intercept_lr": rp.get("q_intercept_lr", 0.2)}
        self.models = {h: {"mean": _pipeline(kind, rp),
                           **{f"q{q}": _pipeline("linear", qp, optim.losses.Quantile(alpha=q)) for q in qs}}
                       for h in range(1, horizon + 1)}
        self.seen = {h: 0 for h in range(1, horizon + 1)}
        self.err = {h: deque(maxlen=self.ERR_WINDOW) for h in range(1, horizon + 1)}  # signed
        self.abs_err = {h: [] for h in range(1, horizon + 1)}

    def _base(self, o: int) -> dict:
        if o not in self.base:
            f = {"y0": self.y[o]}
            for k in range(1, self.p["n_lags"]):
                f[f"d{k}"] = self.y[o - k] - self.y[o]
            for c in self.exog:
                for k in range(self.p["exog_lags"]):
                    f[f"{c}_l{k}"] = self.ex[c][o - k]
            f.update({f"o_{k[2:]}": v for k, v in self.season[o].items()})   # origin season
            self.base[o] = f
        return self.base[o]

    def _x(self, o: int, h: int) -> dict:
        x = dict(self._base(o))
        x.update({f"t_{k[2:]}": v for k, v in self.season[o + h].items()})   # target season
        return x

    def _learn_matured(self, t: int):
        """Targets y[t] just became known: learn from origins o = t-h (progressive validation)."""
        for h in range(1, self.H + 1):
            o = t - h
            if o < self.first:
                break
            x = self._x(o, h)
            tgt = self.y[t] - self.y[o] if self.p["predict_delta"] else self.y[t]
            ms = self.models[h]
            pred = ms["mean"].predict_one(x)                 # predict BEFORE learning
            e = tgt - pred                                   # out-of-sample (progressive) residual
            warm = self.seen[h] >= self.MIN_SEEN
            if warm:
                self.err[h].append(e)
                self.abs_err[h].append(abs(e))
            ms["mean"].learn_one(x, tgt)
            for key, m in ms.items():
                if key != "mean" and warm:                   # quantile models: target = residual
                    m.learn_one(x, e)
            self.seen[h] += 1

    def predict(self, o: int) -> pd.DataFrame:
        """Forecast path from origin index ``o`` (needs o >= first)."""
        lo_q, hi_q = (self.p["quantiles"] if self.p["interval"] else (None, None))
        rows = []
        for h in range(1, self.H + 1):
            x, ms = self._x(o, h), self.models[h]
            base = 0.0 if not self.p["predict_delta"] else self.y[o]
            mean = base + ms["mean"].predict_one(x)
            lo = hi = np.nan
            if self.q:                                       # Quantile-loss regressors
                lo = mean + ms[f"q{lo_q}"].predict_one(x)
                hi = mean + ms[f"q{hi_q}"].predict_one(x)
            elif self.p["interval"] == "residual" and len(self.err[h]) >= 10:
                e = np.asarray(self.err[h])
                lo, hi = mean + np.quantile(e, lo_q), mean + np.quantile(e, hi_q)
            if not np.isnan(lo):                             # band must contain the forecast
                lo, hi = min(lo, mean), max(hi, mean)
            rows.append((mean, lo, hi))
        return pd.DataFrame(rows, columns=["mean", "lower", "upper"],
                            index=self.dates[o + 1:o + 1 + self.H])

    def run(self, eval_origins=()):
        ev = set(eval_origins)
        paths = {}
        for t in range(self.first + 1, self.T):
            self._learn_matured(t)
            if t in ev:
                paths[t] = self.predict(t)
        return paths


# ----------------------------------------------------------------- engine 2: SNARIMAX
class _SnarimaxOnline:
    """River SNARIMAX; recursive forecasts with climatological future exogenous values."""

    def __init__(self, df: pd.DataFrame, exog: list[str], horizon: int, p: dict):
        self.p, self.H, self.exog, self.df = p, horizon, exog, df
        self.y = df["y"].to_numpy(float)
        self.T = len(self.y)
        res = RESOLUTIONS[p["resolution"]]
        self.fut_dates = pd.date_range(df.index[-1], periods=horizon + 1, freq=res["rule"])[1:]
        self.m = res["period"]
        reg = _pipeline(p["regressor"], p["regressor_params"])
        use_season = self.T > 2 * self.m + p["n_lags"]       # need > 2 seasons for seasonal terms
        self.model = time_series.SNARIMAX(
            p=p["n_lags"], d=1 if p["predict_delta"] else 0, q=1,
            m=self.m, sp=1 if use_season else 0, sq=0, regressor=reg)
        self.err1: list[float] = []

    def _key(self, idx: pd.DatetimeIndex):
        return idx.month if self.p["resolution"] == "MS" else np.minimum(idx.isocalendar().week.to_numpy(), 52)

    def _climatology(self, upto: int):
        """Mean of each exogenous variable per week-of-year / month, from data up to ``upto``."""
        if not self.exog:
            return None
        h = self.df.iloc[:upto + 1]
        key = pd.Series(self._key(h.index), index=h.index)
        return h[self.exog].groupby(key.to_numpy()).mean()

    def _future_x(self, clim, dates):
        if clim is None:
            return [{} for _ in dates]
        keys = self._key(dates)
        return [{c: float(clim[c].get(k, clim[c].mean())) for c in self.exog} for k in keys]

    def run(self, eval_origins=()):
        ev, paths = set(eval_origins), {}
        for t in range(self.T):
            x = {c: float(self.df[c].iloc[t]) for c in self.exog}
            if t >= 20:                                       # let the model warm up first
                try:                                          # 1-step progressive error
                    pred = self.model.forecast(1, xs=[x])[0]
                    self.err1.append(self.y[t] - pred)
                except Exception:
                    pass
            self.model.learn_one(self.y[t], x)
            if t in ev:
                paths[t] = self.predict(t)
        return paths

    def predict(self, o: int | None = None) -> pd.DataFrame:
        o = self.T - 1 if o is None else o
        dates = self.fut_dates if o == self.T - 1 else pd.date_range(
            self.df.index[o], periods=self.H + 1, freq=RESOLUTIONS[self.p["resolution"]]["rule"])[1:]
        xs = self._future_x(self._climatology(o), dates)
        mean = np.asarray(self.model.forecast(self.H, xs=xs), float)
        lo = hi = np.full(self.H, np.nan)
        if self.p["interval"] and len(self.err1) >= 20:
            e = np.asarray(self.err1[-260:])
            lq, uq = self.p["quantiles"]
            scale = np.sqrt(np.arange(1, self.H + 1))        # random-walk style widening
            lo, hi = mean + np.quantile(e, lq) * scale, mean + np.quantile(e, uq) * scale
            lo, hi = np.minimum(lo, mean), np.maximum(hi, mean)
        return pd.DataFrame({"mean": mean, "lower": lo, "upper": hi}, index=dates)


# ------------------------------------------------------------------------ public helpers
def _resolve(params: dict) -> dict:
    """Defaults + overrides, validated."""
    p = load_defaults()
    p.update({k: v for k, v in params.items() if v is not None})
    rp = dict(DEFAULTS["regressor_params"])
    rp.update(p.get("regressor_params") or {})
    p["regressor_params"] = rp
    if p["method"] not in ("lag_ml", "snarimax"):
        raise ForecastError("method must be 'lag_ml' or 'snarimax'")
    if p["horizon"] < 1 or p["n_lags"] < 1:
        raise ForecastError("horizon and n_lags must be >= 1")
    if p["interval"] not in ("quantile", "residual", None):
        raise ForecastError("interval must be 'quantile', 'residual' or None")
    lo, hi = p["quantiles"]
    if not 0 < lo < hi < 1:
        raise ForecastError("quantiles must satisfy 0 < lower < upper < 1")
    return p


def _engine(df, exog, p):
    return (_DirectOnline if p["method"] == "lag_ml" else _SnarimaxOnline)(df, exog, p["horizon"], p)


def forecast_fill(fill: pd.Series, weather: pd.DataFrame | None = None, *,
                  exog: list[str] | None = None,
                  train_start=None, train_end=None,
                  **params) -> ForecastResult:
    """
    Forecast the next ``horizon`` steps of reservoir filling for one region.

    Parameters
    ----------
    fill      datetime-indexed Series, fill percentage (weekly NVE data for ONE region).
    weather   optional datetime-indexed DataFrame (hourly or daily) of candidate exogenous variables.
    exog      names of the weather columns to use (data-source selection). [] / None = fill only.
    train_start, train_end
              training window. ``train_end`` is the forecast origin (default: last observation);
              ``train_start`` overrides ``history_years``.
    **params  any key of DEFAULTS: method, resolution, history_years, horizon, n_lags, exog_lags,
              n_harmonics, regressor, regressor_params, predict_delta, interval, quantiles.

    Returns a ForecastResult. Observations after ``train_end`` (if any) are returned as
    ``actual`` and scored (MAE/RMSE) against the forecast and two baselines.
    """
    t0 = time.perf_counter()
    p = _resolve(params)
    exog = list(exog or [])
    df, used, notes = prepare_dataset(fill, weather, p["resolution"], exog)

    # ---- cut the training window ------------------------------------------------------
    origin = df.index[df.index <= pd.Timestamp(train_end)][-1] if train_end is not None else df.index[-1]
    full = df
    obs = df.loc[:origin]
    start = pd.Timestamp(train_start) if train_start is not None else \
        origin - pd.DateOffset(years=int(np.ceil(p["history_years"])))
    obs = obs.loc[obs.index >= start]
    n_min = max(p["n_lags"], 2) + p["horizon"] + 24
    if len(obs) < n_min:
        raise ForecastError(
            f"Only {len(obs)} training points in the selected window; at least {n_min} are "
            f"needed for these settings. Extend the history or reduce lags/horizon.")

    eng = _engine(obs, used, p)
    eng.run()
    path = eng.predict(len(obs) - 1) if p["method"] == "lag_ml" else eng.predict()

    # ---- progressive-validation MAE per horizon step -------------------------------------
    if p["method"] == "lag_ml":
        hm = pd.Series({h: (np.mean(v) if v else np.nan) for h, v in eng.abs_err.items()}, name="MAE")
    else:
        hm = pd.Series(np.nan, index=range(1, p["horizon"] + 1), name="MAE")
    hm.index.name = "horizon"

    # ---- optional hold-out comparison ---------------------------------------------------
    actual = full["y"].loc[full.index > origin].iloc[:p["horizon"]]
    metrics = {}
    if len(actual):
        f = path["mean"].reindex(actual.index)
        ok = f.notna()
        err = (actual[ok] - f[ok])
        per = RESOLUTIONS[p["resolution"]]["period"]
        snaive = pd.Series([full["y"].iloc[full.index.get_loc(d) - per] if full.index.get_loc(d) >= per else np.nan
                            for d in actual.index], index=actual.index)
        metrics = {"mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                   "mae_seasonal_naive": float((actual - snaive).abs().mean()),
                   "mae_persistence": float((actual - obs["y"].iloc[-1]).abs().mean()),
                   "n_compared": int(ok.sum())}
        if path["lower"].notna().any():
            inside = (actual[ok] >= path["lower"].reindex(actual.index)[ok]) & \
                     (actual[ok] <= path["upper"].reindex(actual.index)[ok])
            metrics["coverage"] = float(inside.mean())
    path = path.clip(0, 100)                                 # fill percentage is bounded
    return ForecastResult(forecast=path, history=obs["y"], actual=actual, origin=origin,
                          params=p, horizon_mae=hm, metrics=metrics, exog_used=used,
                          notes=notes, elapsed_s=time.perf_counter() - t0)


def backtest_fill(fill: pd.Series, weather: pd.DataFrame | None = None, *,
                  exog: list[str] | None = None, test_years: float = 5, eval_step: int = 4,
                  holdout_years: float = 0, **params) -> BacktestResult:
    """
    Walk-forward evaluation: one online pass over the data; forecasts are issued every
    ``eval_step`` steps during the last ``test_years`` years and scored against what happened.
    ``history_years`` = how much data precedes the FIRST test origin (training length).
    ``holdout_years`` removes the most recent years from the data completely, so parameters
    can be tuned on an earlier window and then confirmed on the untouched recent years.
    """
    t0 = time.perf_counter()
    p = _resolve(params)
    H = p["horizon"]
    df, used, _ = prepare_dataset(fill, weather, p["resolution"], list(exog or []))
    per = RESOLUTIONS[p["resolution"]]["period"]
    if holdout_years:
        df = df.iloc[:-int(round(holdout_years * per))]
    n_test = int(round(test_years * per))
    first_origin = len(df) - n_test - H
    if first_origin < 30:
        raise ForecastError("Series too short for the requested test period.")
    start = max(0, first_origin - int(round(p["history_years"] * per)))
    sub = df.iloc[start:]
    off = first_origin - start
    origins = list(range(off, len(sub) - H, eval_step))
    eng = _engine(sub, used, p)
    paths = eng.run(origins)

    ys = sub["y"].to_numpy()
    full_y = df["y"].to_numpy()
    rows = {h: {"model": [], "snaive": [], "persist": [], "inside": []} for h in range(1, H + 1)}
    for o, path in paths.items():
        for h in range(1, H + 1):
            act = ys[o + h]
            rows[h]["model"].append(abs(act - np.clip(path["mean"].iloc[h - 1], 0, 100)))
            rows[h]["persist"].append(abs(act - ys[o]))
            g = start + o + h - per                       # same date one year earlier (full series)
            if g >= 0:
                rows[h]["snaive"].append(abs(act - full_y[g]))
            lo, hi = path["lower"].iloc[h - 1], path["upper"].iloc[h - 1]
            if not np.isnan(lo):
                rows[h]["inside"].append(lo <= act <= hi)
    tab = pd.DataFrame({
        "MAE_model": {h: np.mean(r["model"]) for h, r in rows.items()},
        "MAE_seasonal_naive": {h: np.mean(r["snaive"]) if r["snaive"] else np.nan for h, r in rows.items()},
        "MAE_persistence": {h: np.mean(r["persist"]) for h, r in rows.items()},
        "coverage": {h: np.mean(r["inside"]) if r["inside"] else np.nan for h, r in rows.items()},
    })
    tab.index.name = "horizon"
    cov = tab["coverage"].mean()
    return BacktestResult(tab, float(tab["MAE_model"].mean()), float(tab["MAE_seasonal_naive"].mean()),
                          float(tab["MAE_persistence"].mean()), None if np.isnan(cov) else float(cov),
                          len(paths), time.perf_counter() - t0, p)
