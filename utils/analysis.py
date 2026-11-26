"""Sliding-window correlation between a weather series and a reservoir series (weekly)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRANSFORMS = ["Raw", "Weekly change", "Seasonal anomaly"]


def transform(s: pd.Series, how: str) -> pd.Series:
    """Raw series share the annual cycle, which inflates correlation. 'Weekly change' correlates
    week-to-week differences; 'Seasonal anomaly' removes the mean of each ISO week."""
    if how == "Weekly change":
        return s.diff()
    if how == "Seasonal anomaly":
        wk = np.asarray(s.index.isocalendar().week.clip(upper=52))
        return s - s.groupby(wk).transform("mean")
    return s


def lagged_pair(weather: pd.Series, reservoir: pd.Series, lag: int) -> pd.DataFrame:
    """lag > 0: weather is taken `lag` weeks BEFORE each reservoir value (weather leads).
    lag < 0: the reservoir leads."""
    return pd.concat({"weather": weather.shift(lag), "reservoir": reservoir}, axis=1)


def sliding_corr(weather: pd.Series, reservoir: pd.Series, window: int, lag: int, how: str) -> pd.Series:
    pair = lagged_pair(transform(weather, how), transform(reservoir, how), lag)
    return pair["weather"].rolling(window, min_periods=max(8, int(window * 0.8))).corr(pair["reservoir"])


def overall_corr(weather: pd.Series, reservoir: pd.Series, lag: int, how: str) -> float:
    p = lagged_pair(transform(weather, how), transform(reservoir, how), lag).dropna()
    return float(p["weather"].corr(p["reservoir"])) if len(p) > 8 else np.nan


def lag_profile(weather: pd.Series, reservoir: pd.Series, max_lag: int, how: str) -> pd.Series:
    """Correlation over the whole period for every lag in [-max_lag, max_lag]."""
    lags = list(range(-max_lag, max_lag + 1))
    return pd.Series([overall_corr(weather, reservoir, k, how) for k in lags], index=lags, name="r")


def corr_bound(window: int) -> float:
    """95 % bound for 'no correlation' from Fisher's z, assuming independent samples (optimistic
    for autocorrelated weekly data)."""
    return float(np.tanh(1.96 / np.sqrt(max(window - 3, 1))))
