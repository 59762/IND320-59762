# Forecasting of reservoir filling (Project part 4)

## Files
| File | Purpose |
|---|---|
| `forecasting.py` | `forecast_fill()` and `backtest_fill()`, the reusable River functions |
| `data_sources.py` | NVE / Open-Meteo loaders (no Streamlit) + offline synthetic data |
| `forecast_plots.py` | Plotly figures incl. the animated forecast |
| `pages/7_Forecasting.py` | Streamlit interface |
| `tune_forecast.py` | Staged parameter search; `--write` stores the winners in `forecast_defaults.json`, which the app then uses as defaults |

## Function interface
```python
from forecasting import forecast_fill
res = forecast_fill(fill, weather, exog=["temperature_2m_mean"],   # data-source selection
                    train_start="2010-01-01", train_end="2024-06-30",   # history length / origin
                    horizon=6, resolution="MS",                          # horizon + time resolution
                    method="lag_ml", regressor="linear",
                    regressor_params={"lr": 0.003}, n_lags=6, n_harmonics=4,
                    interval="quantile", quantiles=(0.1, 0.9))
res.forecast   # DataFrame: mean, lower, upper      res.metrics  # MAE vs actual + baselines
```

## Model design
* **Direct multi-horizon lag model**: one online regressor per horizon h predicts `y[t+h]-y[t]` from fill lags, weather at the origin, and sin/cos season terms of origin and target date. No weather forecast is needed.
* **SNARIMAX** (`method="snarimax"`): River's seasonal ARIMA; future weather is replaced by climatology.
* **Intervals (bonus)**: extra regressors trained with `optim.losses.Quantile` on the *residual* of the mean model (training on raw targets converged far too slowly and gave collapsed bands).
* **No look-ahead**: time is walked forward once; every sample is predicted *before* it is learned.

## Parameter tuning strategy
1. **Walk-forward backtesting** (never random CV: it leaks the future). Forecasts are issued from many origins and scored per horizon step.
2. **Baselines first**: seasonal-naive ("same month last year") and persistence. A model that doesn't beat seasonal-naive is not useful.
3. **Coordinate search, not a full grid**: sweep one parameter at a time, keep the best, repeat 2 rounds (`tune_forecast.py`). ~60 backtests instead of tens of thousands.
4. **Tune / confirm split**: tune on an earlier window, report the final score on the last 3 years the search never saw.
5. **Order of importance** (what to sweep first): resolution, learning rate, season harmonics, history length, then lags and exogenous set.
6. **Cost-aware ties**: if scores tie, take the faster setting.

## Findings on the offline test data (synthetic reservoir driven by synthetic weather)
| Setting | Result |
|---|---|
| Resolution | monthly MAE 3.27 vs weekly 3.86 (lr 0.01), and **~15x faster** (0.15 s vs 2.4 s per backtest) |
| Learning rate | 0.1 diverges (MAE 49), 0.003 best |
| Season harmonics | 1: 3.76, 2: 2.94, 3: 2.82, 4 best |
| History | 5 y clearly worse (3.65); 15-20 y equal -> 15 |
| Predict change vs level | change 2.81 vs level 3.50 |
| Weather | **no gain** here: the fill level already integrates the weather |
| Models | lag_ml 3.27 < tree 3.54 < SNARIMAX 3.74 << PA 7.06 |
| Tuned default | monthly, lag_ml, linear lr 0.003: MAE 2.68 (tuning), 3.92 (unseen) vs seasonal-naive 3.57 / 6.72 |
| Interval coverage | ~0.6 for a nominal 0.8: intervals are still too narrow |

**Caveat:** the sandbox could not reach NVE/Open-Meteo, so these numbers come from synthetic data. Run
`python tune_forecast.py --area EL --nr 3 --write` on the real data before reporting results or
treating the defaults as final; whether weather helps on real data is an open empirical question.

## Timezone note (for the notebook task)
All series are converted to tz-naive Europe/Oslo dates. At weekly/monthly resolution a UTC vs CET
offset cannot move an observation into another bin, except for hourly ENTSO-E/Open-Meteo data
aggregated to days: aggregate *after* converting to Europe/Oslo.
