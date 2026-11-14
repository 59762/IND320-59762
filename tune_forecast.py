from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

import data_sources as ds
import forecasting as fc

# Search space: parameter -> candidate values (resolution handled specially, see below).
SPACE = {
    "resolution": ["W", "MS"],
    "method": ["lag_ml", "snarimax"],
    "regressor": ["linear", "pa", "tree"],
    "lr": [0.001, 0.003, 0.01, 0.03],         # -> regressor_params["lr"]
    "n_lags": [3, 6, 12],
    "exog_lags": [1, 2, 4],
    "n_harmonics": [1, 2, 3, 4],
    "history_years": [5, 10, 15, 20],
    "predict_delta": [True, False],
    "exog_set": ["none", "temp", "temp+precip", "all"],
}
EXOG_SETS = {
    "none": [],
    "temp": ["temperature_2m_mean"],
    "temp+precip": ["temperature_2m_mean", "precipitation_sum"],
    "all": list(ds.WEATHER_VARS),
}
HORIZON_MONTHS = 6          # compare W and MS on the same calendar horizon (6 months)


def _steps(resolution):     # horizon in model steps for 6 months
    return 26 if resolution == "W" else HORIZON_MONTHS


def evaluate(cfg: dict, fill, weather, **bt) -> fc.BacktestResult:
    """One backtest for a flat config dict (see SPACE)."""
    cfg = dict(cfg)
    exog = EXOG_SETS[cfg.pop("exog_set")]
    rp = {"lr": cfg.pop("lr"), "l2": 0.0}
    res = cfg["resolution"]
    return fc.backtest_fill(fill, weather, exog=exog, horizon=_steps(res), regressor_params=rp,
                            eval_step=4 if res == "W" else 1, **cfg, **bt)


def tune(fill, weather, rounds=2, tune_years=5, holdout_years=3, verbose=True):
    d = fc.DEFAULTS
    cfg = {"resolution": d["resolution"], "method": d["method"], "regressor": d["regressor"],
           "lr": d["regressor_params"]["lr"], "n_lags": d["n_lags"], "exog_lags": d["exog_lags"],
           "n_harmonics": d["n_harmonics"], "history_years": d["history_years"],
           "predict_delta": d["predict_delta"], "exog_set": "temp"}
    log = []
    best = evaluate(cfg, fill, weather, test_years=tune_years, holdout_years=holdout_years)
    for r in range(rounds):
        for name, values in SPACE.items():
            scores = []
            for v in values:
                trial = {**cfg, name: v}
                try:
                    res = evaluate(trial, fill, weather, test_years=tune_years, holdout_years=holdout_years)
                except Exception as e:                        # e.g. too little history: skip
                    if verbose:
                        print(f"  skip {name}={v}: {e}")
                    continue
                scores.append((res.mae, res.elapsed_s, v))
                log.append({"round": r + 1, "param": name, "value": str(v), "MAE": res.mae,
                            "MAE_seasonal_naive": res.mae_seasonal_naive, "coverage": res.coverage,
                            "seconds": res.elapsed_s})
            mae, _, v = min(scores, key=lambda s: (round(s[0], 3), s[1]))   # ties -> faster
            if mae <= best.mae:
                cfg[name], best = v, evaluate({**cfg}, fill, weather, test_years=tune_years,
                                              holdout_years=holdout_years)
            if verbose:
                print(f"round {r + 1}  {name:14s} -> {cfg[name]!s:12s} tuning MAE {best.mae:.3f}")
    # confirmation on the untouched most recent years
    final = evaluate(cfg, fill, weather, test_years=holdout_years, holdout_years=0)
    return cfg, best, final, pd.DataFrame(log)


def to_defaults(cfg: dict) -> dict:
    """Translate the flat tuned config to the DEFAULTS format used by forecasting.py."""
    out = {k: cfg[k] for k in ("resolution", "method", "regressor", "n_lags", "exog_lags",
                               "n_harmonics", "history_years", "predict_delta")}
    out["regressor_params"] = {"lr": cfg["lr"], "l2": 0.0}
    out["horizon"] = _steps(cfg["resolution"])
    return {"params": out, "default_exog": EXOG_SETS[cfg["exog_set"]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic offline data")
    ap.add_argument("--area", default="EL")
    ap.add_argument("--nr", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--write", action="store_true", help="write forecast_defaults.json")
    a = ap.parse_args()

    if a.demo:
        weather = ds.make_synthetic_weather()
        nve = ds.make_synthetic_nve(weather, area_nr=a.nr)
    else:
        nve = ds.load_nve_data()
        weather = ds.load_weather_daily(a.area, a.nr, "2000-01-01",
                                        (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
    fill = ds.get_fill_series(nve, a.area, a.nr)

    cfg, best, final, log = tune(fill, weather, rounds=a.rounds)
    print("\nBEST CONFIG:", cfg)
    print(f"tuning window MAE {best.mae:.2f}  (seasonal-naive {best.mae_seasonal_naive:.2f}, "
          f"persistence {best.mae_persistence:.2f})")
    print(f"confirmation (unseen years) MAE {final.mae:.2f}  (seasonal-naive "
          f"{final.mae_seasonal_naive:.2f}), interval coverage {final.coverage}")
    log.to_csv("tuning_log.csv", index=False)
    print("full sweep log -> tuning_log.csv")
    if a.write:
        Path(__file__).with_name("forecast_defaults.json").write_text(
            json.dumps(to_defaults(cfg), indent=2))
        print("wrote forecast_defaults.json")
