"""Session state shared by the map, correlation and forecast pages (no sidebar anywhere).

Streamlit deletes a widget's state when the widget is not drawn on the current page. So every
shared value lives under a plain key (e.g. ``area_nr``) that survives page changes; widgets get a
private key ``_w_<name>`` that is refreshed from the plain key on each run and written back by an
``on_change`` callback (see ``bind`` / ``commit``).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from utils.config import AREA_TYPES, DEFAULT_SAMPLE_POINTS, NO_AREA_NR
from utils.geo import area_label, area_names

DEFAULTS = {
    "area_type": "EL", "area_nr": 3,
    "period": None,                  # (start_date, end_date): map time frame = training window
    "horizon": 12,                   # forecast horizon in weeks
    "color_mode": "Price area", "weather_geometry": "EL", "weather_prop": "temperature_2m_mean",
    "map_metric": "fill_percentage", "map_stat": "Mean over period", "n_points": DEFAULT_SAMPLE_POINTS,
    "corr_var": "fill_percentage", "corr_lag": 0, "corr_window": 26, "corr_transform": "Raw",
    "fc_var": "fill_percentage",
}
DEFAULT_YEARS = 5


def init() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def get(name: str):
    return st.session_state[name]


def put(name: str, value) -> None:
    st.session_state[name] = value


def bind(name: str) -> str:
    """Widget key for the shared value ``name`` (call right before creating the widget)."""
    key = f"_w_{name}"
    st.session_state[key] = st.session_state[name]
    return key


def commit(name: str) -> None:
    """on_change callback: copy the widget value back to the shared value."""
    st.session_state[name] = st.session_state[f"_w_{name}"]


# ------------------------------------------------------------------------------ selected area
def selected() -> tuple[str, int]:
    return st.session_state["area_type"], int(st.session_state["area_nr"])


def select(area_type: str, area_nr: int) -> None:
    st.session_state["area_type"], st.session_state["area_nr"] = area_type, int(area_nr)


def selected_label() -> str:
    t, n = selected()
    return area_label(t, n)


def area_picker(columns=None) -> None:
    """Area type + area number, bound to the shared selection (same state the map click sets)."""
    c1, c2 = columns or st.columns([1.6, 1.4])
    t = c1.radio("Area type", list(AREA_TYPES), key=bind("area_type"), format_func=AREA_TYPES.get,
                 horizontal=True, on_change=commit, args=("area_type",))
    names = area_names(t)
    nrs = list(names)
    if st.session_state["area_nr"] not in nrs:          # e.g. EL 5 selected, user switches to VASS
        st.session_state["area_nr"] = nrs[0]
    c2.selectbox("Area", nrs, key=bind("area_nr"), format_func=lambda n: names[n],
                 on_change=commit, args=("area_nr",), disabled=(t == "NO"))


# ------------------------------------------------------------------------------ time frame
def ensure_period(lo: date, hi: date) -> None:
    """Default to the last DEFAULT_YEARS years of data; always keep the period inside [lo, hi]."""
    p = st.session_state["period"]
    if p is None:
        p = (max(lo, hi - timedelta(days=365 * DEFAULT_YEARS)), hi)
    a, b = max(lo, min(p[0], hi)), max(lo, min(p[1], hi))
    st.session_state["period"] = (min(a, b), max(a, b))


def period() -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = st.session_state["period"]
    return pd.Timestamp(a), pd.Timestamp(b)


def period_slider(lo: date, hi: date, label: str = "Time frame") -> None:
    st.slider(label, min_value=lo, max_value=hi, key=bind("period"), step=timedelta(days=1),
              format="YYYY-MM-DD", on_change=commit, args=("period",),
              help="Shared by the map, the correlation page and the forecast page. "
                   "On the forecast page this is the training window.")


def horizon_input() -> None:
    st.number_input("Forecast horizon (weeks)", min_value=1, max_value=104, step=1, key=bind("horizon"),
                    on_change=commit, args=("horizon",),
                    help="Forecast covers the weeks after the end of the time frame.")


# ------------------------------------------------------------------------------ map clicks
def _signature(points: list[dict]) -> tuple:
    return tuple((p.get("curve_number"), p.get("point_index", p.get("point_number")), p.get("location"))
                 for p in points)


def handle_map_click(chart_key: str, event, area_type: str, nrs: list[int]) -> bool:
    """Turn a click on the map into the shared selection. Returns True when the selection changed.

    The plot keeps its own selection between runs, so an event is acted on only when it differs
    from the last one handled for this chart; otherwise a stale click would overwrite a selection
    made on another page.
    """
    try:
        points = list(event["selection"]["points"])
    except Exception:
        points = []
    seen = st.session_state.setdefault("_map_seen", {})
    sig = _signature(points)
    if seen.get(chart_key) == sig:
        return False
    seen[chart_key] = sig
    if not points:
        return False
    p = points[-1]
    nr = p.get("location")
    if nr is None:
        idx = p.get("point_index", p.get("point_number"))
        nr = nrs[idx] if isinstance(idx, int) and 0 <= idx < len(nrs) else None
    if nr is None:
        return False
    nr = NO_AREA_NR if area_type == "NO" else int(nr)
    if selected() == (area_type, nr):
        return False
    select(area_type, nr)
    return True
