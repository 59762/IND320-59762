import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import state, nve, weather
from lib.config import (AREA_TYPES, COLOR_MODES, MAP_STYLE, RESERVOIR_VARS, SEQ_BLUE, SERIES,
                        WEATHER_VARS, label)
from lib.geo import area_label, area_names, is_placeholder, load_geojson

st.set_page_config(page_title="Map & Selectors", page_icon="🗺️", layout="wide")
st.title("🗺️ Map & Selectors")
st.caption("Colour Norway's price areas, water areas or a weather property over a chosen period. "
           "Click an area to select it - the selection (and the period) follows you to the "
           "correlation and forecast pages.")

# 1) A click on the map is read BEFORE the sidebar widgets are created, so it can drive them.
mode_now = state.get("color_mode")
geom_now = COLOR_MODES[mode_now] or state.get("weather_geometry")
event = st.session_state.get("map_chart")
if event:
    state.apply_map_click(event.get("selection", {}).get("points", []), geom_now)

state.sidebar()
start, end = state.period()

# 2) Map controls
c1, c2, c3 = st.columns([1.3, 1.4, 1.3])
mode = c1.radio("Colour areas by", list(COLOR_MODES), key=state.bind("color_mode"), horizontal=True,
                on_change=state.commit, args=("color_mode",))
geom = COLOR_MODES[mode]
if geom is None:
    geom = c2.radio("Weather areas", ["EL", "VASS", "NO"], key=state.bind("weather_geometry"),
                    format_func=AREA_TYPES.get, horizontal=True,
                    on_change=state.commit, args=("weather_geometry",))
    var = c2.selectbox("Weather property", list(WEATHER_VARS), key=state.bind("weather_prop"),
                       format_func=lambda v: WEATHER_VARS[v]["label"],
                       on_change=state.commit, args=("weather_prop",))
    stat = "Mean over period"
    unit = WEATHER_VARS[var]["unit"] + ("/week" if WEATHER_VARS[var]["agg"] == "sum" else "")
    legend_title = f"{WEATHER_VARS[var]['label']} ({unit})"
else:
    c2.markdown(f"**Areas:** {AREA_TYPES[geom]}")
    var = c2.selectbox("Magasinstatistikk variable", list(RESERVOIR_VARS), key=state.bind("map_metric"),
                       format_func=lambda v: RESERVOIR_VARS[v]["label"],
                       on_change=state.commit, args=("map_metric",))
    stat = c3.selectbox("Statistic", ["Mean over period", "Latest in period", "Change over period"],
                        key=state.bind("map_stat"), on_change=state.commit, args=("map_stat",))
    unit = RESERVOIR_VARS[var]["unit"]
    legend_title = f"{RESERVOIR_VARS[var]['label']} ({unit})"
overlays = c3.multiselect("Outline overlays", ["NO", "EL", "VASS"], key=state.bind("overlays"),
                          format_func=AREA_TYPES.get, on_change=state.commit, args=("overlays",),
                          help="Boundary lines of the other area types, drawn under the colours.")
st.caption(f"Period: **{start:%Y-%m-%d} → {end:%Y-%m-%d}** (set in the sidebar)")

# 3) Values per area
gj = load_geojson(geom)
nrs = sorted(int(f["properties"]["area_nr"]) for f in gj["features"])
if mode == "Weather property":
    with st.spinner("Fetching ERA5 weather for the areas..."):
        try:
            vals = weather.period_summary(geom, nrs, start, end, var)
        except Exception as e:
            st.error(f"Weather data unavailable: {e}")
            st.stop()
else:
    df = nve.get_nve()
    sub = df[(df.area_type == geom) & (df.date >= start) & (df.date <= end)]
    g = sub.sort_values("date").groupby("area_nr")[var]
    if stat == "Mean over period":
        vals = g.mean()
    elif stat == "Latest in period":
        vals = g.last()
    else:
        vals = g.last() - g.first()
    vals = vals.reindex(nrs)
    vals.index = vals.index.astype(int)

# 4) The map
sel_type, sel_nr = state.get("area_type"), state.get("area_nr")
names = area_names(geom)
fig = go.Figure(go.Choroplethmap(
    geojson=gj, locations=nrs, featureidkey="properties.area_nr", z=[vals.get(n) for n in nrs],
    text=[names[n] for n in nrs], colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
    marker=dict(opacity=0.82, line=dict(width=1.2, color="#ffffff")),
    colorbar=dict(title=dict(text=legend_title, side="right"), thickness=14, len=0.7),
    hovertemplate="<b>%{text}</b><br>" + legend_title + ": %{z:.2f}<extra></extra>",
    selected=dict(marker=dict(opacity=1.0)), unselected=dict(marker=dict(opacity=0.82)),
))
layers = []
for ov in overlays:
    if ov != geom:
        layers.append(dict(sourcetype="geojson", source=load_geojson(ov), type="line",
                           color="#52514e", line=dict(width=1.4)))
if sel_type == geom:   # highlight the selected area
    feat = [f for f in gj["features"] if int(f["properties"]["area_nr"]) == int(sel_nr)]
    if feat:
        layers.append(dict(sourcetype="geojson", source={"type": "FeatureCollection", "features": feat},
                           type="line", color="#eb6834", line=dict(width=4)))
fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=65.2, lon=13.5), zoom=3.1, layers=layers),
                  margin=dict(l=0, r=0, t=0, b=0), height=640, clickmode="event+select")

if (geom == "VASS" or "VASS" in overlays) and is_placeholder("VASS"):
    st.warning("The water-area (VASS) outlines are a **placeholder** (three latitude bands): NVE does not "
               "publish the boundaries with Magasinstatistikk. Replace `data/geo/VASS.geojson` with the real "
               "polygons - the values themselves are real.", icon="⚠️")

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(fig, key="map_chart", on_select="rerun", selection_mode="points",
                    config={"scrollZoom": True})
with right:
    with st.container(border=True):
        st.markdown(f"#### Selected area: {area_label(sel_type, sel_nr)}")
        st.caption("Persisted across pages. Click the map or use the sidebar to change it.")
        s = nve.area_series(sel_type, sel_nr, state.get("map_metric"), start, end)
        figl = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=SERIES[0], width=2),
                                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"))
        figl.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title=label(state.get("map_metric")), hovermode="x")
        st.plotly_chart(figl)
        try:
            st.page_link("pages/8_Sliding_Window_Correlation.py", label="Open correlation →", icon="📈")
            st.page_link("pages/9_Water_Forecasting.py", label="Open forecasting →", icon="🔮")
        except Exception:
            pass
    tbl = pd.DataFrame({"Area": [names[n] for n in nrs], legend_title: [vals.get(n) for n in nrs]})
    st.dataframe(tbl.style.format({legend_title: "{:.2f}"}), hide_index=True)

with st.expander("📄 About the map"):
    st.markdown(
        "- **Price areas (EL)** NO1-NO5 and **Norway (NO)** are simplified outlines derived from the "
        "electricitymaps-contrib zone geometries (`scripts/build_geojson.py`).\n"
        "- **Water areas (VASS)** 1-3: placeholder geometry, see the warning above.\n"
        "- Reservoir values: NVE Magasinstatistikk, weekly. Weather: Open-Meteo ERA5 daily, averaged over "
        "three points per area and over the period.\n"
        "- Forecast horizon and training window on the forecast page are the same settings as the "
        "sidebar here."
    )
