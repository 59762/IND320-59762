import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data, geo, state
from utils.config import (AREA_TYPES, COLOR_MODES, MAP_STYLE, MAX_WEATHER_YEARS, RESERVOIR_VARS,
                          SELECT_COLOR, SERIES_COLORS, WEATHER_VARS)

st.set_page_config(page_title="Map & Selectors", page_icon="🗺️", layout="wide")
state.init()

st.title("🗺️ Map & Selectors")
st.caption("Price areas, water areas or all of Norway, drawn from the NVE shapefiles and coloured by "
           "reservoir filling or a weather property. Click an area to select it: the selection and the "
           "time frame carry over to the correlation and forecast pages.")

try:
    df = data.load_nve()
except data.DataError as e:
    st.error(str(e))
    st.stop()
lo, hi = data.data_range(df)
state.ensure_period(lo, hi)

# time frame
with st.container(border=True):
    state.period_slider(lo, hi)
    c1, c2 = st.columns([1, 3])
    with c1:
        state.horizon_input()
    start, end = state.period()
    fc_end = end + pd.Timedelta(weeks=int(state.get("horizon")))
    c2.markdown(
        f"**Map period / training window:** {start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end - start).days // 7} weeks)  \n"
        f"**Forecast window:** {end:%Y-%m-%d} → {fc_end:%Y-%m-%d}  \n"
        "<span style='color:gray'>Shared with the forecast page.</span>", unsafe_allow_html=True)

# selectors
c1, c2, c3 = st.columns([1.4, 1.5, 1.3])
mode = c1.radio("Colour areas by", list(COLOR_MODES), key=state.bind("color_mode"), horizontal=True,
                on_change=state.commit, args=("color_mode",),
                help="Price area = Magasinstatistikk for the price areas (EL); "
                     "Water area = Magasinstatistikk for the water areas (VASS); "
                     "Weather property = ERA5 weather, aggregated over the time frame.")
geom = COLOR_MODES[mode]
is_weather = geom is None
if is_weather:
    geom = c2.radio("Areas", list(AREA_TYPES), key=state.bind("weather_geometry"), format_func=AREA_TYPES.get,
                    horizontal=True, on_change=state.commit, args=("weather_geometry",))
    var = c2.selectbox("Weather property", list(WEATHER_VARS), key=state.bind("weather_prop"),
                       format_func=lambda v: WEATHER_VARS[v]["label"], on_change=state.commit,
                       args=("weather_prop",))
    c3.number_input("Weather sample points per area", 1, 15, key=state.bind("n_points"),
                    on_change=state.commit, args=("n_points",),
                    help="Random points inside each area polygon; the ERA5 values are averaged. "
                         "More points = better area mean, slower first download.")
    meta = WEATHER_VARS[var]
    unit = data.weather_unit(var)
    stat = "Mean over period"
else:
    c2.markdown(f"**Areas:** {AREA_TYPES[geom]}")
    var = c2.selectbox("Magasinstatistikk variable", list(RESERVOIR_VARS), key=state.bind("map_metric"),
                       format_func=lambda v: RESERVOIR_VARS[v]["label"], on_change=state.commit,
                       args=("map_metric",))
    stat = c3.selectbox("Statistic over the time frame", ["Mean over period", "Latest in period", "Change over period"],
                        key=state.bind("map_stat"), on_change=state.commit, args=("map_stat",))
    meta = RESERVOIR_VARS[var]
    unit = meta["unit"]
legend = f"{meta['label']} ({unit})"

# geometry + values
try:
    gj = geo.load_geojson(geom)                      # shapefile -> GeoJSON (cached)
except Exception as e:
    st.error(f"Could not convert the shapefiles to GeoJSON: {e}")
    st.stop()
nrs = geo.area_numbers(geom)
names = geo.area_names(geom)

if is_weather:
    w_start = max(start, end - pd.DateOffset(years=MAX_WEATHER_YEARS))
    if w_start > start:
        st.info(f"Weather is limited to {MAX_WEATHER_YEARS} years: using {w_start:%Y-%m-%d} → {end:%Y-%m-%d}.")
    n_pts = int(state.get("n_points"))
    try:
        with st.spinner(f"Fetching ERA5 weather for {len(nrs)} area(s) × {n_pts} points …"):
            vals = {n: data.weather_period_value(geom, n, w_start, end, var, n_pts) for n in nrs}
    except data.DataError as e:
        st.error(f"Weather data unavailable: {e}")
        st.stop()
else:
    vals = {n: data.area_value(df, geom, n, var, start, end, stat) for n in nrs}
zs = [vals[n] for n in nrs]
if all(pd.isna(z) for z in zs):
    st.warning("No data for the selected property and time frame. Widen the time frame.")
    st.stop()

# the map
sel_type, sel_nr = state.selected()
fig = go.Figure(go.Choroplethmap(
    geojson=gj, locations=nrs, featureidkey="properties.area_nr", z=zs,
    text=[names[n] for n in nrs], colorscale=meta["scale"], zmid=meta["zmid"],
    marker=dict(opacity=0.6, line=dict(width=1.5, color="#333333")),       # transparent fill + outlines
    colorbar=dict(title=dict(text=legend, side="right"), thickness=14, len=0.75),
    hovertemplate="<b>%{text}</b><br>" + legend + ": %{z:.2f}<extra></extra>",
    selected=dict(marker=dict(opacity=0.6)), unselected=dict(marker=dict(opacity=0.6)),
))
layers = []
if sel_type == geom:                                  # thick contrast outline round the clicked area
    feats = [f for f in gj["features"] if int(f["properties"]["area_nr"]) == sel_nr]
    if feats:
        layers.append(dict(sourcetype="geojson", type="line", color=SELECT_COLOR, line=dict(width=5),
                           source={"type": "FeatureCollection", "features": feats}))
fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=65.0, lon=13.0), zoom=3.1, layers=layers),
                  margin=dict(l=0, r=0, t=0, b=0), height=660, clickmode="event+select",
                  uirevision="map")                   # keep zoom/pan when the figure updates

left, right = st.columns([3, 2], gap="large")
with left:
    chart_key = f"map_{geom}"
    event = st.plotly_chart(fig, key=chart_key, on_select="rerun", selection_mode="points",
                            config={"scrollZoom": True})
    if state.handle_map_click(chart_key, event, geom, nrs):
        st.rerun()                                    # redraw with the new outline
    st.caption("Click an area to select it. Scroll to zoom, drag to pan.")

with right:
    with st.container(border=True):
        st.markdown(f"#### Selected: {state.selected_label()}")
        if sel_type != geom:
            st.caption(f"This area belongs to the {AREA_TYPES[sel_type]} layer, which is not shown now.")
        metric = state.get("map_metric")
        s = data.area_series(df, sel_type, sel_nr, metric, start, end)
        fig_s = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=SERIES_COLORS[0], width=2),
                                     hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"))
        fig_s.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0), hovermode="x",
                            yaxis_title=data.var_label(metric))
        st.plotly_chart(fig_s)
        st.caption("The selection is stored for the other pages.")
        st.page_link("pages/7_Sliding_Window_Correlation.py", label="Open sliding-window correlation →", icon="📈")
        st.page_link("pages/8_Water_Forecasting.py", label="Open forecasting →", icon="🔮")
    table = pd.DataFrame({"Area": [names[n] for n in nrs], legend: zs})
    st.dataframe(table.style.format({legend: "{:.2f}"}), hide_index=True)

with st.expander("📄 About the map"):
    st.markdown(
        "- **Geometry:** the shapefiles in `data/shapefiles` (Budområder = price areas, NVE_Region = NVE "
        "regions) are read with `geopandas`, reprojected from EPSG:25833 to WGS84, simplified (500 m) and "
        "converted to GeoJSON when first needed (`utils/geo.py`). Water areas are the NVE regions merged as "
        "set in `utils/config.py` (assumed mapping: 1 = RS+RØ, 2 = RV, 3 = RM+RN). Norway = union of the price areas.\n"
        "- **Reservoir values:** NVE Magasinstatistikk, weekly; statistic over the time frame as selected.\n"
        "- **Weather:** Open-Meteo ERA5, daily; mean over random sample points inside each polygon "
        "(the price-area polygons include coastal sea), then averaged over the time frame. Precipitation, "
        "snow and radiation are shown as mean weekly totals.\n"
        "- **Colours** are drawn at 60 % opacity so the base map stays visible."
    )
