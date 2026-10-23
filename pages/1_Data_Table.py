import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Data", page_icon="📋", layout="wide")
st.title("📋 Reservoir Data – NVE API")

NVE_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

@st.cache_data(ttl=3600)
def load_nve_data():
    resp = requests.get(f"{NVE_BASE}/HentOffentligData", timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw)
    rename_map = {
        "dato_Id":                 "date",
        "omrType":                 "area_type",
        "omrnr":                   "area_nr",
        "iso_aar":                 "year",
        "iso_uke":                 "week",
        "fyllingsgrad":            "fill_percentage",
        "kapasitet_TWh":           "capacity_TWh",
        "fylling_TWh":             "fill_TWh",
        "neste_Publiseringsdato":  "next_publish_date",
        "fyllingsgrad_forrige_uke":"prev_week_fill_pct",
        "endring_fyllingsgrad":    "change_fill_pct",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_nve_data()

area_types = sorted(df["area_type"].dropna().unique().tolist()) if "area_type" in df.columns else []
if area_types:
    selected_area = st.radio(
        "Select area type",
        options=area_types,
        horizontal=True,
        help="EL = electricity bidding zones, NO = Norway total, VASS = water catchment areas",
    )
    df = df[df["area_type"] == selected_area].copy()
else:
    df = df.copy()

st.caption(f"Loaded **{len(df):,}** rows from NVE API · Showing **{len(df):,}** rows for selected area type.")

st.subheader("Raw data – first 30 rows")
st.dataframe(df.head(30), use_container_width=True, hide_index=True)

st.divider()

st.subheader("First month summary – one row per column")

# Filter to the first calendar month (January 1995 = 4 weekly rows)
first_month_period = df["date"].dt.to_period("M").iloc[0]
first_month = df[df["date"].dt.to_period("M") == first_month_period].copy()

st.write(f"Showing data for **{first_month_period}** — {len(first_month)} weekly measurements.")

# The assignment requires: one row per data column, with a LineChartColumn
# We build a summary table where each row = one numeric column
numeric_cols = [c for c in ["fill_percentage", "capacity_TWh", "fill_TWh",
                             "prev_week_fill_pct", "change_fill_pct"]
                if c in first_month.columns]

# Build the transposed table:
# Each row has the column name, its values as a list (for the line chart),
# and the min/max for context
rows = []
for col in numeric_cols:
    values = first_month[col].tolist()
    rows.append({
        "Column": col,
        "Trend":  values,   # this column becomes the sparkline chart
        "Min":    round(min(values), 4),
        "Max":    round(max(values), 4),
    })

summary_df = pd.DataFrame(rows)

# Display with LineChartColumn — each cell shows a sparkline of the first month
st.dataframe(
    summary_df,
    use_container_width=True,
    column_config={
        "Trend": st.column_config.LineChartColumn(
            label="First month trend",
            width="medium",
        )
    },
    hide_index=True,
)