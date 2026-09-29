import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Imported Data – First Month")

@st.cache_data
def load_data():
    # Read CSV and rename columns from Norwegian to English
    df = pd.read_csv("data/reservoirs.csv")
    df = df.rename(columns={
        "dato_Id":                  "date",
        "fyllingsgrad":             "fill_percentage",
        "kapasitet_TWh":            "capacity_TWh",
        "fylling_TWh":              "fill_TWh",
        "fyllingsgrad_forrige_uke": "prev_week_fill_pct",
        "endring_fyllingsgrad":     "change_fill_pct",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Pick the first month (30 rows) of data
first_month = df.head(30)

st.write(f"Showing the first **{len(first_month)}** rows of the dataset.")

# Transpose so each row = one column of the original data
st.dataframe(first_month, use_container_width=True)


# Filter to the first calendar month (January 1995 = 4 weekly rows)
first_month_period = df["date"].dt.to_period("M").iloc[0]
first_month = df[df["date"].dt.to_period("M") == first_month_period].copy()

st.write(f"Showing data for **{first_month_period}** — {len(first_month)} weekly measurements.")

# The assignment requires: one row per data column, with a LineChartColumn
# We build a summary table where each row = one numeric column
numeric_cols = ["fill_percentage", "capacity_TWh", "fill_TWh",
                "prev_week_fill_pct", "change_fill_pct"]

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