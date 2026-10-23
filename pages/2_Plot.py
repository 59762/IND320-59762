import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from datetime import datetime

st.set_page_config(page_title="Time-Series Plot", page_icon="📈", layout="wide")
st.title("📈 Reservoir Data Plot")

NVE_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

# Read the data
@st.cache_data(ttl=3600, show_spinner="Fetching data from NVE…")
def load_data():
    resp = requests.get(f"{NVE_BASE}/HentOffentligData", timeout=30)
    resp.raise_for_status()

    raw = resp.json()
    df = pd.DataFrame(raw)

    # Rename columns
    rename_map = {
        "dato_Id":                 "date",
        "omrType":                 "area_type",
        "omrnr":                   "area_nr",
        "iso_aar":                 "year",
        "iso_uke":                 "week",
        "fyllingsgrad":            "fill_level",
        "kapasitet_TWh":           "capacity_TWh",
        "fylling_TWh":             "stored_energy_TWh",
        "fyllingsgrad_forrige_uke":"fill_level_previous_week",
        "endring_fyllingsgrad":    "change_in_fill_level",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

df = load_data()

if df.empty:
    st.warning("No data returned for the selected year range.")
    st.stop()

if "area_type" in df.columns:
    area_types = sorted(df["area_type"].dropna().unique().tolist())
    selected_area = st.radio("Area type", area_types, horizontal=True)
    df = df[df["area_type"] == selected_area].copy()

# Sort by date
df = df.sort_values(["date"])

# All numeric columns available for plotting
plot_columns = [c for c in ["fill_level", "capacity_TWh", "stored_energy_TWh",
                              "fill_level_previous_week", "change_in_fill_level"]
                if c in df.columns]

# Choose a column
selected_column = st.selectbox("Select a variable to plot", ["All"] + plot_columns)

# Make a month column for the slider
df["month"] = df["date"].dt.to_period("M").astype(str)

months = sorted(
    df.loc[df["date"].notna(), "month"].unique()
)

# Choose months — default is first month only
selected_months = st.select_slider(
    "Select month range",
    options=months,
    value=(months[0], months[0])
)

start_month, end_month = selected_months

# Keep data from the selected months
filtered_df = df[
    (df["month"] >= start_month) &
    (df["month"] <= end_month)
].copy()

if filtered_df.empty:
    st.warning("No data in selected range.")
    st.stop()

normalize = False
if selected_column == "All":
    normalize = st.checkbox("Normalise (0–1 scale)", value=True)


# Plot one selected column
if selected_column != "All":

    fig, ax = plt.subplots(figsize=(10, 5))

    if selected_column in plot_columns:
        # Numeric column — plot as line
        ax.plot(
            filtered_df["date"],
            filtered_df[selected_column],
            linewidth=1.5,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(selected_column)

    elif selected_column == "date":
        # Date column — count observations per date
        date_counts = filtered_df.groupby("date").size()
        ax.plot(date_counts.index, date_counts.values)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of observations")

    ax.set_title(f"{selected_column} – {start_month} to {end_month}")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Plot all numerical reservoir variables
else:
    all_data = (
        filtered_df
        .groupby("date")[plot_columns]
        .mean()
        .reset_index()
    )

    normalized_df = all_data.copy()

    # Normalize the columns from 0 to 1
    for column in plot_columns:
        minimum = all_data[column].min()
        maximum = all_data[column].max()

        if maximum != minimum:
            normalized_df[column] = (
                all_data[column] - minimum
            ) / (maximum - minimum)
        else:
            normalized_df[column] = 0

    fig, ax = plt.subplots(figsize=(10, 5))

    for column in plot_columns:
        ax.plot(
            normalized_df["date"],
            normalized_df[column],
            label=column
        )

    ax.set_title(f"All reservoir variables – {start_month} to {end_month}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized value (0-1)")
    ax.grid(True)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)