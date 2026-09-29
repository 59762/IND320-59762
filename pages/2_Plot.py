import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Time-Series Plot",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Reservoir Data Plot")

# Path to the csv file — goes up one level from pages/ to find data/
data_path = Path(__file__).resolve().parents[1] / "data" / "reservoirs.csv"

# Read the data
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)

    # Rename columns to understandable English names
    df = df.rename(columns={
        "dato_Id":                  "date",
        "fyllingsgrad":             "fill_level",
        "kapasitet_TWh":            "capacity_TWh",
        "fylling_TWh":              "stored_energy_TWh",
        "fyllingsgrad_forrige_uke": "fill_level_previous_week",
        "endring_fyllingsgrad":     "change_in_fill_level",
    })

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

df = load_data()

# Sort by date
df = df.sort_values(["date"])

# All numeric columns available for plotting
csv_columns = [
    "date",
    "fill_level",
    "capacity_TWh",
    "stored_energy_TWh",
    "fill_level_previous_week",
    "change_in_fill_level",
]

# Choose a column
selected_column = st.selectbox(
    "Select a variable",
    ["All"] + csv_columns
)

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

# Columns to plot (numeric only)
plot_columns = [
    "fill_level",
    "capacity_TWh",
    "stored_energy_TWh",
    "fill_level_previous_week",
    "change_in_fill_level",
]

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