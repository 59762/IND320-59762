import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pymongo import MongoClient

st.set_page_config(page_title="ENTSO-E Cross-Border Flows", page_icon="⚡", layout="wide")
st.title("⚡ Cross-Border Electricity Transfer – ENTSO-E")

# Credentials 
try:
    MONGODB_URI = st.secrets["mongodb_uri"]  # stored in Streamlit secrets
except Exception:
    st.error("MongoDB connection string not found in secrets. Add `mongodb_uri` to your secrets.")
    st.stop()

DB_NAME = "ind320"
YEARLY_COLLECTION = "entsoe_yearly"   # one document per corridor + direction + year
HOURLY_COLLECTION = "entsoe_hourly"   # one document per corridor + direction + hour

# Bidding zones: code -> (EIC code, country name) 
ZONES = {
    "NO1":   ("10YNO-1--------2",   "Norway"),
    "NO2":   ("10YNO-2--------T",   "Norway"),
    "NO3":   ("10YNO-3--------J",   "Norway"),
    "NO4":   ("10YNO-4--------9",   "Norway"),
    "SE1":   ("10Y1001A1001A44P",   "Sweden"),
    "SE2":   ("10Y1001A1001A45N",   "Sweden"),
    "SE3":   ("10Y1001A1001A46L",   "Sweden"),
    "FI":    ("10YFI-1--------U",   "Finland"),
    "DK1":   ("10YDK-1--------W",   "Denmark"),
    "DE-LU": ("10Y1001A1001A82H",   "Germany"),
    "GB":    ("10YGB----------A",   "Great Britain"),
    "NL":    ("10YNL----------L",   "Netherlands"),
}

# Norwegian zone first, foreign zone second. Add a connection = add one line.
CORRIDORS = [
    "NO1 - SE3",
    "NO2 - DE-LU",
    "NO2 - DK1",
    "NO2 - GB",
    "NO2 - NL",
    "NO3 - SE2",
    "NO4 - FI",
    "NO4 - SE1",
    "NO4 - SE2",
]

PIE_COLORS = {"Into Norway": "#0072B2", "Out of Norway": "#D55E00"}
LINE_COLORS = {"Into Norway": "#1f77b4", "Out of Norway": "#ff7f0e"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def corridor_title(corridor: str) -> str:
    """'NO2 - DK1' -> 'Norway - Denmark'"""
    no_zone, foreign_zone = [z.strip() for z in corridor.split(" - ")]
    return f"{ZONES[no_zone][1]} - {ZONES[foreign_zone][1]}"


# MongoDB access 
@st.cache_resource
def get_db():
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    return client[DB_NAME]


@st.cache_data(ttl=3600, show_spinner="Reading available years from MongoDB…")
def load_available_years() -> list[int]:
    return sorted(int(y) for y in get_db()[YEARLY_COLLECTION].distinct("year"))


@st.cache_data(ttl=3600, show_spinner="Reading yearly totals from MongoDB…")
def load_yearly(corridor: str, year: int) -> pd.DataFrame:
    """Yearly energy per direction for one connection (TWh)."""
    docs = list(get_db()[YEARLY_COLLECTION].find(
        {"corridor": corridor, "year": year},
        {"_id": 0, "direction": 1, "energy_twh": 1},
    ))
    return pd.DataFrame(docs)


@st.cache_data(ttl=3600, show_spinner="Reading hourly flows from MongoDB…")
def load_hourly(corridor: str, year: int, months: tuple[int, ...]) -> pd.DataFrame:
    """Hourly flow (MW, UTC) for one connection, year and set of months."""
    if not months:
        return pd.DataFrame()
    docs = list(get_db()[HOURLY_COLLECTION].find(
        {"corridor": corridor, "year": year, "month": {"$in": list(months)}},
        {"_id": 0, "direction": 1, "month": 1, "ts": 1, "flow_mw": 1},
    ))
    df = pd.DataFrame(docs)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
    return df


# Check the database connection / available data
try:
    years = load_available_years()
except Exception as exc:
    st.error(f"Could not read from MongoDB: {exc}")
    st.stop()

if not years:
    st.warning(
        f"The `{YEARLY_COLLECTION}` collection is empty. "
        "Run `notebook_2.ipynb` to download and store the ENTSO-E data first."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Shared controls (one year selector for BOTH charts)
# ═════════════════════════════════════════════════════════════════════════════

corridor = st.radio("Choose a connection:", CORRIDORS, horizontal=True)

ctrl_left, ctrl_right = st.columns([1, 1], gap="large")
with ctrl_left:
    st.subheader("Annual Energy Exchange")
    year = st.selectbox(
        "Year",
        options=years,
        index=len(years) - 1,
    )
with ctrl_right:
    st.subheader("Monthly Flow Profile")
    selected_months = st.pills(
        "Months",
        options=MONTHS,
        selection_mode="multi",
        default=["Jan"],
    )

yearly_df = load_yearly(corridor, year)
has_data = not yearly_df.empty
title_names = corridor_title(corridor)

col_left, col_right = st.columns([1, 1], gap="large")

# Pie chart: yearly transfer 
with col_left:
    if not has_data:
        st.info("No data available for this connection/year.")
    else:
        totals = {
            label: float(yearly_df.loc[yearly_df["direction"] == label, "energy_twh"].sum())
            for label in ["Into Norway", "Out of Norway"]
            if (yearly_df["direction"] == label).any()
        }
        labels = list(totals.keys())
        values = list(totals.values())
        total = sum(values)

        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        wedges, _, autotexts = ax_pie.pie(
            values,
            colors=[PIE_COLORS[k] for k in labels],
            autopct=lambda p: "",
            startangle=90,
            counterclock=False,
            pctdistance=0.62,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"color": "white", "fontsize": 10, "ha": "center"},
        )
        for at, lab, val in zip(autotexts, labels, values):
            at.set_text(f"{lab}\n{val:.2f} TWh\n{val / total * 100:.1f}%")
        ax_pie.legend(
            wedges, labels, title="Direction",
            loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
        )
        ax_pie.set_title(
            f"Yearly electricity transfer - {title_names} ({year})",
            loc="left", fontsize=11,
        )
        st.pyplot(fig_pie)
        plt.close(fig_pie)

# Line plot: physical flow for the selected months 
with col_right:
    if not selected_months:
        st.info("Select at least one month above.")
    elif not has_data:
        st.info("No data available for this connection/year.")
    else:
        month_nums = [MONTHS.index(m) + 1 for m in selected_months]
        hourly_df = load_hourly(corridor, year, tuple(sorted(month_nums)))

        frames = {}
        for label in ["Into Norway", "Out of Norway"]:
            d = hourly_df[hourly_df["direction"] == label] if not hourly_df.empty else hourly_df
            frames[label] = d.rename(columns={"ts": "timestamp", "flow_mw": "value_MW"})

        fig_line, ax_line = plt.subplots(figsize=(9, 4.8))
        plotted = False
        for label, df in frames.items():
            if df.empty:
                continue
            first = True
            for m in month_nums:  # one segment per month, so no line across gaps
                seg = df[df["timestamp"].dt.month == m].sort_values("timestamp")
                if seg.empty:
                    continue
                ax_line.plot(
                    seg["timestamp"], seg["value_MW"],
                    color=LINE_COLORS[label], linewidth=1.2,
                    label=label if first else None,
                )
                first = False
                plotted = True

        if not plotted:
            st.info("No data for the selected months.")
            plt.close(fig_line)
        else:
            ax_line.set_title(f"Electricity flow - {title_names} ({year})")
            ax_line.set_xlabel("Date (UTC)")
            ax_line.set_ylabel("Flow (MW)")
            ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax_line.grid(True, alpha=0.5)
            ax_line.legend(loc="center")
            fig_line.autofmt_xdate()
            fig_line.tight_layout()
            st.pyplot(fig_line)
            plt.close(fig_line)

# Documentation 
st.divider()
with st.expander("📄 Data source documentation", expanded=False):
    st.markdown("""
    ### ENTSO-E Transparency Platform

    The **European Network of Transmission System Operators for Electricity (ENTSO-E)**
    publishes electricity data via its [Transparency Platform](https://transparency.entsoe.eu/).

    **API endpoint used:** `https://web-api.tp.entsoe.eu/api`

    **Document type:** `A11` — Aggregated Energy Data Report (Physical flow)

    This document type returns the measured cross-border electricity flow between two
    bidding zones. For each connection the app requests both directions
    (*Into Norway* and *Out of Norway*). The pie chart shows the yearly energy
    (MW × hours, shown in TWh); the line plot shows the flow in MW for the selected months.

    **Connections shown:** NO1–SE3, NO2–DE-LU, NO2–DK1, NO2–GB, NO2–NL,
    NO3–SE2, NO4–FI, NO4–SE1, NO4–SE2.

    **References:**
    - [ENTSO-E API guide](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
    - [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
    - [NVE Magasinstatistikk API](https://biapi.nve.no/magasinstatistikk/swagger/index.html)
    """)
