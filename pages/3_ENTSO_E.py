import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

st.set_page_config(page_title="ENTSO-E Cross-Border Flows", page_icon="⚡", layout="wide")
st.title("⚡ Cross-Border Electricity Transfer – ENTSO-E")

# Credentials 
try:
    ENTSOE_TOKEN = st.secrets["entsoe_token"]
except Exception:
    st.error("ENTSO-E token not found in secrets. Add `entsoe_token` to your secrets.")
    st.stop()

ENTSOE_API = "https://web-api.tp.entsoe.eu/api"

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


# ENTSO-E parsing / fetching 
def parse_entsoe_xml(xml_text: str) -> list[dict]:
    """Parse ENTSO-E physical flow XML into dicts (MW + interval length in hours)."""
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    ns = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
    q = (lambda name: f"{{{ns}}}{name}") if ns else (lambda name: name)

    records = []
    for ts_elem in root.iter(q("TimeSeries")):
        for period in ts_elem.iter(q("Period")):
            ti = period.find(q("timeInterval"))
            start_str = ti.findtext(q("start")) if ti is not None else None
            res_elem = period.find(q("resolution"))
            res_str = res_elem.text if res_elem is not None else "PT60M"
            if not start_str:
                continue
            try:
                t0 = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            delta = timedelta(hours=1) if res_str == "PT60M" else timedelta(minutes=15)
            for pt in period.iter(q("Point")):
                pos_el = pt.find(q("position"))
                qty_el = pt.find(q("quantity"))
                if pos_el is not None and qty_el is not None:
                    try:
                        ts = t0 + (int(pos_el.text) - 1) * delta
                        records.append({
                            "timestamp": ts.replace(tzinfo=None),
                            "value_MW": float(qty_el.text),
                            "hours": delta.total_seconds() / 3600,
                        })
                    except (ValueError, TypeError):
                        continue
    return records


@st.cache_data(ttl=86400, show_spinner="Fetching ENTSO-E data…")
def fetch_entsoe(out_domain: str, in_domain: str, year: int, token: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, debug_message)."""
    params = {
        "securityToken": token,
        "documentType":  "A11",
        "in_Domain":     in_domain,
        "out_Domain":    out_domain,
        "periodStart":   datetime(year, 1, 1).strftime("%Y%m%d%H%M"),
        "periodEnd":     datetime(year, 12, 31, 23, 59).strftime("%Y%m%d%H%M"),
    }
    try:
        r = requests.get(ENTSOE_API, params=params, timeout=60)
        debug = f"HTTP {r.status_code} · {len(r.text)} chars"
        if r.status_code != 200:
            return pd.DataFrame(), f"{debug} · Response: {r.text[:300]}"
        recs = parse_entsoe_xml(r.text)
        if not recs:
            try:
                root_tag = ET.fromstring(r.text).tag
            except Exception:
                root_tag = "parse error"
            return pd.DataFrame(), f"{debug} · 0 records parsed · root tag: {root_tag} · XML snippet: {r.text[:200]}"
        df = pd.DataFrame(recs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"{debug} · {len(df)} rows"
    except Exception as exc:
        return pd.DataFrame(), f"Exception: {exc}"


def load_corridor(corridor: str, year: int) -> tuple[dict, list[str]]:
    """Return {'Into Norway': df, 'Out of Norway': df} and debug messages."""
    no_zone, foreign_zone = [z.strip() for z in corridor.split(" - ")]
    no_eic, foreign_eic = ZONES[no_zone][0], ZONES[foreign_zone][0]

    frames, msgs = {}, []
    for label, (out_d, in_d) in {
        "Into Norway":   (foreign_eic, no_eic),
        "Out of Norway": (no_eic, foreign_eic),
    }.items():
        df, dbg = fetch_entsoe(out_d, in_d, year, ENTSOE_TOKEN)
        frames[label] = df
        msgs.append(f"**{label}**: {dbg}")
    return frames, msgs


# ═════════════════════════════════════════════════════════════════════════════
# Shared controls (one year selector for BOTH charts)
# ═════════════════════════════════════════════════════════════════════════════
current_year = datetime.now().year

corridor = st.radio("Choose a connection:", CORRIDORS, horizontal=True)

ctrl_left, ctrl_right = st.columns([1, 2], gap="large")
with ctrl_left:
    st.subheader("Annual Energy Exchange")
    year = st.selectbox(
        "Year",
        options=list(range(current_year - 4, current_year + 1)),
        index=4,
    )
with ctrl_right:
    st.subheader("Monthly Flow Profile")
    selected_months = st.pills(
        "Months",
        options=MONTHS,
        selection_mode="multi",
        default=["Jan"],
    )

frames, debug_msgs = load_corridor(corridor, year)
has_data = any(not f.empty for f in frames.values())
title_names = corridor_title(corridor)

col_left, col_right = st.columns([1, 1], gap="large")

# Pie chart: yearly transfer 
with col_left:
    if not has_data:
        st.info("No data available for this connection/year.")
        with st.expander("🔍 Debug info", expanded=True):
            for m in debug_msgs:
                st.markdown(m)
    else:
        totals = {
            label: (df["value_MW"] * df["hours"]).sum() / 1e6  # MWh -> TWh
            for label, df in frames.items() if not df.empty
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
            ax_line.set_xlabel("Date")
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
