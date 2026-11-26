import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide",)


def home_page():
    st.title("⚡ Energy Dashboard – IND320 Project Work")
    st.write(
        "An interactive dashboard that brings together open data on **Norwegian water "
        "reservoirs**, **cross-border electricity flows** and **weather**, so you can "
        "explore how power is stored, traded and affected by the weather. "
        "Pick a topic below or in the sidebar menu to get started."
    )

    # Reservoirs
    st.subheader("💧 Reservoirs")
    st.caption("Weekly reservoir statistics from NVE (Magasinstatistikk).")
    c1, c2, c3 = st.columns(3, gap="medium")
    page_card(
        c1, data_page,
        "Browse the raw NVE reservoir data for any area type and number, with a "
        "one-month summary that shows each variable as a small trend line.",
    )
    page_card(
        c2, plot_page,
        "Plot fill level, capacity and stored energy over time. Choose a variable "
        "and a month range, or compare all variables on a normalised scale.",
    )
    page_card(
        c3, stl_page,
        "Split the fill percentage into trend, seasonal pattern and residual (STL), "
        "and see how the yearly cycle changes over time in a spectrogram.",
    )
    c4, c5, _ = st.columns(3, gap="medium")
    page_card(
        c4, map_page,
        "Select a price area, water area or all of Norway on an interactive map "
        "coloured by reservoir filling or weather. The selection and time frame "
        "carry over to the correlation and forecast pages.",
    )
    page_card(
        c5, forecast_page,
        "Forecast the weekly reservoir series of the selected area with SARIMAX or "
        "ML models, with confidence intervals and an animated forecast.",
    )

    # Electricity and weather
    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("🔌 Electricity")
        st.caption("Physical flows from the ENTSO-E Transparency Platform, stored in MongoDB.")
        page_card(
            st.container(), flows_page,
            "Explore power exchange between Norway and its neighbours. See how much "
            "energy flowed into and out of Norway in a year, and the hourly flow "
            "profile for the months you select.",
        )
    with right:
        st.subheader("🌦️ Weather")
        st.caption("ERA5 reanalysis from Open-Meteo, averaged over an NVE area.")
        page_card(
            st.container(), weather_page,
            "Find unusual days in the weather data: precipitation outliers using a "
            "DCT filter with statistical process control, and temperature anomalies "
            "using Local Outlier Factor (LOF).",
        )
        page_card(
            st.container(), corr_page,
            "Does the weather move with the reservoir? Pearson correlation in a "
            "sliding window, with an adjustable lag between the two series.",
        )

    st.divider()
    st.page_link(about_page, label="About: data sources, author and GitHub repository", icon="ℹ️")
    st.caption("🚧 Work in progress – new pages and features are added during the semester.")


def page_card(container, page, description):
    """A bordered card with a link to the page and a short description."""
    with container.container(border=True):
        st.page_link(page, label=page.title, icon=page.icon)
        st.write(description)


# Pages and menu (grouped by topic)
home = st.Page(home_page, title="Home", icon="🏠", default=True)

data_page = st.Page("pages/1_Data_Table.py", title="Data", icon="📋", url_path="reservoir-data")
plot_page = st.Page("pages/2_Plot.py", title="Plot", icon="📈", url_path="reservoir-plot")
stl_page = st.Page("pages/4_Reservoir_Analysis.py", title="Analysis", icon="🌊", url_path="stl-spectrogram")
map_page = st.Page("pages/6_Map_and_Selectors.py", title="Map", icon="🗺️", url_path="map")
forecast_page = st.Page("pages/8_Water_Forecasting.py", title="Forecast", icon="🔮", url_path="forecast")

flows_page = st.Page("pages/3_Electricity_Transfer.py", title="Cross-Border Flows", icon="⚡", url_path="cross-border-flows")

weather_page = st.Page("pages/5_Weather_Analysis.py", title="Analysis", icon="🔍", url_path="outliers-anomalies")
corr_page = st.Page("pages/7_Sliding_Window_Correlation.py", title="Correlation", icon="📈", url_path="correlation")

about_page = st.Page("pages/9_About.py", title="About", icon="ℹ️", url_path="about")

nav = st.navigation(
    {
        "": [home],
        "Reservoirs": [data_page, plot_page, stl_page, map_page, forecast_page],
        "Electricity": [flows_page],
        "Weather": [weather_page, corr_page],
        "Project": [about_page],
    }
)
nav.run()