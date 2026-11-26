"""About page: project background, data sources, methods and metadata."""
import streamlit as st


# metadata (edit here)
AUTHOR = "Alina Shah"
PROGRAMME = "MSc Data Science, NMBU"
COURSE = "IND320 – Project Work"
APP_VERSION = "Part 4 – Machine learning"
LAST_UPDATED = "2026-11-26"          # update by hand when you deploy; date.today() would change daily
REPO_URL = "https://github.com/59762/IND320-59762"
APP_URL = "https://ind320-59762.streamlit.app/"
LINKEDIN_URL = "https://www.linkedin.com/in/alinas7"

# header
st.title("ℹ️ About this project")
st.caption("What the dashboard is, where the data comes from and how the analysis works.")

main, side = st.columns([2, 1], gap="large")

# main column
with main:
    tab_project, tab_data, tab_methods, tab_tech = st.tabs(
        ["Project", "Data sources", "Methods", "Built with"]
    )

    with tab_project:
        st.markdown("### Background")
        st.write(
            f"This dashboard is the semester project for **{COURSE}**. It was built in four steps, "
            "and every step added a layer: first getting the data on screen, then collecting it from "
            "several sources, then checking its quality, and finally learning from it."
        )
        st.markdown("### The four parts")
        st.markdown(
            """
| Part | Theme | What it added to the app |
|------|-------|--------------------------|
| **1** | Dashboard basics | NVE reservoir data table and interactive plot |
| **2** | Data sources | ENTSO-E cross-border flows, stored in MongoDB (Cassandra and Spark in the notebook) |
| **3** | Data quality | STL decomposition, spectrogram, outlier and anomaly detection |
| **4** | Machine learning | Area map, sliding-window correlation and forecasting |
"""
        )
        st.markdown("### The question behind it")
        st.write(
            "Norwegian electricity depends on water. The dashboard asks how reservoir levels, weather and "
            "cross-border trade relate to each other, and whether the reservoir level of an area can be "
            "forecast a few weeks ahead."
        )

    with tab_data:
        st.markdown("### Where the data comes from")
        st.markdown(
            """
| Source | What it provides | Used on |
|--------|------------------|---------|
| **NVE Magasinstatistikk** | Weekly reservoir filling and capacity for price areas, water areas and Norway | Data Table, Plot, Reservoir Analysis, Map, Correlation, Forecasting |
| **ENTSO-E Transparency Platform** | Physical electricity flows between countries | Electricity Transfer |
| **Open-Meteo (ERA5)** | Daily weather reanalysis, sampled inside each area | Weather Analysis, Map, Correlation, Forecasting |
| **NVE shapefiles** | Price-area and water-area boundaries, converted to GeoJSON on the fly | Map, Weather Analysis |
| **MongoDB** | Stores the processed ENTSO-E data that the app reads | Electricity Transfer |
"""
        )
        st.caption(
            "NVE data is fetched live and cached for about an hour. Data remains subject to each provider's "
            "own terms of use. Weather data is ERA5 reanalysis (Copernicus / ECMWF) delivered by Open-Meteo."
        )

    with tab_methods:
        st.markdown("### What happens to the data")
        st.markdown(
            """
- **STL decomposition** splits the reservoir curve into trend, seasonal pattern and residual.
- **Spectrogram** shows which cycles are strong and how they change over time.
- **Outlier detection** flags unusual weather values with a DCT-based statistical process control (SPC) approach.
- **Local Outlier Factor (LOF)** finds anomalies by comparing each point with its neighbours.
- **Sliding-window correlation** measures how closely weather follows the reservoir level, with an adjustable window and lag.
- **Forecasting** uses SARIMAX or machine-learning regressors (Ridge, random forest, gradient boosting), with confidence intervals around the forecast.
"""
        )
        st.info(
            "Forecasts here are a course exercise. They show how the methods behave, "
            "not an operational prediction of reservoir levels.",
            icon="⚠️",
        )

    with tab_tech:
        st.markdown("### Technology")
        st.markdown(
            "`Python` · `Streamlit` · `pandas` · `NumPy` · `Plotly` · `GeoPandas` · `SciPy` · "
            "`statsmodels` · `scikit-learn` · `PyMongo` · `GitHub` · `Streamlit Community Cloud`"
        )
        st.caption(
            "The notebooks additionally use Cassandra, PySpark and River for the data-source and "
            "streaming-forecast exercises."
        )
        st.markdown("### How it fits together")
        st.code(
            "Public APIs (NVE · ENTSO-E · Open-Meteo)\n"
            "        │\n"
            "        ├─ notebooks → Cassandra / Spark / MongoDB (collect and prepare)\n"
            "        ▼\n"
            "  utils/  (data loading, geo, forecasting)\n"
            "        │\n"
            "        ▼\n"
            "  Streamlit pages  (shared session state: area + time frame)",
            language="text",
        )

# side column
with side:
    with st.container(border=True):
        st.markdown("### 👩‍💻 Author")
        st.write(f"**{AUTHOR}**")
        st.caption(PROGRAMME)
        st.link_button("LinkedIn", LINKEDIN_URL, use_container_width=True)
        st.link_button("GitHub repository", REPO_URL, use_container_width=True)
        st.link_button("Live app", APP_URL, use_container_width=True)

    with st.container(border=True):
        st.markdown("### 🏷️ Project details")
        st.markdown(
            f"**Course:** {COURSE}  \n"
            f"**Stage:** {APP_VERSION}  \n"
            f"**Last updated:** {LAST_UPDATED}"
        )
        
st.divider()
st.caption(f"© {AUTHOR} · Built for {COURSE}")