Project work for the IND320 course at NMBU.


## Part 1

Part 1 explores the reservoir dataset and includes:

* data inspection and visualisation in Jupyter
* a multipage Streamlit application
* a data table with first-month trends
* interactive Plotly visualisations of reservoir data

## Part 2

Part 2 extends the project with:

* reservoir data retrieved directly from the NVE Magasinstatistikk API
* ENTSO-E cross-border electricity flow data
* Cassandra and Spark for storing and processing the data locally
* MongoDB for the data used by the cross-border Streamlit page

## Part 3

Part 3 adds data analysis in Jupyter and Streamlit, including:

* STL decomposition and spectrogram analysis of weekly reservoir filling
* hourly ERA5 weather data downloaded through Open-Meteo
* DCT filtering and SPC for identifying unusual precipitation
* Local Outlier Factor (LOF) for detecting temperature anomalies

## Part 4

Part 4 adds geospatial analysis and forecasting, including:

* an interactive map of price areas and water areas (GeoJSON)
* sliding window correlation between weather and reservoir filling
* Norges Bank exchange rates as an additional variable
* SARIMAX and lag-based forecasting, including online models with `river`

## Streamlit app

[IND320 Energy Project](https://ind320-59762.streamlit.app/)

The app includes reservoir tables and plots, STL and spectrogram analysis, weather analysis using SPC and LOF, cross-border electricity transfer, an area map, sliding window correlation and reservoir forecasting.

## Run locally

Install the project dependencies and start the app:

```
pip install -r requirements.txt
streamlit run app.py
```

The app retrieves reservoir data from NVE and weather data from Open-Meteo. The map and weather pages also use local shapefiles for area sampling. Cross-border Transfers requires a configured MongoDB connection in `.streamlit/secrets.toml`:

```
entsoe_token = "<your ENTSO-E token>"
mongodb_uri = "<your MongoDB connection string>"
```
