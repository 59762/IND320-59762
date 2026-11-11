import streamlit as st
from datetime import date

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About")

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown("### The project")
    st.write(
        "This dashboard was built for **IND320 – Project Work**. It brings together "
        "open energy data into one interactive app, so you can explore how power is "
        "produced, stored and traded across the Nordic region."
    )

    st.markdown("### Data sources")
    st.markdown("""
    | Source | What it provides |
    |--------|------------------|
    | **ENTSO-E Transparency Platform** | Physical electricity flows between countries |
    | **NVE Magasinstatistikk** | Norwegian reservoir levels |
    | **MongoDB / Cassandra** | Storage for the collected data |
    """)

    st.markdown("### Built with")
    st.markdown("`Python` · `Streamlit` · `Pandas` · `Matplotlib` · `MongoDB` · `Cassandra` · `GitHub`")

with right:
    with st.container(border=True):
        st.markdown("### 👩‍💻 Author")
        st.write("**Alina Shah**")
        st.caption("MSc Data Science, NMBU")
        st.link_button("LinkedIn", "https://www.linkedin.com/in/alinas7")
        st.link_button("GitHub repository", "https://github.com/59762/IND320-59762")

st.divider()
st.caption(f"Last updated: {date.today():%Y-%m-%d}")