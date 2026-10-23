import streamlit as st

#st.set_page_config(page_title="IND320 Reservoir Dashboard", layout="wide")
st.set_page_config(page_title="Data", page_icon="🏠", layout="wide",)
st.title("⚡Energy Dashboard – IND320 Project Work")
st.sidebar.success("Select a page above.")

st.markdown("""Welcome to my IND320 project work.""")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("About this app")
    st.write(
        "This dashboard is built step by step during the IND320 course. "
        "It collects open energy data and presents it in interactive charts."
    )
    st.write("Use the **sidebar** to switch between pages.")

with right:
    with st.container(border=True):
        st.subheader("Project status")
        st.markdown("""
        ✅ Dashboard basics  
        ✅ Cross-border electricity flows (ENTSO-E)  
        🚧 More data and analysis coming soon
        """)

st.caption("🚧 Work in progress – new pages and features are added during the semester.")