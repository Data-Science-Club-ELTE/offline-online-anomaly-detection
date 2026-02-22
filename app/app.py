import streamlit as st

from branding import render_brand_footer

st.set_page_config(
    page_title="Offline & Online Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("pages/home.py", title="About the Project", icon=":material/rocket:"),
        st.Page("pages/offline.py", title="Offline detection", icon=":material/eco:"),
        st.Page("pages/online.py", title="Online detection", icon=":material/earthquake:"),
    ]
)

pages.run()

render_brand_footer()
