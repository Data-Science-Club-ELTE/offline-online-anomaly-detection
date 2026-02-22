import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO1_PATH = ASSETS_DIR / "dscelte_logo.jpeg"
LOGO2_PATH = ASSETS_DIR / "ophelia_02.png"

TITLE = "## Offline & Online Anomaly Detection"
CAPTION = "Credit card fraud detection • An *ELTE Data Science Club* ⨉ *Ophelia R&D* project • [GitHub repository](https://github.com/Data-Science-Club-ELTE/offline-online-anomaly-detection)"

def render_home_header() -> None:
    if LOGO1_PATH.exists() and LOGO2_PATH.exists():
        with st.container(horizontal=False, horizontal_alignment="left", vertical_alignment="center"):
            with st.container(horizontal=True):
                st.image(str(LOGO1_PATH), width=100)
                st.image(str(LOGO2_PATH), width=350)

            with st.container():
                st.markdown(TITLE)
                st.caption(CAPTION)
    else:
        st.markdown(TITLE)
        st.caption(CAPTION)

def render_brand_footer() -> None:
    st.divider()
    st.caption("Copyright &copy; 2026 ELTE Data Science Club, Ophelia R&D")
