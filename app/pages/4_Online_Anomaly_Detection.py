import streamlit as st
from pathlib import Path

from branding import render_logo

ASSETS_DIR = Path(__file__).resolve().parent.parent / "work_products"

st.set_page_config(
    page_title="Online Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/earthquake: Stream (Online) Anomaly Detection")
st.markdown("### We are just getting started...")

st.image(f"{ASSETS_DIR}/stream_data_distribution.svg")