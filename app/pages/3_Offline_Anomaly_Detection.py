import streamlit as st

from branding import render_logo

st.set_page_config(
    page_title="Offline Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/potted_plant: Traditional (Offline) Anomaly Detection")
st.markdown("### We are just getting started...")
