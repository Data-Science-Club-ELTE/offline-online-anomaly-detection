import streamlit as st
from branding import render_logo

st.set_page_config(
    page_title="Research",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/genetics: Research")
st.markdown("### Research is in progress...")
