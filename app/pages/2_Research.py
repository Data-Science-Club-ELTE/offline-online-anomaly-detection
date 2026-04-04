from pathlib import Path

import streamlit as st

from streamlit_pdf_viewer import pdf_viewer
from branding import render_logo

st.set_page_config(
    page_title="Research",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

WORK_PRODUCTS_DIR = Path(__file__).resolve().parent.parent / "work_products"

st.title(":material/genetics: Research")

pdf_viewer(WORK_PRODUCTS_DIR / "research_report.pdf")
