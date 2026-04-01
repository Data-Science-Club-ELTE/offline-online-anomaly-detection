import streamlit as st
import importlib
import time
import pandas as pd
import numpy as np
import altair as alt

from pathlib import Path
from branding import render_logo

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "work_products"
MODULE_PATH = BASE_DIR.parent / "offline-detection" / "main.py"

offline_detection = importlib.machinery.SourceFileLoader(
    "offline-detection", str(MODULE_PATH)
).load_module()

st.set_page_config(
    page_title="Offline Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/potted_plant: Offline Anomaly Detection")
st.caption("Traditional anomaly detector for static credit card transaction data.")

@st.cache_data
def get_dataset():
    import kagglehub
    
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = Path(dataset_path) / "creditcard.csv"
    df = pd.read_csv(csv_file)

    data = df.drop(columns=["Class"])
    target = df["Class"]

    return data, target

@st.cache_data
def get_dummy_data():
    data = pd.DataFrame(np.random.randn(1000, 10))
    target = pd.Series(np.random.randint(0, 2, size=1000))
    return data, target

WAIT_SEC = 1

with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="bottom"):
    data_label = st.container(width=500)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        run_btn = st.button(":material/play_circle: Execute pipeline", type="primary")
        use_real = st.toggle("Use full dataset (⚠ data-expensive)", value=False)

        if use_real:
            data_label.warning("This will download ~150MB via kagglehub. May be slow or costly.", icon="⚠️")
        else:
            data_label.info("Using lightweight dummy data.", icon="ℹ️")

if run_btn:
    col1, col2 = st.columns(2)
    message_box = col1.status("Executing Offline pipeline...", expanded=True, state="running")
    chart_placeholder = col2.empty()

    def ui_callback(msg, end=None):
        message_box.write(msg)
        time.sleep(WAIT_SEC)

        if end is True: message_box.update(state="complete", label="Finished!")

    def handle_report(news, **kwargs):
        values, counts = np.unique(news["preds"], return_counts=True)

        df = pd.DataFrame({
            "label": pd.Series(values).map({0: "Normal", 1: "Anomaly"}),
            "count": counts
        })
        chart_placeholder.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x="label:N",
                y="count:Q",
            ).properties(
                title="Predictions of the pipeline"
            )
        )
    
    if use_real:
        cached_dataset = get_dataset()
    else:
        cached_dataset = get_dummy_data()

    offline_detection.pipeline(cached_dataset, msg_callback=ui_callback, report_callback=handle_report)
