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

WAIT_SEC = .25

with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="bottom"):
    run_btn = st.button(":material/play_circle: Execute pipeline", type="primary")

if run_btn:
    col1, col2 = st.columns([1, 2])
    message_box = col1.status("Executing Offline pipeline...", expanded=True, state="running")
    metrics_placeholder = col2.container()
    chart_placeholder = col2.container()

    def ui_callback(msg, end=None):
        message_box.write(msg)
        time.sleep(WAIT_SEC)

        if end is True: message_box.update(state="complete", label="Finished!")

    def handle_report(news, **kwargs):
        metrics = news.get("metrics", {})

        with metrics_placeholder:
            st.subheader("Key Indicators")

            indicator_labels = {
                "pr_auc": "PR AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "predicted_anomaly_rate": "Predicted Anomaly Rate",
                "k_ratio": "K Ratio",
                "k": "K",
                "precision_at_k": "Precision@K",
                "recall_at_k": "Recall@K",
            }

            indicator_values = []
            for key, value in metrics.items():
                if key in {"pr_curve_precision", "pr_curve_recall"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    indicator_values.append((key, float(value)))

            if indicator_values:
                columns = st.columns(4)
                percent_keys = {
                    "precision",
                    "recall",
                    "f1",
                    "predicted_anomaly_rate",
                    "k_ratio",
                    "precision_at_k",
                    "recall_at_k",
                }

                for idx, (key, value) in enumerate(indicator_values):
                    label = indicator_labels.get(key, key.replace("_", " ").title())
                    if key in percent_keys:
                        formatted_value = f"{value:.2%}"
                    elif key == "k":
                        formatted_value = f"{int(round(value))}"
                    else:
                        formatted_value = f"{value:.4f}"
                    columns[idx % 4].metric(label, formatted_value)
            else:
                st.info("No scalar metrics available for indicators.")

        values, counts = np.unique(news["future_predictions"], return_counts=True)

        df = pd.DataFrame({
            "label": pd.Series(values).map({0: "Normal", 1: "Anomaly"}),
            "count": counts
        })

        base_chart = alt.Chart(df).encode(
            x=alt.X("label:N", title="Label"),
            y=alt.Y("count:Q", title="Count"),
        )

        bars = base_chart.mark_bar()
        labels = base_chart.mark_text(dy=-8).encode(text="count:Q")

        with chart_placeholder:
            st.subheader("Predictions of the pipeline")
            st.altair_chart((bars + labels).properties())
    
    cached_dataset = get_dataset()

    offline_detection.pipeline(cached_dataset, msg_callback=ui_callback, report_callback=handle_report)
