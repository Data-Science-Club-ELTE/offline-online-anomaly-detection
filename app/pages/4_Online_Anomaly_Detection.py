import streamlit as st
import importlib
import time
import pandas as pd
import altair as alt

from pathlib import Path
from branding import render_logo

from river import metrics

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "work_products"
MODULE_PATH = BASE_DIR.parent / "online-detection" / "main.py"

online_detection = importlib.machinery.SourceFileLoader(
    "online-detection", str(MODULE_PATH)
).load_module()

st.set_page_config(
    page_title="Online Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/earthquake: Online Anomaly Detection")
st.caption("Online anomaly detector for streamlined credit card transaction monitoring.")

@st.cache_data
def get_dataset():
    from river import datasets
    return datasets.CreditCard()


with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="bottom"):

    global total_n_samples
    total_n_samples = get_dataset().n_samples

    n_selector = st.selectbox(
        "Transaction amount", key="process_n_observations", options=[1000, 5000, "All transactions"], index=2, width=180,
        help="Number of transactions to process. Select 'All transactions' to process the entire dataset (Warning: This may take a while).")

    report_frequency_selector = st.selectbox(
        "Reporting frequency (in minutes)", key="report_every_seconds_elapsed", options=[1, 5, 10, 30], index=2, width=250,
        help="How often to generate reports (in minutes).")

    sleep_time_selector = st.selectbox(
        "Sleep time between reports (in seconds)", key="sleep_time_seconds", options=[.5, 1, 2, 5], index=1, width=275,
        help="How long to sleep between reports (in seconds). This is useful to simulate the time taken to generate a report and to make the UI more responsive.")
    
    run_btn = st.button(":material/play_circle: Execute pipeline", type="primary")
    
    global process_n_observations, report_every_seconds_elapsed
    process_n_observations = n_selector if isinstance(n_selector, int) else total_n_samples
    report_every_seconds_elapsed = report_frequency_selector * 60


if run_btn:

    processed_so_far = 0
    processed_label = st.empty()
    progress_bar = st.progress(0.0, text="Processing incoming transactions")
    col1, col2 = st.columns([1, 2])
    message_box = col1.status("Executing Online pipeline...", expanded=True, state="running")
    metrics_row = col2.container()
    metric_cols = metrics_row.columns(3)
    precision_placeholder = metric_cols[0].empty()
    recall_placeholder = metric_cols[1].empty()
    f1_placeholder = metric_cols[2].empty()
    chart_placeholder = col2.empty()

    def ui_callback(msg, end=None):
        message_box.write(msg)
        time.sleep(sleep_time_selector)

        if end is True:
            message_box.update(state="complete", label="Finished!")

    def handle_report(news, **kwargs):
        data = pd.DataFrame(news["data"])
        scores = pd.DataFrame(news["scores"])
        predictions = pd.DataFrame(news["predictions"])
        df = pd.concat((data, scores, predictions), axis=1)
        df["prediction"] = df["prediction"].map({False: "Normal", True: "Anomaly"})

        global processed_so_far
        processed_so_far += data.shape[0]

        progress_value = min(processed_so_far / max(process_n_observations, 1), 1.0)
        progress_bar.progress(progress_value)
        processed_label.write(f"Processed transactions: {processed_so_far}/{process_n_observations}")

        chart_placeholder.altair_chart(
            alt.Chart(df).mark_point(filled=True, opacity=.9).encode(
                x=alt.X('Time', title="Time (elapsed seconds)",  scale=alt.Scale(domain=(df['Time'].min(), df['Time'].max()))),
                y=alt.Y("score", title="Anomaly score", scale=alt.Scale(domain=(0, 1))),
                color=alt.Color(
                    "prediction:N",
                    title="Prediction",
                    scale=alt.Scale(domain=["Normal", "Anomaly"], range=["green", "red"]),
                ),
                shape=alt.Shape(
                    "prediction:N",
                    scale=alt.Scale(domain=["Normal", "Anomaly"], range=["circle", "triangle-up"]),
                    legend=None,
                ),
                size=alt.condition(
                    alt.datum.prediction == "Anomaly",
                    alt.value(100),
                    alt.value(25)
                )
            ),
            use_container_width=True
        )

        precision: metrics.Precision = news.get("precision")
        recall: metrics.Recall = news.get("recall")
        f1: metrics.F1 = news.get("f1")

        precision_placeholder.metric("Precision", f"{precision.get():.3f}")
        recall_placeholder.metric("Recall", f"{recall.get():.3f}")
        f1_placeholder.metric("F1", f"{f1.get():.3f}")
        
        time.sleep(sleep_time_selector)

    dataset = get_dataset()

    # FIXME
    online_detection.pipeline(
        dataset=dataset,
        process_n_observations=process_n_observations,
        report_every_seconds_elapsed=report_every_seconds_elapsed,
        msg_callback=ui_callback,
        report_callback=handle_report)
