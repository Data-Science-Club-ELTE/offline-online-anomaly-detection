import streamlit as st
import importlib
import time
import pandas as pd
import altair as alt

from pathlib import Path
from branding import render_logo

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

warning_label = st.warning("The execution will download the entire dataset. May be slow or costly for the first time.", icon="⚠️")

with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="bottom"):

    global total_n_samples
    total_n_samples = get_dataset().n_samples

    n_selector = st.selectbox(
        "Transaction amount", key="process_n_observations", options=[1000, 5000, "All transactions"], width=180,
        help="Number of transactions to process. Select 'All transactions' to process the entire dataset (Warning: This may take a while).")

    report_frequency_selector = st.selectbox(
        "Reporting frequency (in minutes)", key="report_every_seconds_elapsed", options=[1, 5, 10], width=250,
        help="How often to generate reports (in minutes).")

    sleep_time_selector = st.selectbox(
        "Sleep time between reports (in seconds)", key="sleep_time_seconds", options=[1, 2, 5], width=275,
        help="How long to sleep between reports (in seconds). This is useful to simulate the time taken to generate a report and to make the UI more responsive.")
    
    run_btn = st.button(":material/play_circle: Execute pipeline", type="primary")
    
    global process_n_observations, report_every_seconds_elapsed
    process_n_observations = n_selector if isinstance(n_selector, int) else total_n_samples
    report_every_seconds_elapsed = report_frequency_selector * 60


if run_btn:

    warning_label.empty()

    processed_so_far = 0
    processed_label = st.empty()
    progress_bar = st.progress(0.0, text="Processing incoming transactions")
    chart_placeholder = st.empty()

    def handle_report(news, **kwargs):
        data = pd.DataFrame(news["data"])
        scores = pd.DataFrame(news["scores"])
        df = pd.concat((data, scores), axis=1)

        global processed_so_far
        processed_so_far += data.shape[0]

        progress_bar.progress(processed_so_far / process_n_observations)
        processed_label.write(f"Processed transactions: {processed_so_far}/{process_n_observations}")

        chart_placeholder.altair_chart(
            alt.Chart(df).mark_point(color="none", opacity=.9).encode(
                x=alt.X('Time', title="Time (elapsed seconds)",  scale=alt.Scale(domain=(df['Time'].min(), df['Time'].max()))),
                y=alt.Y("score", title="Anomaly score", scale=alt.Scale(domain=(0, 1))),
                stroke=alt.condition(
                    alt.datum.score > 0.9,
                    alt.ColorValue('red'),
                    alt.ColorValue('green')
                ),
                shape=alt.condition(
                    alt.datum.score > 0.9,
                    alt.value('triangle-up'),
                    alt.value('circle')
                ),
                size=alt.condition(
                    alt.datum.score > 0.9,
                    alt.value(100),
                    alt.value(25)
                )
            ),
            use_container_width=True
        )
        
        time.sleep(sleep_time_selector)

    dataset = get_dataset()

    # FIXME
    online_detection.pipeline(
        cached_dataset=dataset,
        process_n_observations=process_n_observations,
        report_every_seconds_elapsed=report_every_seconds_elapsed,
        report_callback=handle_report)
