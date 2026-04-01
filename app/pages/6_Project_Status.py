import streamlit as st

from branding import render_logo

st.set_page_config(
    page_title="Project Status",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/detector: Project Status")

st.container(height=25, border=False)

st.caption("27th March | Milestone session")

st.markdown("The **Traditional Modeling** team has addressed data exploration, cleaning, and preprocessing steps. As a next step they are going to create the _Isolation Forest_ model and glue all pre-modeling steps into a single pipeline, execute the algorithm, and then evaluate the performance. Unfinished tasks shall not block us from moving forward, they can be revisited later, if the pipeline calls for improvements. The pipeline is being developed using `sklearn`.")

st.markdown("The **Stream Modeling** team did less data exploration, they started building the pipeline with the `river` library, using the _Half-Space Trees_ algorithm. The pipeline is already executed with the streamlined data, and evaluated via the ROC-AUC metric, however this is not the appropriate way to assess the performance. Currently, we are investigating other metrics to use and introducing thresholding.")

st.markdown("The **Research** team collected and organized a very detailed [documentation](https://prism.openai.com/?u=08092597-64a8-4a16-9f11-b8eff820b58d&pg=1&m=main.tex&d=7) on _Anomaly Detection_, _Isolation Forest_ and how it relates to _Random Forests_, _Stream Mining_, _Half-Space Trees_. From this point on, both development team will **refer to this document** to get familiar with the models created, and to understand how to refine and finetune them.")

st.markdown("To demonstrate the pipelines and our vision, a **demo** has been set up at the [Offline Anomaly Detection](/Offline_Anomaly_Detection) and [Online Anomaly Detection](/Online_Anomaly_Detection) pages.")
