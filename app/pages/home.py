import streamlit as st

from branding import render_home_header

render_home_header()

st.subheader("About the Project")
st.write("Fraudulent transactions related to credit cards either result in financial losses for companies or unjust charges to customers. Detecting these transactions is essential to minimize those outcomes. In this work, our goal is to detect fraudulent transactions in credit card data, in both *offline* and *online* setting, in which the data is given as a large batch (offline), or when it arrives incrementally (online). To do so, we aim to utilize the *Isolation Forest* algorithm and its stream mining counterpart *Half-Space Trees*.")

st.subheader("Challenges")
st.markdown("""
            - Rare labels and class-imbalance
            - Evaluation bias
            - Anomaly detection in Stream Data
            - Overlooked anomalies vs False alarms
            - High-dimensional data
""")

st.subheader("Expectations")
st.markdown("""
            Build an **offline** and an **online** version of the anomaly detection pipeline using the *Isolation Forest* algorithm
            
            We expect that the comparison of the models built in [Scikit-learn](https://scikit-learn.org/stable/index.html) and [River ML](https://riverml.xyz/latest/), respectively, will result in useful insights we, Data Scientists, can leverage in future (anomaly detection) works. After the comparison, a hybrid usage of the pipelines will be demonstrated by building a **prototype** application.
""")
