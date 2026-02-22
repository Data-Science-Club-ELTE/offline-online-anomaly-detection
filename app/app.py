import streamlit as st

st.set_page_config(
    page_title="Offline & Online Anomaly Detection",
    page_icon=":material/detector:",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("pages/home.py", title="About the Project", icon=":material/rocket:"),
        st.Page("pages/research.py", title="Research", icon=":material/genetics:"),
        st.Page("pages/offline.py", title="Offline Anomaly Detection", icon=":material/potted_plant:"),
        st.Page("pages/online.py", title="Online Anomaly Detection", icon=":material/earthquake:"),
        st.Page("pages/team.py", title="Team", icon=":material/group:"),
    ]
)

pages.run()
