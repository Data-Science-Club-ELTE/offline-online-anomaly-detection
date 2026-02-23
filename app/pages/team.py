import streamlit as st

from utils import team_member

st.title(":material/group: Team")

TRADITIONAL_ML_TEAM = {"role": "Traditional ML Team", "color": "orange"}
STREAM_ML_TEAM = {"role": "Stream ML Team", "color": "blue"}
ML_MODELING = {"role": "Data Preparation & Modeling", "color": "yellow"}

st.subheader("Project Leaders")
team_member("Amina Tynybekova", [TRADITIONAL_ML_TEAM])
team_member("Matthew Balogh", [STREAM_ML_TEAM])

st.subheader("Research Team")
team_member("Matthew Balogh", [{"role": "Research Circle Lead", "color": "gray"}])
team_member("Karina Osipova", [{"role": "Research Lead", "color": "primary"}, TRADITIONAL_ML_TEAM])
team_member("Saidul Islam Nayan", [{"role": "Research Lead", "color": "primary"}, STREAM_ML_TEAM])

st.subheader("Traditional (Offline) ML Development Team")
team_member("Aibike Builasheva", [{"role": "Lead Data Scientist", "color": "primary"}, TRADITIONAL_ML_TEAM])
team_member("Gabor Toth", [ML_MODELING, TRADITIONAL_ML_TEAM])
team_member("Muhammad Sufyan", [ML_MODELING, TRADITIONAL_ML_TEAM])
team_member("Jair Alessandro Cupi Olivares", [ML_MODELING, TRADITIONAL_ML_TEAM])
team_member("Nazrin Majidova", [ML_MODELING, TRADITIONAL_ML_TEAM])

st.subheader("Stream (Online) ML Development Team")
team_member("Nursultan Tuleev", [{"role": "Lead Data Scientist", "color": "primary"}, STREAM_ML_TEAM])
team_member("Christian Arinze Okafor", [ML_MODELING, STREAM_ML_TEAM])
team_member("Rafael Ibayev", [ML_MODELING, STREAM_ML_TEAM])

st.subheader("Visualization Team")
st.caption("To be determined...")