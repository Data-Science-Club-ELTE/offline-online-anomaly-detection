import streamlit as st

from utils import team_member

st.title(":material/group: Team")

team_member("Amina Tynybekova", "Project Co-Lead", color="violet")
team_member("Matthew Balogh", "Project Co-Lead", color="violet")

st.subheader("Research Team")
team_member("Matthew Balogh", "Research Circle Lead", color="gray")
team_member("Karina Osipova", "Research Lead in Offline Learning", color="primary")
team_member("Saidul Islam Nayan", "Research Lead in Online Learning", color="primary")

st.subheader("Offline ML Development Team")
team_member("Amina Tynybekova", "Lead", color="gray")
team_member("Aibike Builasheva", "Lead Data Scientist", color="primary")
team_member("Gabor Toth", "Data Preparation & Modeling", color="violet")
team_member("Muhammad Sufyan", "Data Preparation & Modeling", color="violet")
team_member("Jair Alessandro Cupi Olivares", "Data Preparation & Modeling", color="violet")
team_member("Nazrin Majidova", "Data Preparation & Modeling", color="violet")

st.subheader("Online ML Development Team")
team_member("Matthew Balogh", "Lead", color="gray")
team_member("Nursultan Tuleev", "Lead Data Scientist", color="primary")
team_member("Christian Arinze Okafor", "Data Preparation & Modeling", color="violet")
team_member("Rafael Ibayev", "Data Preparation & Modeling", color="violet")

st.subheader("Visualization Team")
team_member("Diana Grigoryan*", "Visualization Lead", color="primary")