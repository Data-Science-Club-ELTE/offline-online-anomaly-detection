import streamlit as st
from branding import render_logo

st.set_page_config(
    page_title="Project Roadmap",
    page_icon=":material/detector:",
    layout="wide",
)

render_logo()

st.title(":material/route: Project Roadmap")

st.subheader("Research Team")

ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
ch.caption("WEEK 1 - WEEK 3", width="content")
ch.badge("Work in Progress", color="orange")
st.write("Research Background and Algorithm (both Traditional and Stream scenarios)")

ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
ch.caption("WEEK 4", width="content")
st.write("Organize Research Materials")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("Traditional (Offline) Model Development Timeline")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 1", width="content")
    ch.badge("Done", color="green")
    st.write("Setup & Data Loading")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 2", width="content")
    ch.badge("Work in Progress", color="orange")
    st.write("Data Cleaning & EDA & Data Preprocessing")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 3", width="content")
    st.write("Modeling, Predicting, Explaining & Evaluating Performance")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 4", width="content")
    st.write("Model Tuning & Create Pipeline")
    st.markdown("_Use research materials to better understand and refine the model_")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 5", width="content")
    st.write("Hybrid Pipeline integration in Streamlit app")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 6", width="content")
    st.write("Assess Results and Limitations & Prepare for Presentation")


with c2:
    st.subheader("Stream (Online) Model Development Timeline")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 1", width="content")
    ch.badge("Done", color="green")
    st.write("Setup & Data Loading & Intro to RiverML")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 2", width="content")
    ch.badge("Work in Progress", color="orange")
    st.write("Creating first Pipeline & Settle on Preprocessing and Evaluation Components")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 3", width="content")
    st.write("Finalize Pipeline & Modeling, Predicting, Explaining & Evaluating Performance")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 4", width="content")
    st.write("Model Tuning & Live Performance Evaluation")
    st.markdown("_Use research materials to better understand and refine the model_")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 5", width="content")
    st.write("Hybrid Pipeline integration in Streamlit app")

    ch = st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center")
    ch.caption("WEEK 6", width="content")
    st.write("Assess Results and Limitations & Prepare for Presentation")

