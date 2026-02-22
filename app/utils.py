import streamlit as st

def team_member(name: str, role: str, color:str="primary") -> None:
    elem = st.container(width="stretch", horizontal=True, horizontal_alignment="left")
    elem.markdown(name, width=175)
    elem.badge(role, color=color)
    return elem
