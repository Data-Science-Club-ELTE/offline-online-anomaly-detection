import streamlit as st

def team_member(name: str, badges:list[dict]=[]) -> None:
    elem = st.container(width="stretch", horizontal=True, horizontal_alignment="left")
    elem.markdown(name, width=200)
    
    for badge in badges:
        elem.badge(badge["role"], color=badge["color"])
    return elem
