"""
File: hasnain_app.py
Project: Intelligent Fruit Classification and Information System
Description:
    starts the hasnain version of the app. app.py is not changed

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration

Run: streamlit run hasnain_app.py
"""
import streamlit as st

from hasnain.pages import run_app

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎")
run_app()
