"""
File: app.py
Project: Intelligent Fruit Classification and Information System

Description:
    Entry point for the Streamlit application. Coordinates the user
    interface and invokes the prediction service to classify uploaded
    fruit images, retrieve fruit information, and display results.

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
Contributors: Group 8
"""
import streamlit as st

from src.fruit_classifier.exceptions import ModelUnavailableError
from src.services.prediction_service import build_prediction_service
from src.ui.classify_page import render_classify
from src.ui.history_page import render_history
from src.ui.home_page import render_home
from src.ui.styles import inject_styles

st.set_page_config(page_title="Intelligent Fruit Classification and Information System", 
                   page_icon="🍎", layout="wide")


@st.cache_resource(show_spinner="Loading the Classification Model...")
def get_service():
    """Load the model once per server process, not on every Streamlit rerun."""
    return build_prediction_service()


inject_styles()

try:
    service = get_service()
except ModelUnavailableError as error:
    # A technical failure, not a prediction outcome: nothing works without the model, so stop here.
    st.error("The classification model could not be loaded, so the application cannot start.")
    st.write(str(error))
    st.caption("Check that the files listed in artifacts/README.md are in the artifacts folder.")
    st.stop()


def home() -> None:
    render_home(on_start=lambda: st.switch_page(classify_page))


def classify() -> None:
    render_classify(service)


def history() -> None:
    render_history(service)


home_page = st.Page(home, title="Home", icon=":material/home:", url_path="home", default=True)
classify_page = st.Page(classify, title="Classify Fruit", icon=":material/photo_camera:", url_path="classify")
history_page = st.Page(history, title="Prediction History", icon=":material/history:", url_path="history")

st.navigation([home_page, classify_page, history_page]).run()