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
from src.ml.model_service import load_model_service
from src.services.prediction_service import PredictionService
from src.ui.classify_page import render_classify
from src.ui.history_page import render_history
from src.ui.home_page import render_home
from src.ui.styles import inject_styles

st.set_page_config(page_title="Intelligent Fruit Classification and Information System", layout="wide")


@st.cache_resource(show_spinner="Loading the Classification Model...")
def get_model_service():
    """Load the trained network once per server process, not on every Streamlit rerun."""
    return load_model_service()

inject_styles()

try:
    model_service = get_model_service()
except ModelUnavailableError as error:
    # A technical failure, not a prediction outcome
    st.error("The classification model could not be loaded, so the application cannot start.")
    st.write(str(error))
    st.caption("Check that the files listed in artifacts/README.md are in the artifacts folder.")
    st.stop()

# Built on every run so it always matches the currently loaded code.
service = PredictionService(model_service)

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