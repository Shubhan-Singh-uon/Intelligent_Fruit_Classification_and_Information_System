"""
File: pages.py
Project: Intelligent Fruit Classification and Information System
Description:
    the three pages of the streamlit app. home page and classify
    fruit page and prediction history page

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
"""
import streamlit as st

from src.fruit_classifier.exceptions import FruitClassifierError, DatabaseUnavailableError
from src.fruit_classifier.schemas import PredictionStatus
from hasnain.prediction_service import PredictionService, get_history, get_summary, HISTORY_NOT_SAVED, FRUIT_INFO_MISSING

REASON_TEXT = {
    "low_confidence": "the model is not sure if it is apple or banana",
    "inconsistent_under_transforms": "the answer change when the photo is turn or flip a little",
    "far_from_training_data": "this image dont look like the apple and banana photos the model was train on",
    "empty_file": "the file is empty",
    "file_too_large": "the file is too big and the max size is 5 MB",
    "unreadable_file": "the file can not be open as a image",
    "unsupported_format": "only JPG and PNG images is allowed",
}


@st.cache_resource
def get_service():
    # model is big so it is load one time only
    return PredictionService()


def percent(value):
    if value == None:
        return "-"
    return str(round(value * 100, 1)) + "%"


def home_page():
    st.title("Intelligent Fruit Classification and Information System")
    st.write("This app tell you if your photo is a apple or a banana and then show you some information about that fruit.")
    st.subheader("How to use")
    st.write("1. Go to the Classify Fruit page from the menu on left side")
    st.write("2. Upload a JPG or PNG photo of one apple or one banana")
    st.write("3. Press the Classify button and wait for the result")
    st.write("4. Open the Prediction History page to see all old results")
    st.info("Only apple and banana is supported. Other things will be rejected by the system.")
    st.button("Start Classify", type="primary", on_click=go_to_classify)


def go_to_classify():
    st.session_state["page"] = "Classify Fruit"


def classify_page():
    st.title("Classify Fruit")
    uploaded = st.file_uploader("Upload a apple or banana photo", type=["jpg", "jpeg", "png"])
    if uploaded == None:
        st.write("Please upload a image to start.")
        return

    try:
        st.image(uploaded, width=300)
    except Exception:
        st.write("This file can not be show as a image.")

    if st.button("Classify", type="primary"):
        try:
            service = get_service()
            with st.spinner("Checking the image..."):
                response = service.classify(uploaded)
        except FruitClassifierError as error:
            # model missing or model crash is a technical error not a result
            st.error("Technical error: the model could not run right now. Please press Classify again or try later.")
            st.caption(str(error))
            return
        show_result(response, uploaded)


def show_result(response, uploaded):
    result = response.prediction
    reason = REASON_TEXT.get(result.reason_code, result.reason_code)

    if result.status == PredictionStatus.INVALID_INPUT:
        st.error("Invalid image: " + reason)
    else:
        if result.status == PredictionStatus.ACCEPTED:
            st.success("This is a " + result.predicted_class + "!")
        elif result.status == PredictionStatus.NEEDS_REVIEW:
            st.warning("Needs review: it may be a " + result.predicted_class + " but " + reason + ".")
            st.write("Try another image with only one fruit and good light.")
        else:
            st.error("Rejected: " + reason + ".")
        st.write("Model scores")
        st.write("Apple: " + percent(result.apple_probability))
        st.write("Banana: " + percent(result.banana_probability))
        st.write("Confidence: " + percent(result.confidence))

    if response.history_saved:
        st.caption("This result is saved in the prediction history.")
    if HISTORY_NOT_SAVED in response.issues:
        st.warning("The result could not be save in the history.")
    if FRUIT_INFO_MISSING in response.issues:
        st.warning("The fruit information is not available right now.")

    if response.fruit_info != None:
        show_fruit_info(response.fruit_info)


def show_fruit_info(info):
    st.subheader("About " + info.fruit["fruit_name"])
    st.write(info.fruit["description"])
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Nutrition", "Health Benefits", "Uses", "Fun Facts", "Sources"])

    with tab1:
        food = info.nutrition
        if food == None:
            st.write("No nutrition information.")
        else:
            st.write("Amount per " + str(food["serving_basis"]))
            st.table([
                {"Nutrient": "Energy (kcal)", "Amount": food["energy_kcal"]},
                {"Nutrient": "Protein (g)", "Amount": food["protein_g"]},
                {"Nutrient": "Carbohydrate (g)", "Amount": food["carbohydrate_g"]},
                {"Nutrient": "Sugars (g)", "Amount": food["sugars_g"]},
                {"Nutrient": "Fibre (g)", "Amount": food["fibre_g"]},
                {"Nutrient": "Fat (g)", "Amount": food["fat_g"]},
                {"Nutrient": "Potassium (mg)", "Amount": food["potassium_mg"]},
                {"Nutrient": "Vitamin C (mg)", "Amount": food["vitamin_c_mg"]},
            ])

    with tab2:
        for item in info.health_benefits:
            st.write("- " + item["benefit_description"])

    with tab3:
        for item in info.uses:
            st.write("- " + item["use_description"])

    with tab4:
        for item in info.fun_facts:
            st.write("- " + item["fact_text"])

    with tab5:
        for item in info.sources:
            st.write("- [" + item["source_name"] + "](" + item["source_url"] + ") accessed " + item["access_date"])


def history_page():
    st.title("Prediction History")
    try:
        history = get_history()
    except DatabaseUnavailableError:
        st.error("The history can not be load. Please run python src/scripts/init_db.py first.")
        return

    if len(history) == 0:
        st.info("No prediction were made yet.")
        return

    show_summary(get_summary(history))

    rows = []
    for item in history:
        rows.append({
            "ID": item["prediction_id"],
            "Date": item.get("created_at") or "-",
            "Status": item["prediction_status"],
            "Fruit": item["predicted_class"] or "-",
            "Apple": percent(item["apple_probability"]),
            "Banana": percent(item["banana_probability"]),
            "Confidence": percent(item["confidence"]),
            "Reason": item["rejection_reason"] or "-",
            "Model": item["model_version"],
        })
    st.dataframe(rows, hide_index=True)


def show_summary(summary):
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", summary["total"])
    col2.metric("Accepted", summary[PredictionStatus.ACCEPTED.value])
    col3.metric("Needs Review", summary[PredictionStatus.NEEDS_REVIEW.value])
    col4, col5, col6 = st.columns(3)
    col4.metric("Rejected", summary[PredictionStatus.REJECTED.value])
    col5.metric("Invalid Input", summary[PredictionStatus.INVALID_INPUT.value])
    col6.metric("Average Confidence", percent(summary["average_confidence"]))
    st.write("Apples accepted: " + str(summary["apples"]) + " and Bananas accepted: " + str(summary["bananas"]))
    st.caption("Average confidence is only count for the result that the model was run on.")
    st.subheader("All Predictions")


def run_app():
    st.sidebar.title("Fruit Classifier")
    page = st.sidebar.radio("Menu", ["Home", "Classify Fruit", "Prediction History"], key="page")
    if page == "Home":
        home_page()
    elif page == "Classify Fruit":
        classify_page()
    else:
        history_page()
