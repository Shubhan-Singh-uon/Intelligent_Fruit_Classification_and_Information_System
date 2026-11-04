"""
File: classify_page.py
Project: Intelligent Fruit Classification and Information System
Description:
    Page 2 - Classify Fruit. Image upload and preview, the classification
    button, and the result for each prediction outcome:
        A accepted      fruit, both probabilities, confidence, fruit information
        B needs review  tentative fruit, probabilities, warning, no fruit information
        C rejected      no fruit confirmed, no fruit information
        D invalid       error and retry, no probabilities
    
Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
Maintainers: Group 8
"""
import io
import logging

import streamlit as st
from PIL import Image, ImageOps

from src.fruit_classifier.config import DISTANCE_REJECT_THRESHOLD, SUPPORTED_FORMATS_LABEL, UPLOAD_EXTENSIONS
from src.fruit_classifier.schemas import ClassificationResponse, FruitInformation, PredictionStatus
from src.ui import messages
from src.ui.styles import banner, bullet_list, chips, esc, html, link, percent, pill, probability_bars

logger = logging.getLogger(__name__)

NUTRITION_ROWS = (
    ("Energy", "energy_kcal", "kcal"),
    ("Protein", "protein_g", "g"),
    ("Carbohydrate", "carbohydrate_g", "g"),
    ("Sugars", "sugars_g", "g"),
    ("Fibre", "fibre_g", "g"),
    ("Fat", "fat_g", "g"),
    ("Potassium", "potassium_mg", "mg"),
    ("Vitamin C", "vitamin_c_mg", "mg"),
)

# Session-state keys
_UPLOADER_KEY = "classify_uploader_key"   # Gives the user a fresh, empty state
_RESPONSE = "classify_response"
_RESPONSE_FOR = "classify_response_for"

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def _upload_id(uploaded):
    """A value that changes when the user picks a different file."""
    if uploaded is None:
        return None
    return getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)


def _preview_image(data: bytes):
    """A display copy of the upload, or None if it cannot be decoded."""
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        image.thumbnail((640, 640))
        return image
    except Exception:   # any decoding problem - no preview
        return None


def _reset() -> None:
    st.session_state[_UPLOADER_KEY] = st.session_state.get(_UPLOADER_KEY, 0) + 1
    st.session_state.pop(_RESPONSE, None)
    st.session_state.pop(_RESPONSE_FOR, None)

# -------------------------------------------------------------------------------------
# Page
# -------------------------------------------------------------------------------------
def render_classify(service) -> None:
    """service: a PredictionService (or any object with the same classify() method)."""
    html('<div class="fa-title">Classify a Fruit</div>')
    html('<div class="fa-sub">Upload an image of an apple or banana and click classify.</div>')

    uploaded = st.file_uploader(
        "Upload an image of an apple or banana",
        type=list(UPLOAD_EXTENSIONS),
        key=f"uploader_{st.session_state.get(_UPLOADER_KEY, 0)}",
        label_visibility="collapsed",
    )
    st.caption(f"Supported formats: {SUPPORTED_FORMATS_LABEL}  |  Max file size: {messages.MAX_MB} MB")

    upload_id = _upload_id(uploaded)
    if st.session_state.get(_RESPONSE_FOR) != upload_id:
        st.session_state.pop(_RESPONSE, None)   # a different or removed file = drop the old result
        st.session_state.pop(_RESPONSE_FOR, None)

    preview = None
    if uploaded is not None:
        preview = _preview_image(uploaded.getvalue())
        if preview is not None:
            st.image(preview, width=300)
        else:
            st.caption("Preview Unavailable: This file could not be displayed as an image.")

    if st.button("Classify Image", type="primary", disabled=uploaded is None, key="classify_button"):
        try:
            response = service.classify(uploaded)
        except Exception:   # InferenceError or an unexpected bug
            logger.exception("Classification failed")
            st.error(messages.INFERENCE_ERROR)
        else:
            st.session_state[_RESPONSE] = response
            st.session_state[_RESPONSE_FOR] = upload_id

    response = st.session_state.get(_RESPONSE) if uploaded is not None else None
    if response is not None:
        st.divider()
        _render_response(response)
        st.button("Try Another Image", on_click=_reset, key="try_another")

# -------------------------------------------------------------------------------------
# Outcomes
# -------------------------------------------------------------------------------------
def _render_response(response: ClassificationResponse) -> None:
    # Dispatch on the status VALUE, Streamlit re-imports every project module whenever a 
    # source file changes, which creates a second copy of PredictionStatus
    status = getattr(response.prediction.status, "value", response.prediction.status)
    renderer = _RENDERERS.get(status)
    if renderer is None:
        # Never fall through to "invalid" - would silently mislabel a good result
        logger.error("Unrecognised Prediction Status %r", status)
        st.error(f"The result could not be displayed (Unrecognised Status: {status!r}).")
        return
    renderer(response)


def _render_issues(response: ClassificationResponse) -> None:
    for code in response.issues:
        st.warning(messages.ISSUE_MESSAGES.get(code, "A system problem occurred."))


def _render_accepted(response: ClassificationResponse) -> None:
    prediction = response.prediction
    html(banner("Prediction Completed Successfully!", "OK"))
    with st.container(border=True):
        html(f'<span class="fa-result-name">{esc(prediction.predicted_class)}</span>{pill("Accepted", "Ok")}')
        html(f'<div class="fa-label">Confidence</div><div class="fa-big">{percent(prediction.confidence)}</div>')
        html('<div class="fa-label">Class Probabilities</div>')
        html(probability_bars(prediction.apple_probability, prediction.banana_probability))
    _render_issues(response)
    if response.fruit_info is not None:
        _render_fruit_information(response.fruit_info)


def _render_needs_review(response: ClassificationResponse) -> None:
    prediction = response.prediction
    html(banner(messages.NEEDS_REVIEW_BANNER, "warn"))
    with st.container(border=True):
        html('<div class="fa-label">Possible Match</div>')
        html(f'<span class="fa-result-name">{esc(prediction.predicted_class)}</span>{pill("Needs Review", "warn")}')
        html(f'<div class="fa-label">Confidence</div><div class="fa-big">{percent(prediction.confidence)}</div>')
        html('<div class="fa-label">Class Probabilities</div>')
        html(probability_bars(prediction.apple_probability, prediction.banana_probability))
        html(
            banner(
                messages.NEEDS_REVIEW_DETAIL.get(prediction.reason_code, messages.NEEDS_REVIEW_DETAIL["low_confidence"]),
                "warn",
            )
        )
    _render_issues(response)   # no fruit information for this outcome, by design


def _render_rejected(response: ClassificationResponse) -> None:
    prediction = response.prediction
    html(banner(messages.REJECTED_BANNER, "bad"))
    with st.container(border=True):
        html(f'<span class="fa-result-name">Unsupported Image</span>{pill("Rejected", "bad")}')
        st.write(messages.REJECTED_DETAIL)
        # Shown for reference only. The app does not confirm a fruit for a rejected image, so the
        # class probabilities stay out of the main view.
        with st.expander("Technical Details (for Reference)"):
            html(probability_bars(prediction.apple_probability, prediction.banana_probability))
            st.caption(
                f"Distance from known apple/banana images: {prediction.feature_distance:.1f} "
                f"(limit {DISTANCE_REJECT_THRESHOLD:g}). These probabilities are not a result."
            )
    _render_issues(response)


def _render_invalid(response: ClassificationResponse) -> None:
    prediction = response.prediction
    html(banner(messages.INVALID_BANNER, "bad"))
    with st.container(border=True):
        html(f'<span class="fa-result-name">Invalid Image</span>{pill("Invalid Input", "bad")}')
        st.write(messages.INVALID_DETAIL.get(prediction.reason_code, messages.INVALID_FALLBACK))
    _render_issues(response)   # no probabilities - the model never ran


# -------------------------------------------------------------------------------------
# Fruit Information Display
# -------------------------------------------------------------------------------------
def _source_note(row) -> str:
    return f' <span class="fa-src">({link(row.get("source_url"), "source")})</span>'


def _render_fruit_information(info: FruitInformation) -> None:
    st.markdown("#### Fruit Information")
    if info.fruit.get("description"):
        st.write(info.fruit["description"])

    basis = info.nutrition["serving_basis"] if info.nutrition else "serving"
    with st.expander(f"Nutrition (per {basis})", expanded=True):
        if info.nutrition is None:
            st.write("No Nutrition Information is Available.")
        else:
            rows = "".join(
                f"<tr><td>{esc(label)}</td><td>{info.nutrition[column]:g} {unit}</td></tr>"
                for label, column, unit in NUTRITION_ROWS
            )
            html(f'<table class="fa-table"><thead><tr><th>Nutrient</th><th>Amount</th></tr></thead><tbody>{rows}</tbody></table>')
            html(f'<div class="fa-src">Source: {link(info.nutrition.get("source_url"), info.nutrition.get("source_name"))}</div>')

    with st.expander("Health Benefits"):
        if info.health_benefits:
            html(bullet_list(esc(row["benefit_description"]) + _source_note(row) for row in info.health_benefits))
            st.caption("General Information from Listed Sources. It is not Medical Advice.")
        else:
            st.write("No Health Benefit Information is Available.")

    with st.expander("Uses"):
        if info.uses:
            html(chips(row["use_description"] for row in info.uses))
        else:
            st.write("No Use Information is Available.")

    with st.expander("Fun Facts"):
        if info.fun_facts:
            html(bullet_list(esc(row["fact_text"]) + _source_note(row) for row in info.fun_facts))
        else:
            st.write("No Fun Facts are Available.")

    with st.expander("Sources"):
        if info.sources:
            html(
                bullet_list(
                    link(row["source_url"], row["source_name"])
                    + f' <span class="fa-src">accessed {esc(row["access_date"])}</span>'
                    for row in info.sources
                )
            )
        else:
            st.write("No Sources are Listed.")

_RENDERERS = {
    PredictionStatus.ACCEPTED.value: _render_accepted,
    PredictionStatus.NEEDS_REVIEW.value: _render_needs_review,
    PredictionStatus.REJECTED.value: _render_rejected,
    PredictionStatus.INVALID_INPUT.value: _render_invalid,
}