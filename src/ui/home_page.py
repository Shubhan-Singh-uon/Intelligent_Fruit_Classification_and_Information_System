"""
File: home_page.py
Project: Intelligent Fruit Classification and Information System
Description:
    Page 1 - Home. Introduces the application, explains how it works, lists the
    supported fruit classes (Apple and Banana) and links to the Classify page.

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
Maintainers: Group 8
"""
import streamlit as st

from src.ui.styles import html

STEPS = (
    ("⬆️", "1. Upload", "Choose a photo of a fruit."),
    ("🧠", "2. Classify", "Our CNN predicts Apple or Banana."),
    ("📚", "3. Learn", "View Nutrition, Benefits and more (from our database)."),
)


def render_home(on_start) -> None:
    """on_start: called when the user presses 'Start Classifying' (navigates to the Classify page)."""
    html('<div class="fa-title">Identify. Learn. Explore.</div>')
    st.write(
        "Upload an image of an apple or banana and let our CNN model identify the fruit and provide "
        "useful information about it."
    )

    if st.button("Start Classifying", type="primary", key="home_start"):
        on_start()

    st.markdown("#### How it works")
    columns = st.columns(len(STEPS))
    for column, (icon, title, text) in zip(columns, STEPS):
        with column:
            html(
                f'<div class="fa-card fa-step"><div class="fa-icon">{icon}</div>'
                f"<b>{title}</b><span>{text}</span></div>"
            )

    st.markdown("#### Supported fruits")
    html('<span class="fa-chip">🍎 Apple</span><span class="fa-chip">🍌 Banana</span>')
    st.caption(
        "The application only tells apples and bananas apart. It does not assess ripeness, quality "
        "or whether a fruit is fresh or spoiled, and it will not name any other fruit. If an image "
        "does not look like an apple or banana, or the result is uncertain, you will be told so."
    )
