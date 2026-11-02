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
import base64
from pathlib import Path

from src.ui.styles import html

# Helper function to convert local images to Base64 with correct MIME type
def get_image_data_uri(img_path):
    try:
        with open(img_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            
        # Determine the MIME type based on the file extension
        suffix = img_path.suffix.lower()
        if suffix == ".svg":
            mime = "image/svg+xml"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        else:
            mime = "image/png" # Fallback
            
        return f"data:{mime};base64,{encoded}"
    except FileNotFoundError:
        return ""

# Because home_page.py is inside src/ui/, parents[1] points to the 'src' folder
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

BANNER_PATH = ASSETS_DIR / "banner.png"

# Update these file names to match exactly what you saved in src/assets/
STEPS = (
    (ASSETS_DIR / "icon_upload.svg", "1. Upload", "Choose a photo of a fruit."),
    (ASSETS_DIR / "icon_brain.svg", "2. Classify", "Our CNN predicts Apple or Banana."),
    (ASSETS_DIR / "icon_book.svg", "3. Learn", "View Nutrition, Benefits and more (from our database)."),
)


def render_home(on_start) -> None:
    """on_start: called when the user presses 'Start Classifying' (navigates to the Classify page)."""
    
    # Display the Banner Image
    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), width="stretch")
    else:
        st.warning(f"Banner image not found at {BANNER_PATH}")

    # Heading and Button
    html('<div class="fa-title">Identify. Learn. Explore.</div>')
    st.write(
        "Upload an image of an apple or banana and let our CNN model identify the fruit and provide useful information about it."
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        if st.button("Start Classifying", type="primary", width="stretch", key="home_start"):
            on_start()

    # The Steps 
    st.markdown("#### How it works")
    columns = st.columns(len(STEPS))
    
    for column, (icon_path, title, text) in zip(columns, STEPS):
        with column:
            # Get the data URI
            img_data_uri = get_image_data_uri(icon_path)
            
            # If the image exists, use an <img> tag
            if img_data_uri:
                icon_html = f'<img src="{img_data_uri}" style="width: 60px; height: 60px; object-fit: contain; margin-bottom: 10px;">'
            else:
                # Fallback if the SVG is missing
                icon_html = '<div style="font-size: 40px; margin-bottom: 10px;">❓</div>'

            html(
                f'<div class="fa-card fa-step"><div class="fa-icon">{icon_html}</div>'
                f"<b>{title}</b><span>{text}</span></div>"
            )

    st.markdown("#### Supported fruits")
    html('<span class="fa-chip">🍎 Apple</span><span class="fa-chip">🍌 Banana</span>')
    st.caption(
        "The application only tells apples and bananas apart. It does not assess ripeness, quality "
        "or whether a fruit is fresh or spoiled, and it will not name any other fruit. If an image "
        "does not look like an apple or banana, or the result is uncertain, you will be told so."
    )