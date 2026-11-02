"""
File: styles.py
Project: Intelligent Fruit Classification and Information System
Description:
    Shared look and feel for the Streamlit pages: the stylesheet and small HTML
    builders (status pills, banners, probability bars, tiles, link lists).
    Every piece of text passed to a builder is HTML-escaped here, so pages can
    pass database content straight in.

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
"""
from html import escape
import streamlit as st

GREEN = "#17483a"

CSS = """
<style>
section[data-testid="stSidebar"] { background: #17483a; }
section[data-testid="stSidebar"] * { color: #e6f2ed; }
[data-testid="stSidebarNav"]::before {
    content: "FruitAI"; display: block; font-size: 1.35rem; font-weight: 700;
    padding: 1.1rem 1rem 0.6rem 1rem; color: #ffffff;
}
.block-container { max-width: 1100px; padding-top: 4.5rem; }

.fa-title { font-size: 1.9rem; font-weight: 700; margin: 0 0 .2rem 0; color: #173f33; }
.fa-sub   { color: #5b6b66; margin: 0 0 1.2rem 0; }

.fa-card { border: 1px solid #d9e6e0; border-radius: 12px; padding: 1rem 1.2rem; background: #ffffff; }
.fa-step { text-align: center; min-height: 9.5rem; }
.fa-step .fa-icon { font-size: 2rem; }
.fa-step b { display: block; margin: .3rem 0 .15rem 0; }
.fa-step span { color: #5b6b66; font-size: .9rem; }

.fa-pill { display: inline-block; padding: .12rem .65rem; border-radius: 999px; font-size: .8rem;
           font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.fa-ok   { background: #e3f4ea; color: #1b7f4b; border-color: #bfe3cd; }
.fa-warn { background: #fff3d6; color: #9a6200; border-color: #f1d9a0; }
.fa-bad  { background: #fde6e6; color: #b3261e; border-color: #f4bcbc; }
.fa-info { background: #e8f1fb; color: #1d5a96; border-color: #c4d9f0; }

.fa-banner { padding: .65rem 1rem; border-radius: 8px; font-size: .95rem; margin: .2rem 0 1rem 0;
             border: 1px solid transparent; }
.fa-banner.fa-ok   { background: #e8f6ee; }
.fa-banner.fa-warn { background: #fff6e0; }
.fa-banner.fa-bad  { background: #fdebeb; }
.fa-banner.fa-info { background: #eef5fc; }

.fa-result-name { font-size: 1.5rem; font-weight: 700; margin: 0 .5rem .2rem 0; display: inline-block; }
.fa-label { color: #5b6b66; font-size: .85rem; margin-top: .6rem; }
.fa-big   { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }

.fa-bar-row { display: flex; align-items: center; gap: .6rem; margin: .3rem 0; font-size: .9rem; }
.fa-bar-row .fa-bar-name { width: 4.5rem; }
.fa-bar-track { flex: 1; height: 9px; background: #e6eeea; border-radius: 6px; overflow: hidden; }
.fa-bar-fill  { height: 100%; background: #bcc9c3; border-radius: 6px; }
.fa-bar-fill.fa-lead { background: #1f9e6f; }
.fa-bar-row .fa-bar-pct { width: 3.6rem; text-align: right; color: #44524d; }

.fa-tile { border: 1px solid #d9e6e0; border-radius: 10px; padding: .7rem 1rem; background: #fff; }
.fa-tile .fa-tile-label { font-size: .85rem; color: #5b6b66; }
.fa-tile .fa-tile-value { font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.fa-tile.fa-ok   .fa-tile-value { color: #1b7f4b; }
.fa-tile.fa-warn .fa-tile-value { color: #9a6200; }
.fa-tile.fa-bad  .fa-tile-value { color: #b3261e; }

.fa-table { width: 100%; border-collapse: collapse; font-size: .92rem; background: #fff;
            border: 1px solid #d9e6e0; border-radius: 10px; overflow: hidden; }
.fa-table th { text-align: left; background: #f1f6f4; color: #44524d; font-weight: 600; padding: .6rem .8rem; }
.fa-table td { padding: .6rem .8rem; border-top: 1px solid #e6eeea; }

.fa-chip { display: inline-block; background: #eef5f1; color: #1d5b46; border-radius: 999px;
           padding: .15rem .7rem; margin: 0 .35rem .35rem 0; font-size: .88rem; }
.fa-list { margin: .2rem 0 .6rem 0; padding-left: 1.2rem; }
.fa-list li { margin-bottom: .35rem; }
.fa-src  { font-size: .8rem; color: #6a7a74; }
.fa-src a, .fa-list a { color: #1d6e52; }
</style>
"""


def esc(value) -> str:
    return escape("" if value is None else str(value), quote=True)


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------- 
# Builders
# -------------------------------------------------------------------------------------
def pill(text, kind="info") -> str:
    return f'<span class="fa-pill fa-{kind}">{esc(text)}</span>'

def banner(text, kind="info") -> str:
    return f'<div class="fa-banner fa-{kind}">{esc(text)}</div>'

def tile(label, value, kind="") -> str:
    return (
        f'<div class="fa-tile fa-{kind}"><div class="fa-tile-label">{esc(label)}</div>'
        f'<div class="fa-tile-value">{esc(value)}</div></div>'
    )

def percent(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"

def probability_bars(apple, banana) -> str:
    """Two horizontal bars; the more probable class is highlighted."""
    rows = []
    for name, value, other in (("Apple", apple, banana), ("Banana", banana, apple)):
        lead = " fa-lead" if value >= other else ""
        rows.append(
            f'<div class="fa-bar-row"><span class="fa-bar-name">{name}</span>'
            f'<div class="fa-bar-track"><div class="fa-bar-fill{lead}" style="width:{value * 100:.1f}%"></div></div>'
            f'<span class="fa-bar-pct">{percent(value)}</span></div>'
        )
    return "".join(rows)


def link(url, text) -> str:
    """An anchor for http(s) URLs only; anything else is shown as plain text."""
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(text)}</a>'
    return esc(text)


def bullet_list(items_html) -> str:
    """items_html: already-escaped HTML fragments."""
    return '<ul class="fa-list">' + "".join(f"<li>{item}</li>" for item in items_html) + "</ul>"


def chips(texts) -> str:
    return "".join(f'<span class="fa-chip">{esc(text)}</span>' for text in texts)
