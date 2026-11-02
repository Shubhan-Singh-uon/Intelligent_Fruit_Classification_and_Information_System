"""
File: history_page.py
Project: Intelligent Fruit Classification and Information System
Description:
    Page 3 - Prediction History. Summary tiles and a paginated table of the
    attempts stored in SQLite database.

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
Maintainers: Group 8
"""
from datetime import datetime, timezone

import streamlit as st

from src.fruit_classifier.exceptions import DatabaseUnavailableError
from src.ui import messages
from src.ui.styles import esc, html, percent, pill, tile

PAGE_SIZE = 10
_PAGE_KEY = "history_page_number"

# Name of the timestamp column in PREDICTION_HISTORY. 
TIMESTAMP_COLUMN = "created_at"


def format_timestamp(value) -> str:
    """SQLite's CURRENT_TIMESTAMP is UTC ('YYYY-MM-DD HH:MM:SS'); show it in machine's local time."""
    try:
        moment = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return moment.astimezone().strftime("%d %b %Y %H:%M")
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def _row_html(record, with_time: bool) -> str:
    label, kind = messages.STATUS_LABEL.get(record["prediction_status"], (record["prediction_status"], "info"))
    detail = messages.REASON_SHORT.get(record.get("rejection_reason"), record.get("rejection_reason") or "")
    cells = []
    if with_time:
        cells.append(esc(format_timestamp(record.get(TIMESTAMP_COLUMN))))
    cells += [
        esc(f'#{record["prediction_id"]}'),
        esc(record.get("predicted_class") or "—"),
        esc(percent(record.get("confidence"))),
        esc(percent(record.get("apple_probability"))),
        esc(percent(record.get("banana_probability"))),
        pill(label, kind),
        esc(detail),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def history_table_html(records) -> str:
    with_time = bool(records) and TIMESTAMP_COLUMN in records[0]
    headers = (["Date &amp; Time"] if with_time else []) + [
        "ID", "Result", "Confidence", "Apple", "Banana", "Status", "Details",
    ]
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(_row_html(record, with_time) for record in records)
    return f'<table class="fa-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render_history(service) -> None:
    """service: a PredictionService (or any object with get_history() and summarise_history())."""
    html('<div class="fa-title">Prediction History</div>')
    html('<div class="fa-sub">View your past classification attempts and results.</div>')

    try:
        records = service.get_history()
    except DatabaseUnavailableError:
        st.error(
            "The prediction history could not be read from the database. "
            "If this is a new installation, create it first with: python src/scripts/init_db.py"
        )
        return

    summary = service.summarise_history(records)
    columns = st.columns(4)
    for column, (label, key, kind) in zip(
        columns,
        (
            ("Total Attempts", "total", ""),
            ("Accepted", "accepted", "ok"),
            ("Needs Review", "needs_review", "warn"),
            ("Rejected / Invalid", "rejected_or_invalid", "bad"),
        ),
    ):
        with column:
            html(tile(label, summary[key], kind))

    if not records:
        st.info("No Predictions Yet. Classify an image and it will appear here.")
        return

    pages = max(1, -(-len(records) // PAGE_SIZE))   # ceiling division
    current = min(max(st.session_state.get(_PAGE_KEY, 1), 1), pages)
    st.session_state[_PAGE_KEY] = current

    start = (current - 1) * PAGE_SIZE
    st.write("")
    html(history_table_html(records[start : start + PAGE_SIZE]))

    if pages > 1:
        previous_col, label_col, next_col = st.columns([1, 2, 1])
        with previous_col:
            if st.button("← Previous", disabled=current <= 1, key="history_prev"):
                st.session_state[_PAGE_KEY] = current - 1
                st.rerun()
        with label_col:
            st.caption(f"Page {current} of {pages}  ({len(records)} attempts)")
        with next_col:
            if st.button("Next →", disabled=current >= pages, key="history_next"):
                st.session_state[_PAGE_KEY] = current + 1
                st.rerun()
