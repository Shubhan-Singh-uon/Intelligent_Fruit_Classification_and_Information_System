"""Headless tests of the Streamlit pages using Streamlit's AppTest (no browser, no model, no database).

The Classify page is exercised through a fake service that returns a preset response, so these tests
check what the user is SHOWN for each outcome: which messages, which sections, and (for needs review,
rejected and invalid) that fruit information and probabilities stay hidden.
"""
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_fakes as fakes  # noqa: E402
from src.fruit_classifier.exceptions import DatabaseUnavailableError, InferenceError  # noqa: E402
from src.ui import messages  # noqa: E402
from src.ui.history_page import format_timestamp  # noqa: E402

TIMEOUT = 20

# -------------------------------------------------------------------------------------
# Scripts
# -------------------------------------------------------------------------------------
def classify_script():
    import streamlit as st
    import ui_fakes
    from src.ui.classify_page import render_classify

    st.file_uploader = ui_fakes.fake_file_uploader
    render_classify(ui_fakes.service)


def home_script():
    import ui_fakes
    from src.ui.home_page import render_home

    render_home(on_start=ui_fakes.note_start)


def history_script():
    import ui_fakes
    from src.ui.history_page import render_history

    render_history(ui_fakes.history_service)


@pytest.fixture(autouse=True)
def clean_state():
    original_uploader = st.file_uploader
    fakes.service = fakes.FakeClassifyService()
    fakes.current_upload = None
    fakes.started = []
    fakes.note_start = lambda: fakes.started.append(True)
    yield
    st.file_uploader = original_uploader


def run_classify(upload=None, response=None, error=None):
    fakes.current_upload = upload
    fakes.service.response, fakes.service.error = response, error
    return AppTest.from_function(classify_script, default_timeout=TIMEOUT).run()


def click_classify(at):
    at.button(key="classify_button").click().run()
    return at


def page_text(at):
    """All visible text: markdown (including our HTML), alerts, captions and expander labels."""
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [c.value for c in at.caption] + [x.label for x in at.expander]
    return "\n".join(parts)


def expander_labels(at):
    return [x.label for x in at.expander]


# -------------------------------------------------------------------------------------
# Home Page
# -------------------------------------------------------------------------------------
def test_home_explains_the_app_and_states_the_scope():
    at = AppTest.from_function(home_script, default_timeout=TIMEOUT).run()
    text = page_text(at)
    assert not at.exception
    assert "Apple" in text and "Banana" in text
    assert "1. Upload" in text and "2. Classify" in text and "3. Learn" in text
    assert "does not assess ripeness" in text


def test_home_start_button_navigates():
    at = AppTest.from_function(home_script, default_timeout=TIMEOUT).run()
    at.button(key="home_start").click().run()
    assert fakes.started == [True]


# -------------------------------------------------------------------------------------- 
# Classify: Idle
# -------------------------------------------------------------------------------------
def test_classify_button_is_disabled_until_a_file_is_chosen():
    at = run_classify(upload=None)
    assert not at.exception
    assert at.button(key="classify_button").disabled
    assert fakes.service.calls == 0


# -------------------------------------------------------------------------------------
# Classify: Outcome A
# -------------------------------------------------------------------------------------
def test_accepted_shows_result_probabilities_and_all_five_information_sections():
    at = run_classify(fakes.FakeUpload(), fakes.ACCEPTED())
    click_classify(at)
    text = page_text(at)

    assert not at.exception and fakes.service.calls == 1
    assert "Prediction Completed Successfully!" in text
    assert "Apple" in text and "97.0%" in text and "3.0%" in text   # confidence and both probabilities
    assert "Fruit Information" in text and "An apple is a fruit." in text
    assert expander_labels(at) == ["Nutrition (per 100g)", "Health Benefits", "Uses", "Fun Facts", "Sources"]
    assert "57.8 kcal" in text and "115 mg" in text
    assert "ripe" not in text.lower()                                  # ripeness is out of scope


def test_result_survives_reruns_without_classifying_or_saving_again():
    at = run_classify(fakes.FakeUpload(), fakes.ACCEPTED())
    click_classify(at)
    at.run()
    at.run()
    assert fakes.service.calls == 1                                    # one click = one history row
    assert "Prediction Completed Successfully!" in page_text(at)


def test_database_text_is_html_escaped():
    info = fakes.fruit_info()           # its source name contains <b>One</b>
    at = run_classify(fakes.FakeUpload(), fakes.ACCEPTED(fruit_info=info))
    click_classify(at)
    text = page_text(at)
    assert "Source &lt;b&gt;One&lt;/b&gt;" in text
    assert "Source <b>One</b>" not in text


def test_links_are_only_made_for_http_urls():
    bad = {**fakes.SRC, "source_url": "javascript:alert(1)"}
    info = fakes.fruit_info(health_benefits=({"benefit_description": "x", **bad},))
    at = run_classify(fakes.FakeUpload(), fakes.ACCEPTED(fruit_info=info))
    click_classify(at)
    assert "javascript:" not in page_text(at)


# -------------------------------------------------------------------------------------
# Classify: Outcome B
# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("reason", ["low_confidence", "inconsistent_under_transforms"])
def test_needs_review_warns_and_shows_no_fruit_information(reason):
    at = run_classify(fakes.FakeUpload(), fakes.NEEDS_REVIEW(reason=reason))
    click_classify(at)
    text = page_text(at)

    assert messages.NEEDS_REVIEW_BANNER in text
    assert messages.NEEDS_REVIEW_DETAIL[reason] in text
    assert "Possible Match" in text and "Banana" in text and "52.0%" in text and "48.0%" in text
    assert "Fruit Information" not in text
    assert not any("Nutrition" in label for label in expander_labels(at))
    assert at.button(key="try_another")


# -------------------------------------------------------------------------------------
# Classify: Outcome C
# -------------------------------------------------------------------------------------
def test_rejected_confirms_no_fruit_and_keeps_probabilities_out_of_the_main_view():
    at = run_classify(fakes.FakeUpload(), fakes.REJECTED())
    click_classify(at)
    text = page_text(at)

    assert messages.REJECTED_BANNER in text and messages.REJECTED_DETAIL in text
    assert "Unsupported Image" in text and "Rejected" in text
    assert "Fruit Information" not in text and "Confidence" not in text
    assert expander_labels(at) == ["Technical Details (for reference)"]


# -------------------------------------------------------------------------------------
# Classify: Outcome D
# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("reason", sorted(messages.INVALID_DETAIL))
def test_invalid_input_has_a_specific_message_and_no_probabilities(reason):
    at = run_classify(fakes.FakeUpload(), fakes.INVALID(reason))
    click_classify(at)
    text = page_text(at)

    assert messages.INVALID_BANNER in text and messages.INVALID_DETAIL[reason] in text
    assert "Invalid Image" in text
    assert "Class Probabilities" not in text and "Fruit Information" not in text


def test_unreadable_and_unsupported_files_get_different_messages():
    assert messages.INVALID_DETAIL["unreadable_file"] != messages.INVALID_DETAIL["unsupported_format"]
    assert messages.REJECTED_DETAIL != messages.INVALID_DETAIL["unreadable_file"]


def test_a_file_that_cannot_be_previewed_can_still_be_classified():
    at = run_classify(fakes.FakeUpload(data=b"garbage" * 30, name="broken.jpg"), fakes.INVALID("unreadable_file"))
    assert "Preview Unavailable" in page_text(at)
    assert not at.button(key="classify_button").disabled
    click_classify(at)
    assert messages.INVALID_DETAIL["unreadable_file"] in page_text(at)


# ------------------------------------------------------------------------------------- 
# Technical Issues
# -------------------------------------------------------------------------------------
def test_technical_issues_appear_as_warnings_next_to_a_valid_result():
    response = fakes.ACCEPTED(history_saved=False, issues=["history_not_saved"], fruit_info=None)
    at = run_classify(fakes.FakeUpload(), response)
    click_classify(at)
    text = page_text(at)
    assert "Prediction Completed Successfully!" in text
    assert messages.ISSUE_MESSAGES["history_not_saved"] in [w.value for w in at.warning]
    assert "Fruit Information" not in text


def test_model_failure_is_shown_as_a_system_error_not_an_outcome():
    at = run_classify(fakes.FakeUpload(), error=InferenceError("boom"))
    click_classify(at)
    assert messages.INFERENCE_ERROR in [e.value for e in at.error]
    assert not any(label in page_text(at) for label in ("Rejected", "Needs Review", "Invalid Input"))
    assert not at.exception


# -------------------------------------------------------------------------------------
# State Handling
# -------------------------------------------------------------------------------------
def test_choosing_a_different_file_clears_the_previous_result():
    at = run_classify(fakes.FakeUpload(file_id="a"), fakes.ACCEPTED())
    click_classify(at)
    assert "Prediction Completed Successfully!" in page_text(at)
    fakes.current_upload = fakes.FakeUpload(file_id="b")
    at.run()
    assert "Prediction Completed Successfully!" not in page_text(at)


def test_try_another_image_resets_the_page():
    at = run_classify(fakes.FakeUpload(), fakes.NEEDS_REVIEW())
    click_classify(at)
    fakes.current_upload = None            # the fresh uploader is empty
    at.button(key="try_another").click().run()
    assert messages.NEEDS_REVIEW_BANNER not in page_text(at)
    assert at.session_state["classify_uploader_key"] == 1


# -------------------------------------------------------------------------------------
# Prediction History
# -------------------------------------------------------------------------------------
def record(i, status="accepted", **kw):
    base = dict(prediction_id=i, fruit_id=None, predicted_class="Apple", apple_probability=0.97,
                banana_probability=0.03, confidence=0.97, prediction_status=status,
                rejection_reason=None, model_version="M4_PC1-test")
    base.update(kw)
    return base


def run_history(records=(), error=None):
    fakes.history_service = fakes.FakeHistoryService(records, error)
    return AppTest.from_function(history_script, default_timeout=TIMEOUT).run()


def test_history_shows_tiles_and_one_row_per_record():
    records = [
        record(4, "invalid_input", predicted_class=None, confidence=None, apple_probability=None,
               banana_probability=None, rejection_reason="unreadable_file"),
        record(3, "rejected", predicted_class=None, rejection_reason="far_from_training_data"),
        record(2, "needs_review", predicted_class="Banana", confidence=0.52, rejection_reason="low_confidence"),
        record(1),
    ]
    at = run_history(records)
    text = page_text(at)
    assert not at.exception
    for label in ("Total Attempts", "Accepted", "Needs Review", "Rejected / Invalid"):
        assert label in text
    table = next(m.value for m in at.markdown if "fa-table" in m.value)
    assert table.count("<tr>") == 1 + 4                                   # header + 4 rows
    assert "Unreadable file" in table and "Unlike Supported Images" in table and "Low Confidence" in table
    assert table.index("#4") < table.index("#1")                          # order as supplied: newest first


def test_history_adds_a_date_column_only_when_the_database_has_timestamps():
    without = next(m.value for m in run_history([record(1)]).markdown if "fa-table" in m.value)
    with_time = next(m.value for m in run_history([record(1, created_at="2026-10-20 08:30:00")]).markdown
                     if "fa-table" in m.value)
    assert "Date &amp; Time" not in without and "Date &amp; Time" in with_time and "Oct 2026" in with_time


def test_history_empty_state():
    at = run_history([])
    assert any("No Predictions Yet" in i.value for i in at.info)


def test_history_database_failure_shows_a_clear_error():
    at = run_history(error=DatabaseUnavailableError("down"))
    assert not at.exception
    assert any("init_db.py" in e.value for e in at.error)


def test_history_paginates():
    at = run_history([record(i) for i in range(25, 0, -1)])
    assert "Page 1 of 3" in page_text(at)
    assert at.button(key="history_prev").disabled
    table = next(m.value for m in at.markdown if "fa-table" in m.value)
    assert table.count("<tr>") == 1 + 10
    at.button(key="history_next").click().run()
    assert "Page 2 of 3" in page_text(at)


def test_format_timestamp():
    assert "2026" in format_timestamp("2026-10-20 08:30:00") and "Oct" in format_timestamp("2026-10-20 08:30:00")
    assert format_timestamp("not a date") == "not a date"
    assert format_timestamp(None) == ""
