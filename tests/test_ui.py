import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parents[1] / "ui"))
from render import answer_markdown, cited_sources  # noqa: E402

APP = str(Path(__file__).parents[1] / "ui" / "streamlit_app.py")

ANSWER = {
    "status": "answered", "summary": "Resets to defaults.", "warnings": ["Back up first."],
    "steps": [{"number": 1, "instruction": "Hold power.", "citation": "S2", "verified": True},
              {"number": 2, "instruction": "Wait for blue light.", "citation": "S1", "verified": False}],
    "sources": [{"id": "S1", "title": "Help Guide", "url": "https://sony.net/a", "page": 12, "trust_score": 1.0},
                {"id": "S2", "title": "Support", "url": "https://sony.com/b", "page": None, "trust_score": 0.95},
                {"id": "S3", "title": "Unused", "url": "https://x.com", "page": None, "trust_score": 0.3}],
}


def test_answer_markdown_cites_every_step():
    md = answer_markdown(ANSWER)
    assert "1. Hold power. [[S2]](https://sony.com/b)" in md
    assert "2. Wait for blue light. [[S1]](https://sony.net/a) :orange[(not verified in source)]" in md


def test_cited_sources_order_and_filter():
    assert [s["id"] for s in cited_sources(ANSWER)] == ["S2", "S1"]


def test_non_answer_statuses():
    assert answer_markdown({"status": "needs_clarification", "clarification_question": "Which model?"}) == "Which model?"
    assert answer_markdown({"status": "not_found", "summary": "Nothing."}) == "Nothing."


def test_app_renders_empty_state():
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert at.title[0].value.endswith("Device Guide")
    assert len(at.chat_input) == 1


def test_app_handles_unreachable_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.chat_input[0].set_value("reset my sony xm5").run()
    assert not at.exception
    assert "unavailable" in at.session_state.messages[-1]["content"]
