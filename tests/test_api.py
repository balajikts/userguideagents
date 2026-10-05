import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import get_settings
from app.schemas import GuideAnswer, GuideStep


class StubManager:
    """Scripted Manager: records conversations, returns queued answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    async def ask(self, turns):
        self.calls.append(list(turns))
        return self.answers.pop(0)


ANSWERED = GuideAnswer(status="answered", summary="ok", steps=[GuideStep(number=1, instruction="Hold power.", citation="S1")])
CLARIFY = GuideAnswer(status="needs_clarification", clarification_question="Which model?")


@pytest.fixture
def make_client(monkeypatch):
    def _make(*answers, rate=100):
        monkeypatch.setenv("RATE_LIMIT_PER_MIN", str(rate))
        get_settings.cache_clear()
        mgr = StubManager(*answers)
        app = create_app(manager=mgr, redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
        return TestClient(app), mgr
    yield _make
    get_settings.cache_clear()


def test_health(make_client):
    c, _ = make_client()
    with c:
        assert c.get("/health").json() == {"status": "ok", "redis": True}


def test_answer_is_cached(make_client):
    c, mgr = make_client(ANSWERED)
    with c:
        r1 = c.post("/ask", json={"question": "Reset my Sony XM5?"}).json()
        r2 = c.post("/ask", json={"question": "reset my sony xm5"}).json()  # normalises case/punctuation
    assert r1["cached"] is False and r2["cached"] is True
    assert r2["answer"]["steps"][0]["citation"] == "S1"
    assert len(mgr.calls) == 1


def test_clarification_session_round_trip(make_client):
    c, mgr = make_client(CLARIFY, ANSWERED)
    with c:
        r1 = c.post("/ask", json={"question": "reset my headphones"}).json()
        assert r1["answer"]["status"] == "needs_clarification"
        r2 = c.post("/ask", json={"question": "Sony WH-1000XM5", "session_id": r1["session_id"]}).json()
    assert r2["answer"]["status"] == "answered"
    assert mgr.calls[1] == [["user", "reset my headphones"], ["assistant", "Which model?"], ["user", "Sony WH-1000XM5"]] or \
        mgr.calls[1] == [("user", "reset my headphones"), ("assistant", "Which model?"), ("user", "Sony WH-1000XM5")]


def test_rate_limit(make_client):
    c, _ = make_client(ANSWERED, rate=2)
    with c:
        codes = [c.post("/ask", json={"question": "reset sony xm5"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_pipeline_error_is_502(make_client):
    class Boom:
        async def ask(self, turns):
            raise RuntimeError("x")

    get_settings.cache_clear()
    app = create_app(manager=Boom(), redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    with TestClient(app) as c:
        assert c.post("/ask", json={"question": "reset sony"}).status_code == 502


def test_validation(make_client):
    c, _ = make_client()
    with c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "x", "session_id": "../etc"}).status_code == 422
