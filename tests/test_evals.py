"""Eval metrics and the offline runner with a fake judge and fake target."""

from evals.metrics import (
    FaithfulnessVerdict, RelevanceVerdict, StepVerdict,
    citation_coverage, citation_trust, key_fact_recall, make_faithfulness, make_relevance,
    official_source, status_match,
)
from evals.run_evals import DETERMINISTIC, load_dataset, run_offline

OUT = {
    "status": "answered", "summary": "Reset the headset.",
    "steps": [{"number": 1, "instruction": "Hold the power and custom buttons for 7 seconds.", "citation": "S1"},
              {"number": 2, "instruction": "Wait for the indicator.", "citation": "S9"}],
    "sources": [{"id": "S1", "url": "https://helpguide.sony.net/x", "trust_score": 1.0, "snippet": "hold power and custom 7 s"},
                {"id": "S2", "url": "https://blog.io/x", "trust_score": 0.3, "snippet": "..."}],
}
REF = {"status": "answered", "must_mention": ["power", "custom", "7", "blue"], "official_domains": ["sony.net"]}


def test_deterministic_metrics():
    assert status_match({}, OUT, REF)["score"] == 1.0
    assert citation_coverage({}, OUT, REF)["score"] == 0.5  # S9 doesn't exist
    assert citation_trust({}, OUT, REF)["score"] == 1.0
    assert official_source({}, OUT, REF)["score"] == 1.0
    assert key_fact_recall({}, OUT, REF)["score"] == 0.75


def test_metrics_skip_when_not_answered():
    out = {"status": "rejected", "steps": []}
    assert citation_coverage({}, out, REF)["score"] is None
    assert status_match({}, out, {"status": "rejected"})["score"] == 1.0


def fake_judge(verdict):
    calls = []

    async def judge(prompt, schema):
        calls.append(prompt)
        return verdict

    judge.calls = calls
    return judge


async def test_faithfulness_uses_cited_snippets():
    j = fake_judge(FaithfulnessVerdict(verdicts=[StepVerdict(step=1, supported=True, reason=""),
                                                 StepVerdict(step=2, supported=False, reason="no source")]))
    r = await make_faithfulness(j)({}, OUT, REF)
    assert r["score"] == 0.5 and "no source" in r["comment"]
    assert "hold power and custom 7 s" in j.calls[0] and "(missing)" in j.calls[0]


async def test_relevance_normalised():
    r = await make_relevance(fake_judge(RelevanceVerdict(score=5, reason="ok")))({"question": "q"}, OUT, REF)
    assert r["score"] == 1.0


def test_golden_dataset_is_well_formed():
    ds = load_dataset()
    assert len(ds) >= 12 and len({e["id"] for e in ds}) == len(ds)
    statuses = {e["outputs"]["status"] for e in ds}
    assert statuses == {"answered", "needs_clarification", "rejected"}
    for e in ds:
        if e["outputs"]["status"] == "answered":
            assert e["outputs"]["must_mention"] and e["outputs"]["official_domains"]


async def test_offline_runner_survives_target_errors():
    async def target(inputs):
        if "poem" in inputs["question"]:
            raise RuntimeError("boom")
        return OUT

    examples = [{"id": "a", "inputs": {"question": "reset"}, "outputs": REF},
                {"id": "b", "inputs": {"question": "poem"}, "outputs": {"status": "rejected"}}]
    report = await run_offline(target, DETERMINISTIC, examples)
    assert report["rows"][1]["error"].startswith("RuntimeError")
    assert report["summary"]["status_match"] == 0.5
