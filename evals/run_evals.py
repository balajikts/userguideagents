"""Run the golden-dataset evaluation.

LangSmith (results land in a LangSmith experiment; needs LANGSMITH_API_KEY):
    python -m evals.run_evals --mode langsmith

Offline (prints a table, writes evals/results/<timestamp>.json):
    python -m evals.run_evals --mode offline [--no-judge]

Both modes run the real pipeline (LLM + Tavily + manual store from .env), so
every run costs API calls: 14 examples x (2 pipeline calls + 2 judge calls).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.agents.manager import ASSISTANT, USER, ManagerAgent
from app.config import get_settings
from app.factory import build_manager
from evals.metrics import DETERMINISTIC, langchain_judge, make_faithfulness, make_relevance

HERE = Path(__file__).parent
DATASET = HERE / "golden_dataset.jsonl"


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_target(manager: ManagerAgent):
    async def target(inputs: dict) -> dict:
        turns = [(USER, inputs["question"])]
        ans = await manager.ask(turns)
        if ans.status == "needs_clarification" and inputs.get("followup"):
            ans = await manager.ask([*turns, (ASSISTANT, ans.clarification_question or ""), (USER, inputs["followup"])])
        return ans.model_dump(mode="json")

    return target


def evaluators(use_judge: bool, judge_model: str | None):
    evs = list(DETERMINISTIC)
    if use_judge:
        j = langchain_judge(get_settings().llm_provider, judge_model)
        evs += [make_faithfulness(j), make_relevance(j)]
    return evs


async def _call(ev, inputs, outputs, ref) -> dict:
    r = ev(inputs, outputs, ref)
    return await r if asyncio.iscoroutine(r) else r


async def run_offline(target, evs, examples: list[dict], concurrency: int = 4) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(ex: dict) -> dict:
        async with sem:
            t = time.perf_counter()
            try:
                out = await target(ex["inputs"])
                err = None
            except Exception as e:  # keep going; an exception is a failed example
                out, err = {"status": "error"}, f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t
            scores = [await _call(ev, ex["inputs"], out, ex["outputs"]) for ev in evs]
            return {"id": ex["id"], "latency_s": round(latency, 2), "error": err, "outputs": out, "scores": scores}

    rows = await asyncio.gather(*(one(ex) for ex in examples))
    summary: dict[str, float] = {}
    for key in {s["key"] for r in rows for s in r["scores"]}:
        vals = [s["score"] for r in rows for s in r["scores"] if s["key"] == key and s["score"] is not None]
        if vals:
            summary[key] = round(statistics.mean(vals), 3)
    summary["p50_latency_s"] = round(statistics.median(r["latency_s"] for r in rows), 2)
    return {"summary": summary, "rows": rows}


def print_report(report: dict) -> None:
    keys = sorted({s["key"] for r in report["rows"] for s in r["scores"]})
    print(f"{'example':28}" + "".join(f"{k[:14]:>15}" for k in keys))
    for r in report["rows"]:
        by = {s["key"]: s["score"] for s in r["scores"]}
        cells = "".join(f"{'-' if by.get(k) is None else f'{by[k]:.2f}':>15}" for k in keys)
        print(f"{r['id'][:28]:28}{cells}" + (f"  ERROR {r['error']}" if r["error"] else ""))
    print("\nsummary:", json.dumps(report["summary"], indent=2))


def ensure_langsmith_dataset(client, name: str, examples: list[dict]) -> None:
    if client.has_dataset(dataset_name=name):
        return
    ds = client.create_dataset(name, description="Electronics user-guide golden set")
    client.create_examples(
        dataset_id=ds.id,
        examples=[{"inputs": e["inputs"], "outputs": e["outputs"], "metadata": {"id": e["id"]}} for e in examples],
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "langsmith"], default="offline")
    ap.add_argument("--dataset-name", default="electronics-guide-golden")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    ap.add_argument("--judge-model", default=None, help="defaults to the provider's default model")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    examples = load_dataset()
    evs = evaluators(not a.no_judge, a.judge_model)
    async with build_manager() as manager:
        target = make_target(manager)
        if a.mode == "langsmith":
            from langsmith import Client, aevaluate

            client = Client()
            ensure_langsmith_dataset(client, a.dataset_name, examples)
            await aevaluate(target, data=a.dataset_name, evaluators=evs, experiment_prefix="guide",
                            max_concurrency=a.concurrency, client=client)
            print(f"done; see the '{a.dataset_name}' dataset in LangSmith")
        else:
            report = await run_offline(target, evs, examples, a.concurrency)
            print_report(report)
            out = HERE / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
