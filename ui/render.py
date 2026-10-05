"""Pure formatting helpers for the Streamlit UI (unit-tested without Streamlit)."""

from __future__ import annotations


def source_label(src: dict) -> str:
    page = f", p. {src['page']}" if src.get("page") else ""
    return f"{src['title']}{page}"


def answer_markdown(answer: dict) -> str:
    """Steps as a numbered list with an inline citation link per step."""
    status = answer.get("status")
    if status == "needs_clarification":
        return answer.get("clarification_question") or "Could you tell me the brand and model?"
    if status in ("rejected", "not_found") or not answer.get("steps"):
        return answer.get("summary") or "I couldn't find an answer in the official manuals."

    by_id = {s["id"]: s for s in answer.get("sources", [])}
    lines = [answer["summary"], ""] if answer.get("summary") else []
    for step in answer["steps"]:
        src = by_id.get(step["citation"])
        cite = f"[[{step['citation']}]]({src['url']})" if src else f"[{step['citation']}]"
        flag = " :orange[(not verified in source)]" if not step.get("verified", True) else ""
        lines.append(f"{step['number']}. {step['instruction']} {cite}{flag}")
    return "\n".join(lines)


def cited_sources(answer: dict) -> list[dict]:
    """Sources actually cited by a step, in citation order; all sources if none cited."""
    used = [s["citation"] for s in answer.get("steps", [])]
    srcs = answer.get("sources", [])
    if not used:
        return srcs
    order = {sid: i for i, sid in enumerate(dict.fromkeys(used))}
    return sorted((s for s in srcs if s["id"] in order), key=lambda s: order[s["id"]])
