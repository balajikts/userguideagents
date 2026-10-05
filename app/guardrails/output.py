"""Output guardrails: grounding and safety.

Grounding (per step):
* the citation must name a source we actually returned;
* every number in the step ("7 seconds", "192.168.0.1") must appear in that
  source, since a wrong hold-time or voltage is the worst kind of hallucination;
* enough of the step's content words must appear in the source.

A step failing the number or citation check, or too low a share of verified
steps, downgrades the whole answer to ``not_found``. Steps that only miss on
word overlap (paraphrase) are kept but flagged ``verified=False``.

Safety: deterministic rules add standard warnings for risky procedures and
block a small set of procedures that should only be done by a technician.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import GuideAnswer, GuideStep, Source

_STOP = set(
    "a an the and or of to in on at for with by from your you it its this that is are be as then "
    "if into onto out up down until about again once more any all each both can will should may "
    "do does done not no yes so than too very just also when while press tap select choose go open "
    "make sure using use".split()
)
_NUM = re.compile(r"\d+(?:[.:]\d+)*")

_SAFETY_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(factory|hard|master)\s+reset|reset\s+to\s+factory|initiali[sz]e|erase all", re.I), "data",
     "A factory reset erases your settings and data. Back up anything you want to keep first."),
    (re.compile(r"\bfirmware|software update|system update|bios", re.I), "power",
     "Don't unplug or power off the device while an update is installing; it can leave the device unusable."),
    (re.compile(r"\b(open(ing)?|remove|unscrew)\b.{0,30}\b(case|cover|back panel|housing|chassis)", re.I), "shock",
     "Unplug the device from mains power before opening it. Opening the case may void your warranty."),
    (re.compile(r"\b(battery|batteries)\b.{0,40}\b(replace|remove|swollen|swelling|bulging|puncture)|\b(replace|remove)\b.{0,30}\bbattery", re.I), "battery",
     "Lithium batteries can catch fire if punctured or bent. Stop using a swollen battery and have it replaced by an authorized service center."),
    (re.compile(r"\b(clean|wipe|wash|descale|liquid|water)\b", re.I), "liquid",
     "Unplug the device and let it dry completely before reconnecting power."),
    (re.compile(r"\b(mains|240\s?v|230\s?v|120\s?v|110\s?v|wiring|hardwire|electrical panel|breaker)\b", re.I), "mains",
     "Work involving mains wiring should be done by a qualified electrician."),
]
_WARNING_PRESENT = {
    "data": re.compile(r"\b(back ?up|erase|data loss|lose)", re.I),
    "power": re.compile(r"\b(unplug|power off|turn off).{0,40}(update|install)|(update|install).{0,40}(unplug|power off|turn off)", re.I),
    "shock": re.compile(r"\b(shock|unplug)", re.I),
    "battery": re.compile(r"\b(fire|swollen|puncture)", re.I),
    "liquid": re.compile(r"\b(unplug|dry)", re.I),
    "mains": re.compile(r"\belectrician", re.I),
}
# Procedures we refuse to walk users through regardless of sources.
_BLOCK = re.compile(
    r"(discharg\w*\s+(the\s+)?(high[- ]voltage\s+)?capacitor|microwave.{0,40}(open|repair|magnetron|interior)|"
    r"bypass\w*\s+(the\s+)?(door\s+)?(safety\s+)?interlock|crt\s+(repair|flyback)|flyback\s+transformer)",
    re.I,
)
BLOCKED_MESSAGE = (
    "This repair involves components that can hold a lethal charge even when unplugged. "
    "Please contact the manufacturer or an authorized service technician."
)


def is_blocked_procedure(text: str) -> bool:
    return bool(_BLOCK.search(text))


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9-]+", text.lower())
    return {w.rstrip("s") for w in words if w not in _STOP and len(w) > 2}


@dataclass
class StepCheck:
    number: int
    citation_ok: bool
    numbers_ok: bool
    overlap: float

    def hard_fail(self) -> bool:
        return not (self.citation_ok and self.numbers_ok)


@dataclass
class GroundingReport:
    steps: list[StepCheck] = field(default_factory=list)
    verified_ratio: float = 1.0
    passed: bool = True


def check_step(step: GuideStep, sources: dict[str, Source], min_overlap: float) -> StepCheck:
    src = sources.get(step.citation)
    if src is None:
        return StepCheck(step.number, False, False, 0.0)
    haystack = f"{src.title} {src.snippet}"
    nums_ok = set(_NUM.findall(step.instruction)) <= set(_NUM.findall(haystack))
    words = _content_words(step.instruction)
    overlap = len(words & _content_words(haystack)) / len(words) if words else 1.0
    return StepCheck(step.number, True, nums_ok, round(overlap, 3))


def check_grounding(answer: GuideAnswer, *, min_overlap: float = 0.2, min_verified_ratio: float = 0.75) -> GroundingReport:
    by_id = {s.id: s for s in answer.sources}
    checks = [check_step(s, by_id, min_overlap) for s in answer.steps]
    if not checks:
        return GroundingReport([], 1.0, True)
    verified = [c for c in checks if not c.hard_fail() and c.overlap >= min_overlap]
    ratio = len(verified) / len(checks)
    passed = not any(c.hard_fail() for c in checks) and ratio >= min_verified_ratio
    return GroundingReport(checks, round(ratio, 3), passed)


def safety_warnings(answer: GuideAnswer) -> list[str]:
    text = " ".join([answer.query.question if answer.query else "", answer.summary, *(s.instruction for s in answer.steps)])
    existing = " ".join(answer.warnings)
    added = []
    for pattern, kind, warning in _SAFETY_RULES:
        if pattern.search(text) and not _WARNING_PRESENT[kind].search(existing):
            added.append(warning)
            existing += " " + warning
    return added


def apply_output_guardrails(
    answer: GuideAnswer, *, min_overlap: float = 0.2, min_verified_ratio: float = 0.75
) -> tuple[GuideAnswer, GroundingReport]:
    if answer.status != "answered":
        return answer, GroundingReport()

    text = " ".join([answer.query.question if answer.query else "", answer.summary, *(s.instruction for s in answer.steps)])
    if _BLOCK.search(text):
        return GuideAnswer(status="rejected", summary=BLOCKED_MESSAGE, warnings=[BLOCKED_MESSAGE], query=answer.query), GroundingReport()

    report = check_grounding(answer, min_overlap=min_overlap, min_verified_ratio=min_verified_ratio)
    if not report.passed:
        return answer.model_copy(update={
            "status": "not_found",
            "summary": "I found possible sources but couldn't verify step-by-step instructions in them. "
                       "Please check the linked manuals directly.",
            "steps": [],
        }), report

    by_num = {c.number: c for c in report.steps}
    steps = [s.model_copy(update={"verified": by_num[s.number].overlap >= min_overlap}) for s in answer.steps]
    return answer.model_copy(update={"steps": steps, "warnings": answer.warnings + safety_warnings(answer)}), report
