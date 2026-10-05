"""Input guardrails: prompt-injection detection, PII redaction, and scope check.

All checks are deterministic so they run before any LLM call and cost nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
    r"forget\s+(everything|all|your)\s+(instructions|rules|previous)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"\b(act|behave)\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(unrestricted|unfiltered|jailbroken|evil|different\s+(ai|assistant|model))\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\b(developer|dan|jailbreak|god)\s+mode\b",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>",
    r"\[\s*(system|inst)\s*\]",
    r"new\s+instructions\s*:",
    r"override\s+(your|the)\s+(rules|guardrails|safety)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?![\w-])")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_PRIVATE_IP = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|127\.)")

# Words that strongly suggest the question is about a device.
_SCOPE_TERMS = {
    "tv", "television", "phone", "smartphone", "iphone", "android", "laptop", "notebook",
    "macbook", "tablet", "ipad", "headphone", "headphones", "earbuds", "airpods", "speaker",
    "soundbar", "camera", "router", "modem", "wifi", "wi-fi", "bluetooth", "watch",
    "smartwatch", "console", "playstation", "ps5", "xbox", "switch", "printer", "monitor",
    "remote", "firmware", "reset", "pair", "pairing", "charger", "charging", "battery",
    "hdmi", "usb", "device", "settings", "screen", "display", "update", "install", "setup",
    "microwave", "fridge", "refrigerator", "washer", "dishwasher", "dryer", "vacuum", "projector",
    "keyboard", "mouse", "drone", "receiver", "amplifier", "turntable", "thermostat", "doorbell",
    "manual", "model", "volume", "mute", "power", "factory", "app", "connect", "mode",
}
_BRANDS = {
    "apple", "samsung", "sony", "lg", "panasonic", "philips", "bose", "jbl", "sennheiser",
    "google", "pixel", "oneplus", "xiaomi", "huawei", "motorola", "nokia", "dell", "hp", "lenovo",
    "asus", "acer", "msi", "microsoft", "nintendo", "canon", "nikon", "fujifilm", "gopro",
    "netgear", "tp-link", "tplink", "linksys", "asus", "garmin", "fitbit", "epson", "brother",
    "roku", "vizio", "tcl", "hisense", "sharp", "toshiba", "whirlpool", "bosch", "dyson",
    "logitech", "anker", "beats", "sonos", "nest", "ring", "amazon", "kindle", "echo", "dji",
}
_OFF_TOPIC = re.compile(
    r"\b(write (me )?(a|an) (poem|essay|story|song)|recipe|stock price|weather forecast|"
    r"horoscope|homework|translate|who won|politic|election|medical advice|diagnos)\w*",
    re.IGNORECASE,
)


@dataclass
class InputCheck:
    allowed: bool
    sanitized_text: str
    reasons: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def sub(pattern: re.Pattern, label: str, s: str, predicate=None) -> str:
        def repl(m: re.Match) -> str:
            if predicate and not predicate(m.group(0)):
                return m.group(0)
            found.append(label)
            return f"[{label}]"

        return pattern.sub(repl, s)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_SSN, "SSN", text)
    text = sub(
        _CARD_CANDIDATE,
        "CARD",
        text,
        lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn_ok(re.sub(r"\D", "", s)),
    )
    text = sub(_PHONE, "PHONE", text)
    # Private/LAN IPs are useful for router questions; only public IPs are redacted.
    text = sub(_IP, "IP", text, lambda s: not _PRIVATE_IP.match(s))
    return text, sorted(set(found))


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def in_scope(text: str) -> bool:
    if _OFF_TOPIC.search(text):
        return False
    tokens = set(re.findall(r"[a-z0-9][a-z0-9+-]*", text.lower()))
    if tokens & (_SCOPE_TERMS | _BRANDS):
        return True
    # Model-number-looking tokens (letters+digits, e.g. "wh1000xm5", "sm-s918b").
    return any(re.search(r"[a-z]", t) and re.search(r"\d", t) and len(t) >= 4 for t in tokens)


def check_input(text: str) -> InputCheck:
    text = (text or "").strip()
    if not text:
        return InputCheck(False, "", ["empty_input"])
    if len(text) > MAX_INPUT_CHARS:
        return InputCheck(False, text[:MAX_INPUT_CHARS], ["too_long"])

    sanitized, pii = redact_pii(text)
    reasons: list[str] = []
    if detect_injection(text):
        reasons.append("prompt_injection")
    if not in_scope(sanitized):
        reasons.append("out_of_scope")
    return InputCheck(not reasons, sanitized, reasons, pii)


REFUSAL_MESSAGES = {
    "empty_input": "Please type a question about an electronic device.",
    "too_long": f"Please keep your question under {MAX_INPUT_CHARS} characters.",
    "prompt_injection": "I can only help with questions about using electronic devices.",
    "out_of_scope": (
        "I answer questions about using and setting up electronic devices "
        "(e.g. 'How do I reset my Sony WH-1000XM5?')."
    ),
}
