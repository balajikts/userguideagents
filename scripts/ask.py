"""Ask one question through the full pipeline from the command line.

    python -m scripts.ask "How do I factory reset my Sony WH-1000XM5?"

Uses the provider and keys from .env. Answers clarification questions interactively.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.agents.manager import ASSISTANT, USER
from app.factory import build_manager


async def main(question: str) -> None:
    turns = [(USER, question)]
    async with build_manager() as manager:
        while True:
            ans = await manager.ask(turns)
            if ans.status != "needs_clarification":
                break
            reply = input(f"? {ans.clarification_question}\n> ").strip()
            turns += [(ASSISTANT, ans.clarification_question or ""), (USER, reply)]

    print(f"\n[{ans.status}] {ans.summary}")
    for s in ans.steps:
        mark = "" if s.verified else "  (not verified in source)"
        print(f"  {s.number}. {s.instruction} [{s.citation}]{mark}")
    for w in ans.warnings:
        print(f"  ! {w}")
    for src in ans.sources:
        print(f"  {src.id}: {src.title} — {src.url}" + (f" (p. {src.page})" if src.page else ""))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m scripts.ask "your question"')
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(" ".join(sys.argv[1:])))
