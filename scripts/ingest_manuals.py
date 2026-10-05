"""Ingest official manuals (PDF or text) into the manual store (VECTOR_STORE: chroma | pgvector).

    python -m scripts.ingest_manuals --brand Sony --model WH-1000XM5 --device-type headphones \
        --url https://helpguide.sony.net/mdr/wh1000xm5/v1/en/print.pdf --file manuals/wh1000xm5.pdf

--url is the public link users will see in citations; --file is the local copy
(omit it to download from --url).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
from pathlib import Path

import httpx

from app.config import get_settings
from app.tools.manual_rag import ManualChunk, create_manual_store


def chunk_text(text: str, *, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries into ~size-char chunks with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p
        else:
            cur = f"{cur}\n{p}" if cur else p
        while len(cur) > size * 1.5:  # one giant paragraph
            chunks.append(cur[:size])
            cur = cur[size - overlap :]
    if cur:
        chunks.append(cur)
    return chunks


def pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]


def build_chunks(pages: list[str], *, brand: str, model: str | None, device_type: str | None, url: str, title: str) -> list[ManualChunk]:
    out = []
    for page_no, text in enumerate(pages, 1):
        for c in chunk_text(text):
            if len(c) < 40:
                continue
            out.append(ManualChunk(brand=brand, model=model, device_type=device_type, title=title,
                                   url=url, page=page_no if len(pages) > 1 else None, content=c))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--model")
    ap.add_argument("--device-type")
    ap.add_argument("--url", required=True)
    ap.add_argument("--file")
    ap.add_argument("--title")
    a = ap.parse_args()

    data = Path(a.file).read_bytes() if a.file else httpx.get(a.url, follow_redirects=True, timeout=60).content
    is_pdf = data[:4] == b"%PDF"
    pages = pdf_pages(data) if is_pdf else [data.decode("utf-8", "ignore")]
    title = a.title or f"{a.brand} {a.model or ''} manual".replace("  ", " ")
    chunks = build_chunks(pages, brand=a.brand, model=a.model, device_type=a.device_type, url=a.url, title=title)

    s = get_settings()
    store = create_manual_store(s)
    await store.open()
    try:
        n = await store.add(chunks)
    finally:
        await store.close()
    print(f"processed {n} chunks from {len(pages)} page(s) (already-ingested chunks are skipped)")


if __name__ == "__main__":
    asyncio.run(main())
