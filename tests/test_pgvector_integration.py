"""Runs against a real pgvector Postgres when TEST_DATABASE_URL is set, e.g.

    docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=t pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:t@localhost:5433/postgres pytest tests/test_pgvector_integration.py
"""

import os
from pathlib import Path

import pytest

from app.schemas import DeviceQuery
from app.tools.manual_store import ManualChunk, PgVectorManualStore
from tests.fakes import HashEmbedder

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")


class Embedder384(HashEmbedder):
    dim = 384


@pytest.fixture
async def store():
    import psycopg

    sql = (Path(__file__).parents[1] / "db" / "init.sql").read_text()
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as conn:
        await conn.execute("DROP TABLE IF EXISTS manual_chunks")
        await conn.execute(sql)
    s = PgVectorManualStore(DSN, Embedder384())
    await s.open()
    yield s
    await s.close()


async def test_add_search_brand_filter_and_dedupe(store):
    chunks = [
        ManualChunk(brand="Sony", model="WH-1000XM5", title="Reset", url="u1", page=None,
                    content="factory reset hold power and custom buttons 7 seconds"),
        ManualChunk(brand="Sony", model="WH-1000XM4", title="Charge", url="u2", page=3, content="charge battery usb"),
        ManualChunk(brand="Bose", model="QC45", title="Reset", url="u3", page=1, content="factory reset hold power"),
    ]
    await store.add(chunks)
    await store.add(chunks)  # idempotent, including the page=None chunk

    q = DeviceQuery(brand="sony", model="WH1000XM5", device_type="headphones", question="factory reset", confidence=0.9)
    hits = await store.search(q, k=5)
    assert len(hits) == 2 and {h.chunk.brand for h in hits} == {"Sony"}
    assert hits[0].chunk.url == "u1"
