"""VECTOR_STORE selection, the Chroma backend, and psycopg import isolation."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.config import Settings
from app.schemas import DeviceQuery
from app.tools.chroma_store import ChromaManualStore
from app.tools.manual_rag import ManualChunk, create_manual_store, model_bonus
from tests.fakes import HashEmbedder

ROOT = Path(__file__).parents[1]
Q = DeviceQuery(brand="Sony", model="WH1000XM5", device_type="headphones", question="factory reset", confidence=0.9)
CHUNKS = [
    ManualChunk(brand="Sony", model="WH-1000XM5", title="Reset", url="https://sony.net/m.pdf", page=112,
                content="factory reset hold power and custom buttons 7 seconds"),
    ManualChunk(brand="Sony", model="WH-1000XM4", title="Charge", url="https://sony.net/m4.pdf", page=None,
                content="charge battery usb cable"),
    ManualChunk(brand="Bose", model="QC45", title="Reset", url="https://bose.com/qc.pdf", page=3,
                content="factory reset hold power"),
]


def settings(tmp_path, **kw) -> Settings:
    return Settings(_env_file=None, chroma_path=str(tmp_path / "chroma"), **kw)


@pytest.fixture
async def chroma(tmp_path):
    store = ChromaManualStore(str(tmp_path / "chroma"), HashEmbedder())
    await store.open()
    yield store
    await store.close()


def test_default_store_is_chroma(tmp_path, monkeypatch):
    monkeypatch.delenv("VECTOR_STORE", raising=False)  # the Docker image sets pgvector
    assert Settings(_env_file=None).vector_store == "chroma"
    assert isinstance(create_manual_store(settings(tmp_path), HashEmbedder()), ChromaManualStore)


async def test_chroma_search_filters_brand_and_ranks(chroma):
    await chroma.add(CHUNKS)
    hits = await chroma.search(Q, k=5)
    assert {h.chunk.brand for h in hits} == {"Sony"} and len(hits) == 2
    assert hits[0].chunk.page == 112 and hits[0].chunk.model == "WH-1000XM5"
    assert hits[1].chunk.page is None  # None metadata round-trips
    assert all(0.0 <= h.score <= 1.0 for h in hits)


async def test_chroma_unknown_brand_searches_everything(chroma):
    await chroma.add(CHUNKS)
    hits = await chroma.search(Q.model_copy(update={"brand": None}), k=5)
    assert len(hits) == 3


async def test_chroma_reingest_is_idempotent(chroma):
    await chroma.add(CHUNKS)
    await chroma.add(CHUNKS + CHUNKS[:1])  # duplicate inside one batch too
    assert len(await chroma.search(Q.model_copy(update={"brand": None}), k=10)) == 3


async def test_chroma_persists_across_instances(tmp_path):
    first = ChromaManualStore(str(tmp_path / "c"), HashEmbedder())
    await first.open()
    await first.add(CHUNKS[:1])
    await first.close()
    second = ChromaManualStore(str(tmp_path / "c"), HashEmbedder())
    await second.open()
    assert len(await second.search(Q, k=3)) == 1


async def test_chroma_empty_and_unopened(tmp_path, chroma):
    assert await chroma.search(Q) == []
    with pytest.raises(RuntimeError, match="open"):
        await ChromaManualStore(str(tmp_path / "x"), HashEmbedder()).search(Q)


def test_model_bonus():
    assert model_bonus("WH1000XM5", "WH-1000XM5/B") > 0
    assert model_bonus("WH1000XM5", "QC45") == 0 and model_bonus(None, "QC45") == 0


# --- psycopg isolation -------------------------------------------------------------
# Setting sys.modules[name] = None makes `import name` raise ImportError, which is how
# the Windows Application Control block surfaces. These run in a subprocess so the
# result doesn't depend on what this test process already imported.

BLOCK_PSYCOPG = """
import sys
for name in ("psycopg", "psycopg_pool", "psycopg_binary", "pgvector"):
    sys.modules[name] = None
"""


def run_py(code: str, tmp_path) -> subprocess.CompletedProcess:
    # Normal environment minus the app's own settings, so nothing in the shell leaks in.
    app_vars = {k.upper() for k in Settings.model_fields}
    env = {k: v for k, v in os.environ.items() if k.upper() not in app_vars}
    env["CHROMA_PATH"] = str(tmp_path / "c")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(BLOCK_PSYCOPG + code)],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
    )


def test_app_imports_and_builds_with_chroma_when_psycopg_is_blocked(tmp_path):
    r = run_py("""
    import asyncio
    import app, app.factory, app.api.main, scripts.ingest_manuals, evals.run_evals
    from app.config import Settings
    from app.factory import build_manager

    async def main():
        s = Settings(_env_file=None, vector_store="chroma", llm_provider="openai", openai_api_key="sk-x")
        async with build_manager(s) as m:
            assert m.searcher._store is not None, "chroma store should be up"
    asyncio.run(main())
    loaded = [m for m in sys.modules if m.split(".")[0] in ("psycopg", "psycopg_pool", "pgvector") and sys.modules[m] is not None]
    assert not loaded, loaded
    print("OK")
    """, tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]


def test_pgvector_selected_but_blocked_degrades_instead_of_crashing(tmp_path):
    r = run_py("""
    import asyncio
    from app.config import Settings
    from app.factory import build_manager
    from app.tools.manual_rag import create_manual_store

    s = Settings(_env_file=None, vector_store="pgvector", llm_provider="openai", openai_api_key="sk-x")
    try:
        create_manual_store(s)
        raise SystemExit("expected ImportError")
    except ImportError:
        pass

    async def main():
        async with build_manager(s) as m:
            assert m.searcher._store is None
    asyncio.run(main())
    print("OK")
    """, tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
