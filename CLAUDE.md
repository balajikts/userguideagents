# CLAUDE.md

Multi-agent electronics user-guide assistant (AutoGen agentchat 0.7). Design: `PLAN.md`; usage and deploy: `README.md`.

## Commands

```bash
.venv/Scripts/python -m pytest -q                 # offline; no API keys or Postgres needed
.venv/Scripts/python -m scripts.ask "question"    # one real question end to end (uses .env)
uvicorn --factory app.api.main:create_app --reload
streamlit run ui/streamlit_app.py
python -m scripts.ingest_manuals --brand ... --url ... [--file ...]
python -m evals.run_evals --mode offline|langsmith [--no-judge]   # costs API calls
```

## Layout

- `app/agents/` — `query_verifier`, `searcher` (no LLM), `filter`, `manager` (fixed-order orchestrator)
- `app/guardrails/` — `input.py` (scope, injection, PII, dangerous procedures), `output.py` (grounding, safety)
- `app/tools/manual_rag.py` — the manual-store interface and `create_manual_store()`; backends in
  `chroma_store.py` and `pgvector_store.py`
- `app/llm.py` — model clients; `app/factory.py` wires everything for the API, scripts and evals

## Configuration (`.env`, see `.env.example`)

| Variable | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` \| `anthropic` | plus `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; `LLM_MODEL` optional |
| `VECTOR_STORE` | `chroma` (default) \| `pgvector` | docker-compose sets `pgvector` for the api service |
| `CHROMA_PATH` | dir, default `./data/chroma` | only used with `chroma` |
| `DATABASE_URL` | Postgres DSN | only used with `pgvector` |
| `TAVILY_API_KEY` | optional | without it, only ingested manuals are searched |

## Rules and gotchas

- **Never import psycopg/pgvector outside `app/tools/pgvector_store.py`**, and import that module only
  via `create_manual_store()`. On the dev Windows machine an Application Control policy blocks the
  psycopg-binary DLL; `tests/test_manual_rag.py` fails if app startup imports it.
- Dependencies: `requirements.txt` (local + base) and `requirements-prod.txt` (adds psycopg/pgvector;
  Dockerfile only). Keep them in sync with `pyproject.toml`; `tests/test_requirements.py` checks this.
- `protobuf~=5.29.3` and `googleapis-common-protos<=1.75.0` are pinned on purpose: autogen-core needs
  protobuf 5.29, and newer googleapis-common-protos (via chromadb's OpenTelemetry) fail to import on it.
- Tests must stay offline. LLMs use AutoGen's `ReplayChatCompletionClient`; HTTP via respx. The OpenAI
  SDK v3 uses `httpx2`, which respx does not patch; inject an `httpx2.MockTransport` instead
  (see `tests/test_openai_provider.py`).
- LLM-facing Pydantic models (`DeviceQueryDraft`, `DraftAnswer`) must have no defaults or constraints,
  so OpenAI strict structured outputs accept them; validate into the real schema afterwards.
- Inner `AssistantAgent`s are created per call: they hold conversation state and the API serves
  concurrent requests.
- Do not print or commit `.env`; it holds real keys.
