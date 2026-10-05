# Electronics User-Guide Assistant — Plan

## Goal
Answer "how do I …" questions about consumer electronics with step-by-step
instructions grounded in **official manuals**, one citation per step.

## Request flow

```
user ─► FastAPI /ask ─► Redis cache? ─► Manager (AutoGen BaseChatAgent)
                                         │
             ┌───────────────────────────┤
             ▼                           │
   Input guardrails (scope, injection, PII redaction)   ── reject ─► refusal
             ▼
   QueryVerifier  (LLM → DeviceQuery JSON)  ── low confidence ─► clarification question
             ▼
   Searcher (no LLM, asyncio.gather) ─┬─ WebSearch (Tavily, official-domain filter)
                                      └─ ManualStore (Chroma locally / pgvector in prod, fastembed)
             ▼
   Filter  (deterministic trust rank + dedupe → LLM writes steps citing [S#])
             ▼
   Output guardrails (every step cites a real source, lexical grounding,
                      safety warnings injected for risky procedures)
             ▼
   GuideAnswer JSON ─► Redis cache ─► client (Streamlit UI / API)
```

## Design decisions
* **Each agent is an AutoGen `BaseChatAgent`.** They speak AutoGen messages, so
  they can be dropped into a team, but the Manager drives them in a fixed order:
  the pipeline is known in advance, so deterministic orchestration is cheaper
  and more testable than letting an LLM select the next speaker.
* **LLM work is isolated** in the Verifier and Filter. Both wrap an AutoGen
  `AssistantAgent` with `output_content_type`, mapped onto Claude structured outputs.
  Tests use AutoGen's `ReplayChatCompletionClient`, so no API key is needed.
* **Trust ranking and dedupe are deterministic** (domain allow-list per brand,
  manual-store hits rank highest, shingle-Jaccard dedupe). The LLM only writes
  prose from already-ranked sources.
* **Grounding is enforced after generation**: a step whose citation is missing,
  unknown, or lexically unsupported by its source is dropped; if too few steps
  survive, the answer is downgraded to "couldn't find this in the manual".
* **Model provider is configurable** (`LLM_PROVIDER=anthropic|openai`); default
  is Claude (`claude-opus-5-5`). Embeddings are local (fastembed, bge-small, 384-d)
  so the VPS needs no extra embedding API.

## Folder structure

```
electronics-guide-assistant/
├── app/
│   ├── config.py              # pydantic-settings
│   ├── schemas.py             # DeviceQuery, Source, GuideStep, GuideAnswer …
│   ├── llm.py                 # model-client factory (Anthropic / OpenAI)
│   ├── agents/
│   │   ├── query_verifier.py  # 1. extract + clarify
│   │   ├── searcher.py        # 2. parallel web + manual store
│   │   ├── filter.py          # 3. rank, dedupe, write cited steps
│   │   └── manager.py         # orchestrator
│   ├── guardrails/
│   │   ├── input.py           # scope, injection, PII
│   │   └── output.py          # grounding, safety warnings
│   ├── tools/
│   │   ├── web_search.py      # Tavily client + official-domain map
│   │   ├── manual_rag.py      # store interface + VECTOR_STORE factory
│   │   ├── chroma_store.py    # Chroma backend (local default)
│   │   ├── pgvector_store.py  # pgvector backend (prod; only module importing psycopg)
│   │   └── embeddings.py      # fastembed wrapper
│   ├── cache.py               # Redis answer cache + rate limit
│   └── api/main.py            # FastAPI
├── ui/streamlit_app.py
├── scripts/ingest_manuals.py  # PDF/text → chunks → manual store
├── db/init.sql
├── evals/
│   ├── golden_dataset.jsonl
│   ├── metrics.py             # faithfulness, relevance, citation metrics
│   └── run_evals.py           # LangSmith evaluate() or offline mode
├── tests/                     # one test module per component
├── deploy/Caddyfile
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Build order (each step lands with tests)
1. Scaffolding, config, schemas
2. Input guardrails
3. Query Verifier
4. Searcher (+ web search + manual store tools)
5. Filter
6. Output guardrails
7. Manager
8. FastAPI + Redis cache
9. Streamlit UI
10. Evals (LangSmith)
11. Docker / compose / Caddy for Hostinger VPS
