# Electronics User-Guide Assistant

Multi-agent assistant (AutoGen `autogen-agentchat` 0.7) that answers "how do I…"
questions about consumer electronics with step-by-step instructions grounded in
official manuals, **one citation per step**. See [PLAN.md](PLAN.md) for the design.

```
Manager ─► input guardrails ─► Query Verifier ─► Searcher ─► Filter ─► output guardrails
            scope/injection/PII   brand/model/     web + manuals   trust rank,   grounding +
            dangerous procedures  clarification    in parallel     dedupe, steps safety warnings
```

| Component | File | LLM? |
|---|---|---|
| Query Verifier | `app/agents/query_verifier.py` | yes (structured output → `DeviceQuery`) |
| Searcher | `app/agents/searcher.py` | no (Tavily + manual store via `asyncio.gather`) |
| Filter | `app/agents/filter.py` | yes (writes steps from ranked sources) |
| Manager | `app/agents/manager.py` | no (fixed pipeline) |
| Input guardrails | `app/guardrails/input.py` | no |
| Output guardrails | `app/guardrails/output.py` | no |

**LLM provider** (`LLM_PROVIDER` in `.env`), one key needed:

| Provider | Key | Default model | Client |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-5` | AutoGen `OpenAIChatCompletionClient` (strict structured outputs, `reasoning_effort`) |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | `app/llm.py` on the official `anthropic` SDK (structured outputs, `effort`, refusal fallbacks) |

Override the model with `LLM_MODEL`. The eval judge uses the same provider.

## Local development

```bash
py -3.11 -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt                       # no psycopg; uses Chroma locally
pip install pytest pytest-asyncio respx fakeredis     # test tools
cp .env.example .env                                 # set LLM_PROVIDER + its key (TAVILY_API_KEY optional)
pytest                                               # no API keys needed

docker compose up -d redis                           # or point REDIS_URL elsewhere
uvicorn --factory app.api.main:create_app --reload
streamlit run ui/streamlit_app.py
```

**Manual store** (`VECTOR_STORE`): `chroma` (default) is embedded and stored in
`./data/chroma`, so local runs need no Postgres and no psycopg. That matters on Windows
machines where Application Control blocks psycopg's DLL. docker-compose sets
`pgvector` for the API. `requirements-prod.txt` adds the Postgres driver and is used
only by the Dockerfile.

Ingest a manual into the configured store:

```bash
python -m scripts.ingest_manuals --brand Sony --model WH-1000XM5 --device-type headphones \
  --url https://helpguide.sony.net/mdr/wh1000xm5/v1/en/print.pdf --file manuals/wh1000xm5.pdf
```

API:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do I factory reset my Sony WH-1000XM5?"}'
# status=needs_clarification → resend the reply with the returned session_id
```

## Evaluations (`evals/`)

`golden_dataset.jsonl` holds 14 cases: answers, a clarification, a clarify-then-answer,
an off-topic question, a prompt injection, and a dangerous repair.

| Metric | Type |
|---|---|
| `faithfulness` | LLM judge: share of steps supported by their cited source |
| `answer_relevance` | LLM judge: does the answer do the task (1–5 → 0–1) |
| `citation_coverage`, `citation_trust`, `official_source` | deterministic |
| `status_match`, `key_fact_recall` | deterministic |

```bash
python -m evals.run_evals --mode offline            # table + evals/results/*.json
python -m evals.run_evals --mode langsmith          # LangSmith experiment (LANGSMITH_API_KEY)
python -m evals.run_evals --mode offline --no-judge # deterministic metrics only
```

Each run calls the real pipeline (LLM + Tavily), so it costs money.

## Deploying to a Hostinger VPS

1. Use a KVM VPS (2 vCPU / 8 GB is comfortable) with the Ubuntu + Docker template,
   or install Docker: `curl -fsSL https://get.docker.com | sh`.
2. Point your domain's **A record** at the VPS IP (Hostinger hPanel → DNS).
3. Open ports 80 and 443 in the Hostinger firewall (and `ufw allow 80,443/tcp` if ufw is on).
4. On the VPS:
   ```bash
   git clone <your repo> guide && cd guide
   cp .env.example .env   # set LLM_PROVIDER + key, POSTGRES_PASSWORD, DOMAIN, ACME_EMAIL (TAVILY_API_KEY optional)
   docker compose up -d --build
   docker compose exec api python -m scripts.ingest_manuals --brand ... --url ...
   ```
5. Caddy obtains a Let's Encrypt certificate automatically. The UI is at
   `https://DOMAIN/` and the API at `https://DOMAIN/api/ask`.

Only Caddy publishes ports; Postgres and Redis are reachable only on the
compose network. Data lives in named volumes (`pgdata`, `redisdata`, `caddy_data`).
