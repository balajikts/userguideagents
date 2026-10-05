CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS manual_chunks (
    id           BIGSERIAL PRIMARY KEY,
    brand        TEXT NOT NULL,
    model        TEXT,
    model_key    TEXT NOT NULL DEFAULT '',   -- lower-case alphanumerics of model, for fuzzy model match
    device_type  TEXT,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    page         INT,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,       -- BAAI/bge-small-en-v1.5
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS manual_chunks_dedupe ON manual_chunks (url, COALESCE(page, 0), md5(content));
CREATE INDEX IF NOT EXISTS manual_chunks_brand ON manual_chunks (lower(brand));
CREATE INDEX IF NOT EXISTS manual_chunks_embedding ON manual_chunks USING hnsw (embedding vector_cosine_ops);
