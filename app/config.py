from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_max_tokens: int = 16000

    # Agents
    clarify_threshold: float = 0.6
    max_sources: int = 8
    min_grounded_steps: int = 1
    grounding_min_overlap: float = 0.2

    # Retrieval
    tavily_api_key: str | None = None
    web_search_results: int = 6
    database_url: str = "postgresql://guide:guide@localhost:5432/guide"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    manual_store_k: int = 6
    search_timeout_s: float = 15.0

    # Infra
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_s: int = 60 * 60 * 24
    rate_limit_per_min: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
