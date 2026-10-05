"""Shared Pydantic models passed between agents."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DeviceCategory(str, Enum):
    TV = "tv"
    PHONE = "phone"
    LAPTOP = "laptop"
    TABLET = "tablet"
    HEADPHONES = "headphones"
    SPEAKER = "speaker"
    CAMERA = "camera"
    ROUTER = "router"
    WEARABLE = "wearable"
    GAME_CONSOLE = "game_console"
    APPLIANCE = "appliance"
    PRINTER = "printer"
    MONITOR = "monitor"
    OTHER = "other"


class DeviceQuery(BaseModel):
    """Structured form of the user's question, produced by the Query Verifier."""

    brand: str | None = Field(None, description="Manufacturer, e.g. 'Sony'")
    model: str | None = Field(None, description="Model number/name, e.g. 'WH-1000XM5'")
    device_type: DeviceCategory = DeviceCategory.OTHER
    question: str = Field(..., min_length=3, description="The task the user wants to do")
    confidence: float = Field(..., ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_question: str | None = None

    @field_validator("brand", "model", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip().lower() in {"", "unknown", "n/a", "none", "null"}:
            return None
        return v.strip() if isinstance(v, str) else v

    @property
    def search_text(self) -> str:
        parts = [self.brand, self.model, self.device_type.value if self.device_type != DeviceCategory.OTHER else None]
        return " ".join(p for p in parts if p) + f" {self.question}"


class SourceOrigin(str, Enum):
    MANUAL_STORE = "manual_store"
    WEB = "web"


class Source(BaseModel):
    id: str = Field(..., description="Stable citation id, e.g. 'S1'")
    title: str
    url: str
    snippet: str
    origin: SourceOrigin
    retrieval_score: float = 0.0
    trust_score: float = 0.0
    page: int | None = None


class SearchResults(BaseModel):
    query: DeviceQuery
    sources: list[Source]
    errors: list[str] = Field(default_factory=list)


class GuideStep(BaseModel):
    number: int = Field(..., ge=1)
    instruction: str = Field(..., min_length=3)
    citation: str = Field(..., description="Source id, e.g. 'S2'")
    verified: bool = Field(True, description="Output guardrail found support for this step in the cited source")


class GuideAnswer(BaseModel):
    status: str = Field("answered", pattern="^(answered|needs_clarification|not_found|rejected)$")
    summary: str = ""
    steps: list[GuideStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    clarification_question: str | None = None
    query: DeviceQuery | None = None
