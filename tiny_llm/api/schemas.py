"""HTTP request and response schemas for the inference API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Parameters accepted by the generation endpoint."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=2000)


class GenerateResponse(BaseModel):
    """Generated answer and measurable inference metadata."""

    output: str
    prompt_tokens: int
    tokens_generated: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
