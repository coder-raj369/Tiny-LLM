"""FastAPI application for Tiny LLM inference."""

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from tiny_llm.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from tiny_llm.api.metrics import APIMetrics
from tiny_llm.config import load_config
from tiny_llm.inference import GenerationError, Generator, ModelLoader

LOGGER = logging.getLogger("tiny_llm.api")
CONFIG = load_config()

CHECKPOINT_PATH = os.getenv(
    "TINY_LLM_CHECKPOINT", str(Path("checkpoints") / "best_model.pt")
)
TOKENIZER_PATH = os.getenv("TINY_LLM_TOKENIZER", str(Path("models") / "tokenizer.json"))
DEVICE = os.getenv("TINY_LLM_DEVICE", CONFIG.training.device)

loader = ModelLoader(
    checkpoint_path=CHECKPOINT_PATH,
    tokenizer_path=TOKENIZER_PATH,
    model_config=CONFIG.model,
    device=DEVICE,
)
generator = Generator(loader)
metrics = APIMetrics()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add a request ID and baseline security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            metrics.observe(request.url.path, 500, (time.perf_counter() - start) * 1000)
            raise
        metrics.observe(request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

app = FastAPI(
    title="Tiny LLM Quantum Computing API",
    version="0.1.0",
    description="Inference API for the domain-specific quantum computing assistant.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("TINY_LLM_CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness endpoint; it remains available before model artifacts are built."""
    return HealthResponse(status="ok", model_loaded=loader.is_loaded)


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """Expose process-local metrics in Prometheus text format."""
    return Response(content=metrics.prometheus_text(), media_type="text/plain; version=0.0.4")


@app.get("/api/v1/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness endpoint for a load balancer or deployment probe."""
    if not loader.is_loaded:
        try:
            loader.load()
        except Exception as exc:
            LOGGER.warning("Model is not ready: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model artifacts are not available",
            ) from exc
    return HealthResponse(status="ready", model_loaded=True)


@app.post("/api/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    """Generate a domain answer from a validated prompt."""
    try:
        result = generator.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(
            "Inference failed request_id=%s client=%s",
            http_request.state.request_id,
            http_request.client,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference service is unavailable",
        ) from exc

    LOGGER.info(
        "generation_completed request_id=%s prompt_tokens=%d generated_tokens=%d latency_ms=%.2f",
        http_request.state.request_id,
        result.prompt_tokens,
        result.tokens_generated,
        result.latency_ms,
    )
    return GenerateResponse(
        output=result.output,
        prompt_tokens=result.prompt_tokens,
        tokens_generated=result.tokens_generated,
        latency_ms=result.latency_ms,
    )
