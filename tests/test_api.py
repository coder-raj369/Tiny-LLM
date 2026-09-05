"""Contract tests for the HTTP API."""

from fastapi.testclient import TestClient
import pytest

from tiny_llm.api.server import app, loader
from tiny_llm.config import load_config


client = TestClient(app)


def test_health_is_available_without_checkpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is loader.is_loaded


def test_health_returns_request_id_and_security_headers():
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-request-1"})

    assert response.headers["X-Request-ID"] == "test-request-1"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"


def test_generate_rejects_empty_prompt():
    response = client.post("/api/v1/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_generate_rejects_invalid_sampling_parameters():
    response = client.post(
        "/api/v1/generate",
        json={"prompt": "What is a qubit?", "temperature": 0},
    )

    assert response.status_code == 422


def test_config_rejects_invalid_split_ratios(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  train_ratio: 0.8\n  val_ratio: 0.1\n  test_ratio: 0.2\n")

    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        load_config(str(config_path))
