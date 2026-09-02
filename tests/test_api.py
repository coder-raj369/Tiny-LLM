"""Contract tests for the HTTP API."""

from fastapi.testclient import TestClient

from tiny_llm.api.server import app, loader


client = TestClient(app)


def test_health_is_available_without_checkpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is loader.is_loaded


def test_generate_rejects_empty_prompt():
    response = client.post("/api/v1/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_generate_rejects_invalid_sampling_parameters():
    response = client.post(
        "/api/v1/generate",
        json={"prompt": "What is a qubit?", "temperature": 0},
    )

    assert response.status_code == 422
