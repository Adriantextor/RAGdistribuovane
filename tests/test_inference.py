import json

import httpx
from fastapi.testclient import TestClient

from services.inference import main
from shared.schemas import GenerateInput, Source


SOURCE = Source(id="chunk", document="paper.pdf", page=1, text="Double quantization quantizes the quantization constants.")


def test_inference_requires_token_and_handles_empty_context_without_model():
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 401
        response = client.post("/generate", headers={"Authorization": "Bearer test-only-token-abcdefghijklmnopqrstuvwxyz"},
                               json={"question": "What is my password?", "sources": [], "language": "en"})
        assert response.status_code == 200
        assert "sufficient evidence" in response.json()["answer"]


def test_real_ndjson_protocol_and_language_instruction(monkeypatch):
    real_client = httpx.AsyncClient
    def respond(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert "výhradne po slovensky" in body["messages"][0]["content"]
        frames = [{"message": {"content": "Kvantizujú sa aj "}, "done": False},
                  {"message": {"content": "konštanty [1]."}, "done": False},
                  {"message": {"content": ""}, "done": True, "eval_count": 10}]
        return httpx.Response(200, text="\n".join(map(json.dumps, frames)))
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(respond)))
    with TestClient(main.app) as client:
        response = client.post("/generate/stream", headers={"Authorization": "Bearer test-only-token-abcdefghijklmnopqrstuvwxyz"},
                               json={"question": "Čo znamená double quantization?", "sources": [SOURCE.model_dump()], "language": "sk"})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [event["type"] for event in events] == ["token", "token", "done"]
        assert events[-1]["answer"] == "Kvantizujú sa aj konštanty [1]."


def test_invented_citation_is_not_published_as_a_completed_answer():
    body = GenerateInput(question="What is quantization?", sources=[SOURCE], language="en")
    assert "sufficient evidence" in main.validate_answer("It saves 99% [5].", body)
    assert "sufficient evidence" in main.validate_answer("A completely uncited claim.", body)
    assert main.validate_answer("Quantization constants are quantized [1].", body).endswith("[1].")
