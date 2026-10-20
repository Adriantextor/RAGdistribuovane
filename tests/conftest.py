import pytest

from shared.config import settings


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("NODE_API_TOKEN", "test-only-token-abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("WORKER_POLL_SECONDS", "0.05")
    monkeypatch.setenv("RETRY_BASE_SECONDS", "0.05")
    settings.cache_clear()
    yield
    settings.cache_clear()
