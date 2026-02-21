import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
