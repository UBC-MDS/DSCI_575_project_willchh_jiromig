import os
from unittest.mock import MagicMock, patch


def test_web_search_returns_empty_string_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from src.tools import web_search

    result = web_search.invoke({"query": "vitamin c serum"})

    assert result == ""
