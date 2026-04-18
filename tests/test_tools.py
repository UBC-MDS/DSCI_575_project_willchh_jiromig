import os
from unittest.mock import MagicMock, patch


def test_web_search_returns_empty_string_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from src.tools import web_search

    result = web_search.invoke({"query": "vitamin c serum"})

    assert result == ""


def test_web_search_concatenates_snippets_when_tavily_returns_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    fake_response = {
        "results": [
            {"content": "snippet one"},
            {"content": "snippet two"},
            {"content": "snippet three"},
        ]
    }
    fake_client = MagicMock()
    fake_client.search.return_value = fake_response

    with patch("src.tools.TavilyClient", return_value=fake_client) as ctor:
        from src.tools import web_search

        result = web_search.invoke({"query": "retinol", "max_results": 3})

    ctor.assert_called_once_with(api_key="fake-key")
    fake_client.search.assert_called_once_with("retinol", max_results=3)
    assert result == "snippet one\nsnippet two\nsnippet three"


def test_web_search_returns_empty_string_when_results_missing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}

    with patch("src.tools.TavilyClient", return_value=fake_client):
        from src.tools import web_search

        result = web_search.invoke({"query": "anything"})

    assert result == ""


def test_tools_export_includes_web_search():
    from src.tools import TOOLS, web_search

    assert web_search in TOOLS
    assert len(TOOLS) == 1
