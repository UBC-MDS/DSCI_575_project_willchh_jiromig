from unittest.mock import MagicMock, patch


def test_web_search_returns_empty_string_when_api_key_missing(monkeypatch):
    """web_search returns an empty string when TAVILY_API_KEY is unset."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from src.tools import web_search

    result = web_search.invoke({"query": "vitamin c serum"})

    assert result == ""


def test_web_search_concatenates_snippets_when_tavily_returns_results(monkeypatch):
    """web_search joins Tavily result snippets with newlines."""
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
    """web_search returns an empty string when Tavily yields no results."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}

    with patch("src.tools.TavilyClient", return_value=fake_client):
        from src.tools import web_search

        result = web_search.invoke({"query": "anything"})

    assert result == ""


def test_tools_export_includes_web_search():
    """The public TOOLS list contains exactly the web_search tool."""
    from src.tools import TOOLS, web_search

    assert web_search in TOOLS
    assert len(TOOLS) == 1


def test_web_search_snippets_returns_empty_list_when_api_key_missing(monkeypatch):
    """web_search_snippets returns an empty list when TAVILY_API_KEY is unset."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from src.tools import web_search_snippets

    assert web_search_snippets("vitamin c serum") == []


def test_web_search_snippets_returns_list_when_tavily_returns_results(monkeypatch):
    """web_search_snippets returns the Tavily snippets as a list of strings."""
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
        from src.tools import web_search_snippets

        result = web_search_snippets("retinol", max_results=3)

    ctor.assert_called_once_with(api_key="fake-key")
    fake_client.search.assert_called_once_with("retinol", max_results=3)
    assert result == ["snippet one", "snippet two", "snippet three"]


def test_web_search_snippets_returns_empty_list_when_results_missing(monkeypatch):
    """web_search_snippets returns an empty list when Tavily yields no results."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}

    with patch("src.tools.TavilyClient", return_value=fake_client):
        from src.tools import web_search_snippets

        assert web_search_snippets("anything") == []


def test_web_search_snippets_filters_empty_content(monkeypatch):
    """web_search_snippets drops results whose content field is empty or missing."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [{"content": "kept"}, {"content": ""}, {"content": "also kept"}]
    }

    with patch("src.tools.TavilyClient", return_value=fake_client):
        from src.tools import web_search_snippets

        assert web_search_snippets("q") == ["kept", "also kept"]
