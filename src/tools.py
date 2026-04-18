import os

from langchain_core.tools import tool

try:
    from tavily import TavilyClient

    _HAS_TAVILY = True
except ImportError:
    TavilyClient = None  # type: ignore[assignment]
    _HAS_TAVILY = False


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for current information about a beauty product, brand, or
    ingredient. Use this when the provided reviews/context lack specific or recent
    information. Returns concatenated text snippets, or an empty string if the
    Tavily API key is unavailable.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or not _HAS_TAVILY:
        return ""
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=max_results)
    snippets = [r.get("content", "") for r in results.get("results", [])]
    return "\n".join(snippets)


TOOLS = [web_search]
