# Wire web-search tool into Streamlit app — design

**Date:** 2026-04-22
**Branch:** feat/web-tool
**Status:** approved by user, implementation to proceed immediately

## Problem

`src/tools.py` exposes a Tavily-backed `web_search` LangChain `@tool` and ships with four unit tests. `app/app.py:128-134` already renders an "Enable web search (Tavily)" checkbox in the RAG tab's sidebar, and the checkbox is even disabled when `TAVILY_API_KEY` is unset. However, the checkbox value (`st.session_state["rag_tools"]`) is **never read** — `pipeline.answer(query)` at `app/app.py:148` ignores it. `RAGPipeline.answer()` has no tool-calling path at all.

`results/milestone2_discussion.md:60` explicitly flagged this as unfinished: *"The tool is not wired into the LCEL chain yet — exposing it to the LLM via `bind_tools` so the model can decide when to call it is a natural extension beyond Milestone 2."*

Goal: make the checkbox functional — when on, the web search runs and its snippets show up in the LLM's prompt context and the UI.

## Decisions (resolved during brainstorming)

1. **User-gated, not agentic.** The checkbox is the decision point. When on, `web_search(query)` runs unconditionally; when off, it does not. No `bind_tools` loop. Rationale: deterministic, demoable, uniform across all three LLM backends (HF Inference, Ollama Qwen, Ollama Llama) regardless of tool-calling fidelity.
2. **Labeled prompt block + separate UI section.** A new `Web Context:` block sits below the existing `Context:` block in the prompt, with snippets cited as `[W1]`, `[W2]`, … distinct from product `[B…]` ASIN citations. The UI renders a "Web Sources (Tavily)" `st.expander` below the product Sources.
3. **Fail soft on Tavily errors.** A try/except inside the pipeline catches any exception from the Tavily call, produces a `web_warning` string, and proceeds with RAG-only context. The app surfaces the warning via `st.warning` but still renders the RAG answer. No retry. `structured_json` prompt variant is unchanged — its JSON schema stays tight; web text may influence its `reasoning` field but gets no new schema key.
4. **Shape 1 implementation.** Changes live in `src/rag_pipeline.py` (new param + return keys), `src/prompts.py` (new template variable + helper), `src/tools.py` (new snippet-list helper alongside the existing `@tool`), `app/app.py` (read checkbox, render warning/expander).
5. **Tavily client returns a list; keep it structured.** Introduce `web_search_snippets(query, max_results) -> list[str]` in `src/tools.py` so the pipeline gets snippet boundaries cleanly for `[Wn]` labelling. The existing `@tool` `web_search` stays untouched for LangChain/notebook compatibility.

## Architecture

```
User query ─► Streamlit RAG tab
                │
                ▼
     [checkbox rag_tools?]
        │              │
       no             yes
        │              │
        │              ▼
        │      web_search_snippets(query)  ◄── try/except → warning on fail
        │              │
        ▼              ▼
     retriever     list[str]
        │              │
        └──────┬───────┘
               ▼
     prompt.invoke({context, web_context, question})
               │
               ▼
              LLM ─► answer (with [B…] and [W…] citations)
               │
               ▼
     UI: Answer · Sources (product cards) · Web Sources (expander) | Warning
```

Retriever and web search run independently; both complete before prompt assembly. The pipeline's existing LCEL chain is preserved in spirit but collapsed — today the retriever is invoked twice (directly in `answer()` and again inside `self.chain`); the new `answer()` runs it once and assembles the prompt input dict explicitly.

## Components

### `src/tools.py`

Add a new helper alongside the existing `@tool`:

```python
def web_search_snippets(query: str, max_results: int = 3) -> list[str]:
    """Return Tavily snippets as a list. Empty list when TAVILY_API_KEY
    is unset or the Tavily client is unavailable."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or not _HAS_TAVILY:
        return []
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=max_results)
    return [r.get("content", "") for r in results.get("results", []) if r.get("content")]
```

The existing `web_search` `@tool` and `TOOLS` list are unchanged.

### `src/prompts.py`

1. Add a helper:

```python
def build_web_context(snippets: Iterable[str]) -> str:
    """Format Tavily snippets into a labeled prompt block with [Wn] citations.
    Returns '' when snippets is empty so the block collapses cleanly."""
    items = [s for s in snippets if s]
    if not items:
        return ""
    labeled = "\n".join(f"[W{i}] {s}" for i, s in enumerate(items, start=1))
    return f"\nWeb Context:\n{labeled}\n"
```

2. Extend the user template:

```python
_USER_TEMPLATE = (
    "Context:\n{context}\n"
    "{web_context}"
    "\nQuestion: {question}\n\nAnswer:"
)
```

When `web_context == ""`, the template produces a clean Context → Question flow identical to today. When populated, it produces a labeled Web Context block between them.

3. System-prompt additions for two variants:
- `strict_citation` gains: *"If Web Context is provided, cite web snippets as [W1], [W2], …"*
- `helpful_shopper` gains: *"If Web Context is provided, you may use it for ingredient, safety, or recency details."*
- `structured_json` is unchanged (schema stays tight; `web_references` is *not* added to the JSON keys).

### `src/rag_pipeline.py`

Rework `RAGPipeline.__init__` to build the prompt→llm→parse chain without the retriever front-stage:

```python
self.chain = self.prompt | self.llm | StrOutputParser()
```

Rework `answer()`:

```python
def answer(self, query: str, use_web_search: bool = False) -> dict:
    from src.prompts import build_web_context
    from src.tools import web_search_snippets

    docs = self.retriever.invoke(query)
    web_sources: list[str] = []
    web_warning: str | None = None
    if use_web_search:
        try:
            web_sources = web_search_snippets(query)
        except Exception as exc:  # noqa: BLE001
            web_warning = f"Web search failed: {exc}"

    text = self.chain.invoke({
        "context": build_context(docs),
        "web_context": build_web_context(web_sources),
        "question": query,
    })
    return {
        "answer": text,
        "sources": [{"page_content": d.page_content, **d.metadata} for d in docs],
        "web_sources": web_sources,
        "web_warning": web_warning,
    }
```

The off path returns `web_sources=[]`, `web_warning=None`. The return-dict shape is a superset of today's — the app's existing rendering is unaffected for the off path.

### `app/app.py`

Three line-level edits inside `_render_rag_tab`:

1. Read the checkbox value:
   ```python
   use_web_search = st.session_state.get("rag_tools", False)
   ```
2. Pass to the pipeline:
   ```python
   result = pipeline.answer(query, use_web_search=use_web_search)
   ```
3. After the Sources section, render:
   ```python
   if result.get("web_warning"):
       st.warning(result["web_warning"])
   elif result.get("web_sources"):
       with st.expander("Web Sources (Tavily)"):
           for i, snippet in enumerate(result["web_sources"], start=1):
               st.markdown(f"**[W{i}]** {snippet}")
   ```

The `@st.cache_resource` on `get_rag_pipeline` is unchanged — `use_web_search` is a per-call arg, not a construction param.

## Data flow

1. User submits in the RAG tab with the toggle on.
2. `_render_rag_tab` reads `st.session_state["rag_tools"]`, calls `pipeline.answer(query, use_web_search=True)`.
3. `self.retriever.invoke(query)` → `list[Document]`.
4. `web_search_snippets(query, max_results=3)` → `list[str]` (or `[]` on missing key, or raises → caught).
5. `build_context(docs)` + `build_web_context(snippets)` → two strings.
6. `self.chain.invoke({context, web_context, question})` → LLM output string.
7. Return `{answer, sources, web_sources, web_warning}`.
8. App renders Answer + Product Sources + (Warning | Web Sources expander).

Off path: step 4 skipped, `web_sources=[]`, `web_warning=None`, step 8's web section collapses.

## Error handling

| Condition | Pipeline behavior | UI behavior |
|---|---|---|
| Toggle off | No web call | Identical to today |
| Toggle on, `TAVILY_API_KEY` missing | `web_search_snippets` returns `[]` | Checkbox already disabled — unreachable in practice |
| Toggle on, Tavily returns no hits | `web_sources=[]`, `web_warning=None` | No web section rendered |
| Toggle on, Tavily raises | `web_warning="Web search failed: …"`, `web_sources=[]`, LLM still called with RAG-only context | `st.warning(...)` below Sources; RAG answer still shown |
| LLM raises | Exception propagates (today's behavior, unchanged) | Existing `st.error` path at `app/app.py:149-154` |

Explicitly **not** included: retry, circuit-breaker, rate-limit bookkeeping, cascading failures to `st.error` on web failure. The answer is still grounded in reviews, so a web outage should not break the query.

## Testing

Coverage bar: `pytest --cov-fail-under=90` (project-enforced per user memory). Every new branch gets a corresponding test. All tests mock external calls — no network.

### `tests/test_tools.py` (add)

- `test_web_search_snippets_returns_empty_list_when_api_key_missing` — delete `TAVILY_API_KEY`, assert `web_search_snippets("q") == []`.
- `test_web_search_snippets_returns_list_when_tavily_returns_results` — monkeypatch `TAVILY_API_KEY`, `patch("src.tools.TavilyClient")` with a fake whose `.search` returns three results, assert `web_search_snippets("retinol", max_results=3) == ["snippet one", "snippet two", "snippet three"]` and the client was constructed with the key.
- `test_web_search_snippets_returns_empty_list_when_results_missing` — fake client returns `{"results": []}`, assert `[]`.

### `tests/test_prompts.py` (add + one update)

- `test_build_web_context_returns_empty_string_for_empty_list` — `build_web_context([]) == ""`.
- `test_build_web_context_labels_snippets_with_w_citations` — input `["alpha", "beta"]`, assert output contains `"[W1] alpha"` and `"[W2] beta"` and begins/ends with newline wrapping.
- `test_build_web_context_drops_empty_snippets` — input `["a", "", "b"]`, assert labels `[W1] a`, `[W2] b` (no `[W3]` for the empty).
- `test_each_prompt_variant_accepts_web_context_variable` — calls `format_messages(context="x", web_context="", question="y")` on all three; asserts messages render and the user content does NOT contain `"Web Context:"`.
- `test_each_prompt_variant_renders_web_context_when_populated` — same call with `web_context="\nWeb Context:\n[W1] snip\n"`; asserts user content contains `"Web Context:"` and `"[W1] snip"`.
- `test_strict_citation_system_message_explains_w_citation_format` — asserts `"[W1]"` appears in the updated system prompt.
- **Update** existing `test_each_prompt_variant_renders_context_and_question` to pass `web_context=""` alongside `context` and `question`.

### `tests/test_rag.py` (add + one update)

- `test_rag_pipeline_answer_without_web_search_has_empty_web_fields` — existing stub retrievers + `FakeListChatModel`; call `pipeline.answer("q")`; assert `result["web_sources"] == []` and `result["web_warning"] is None`.
- `test_rag_pipeline_answer_with_web_search_includes_snippets` — `patch("src.rag_pipeline.web_search_snippets", return_value=["s1", "s2"])`; call `pipeline.answer("q", use_web_search=True)`; assert `result["web_sources"] == ["s1", "s2"]`, `result["web_warning"] is None`, and the answer text matches the fake LLM response.
- `test_rag_pipeline_answer_catches_web_search_exception_and_warns` — `patch("src.rag_pipeline.web_search_snippets", side_effect=RuntimeError("tavily boom"))`; assert `"tavily boom"` appears in `result["web_warning"]`, `result["web_sources"] == []`, and `result["answer"]` is still the fake LLM response.
- **Update** existing `test_rag_pipeline_answer_returns_text_and_sources` to also assert the two new keys exist with default empty values.

### Not in scope (manual verification only)

`app/app.py` has no existing unit-test harness; adding one is out of scope. Three lines change there. Manual smoke matrix:

| Toggle | `TAVILY_API_KEY` | Expected |
|---|---|---|
| off | anything | No web block, identical to today |
| on | missing | Checkbox disabled; unreachable |
| on | valid, Tavily returns hits | Answer cites `[W1]`/`[W2]`; Web Sources expander visible |
| on | valid, Tavily raises | `st.warning` shown; answer from reviews only |

## Out of scope

- Agentic tool calling via `bind_tools` / the LLM choosing when to invoke (may be a future milestone).
- Schema changes to `structured_json` (no `web_references` key).
- Retry / circuit-breaker / rate-limit handling.
- Streamlit UI unit tests.
- Caching Tavily results across queries (each submission hits the live API when toggle is on).
- Refactoring the product-card renderer to also display web snippets as cards.
