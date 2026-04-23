from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def _stub_retrievers():
    bm25 = MagicMock()
    bm25.search.return_value = [
        {
            "parent_asin": "B001",
            "title": "Vitamin C Serum",
            "text": "brightening serum with vitamin C",
            "price": 24.99,
            "average_rating": 4.5,
            "score": 12.3,
        }
    ]
    semantic = MagicMock()
    semantic.search.return_value = [
        {
            "parent_asin": "B002",
            "title": "Pure Vitamin C",
            "text": "pure vitamin C concentrate",
            "price": 30.0,
            "average_rating": 4.7,
            "score": 0.91,
        }
    ]
    return bm25, semantic


def test_rag_pipeline_requires_hf_token_when_no_llm_injected(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from src.rag_pipeline import RAGPipeline

    bm25 = MagicMock()
    semantic = MagicMock()

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        RAGPipeline(bm25=bm25, semantic=semantic)


def test_rag_pipeline_answer_returns_text_and_sources():
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["Use [B001] for brightening."])

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(bm25=bm25, semantic=semantic, retriever_name="BM25", llm=fake_llm)
    result = pipeline.answer("vitamin c")

    assert result["answer"] == "Use [B001] for brightening."
    assert len(result["sources"]) == 1
    src = result["sources"][0]
    assert src["parent_asin"] == "B001"
    assert src["title"] == "Vitamin C Serum"
    assert src["page_content"] == "brightening serum with vitamin C"
    assert result["web_sources"] == []
    assert result["web_warning"] is None


def test_rag_pipeline_answer_without_web_search_skips_tool_call(monkeypatch):
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["ok"])

    from src import rag_pipeline as rag_pipeline_module
    from src.rag_pipeline import RAGPipeline

    def _boom(*args, **kwargs):
        raise AssertionError("web_search_snippets should not be called when toggle is off")

    monkeypatch.setattr(rag_pipeline_module, "web_search_snippets", _boom)

    pipeline = RAGPipeline(bm25=bm25, semantic=semantic, retriever_name="BM25", llm=fake_llm)
    result = pipeline.answer("vitamin c")

    assert result["web_sources"] == []
    assert result["web_warning"] is None


def test_rag_pipeline_answer_with_web_search_includes_snippets(monkeypatch):
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["answer with [W1]"])

    from src import rag_pipeline as rag_pipeline_module
    from src.rag_pipeline import RAGPipeline

    monkeypatch.setattr(
        rag_pipeline_module, "web_search_snippets", lambda q: ["snippet alpha", "snippet beta"]
    )

    pipeline = RAGPipeline(bm25=bm25, semantic=semantic, retriever_name="BM25", llm=fake_llm)
    result = pipeline.answer("vitamin c", use_web_search=True)

    assert result["web_sources"] == ["snippet alpha", "snippet beta"]
    assert result["web_warning"] is None
    assert result["answer"] == "answer with [W1]"


def test_rag_pipeline_answer_passes_web_context_into_prompt(monkeypatch):
    bm25, semantic = _stub_retrievers()

    captured = {}

    class _CapturingLLM(FakeListChatModel):
        def invoke(self, messages, *args, **kwargs):
            captured["messages"] = messages
            return super().invoke(messages, *args, **kwargs)

    fake_llm = _CapturingLLM(responses=["ok"])

    from src import rag_pipeline as rag_pipeline_module
    from src.rag_pipeline import RAGPipeline

    monkeypatch.setattr(rag_pipeline_module, "web_search_snippets", lambda q: ["fresh data"])

    pipeline = RAGPipeline(bm25=bm25, semantic=semantic, retriever_name="BM25", llm=fake_llm)
    pipeline.answer("q", use_web_search=True)

    rendered = captured["messages"]
    user_text = (
        rendered.messages[-1].content if hasattr(rendered, "messages") else rendered[-1].content
    )
    assert "Web Context:" in user_text
    assert "[W1] fresh data" in user_text


def test_rag_pipeline_answer_catches_web_search_exception_and_warns(monkeypatch):
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["fallback answer"])

    from src import rag_pipeline as rag_pipeline_module
    from src.rag_pipeline import RAGPipeline

    def _raise(q):
        raise RuntimeError("tavily boom")

    monkeypatch.setattr(rag_pipeline_module, "web_search_snippets", _raise)

    pipeline = RAGPipeline(bm25=bm25, semantic=semantic, retriever_name="BM25", llm=fake_llm)
    result = pipeline.answer("vitamin c", use_web_search=True)

    assert result["web_sources"] == []
    assert result["web_warning"] is not None
    assert "tavily boom" in result["web_warning"]
    assert result["answer"] == "fallback answer"


def test_rag_pipeline_uses_selected_prompt_variant():
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["x"])

    from src.rag_pipeline import RAGPipeline

    strict = RAGPipeline(
        bm25=bm25,
        semantic=semantic,
        retriever_name="BM25",
        prompt_name="strict_citation",
        llm=fake_llm,
    )
    json_pipe = RAGPipeline(
        bm25=bm25,
        semantic=semantic,
        retriever_name="BM25",
        prompt_name="structured_json",
        llm=fake_llm,
    )

    assert strict.prompt is not json_pipe.prompt
    s_sys = strict.prompt.format_messages(context="x", web_context="", question="y")[0].content
    j_sys = json_pipe.prompt.format_messages(context="x", web_context="", question="y")[0].content
    assert "ASIN" in s_sys
    assert "JSON" in j_sys.upper()


def test_rag_pipeline_rejects_unknown_prompt_name():
    bm25, semantic = _stub_retrievers()
    fake_llm = FakeListChatModel(responses=["x"])

    from src.rag_pipeline import RAGPipeline

    with pytest.raises(KeyError):
        RAGPipeline(
            bm25=bm25,
            semantic=semantic,
            retriever_name="BM25",
            prompt_name="bogus",
            llm=fake_llm,
        )


def test_load_llm_uses_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    from unittest.mock import patch

    from src.rag_pipeline import load_llm

    with (
        patch("src.rag_pipeline.HuggingFaceEndpoint") as endpoint_cls,
        patch("src.rag_pipeline.ChatHuggingFace") as chat_cls,
    ):
        endpoint_cls.return_value = MagicMock()
        chat_cls.return_value = MagicMock()

        load_llm(model_id="some/model", max_new_tokens=100, provider="auto")

    endpoint_cls.assert_called_once()
    kwargs = endpoint_cls.call_args.kwargs
    assert kwargs["repo_id"] == "some/model"
    assert kwargs["huggingfacehub_api_token"] == "fake-token"
    assert kwargs["max_new_tokens"] == 100
    chat_cls.assert_called_once()
