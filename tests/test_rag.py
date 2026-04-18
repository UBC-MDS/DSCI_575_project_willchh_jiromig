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
    s_sys = strict.prompt.format_messages(context="x", question="y")[0].content
    j_sys = json_pipe.prompt.format_messages(context="x", question="y")[0].content
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
