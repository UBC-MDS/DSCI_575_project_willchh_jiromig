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
