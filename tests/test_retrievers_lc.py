import pytest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from unittest.mock import MagicMock

from src.retrievers_lc import BM25LCRetriever, SemanticLCRetriever, build_ensemble_retriever, wrap_retriever


def test_bm25_lc_returns_documents_with_m1_metadata():
    underlying = MagicMock()
    underlying.search.return_value = [
        {
            "parent_asin": "B001",
            "title": "Vitamin C Serum",
            "text": "brightening serum...",
            "price": 24.99,
            "average_rating": 4.5,
            "score": 12.34,
        },
    ]
    retriever = BM25LCRetriever(underlying=underlying, top_k=5)

    docs = retriever.invoke("vitamin c")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "brightening serum..."
    assert docs[0].metadata == {
        "parent_asin": "B001",
        "title": "Vitamin C Serum",
        "price": 24.99,
        "average_rating": 4.5,
        "score": 12.34,
    }
    underlying.search.assert_called_once_with("vitamin c", top_k=5)


def test_semantic_lc_returns_documents_with_m1_metadata():
    underlying = MagicMock()
    underlying.search.return_value = [
        {
            "parent_asin": "B003",
            "title": "SPF 50 Sunscreen",
            "text": "broad spectrum sunscreen...",
            "price": 15.99,
            "average_rating": 4.8,
            "score": 0.87,
        },
    ]
    retriever = SemanticLCRetriever(underlying=underlying, top_k=3)

    docs = retriever.invoke("sunscreen")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "broad spectrum sunscreen..."
    assert docs[0].metadata == {
        "parent_asin": "B003",
        "title": "SPF 50 Sunscreen",
        "price": 15.99,
        "average_rating": 4.8,
        "score": 0.87,
    }
    underlying.search.assert_called_once_with("sunscreen", top_k=3)


def test_build_ensemble_returns_ensemble_with_two_retrievers_and_weights():
    bm25 = MagicMock()
    bm25.search.return_value = [
        {"parent_asin": "B001", "title": "A", "text": "a", "price": 1.0,
         "average_rating": 4.0, "score": 5.0},
    ]
    semantic = MagicMock()
    semantic.search.return_value = [
        {"parent_asin": "B002", "title": "B", "text": "b", "price": 2.0,
         "average_rating": 4.5, "score": 0.9},
    ]

    ensemble = build_ensemble_retriever(bm25, semantic, weights=(0.4, 0.6), top_k=2)

    assert isinstance(ensemble, EnsembleRetriever)
    assert ensemble.weights == [0.4, 0.6]
    assert len(ensemble.retrievers) == 2

    docs = ensemble.invoke("test")
    asins = {d.metadata["parent_asin"] for d in docs}
    assert asins == {"B001", "B002"}
    bm25.search.assert_called_once_with("test", top_k=2)
    semantic.search.assert_called_once_with("test", top_k=2)


def test_wrap_retriever_dispatches_by_name():
    bm25 = MagicMock()
    semantic = MagicMock()

    assert isinstance(wrap_retriever("BM25", bm25, semantic), BM25LCRetriever)
    assert isinstance(wrap_retriever("Semantic", bm25, semantic), SemanticLCRetriever)
    assert isinstance(wrap_retriever("Hybrid", bm25, semantic), EnsembleRetriever)


def test_wrap_retriever_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown retriever name"):
        wrap_retriever("Bogus", MagicMock(), MagicMock())
