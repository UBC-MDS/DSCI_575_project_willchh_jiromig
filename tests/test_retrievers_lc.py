from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retrievers_lc import BM25LCRetriever, SemanticLCRetriever


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
    assert docs[0].metadata["parent_asin"] == "B003"
    assert docs[0].metadata["score"] == 0.87
    underlying.search.assert_called_once_with("sunscreen", top_k=3)
