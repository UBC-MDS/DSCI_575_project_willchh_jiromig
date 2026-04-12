import pytest

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.semantic import SemanticRetriever


@pytest.fixture()
def hybrid_retriever(fake_corpus):
    bm25 = BM25Retriever()
    bm25.build_index(fake_corpus)
    semantic = SemanticRetriever()
    semantic.build_index(fake_corpus)
    return HybridRetriever(bm25, semantic)


def test_search_returns_correct_shape(hybrid_retriever):
    results = hybrid_retriever.search("vitamin C serum", top_k=3)
    assert len(results) == 3
    for result in results:
        assert "parent_asin" in result
        assert "title" in result
        assert "score" in result


def test_search_scores_sorted_descending(hybrid_retriever):
    results = hybrid_retriever.search("moisturizer for dry skin", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_full_bm25_weight(hybrid_retriever):
    hybrid_results = hybrid_retriever.search("coconut oil shampoo", top_k=3, bm25_weight=1.0)
    assert hybrid_results[0]["parent_asin"] == "B002"


def test_full_semantic_weight(hybrid_retriever):
    hybrid_results = hybrid_retriever.search(
        "something to protect from sun damage", top_k=3, bm25_weight=0.0
    )
    top_asins = [r["parent_asin"] for r in hybrid_results]
    assert "B003" in top_asins


def test_default_weight_is_balanced(hybrid_retriever):
    results = hybrid_retriever.search("face care", top_k=5)
    assert len(results) == 5
    scores = [r["score"] for r in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
