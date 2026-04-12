import pytest
from src.semantic import SemanticRetriever


@pytest.fixture()
def semantic_retriever(fake_corpus):
    retriever = SemanticRetriever()
    retriever.build_index(fake_corpus)
    return retriever


def test_build_index(semantic_retriever):
    assert semantic_retriever.faiss_index is not None
    assert semantic_retriever.faiss_index.ntotal == 10


def test_search_returns_correct_shape(semantic_retriever):
    results = semantic_retriever.search("vitamin C serum", top_k=3)
    assert len(results) == 3
    for result in results:
        assert "parent_asin" in result
        assert "title" in result
        assert "score" in result


def test_search_scores_sorted_descending(semantic_retriever):
    results = semantic_retriever.search("moisturizer for oily skin", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_semantic_relevance(semantic_retriever):
    results = semantic_retriever.search("something to protect from sun damage", top_k=3)
    top_asins = [r["parent_asin"] for r in results]
    assert "B003" in top_asins


def test_search_top_k_limits_results(semantic_retriever):
    results = semantic_retriever.search("hair treatment", top_k=2)
    assert len(results) == 2


def test_save_and_load(fake_corpus, tmp_path):
    retriever = SemanticRetriever()
    retriever.build_index(fake_corpus)
    save_dir = str(tmp_path / "faiss_index")
    retriever.save(save_dir)

    loaded = SemanticRetriever()
    loaded.load(save_dir)
    results = loaded.search("vitamin C serum", top_k=3)
    assert len(results) == 3
    assert results[0]["title"] == "Vitamin C Serum"
