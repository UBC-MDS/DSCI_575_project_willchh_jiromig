from src.bm25 import BM25Retriever


def test_build_index(fake_corpus):
    """Building an index populates the retriever with the expected doc count."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    assert retriever.index is not None
    assert len(retriever.corpus) == 10


def test_search_returns_correct_shape(fake_corpus):
    """Search results expose parent_asin, title, and score for each hit."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    results = retriever.search("vitamin C serum", top_k=3)
    assert len(results) == 3
    for result in results:
        assert "parent_asin" in result
        assert "title" in result
        assert "score" in result


def test_search_scores_sorted_descending(fake_corpus):
    """Search results are returned in descending order by score."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    results = retriever.search("coconut oil shampoo", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_relevant_result_first(fake_corpus):
    """The obvious keyword match ranks first."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    results = retriever.search("coconut oil shampoo", top_k=3)
    assert results[0]["parent_asin"] == "B002"


def test_search_top_k_limits_results(fake_corpus):
    """top_k caps the number of returned results."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    results = retriever.search("skin", top_k=2)
    assert len(results) == 2


def test_save_and_load(fake_corpus, tmp_path):
    """A BM25 index survives a save/load round trip with identical top results."""
    retriever = BM25Retriever()
    retriever.build_index(fake_corpus)
    save_path = str(tmp_path / "bm25_index.pkl")
    retriever.save(save_path)

    loaded = BM25Retriever()
    loaded.load(save_path)
    results = loaded.search("vitamin C serum", top_k=3)
    assert len(results) == 3
    assert results[0]["parent_asin"] == "B001"
