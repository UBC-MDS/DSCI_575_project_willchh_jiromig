import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.utils import tokenize


class BM25Retriever:
    """Keyword-based retriever using BM25Okapi scoring."""

    def __init__(self):
        """Initialize with empty index and corpus."""
        self.index = None
        self.corpus = []
        self.tokenized_corpus = []

    def build_index(self, corpus: list[dict]) -> None:
        """Build BM25 index from corpus."""
        self.corpus = corpus
        self.tokenized_corpus = [tokenize(doc["text"]) for doc in corpus]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index and return top-k results with scores."""
        tokenized_query = tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            result = {**self.corpus[idx], "score": float(scores[idx])}
            results.append(result)
        return results

    def save(self, path: str) -> None:
        """Persist the BM25 index and corpus to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "corpus": self.corpus,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load a persisted BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.corpus = data["corpus"]
        self.tokenized_corpus = data["tokenized_corpus"]
