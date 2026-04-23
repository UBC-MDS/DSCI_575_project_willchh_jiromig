from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic scores with min-max normalization."""

    def __init__(self, bm25: BM25Retriever, semantic: SemanticRetriever):
        """Initialize with BM25 and semantic retriever instances."""
        self.bm25 = bm25
        self.semantic = semantic

    def search(self, query: str, top_k: int = 5, bm25_weight: float = 0.5) -> list[dict]:
        """Run both retrievers, normalize scores, combine with weighted sum."""
        n_candidates = min(top_k * 3, len(self.bm25.corpus))

        bm25_results = self.bm25.search(query, top_k=n_candidates)
        semantic_results = self.semantic.search(query, top_k=n_candidates)

        bm25_scores = self._normalize([r["score"] for r in bm25_results])
        semantic_scores = self._normalize([r["score"] for r in semantic_results])

        combined = {}
        for result, score in zip(bm25_results, bm25_scores):
            asin = result["parent_asin"]
            combined[asin] = {**result, "score": bm25_weight * score}

        semantic_weight = 1.0 - bm25_weight
        for result, score in zip(semantic_results, semantic_scores):
            asin = result["parent_asin"]
            if asin in combined:
                combined[asin]["score"] += semantic_weight * score
            else:
                combined[asin] = {**result, "score": semantic_weight * score}

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """Min-max normalize a list of scores to [0, 1]."""
        if not scores:
            return scores
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
