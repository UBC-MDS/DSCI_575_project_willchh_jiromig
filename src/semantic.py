import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.faiss_index = None
        self.corpus = []

    def build_index(self, corpus: list[dict]) -> None:
        """Encode all documents and build a FAISS inner-product index."""
        self.corpus = corpus
        texts = [doc["text"] for doc in corpus]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Encode query, search FAISS index, return top-k results with scores."""
        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            result = {**self.corpus[idx], "score": float(score)}
            results.append(result)
        return results

    def save(self, directory: str) -> None:
        """Save FAISS index and corpus to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(path / "index.faiss"))
        with open(path / "corpus.pkl", "wb") as f:
            pickle.dump(self.corpus, f)

    def load(self, directory: str) -> None:
        """Load FAISS index and corpus from disk."""
        path = Path(directory)
        self.faiss_index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "corpus.pkl", "rb") as f:
            self.corpus = pickle.load(f)
