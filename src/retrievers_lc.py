from typing import Any, List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


def _to_document(result: dict) -> Document:
    """Convert a raw retriever result dict to a LangChain Document."""
    text = result.get("text", "")
    metadata = {
        "parent_asin": result.get("parent_asin"),
        "title": result.get("title"),
        "price": result.get("price"),
        "average_rating": result.get("average_rating"),
        "score": result.get("score"),
    }
    return Document(page_content=text, metadata=metadata)


class BM25LCRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper around the M1 BM25Retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: Any
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Delegate to the underlying BM25 retriever and wrap results as Documents."""
        results = self.underlying.search(query, top_k=self.top_k)
        return [_to_document(r) for r in results]


class SemanticLCRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper around the M1 SemanticRetriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: Any
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Delegate to the underlying semantic retriever and wrap results as Documents."""
        results = self.underlying.search(query, top_k=self.top_k)
        return [_to_document(r) for r in results]


def build_ensemble_retriever(
    bm25_underlying: Any,
    semantic_underlying: Any,
    weights: tuple[float, float] = (0.4, 0.6),
    top_k: int = 5,
) -> EnsembleRetriever:
    """Create an EnsembleRetriever combining BM25 and semantic via RRF."""
    return EnsembleRetriever(
        retrievers=[
            BM25LCRetriever(underlying=bm25_underlying, top_k=top_k),
            SemanticLCRetriever(underlying=semantic_underlying, top_k=top_k),
        ],
        weights=list(weights),
    )


def wrap_retriever(
    name: str,
    bm25_underlying: Any,
    semantic_underlying: Any,
    top_k: int = 5,
) -> BaseRetriever:
    """Factory that returns a LangChain retriever by name (BM25/Semantic/Hybrid)."""
    if name == "BM25":
        return BM25LCRetriever(underlying=bm25_underlying, top_k=top_k)
    if name == "Semantic":
        return SemanticLCRetriever(underlying=semantic_underlying, top_k=top_k)
    if name == "Hybrid":
        return build_ensemble_retriever(bm25_underlying, semantic_underlying, top_k=top_k)
    raise ValueError(f"Unknown retriever name: {name!r}")
