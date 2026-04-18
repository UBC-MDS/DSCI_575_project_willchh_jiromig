from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


def _to_document(result: dict) -> Document:
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: Any
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.underlying.search(query, top_k=self.top_k)
        return [_to_document(r) for r in results]


class SemanticLCRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: Any
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.underlying.search(query, top_k=self.top_k)
        return [_to_document(r) for r in results]
