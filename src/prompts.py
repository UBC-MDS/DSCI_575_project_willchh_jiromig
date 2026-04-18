from typing import Iterable

from langchain_core.documents import Document

MAX_REVIEW_CHARS = 300


def build_context(docs: Iterable[Document], max_chars: int = MAX_REVIEW_CHARS) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        text = (doc.page_content or "")[:max_chars]
        rating = meta.get("average_rating")
        price = meta.get("price")
        try:
            rating_str = f"{float(rating):.1f}/5"
        except (TypeError, ValueError):
            rating_str = "N/A"
        try:
            price_str = f"${float(price):.2f}"
        except (TypeError, ValueError):
            price_str = "N/A"
        blocks.append(
            f"[{i}] ASIN: {meta.get('parent_asin', 'N/A')} | "
            f"Title: {meta.get('title', 'N/A')} | "
            f"Rating: {rating_str} | Price: {price_str}\n"
            f"Review/Description: {text}"
        )
    return "\n\n".join(blocks)
