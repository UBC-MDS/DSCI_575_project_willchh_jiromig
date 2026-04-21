from typing import Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

MAX_REVIEW_CHARS = 300


def build_context(docs: Iterable[Document], max_chars: int = MAX_REVIEW_CHARS) -> str:
    """Format retrieved documents into a numbered, prompt-ready context block."""
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


DEFAULT_PROMPT_NAME = "strict_citation"

_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

PROMPT_VARIANTS: dict[str, ChatPromptTemplate] = {
    "strict_citation": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an Amazon Beauty product assistant. Answer ONLY using the "
                "provided reviews and metadata. Cite the product ASIN like [B001] for "
                "every claim. If the context is insufficient, reply: "
                "'I don't have enough information.'",
            ),
            ("user", _USER_TEMPLATE),
        ]
    ),
    "helpful_shopper": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly Amazon shopping assistant. Use the reviews to "
                "recommend 1-2 products. Mention price and rating when relevant.",
            ),
            ("user", _USER_TEMPLATE),
        ]
    ),
    "structured_json": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a product recommendation engine. Respond as a JSON object "
                "with keys 'recommendation' (string), 'reasoning' (string), and "
                "'asins' (list of ASIN strings). Output ONLY valid JSON, no preamble.",
            ),
            ("user", _USER_TEMPLATE),
        ]
    ),
}
