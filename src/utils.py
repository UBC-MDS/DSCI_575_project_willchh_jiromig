import gzip
import json
import re
from pathlib import Path

import pandas as pd


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _to_str_list(value) -> list[str]:
    """Convert a field value to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if v and str(v).strip()]
    return []


def build_text(product: dict) -> str:
    """Concatenate title + description + features into a single text field."""
    parts = []
    title = product.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title)
    parts.extend(_to_str_list(product.get("description")))
    parts.extend(_to_str_list(product.get("features")))
    return " ".join(parts)


def load_metadata(filepath: str) -> list[dict]:
    """Load product metadata from a gzipped JSONL file and build corpus."""
    corpus = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            product = json.loads(line)
            text = build_text(product)
            if not text.strip():
                continue
            corpus.append(
                {
                    "parent_asin": product.get("parent_asin", ""),
                    "title": product.get("title", ""),
                    "text": text,
                    "price": product.get("price"),
                    "average_rating": product.get("average_rating"),
                    "images": product.get("images", []),
                }
            )
    return corpus


def save_corpus(corpus: list[dict], filepath: str) -> None:
    """Save corpus to a parquet file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(corpus).to_parquet(filepath, compression="zstd", index=False)


def load_corpus(filepath: str) -> list[dict]:
    """Load corpus from a parquet file."""
    return pd.read_parquet(filepath).to_dict(orient="records")
