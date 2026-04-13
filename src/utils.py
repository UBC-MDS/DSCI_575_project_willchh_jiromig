import gzip
import json
import re
from pathlib import Path

import duckdb
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    """Lowercase, split hyphens, remove punctuation, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return [LEMMATIZER.lemmatize(t) for t in text.split() if t not in STOPWORDS]


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
    """Concatenate title + description + features + reviews into a single text field."""
    parts = []
    title = product.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title)
    parts.extend(_to_str_list(product.get("description")))
    parts.extend(_to_str_list(product.get("features")))
    reviews_text = product.get("reviews_text", "")
    if isinstance(reviews_text, str) and reviews_text.strip():
        parts.append(reviews_text)
    return " ".join(parts)


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
CATEGORY = "All_Beauty"


def download_raw_data(data_raw_dir: str) -> None:
    """Download metadata and reviews from McAuley Lab and save as parquet."""
    data_raw = Path(data_raw_dir)
    data_raw.mkdir(parents=True, exist_ok=True)

    meta_url = f"{BASE_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"
    reviews_url = f"{BASE_URL}/review_categories/{CATEGORY}.jsonl.gz"

    con = duckdb.connect()

    meta_path = data_raw / f"meta_{CATEGORY}.parquet"
    if not meta_path.exists():
        print(f"Downloading metadata to {meta_path}...")
        con.execute(
            f"""
            COPY (SELECT * FROM read_json_auto('{meta_url}'))
            TO '{meta_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        )
        print("Metadata download complete.")
    else:
        print(f"Already exists, skipping: {meta_path}")

    reviews_path = data_raw / f"reviews_{CATEGORY}.parquet"
    if not reviews_path.exists():
        print(f"Downloading reviews to {reviews_path}...")
        con.execute(
            f"""
            COPY (SELECT * FROM read_json_auto('{reviews_url}'))
            TO '{reviews_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        )
        print("Reviews download complete.")
    else:
        print(f"Already exists, skipping: {reviews_path}")

    con.close()


def build_processed_corpus(data_raw_dir: str, data_processed_dir: str) -> None:
    """Load raw parquets, aggregate reviews, build corpus, and save."""
    data_raw = Path(data_raw_dir)
    data_processed = Path(data_processed_dir)
    data_processed.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_parquet(data_raw / f"meta_{CATEGORY}.parquet")
    reviews_df = pd.read_parquet(data_raw / f"reviews_{CATEGORY}.parquet")

    # Keep only the most helpful review per product
    reviews_sorted = reviews_df.sort_values("helpful_vote", ascending=False)
    reviews_agg = (
        reviews_sorted.groupby("parent_asin")["text"]
        .apply(lambda texts: " ".join(t for t in texts.dropna().head(1) if str(t).strip()))
        .reset_index()
        .rename(columns={"text": "reviews_text"})
    )
    reviews_lookup = dict(zip(reviews_agg["parent_asin"], reviews_agg["reviews_text"]))

    records = []
    for _, row in meta_df.iterrows():
        product = row.to_dict()
        asin = product.get("parent_asin", "")
        product["reviews_text"] = reviews_lookup.get(asin, "")
        text = build_text(product)
        if not text.strip():
            continue
        records.append(
            {
                "parent_asin": asin,
                "title": product.get("title", ""),
                "text": text,
                "price": product.get("price"),
                "average_rating": product.get("average_rating"),
            }
        )

    corpus_path = data_processed / "product_corpus.parquet"
    pd.DataFrame(records).to_parquet(corpus_path, compression="zstd", index=False)
    print(f"Corpus saved: {len(records):,} products to {corpus_path}")


def build_indices(corpus_path: str, indices_dir: str) -> None:
    """Build BM25 and FAISS indices from a processed corpus and save to disk."""
    from src.bm25 import BM25Retriever
    from src.semantic import SemanticRetriever

    indices = Path(indices_dir)
    indices.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_path)
    print(f"Loaded corpus: {len(corpus):,} products")

    bm25_path = indices / "bm25_index.pkl"
    if not bm25_path.exists():
        print("Building BM25 index...")
        bm25 = BM25Retriever()
        bm25.build_index(corpus)
        bm25.save(str(bm25_path))
        print(f"BM25 index saved to {bm25_path}")
    else:
        print(f"Already exists, skipping: {bm25_path}")

    faiss_path = indices / "faiss_index"
    if not faiss_path.exists():
        print("Building semantic (FAISS) index...")
        semantic = SemanticRetriever()
        semantic.build_index(corpus)
        semantic.save(str(faiss_path))
        print(f"Semantic index saved to {faiss_path}")
    else:
        print(f"Already exists, skipping: {faiss_path}")


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
