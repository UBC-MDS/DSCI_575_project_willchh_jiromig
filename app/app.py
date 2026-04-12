import csv
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.bm25 import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent.parent
INDICES_DIR = BASE_DIR / "indices"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.csv"


@st.cache_resource
def load_retrievers():
    """Load persisted indices and return available retrievers."""
    bm25 = BM25Retriever()
    bm25.load(str(INDICES_DIR / "bm25_index.pkl"))

    return {"BM25": bm25}

def save_feedback(query: str, mode: str, product_asin: str, feedback: str) -> None:
    """Append feedback to CSV file."""
    file_exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "mode", "product_asin", "feedback"])
        writer.writerow([datetime.now().isoformat(), query, mode, product_asin, feedback])

