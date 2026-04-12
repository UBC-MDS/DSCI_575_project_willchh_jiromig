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

def display_result(result: dict, idx: int, query: str, mode: str) -> None:
    """Display a single product result as a card."""
    with st.container(border=True):
        st.subheader(result.get("title", "No title"))

        text = result.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."
        st.write(text)

        col1, col2, col3 = st.columns(3)
        with col1:
            price = result.get("price")
            st.write(f"**Price:** ${price:.2f}" if price else "**Price:** N/A")
        with col2:
            rating = result.get("average_rating")
            if rating:
                stars = "\u2605" * int(round(rating)) + "\u2606" * (5 - int(round(rating)))
                st.write(f"**Rating:** {stars} ({rating})")
            else:
                st.write("**Rating:** N/A")
        with col3:
            st.write(f"**Score:** {result.get('score', 0):.4f}")

        col_up, col_down, _ = st.columns([1, 1, 8])
        with col_up:
            if st.button("\U0001f44d", key=f"up_{idx}"):
                save_feedback(query, mode, result.get("parent_asin", ""), "positive")
                st.toast("Thanks for your feedback!")
        with col_down:
            if st.button("\U0001f44e", key=f"down_{idx}"):
                save_feedback(query, mode, result.get("parent_asin", ""), "negative")
                st.toast("Thanks for your feedback!")
