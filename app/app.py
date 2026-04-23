import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.prompts import PROMPT_VARIANTS
from src.semantic import SemanticRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
INDICES_DIR = BASE_DIR / "indices"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SAMPLE_PLACEHOLDER = "— choose a sample query —"

WEB_DEMO_QUERIES = [
    "best drugstore sunscreen 2026 dermatologist recommended",
    "best retinol cream 2026 under $30",
    "best vitamin C serum 2026 for hyperpigmentation",
]


@st.cache_data
def load_search_samples() -> dict[str, str]:
    """Load curated retrieval queries from ground_truth.csv, labeled by difficulty."""
    path = PROCESSED_DIR / "ground_truth.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[f"[{row['difficulty']}] {row['query']}"] = row["query"]
    return out


@st.cache_data
def load_rag_samples() -> dict[str, str]:
    """Load curated RAG queries plus product-focused web-demo queries."""
    path = PROCESSED_DIR / "rag_queries.csv"
    out: dict[str, str] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[f"[{row['category']}] {row['query']}"] = row["query"]
    for q in WEB_DEMO_QUERIES:
        out[f"[web-demo] {q}"] = q
    return out


def _apply_sample(mapping: dict[str, str], target_key: str, selector_key: str) -> None:
    """on_change callback: copy the selected sample query into the text-input state."""
    choice = st.session_state.get(selector_key)
    if choice and choice in mapping:
        st.session_state[target_key] = mapping[choice]


@st.cache_resource
def load_retrievers():
    """Load persisted indices and return available retrievers."""
    bm25 = BM25Retriever()
    bm25.load(str(INDICES_DIR / "bm25_index.pkl"))

    semantic = SemanticRetriever()
    semantic.load(str(INDICES_DIR / "faiss_index"))

    hybrid = HybridRetriever(bm25, semantic)

    return {"BM25": bm25, "Semantic": semantic, "Hybrid": hybrid}


@st.cache_resource
def get_rag_pipeline(retriever_name: str, prompt_name: str, top_k: int):
    """Build the RAG pipeline lazily so HF_TOKEN is only required on query submit."""
    from src.rag_pipeline import RAGPipeline

    retrievers = load_retrievers()
    return RAGPipeline(
        bm25=retrievers["BM25"],
        semantic=retrievers["Semantic"],
        retriever_name=retriever_name,
        prompt_name=prompt_name,
        top_k=top_k,
    )


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
            if st.button("\U0001f44d", key=f"up_{mode}_{idx}"):
                save_feedback(query, mode, result.get("parent_asin", ""), "positive")
                st.toast("Thanks for your feedback!")
        with col_down:
            if st.button("\U0001f44e", key=f"down_{mode}_{idx}"):
                save_feedback(query, mode, result.get("parent_asin", ""), "negative")
                st.toast("Thanks for your feedback!")


def _render_search_tab(retrievers: dict) -> None:
    """Render the Search tab: sidebar controls, sample selector, and results list."""
    with st.sidebar:
        st.header("Search Settings")
        mode = st.radio("Search Mode", list(retrievers.keys()), key="search_mode")
        top_k = st.slider(
            "Number of results", min_value=1, max_value=10, value=5, key="search_topk"
        )

    samples = load_search_samples()
    st.selectbox(
        "Try a sample query",
        options=[SAMPLE_PLACEHOLDER] + list(samples.keys()),
        key="search_sample",
        on_change=_apply_sample,
        args=(samples, "search_query", "search_sample"),
        help="Selecting a sample populates the search box; you can still edit it before submitting.",
    )

    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., best moisturizer for sensitive skin",
        key="search_query",
    )

    if query:
        retriever = retrievers[mode]
        results = retriever.search(query, top_k=top_k)
        st.subheader(f"Results ({mode})")
        for idx, result in enumerate(results):
            display_result(result, idx, query, mode)


def _render_rag_tab(retrievers: dict) -> None:
    """Render the RAG tab: sidebar controls, query input, answer, and sources."""
    has_tavily = bool(os.environ.get("TAVILY_API_KEY"))

    with st.sidebar:
        st.header("RAG Settings")
        retriever_name = st.radio(
            "Retriever", ["BM25", "Semantic", "Hybrid"], index=2, key="rag_retriever"
        )
        prompt_name = st.radio("Prompt variant", list(PROMPT_VARIANTS.keys()), key="rag_prompt")
        top_k = st.slider("Number of sources", min_value=1, max_value=10, value=5, key="rag_topk")
        st.checkbox(
            "Enable web search (Tavily)",
            value=False,
            disabled=not has_tavily,
            help="Set TAVILY_API_KEY in .env to enable.",
            key="rag_tools",
        )

    samples = load_rag_samples()
    st.selectbox(
        "Try a sample query",
        options=[SAMPLE_PLACEHOLDER] + list(samples.keys()),
        key="rag_sample",
        on_change=_apply_sample,
        args=(samples, "rag_query", "rag_sample"),
        help=(
            "Selecting a sample populates the question box; you can still edit it before "
            "submitting. [web-demo] queries need the Tavily toggle on to be answered well."
        ),
    )

    query = st.text_input(
        "Ask a question about Amazon Beauty products",
        placeholder="e.g., what's a good vitamin C serum under $25?",
        key="rag_query",
    )

    if not query:
        return

    use_web_search = bool(st.session_state.get("rag_tools", False))
    pipeline = get_rag_pipeline(retriever_name, prompt_name, top_k)
    with st.spinner("Generating answer..."):
        try:
            result = pipeline.answer(query, use_web_search=use_web_search)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM call failed: {exc}")
            return

    st.subheader("Answer")
    if prompt_name == "structured_json":
        try:
            parsed = json.loads(result["answer"])
            st.json(parsed)
        except json.JSONDecodeError:
            st.code(result["answer"], language="json")
    else:
        st.info(result["answer"])

    if result.get("web_warning"):
        st.warning(result["web_warning"])

    mode_label = f"rag:{retriever_name}:{prompt_name}"
    has_web = bool(result.get("web_sources")) and not result.get("web_warning")

    def _render_product_sources() -> None:
        """Render the retrieved product-source cards beneath the answer."""
        for idx, src in enumerate(result["sources"], start=1):
            result_dict = {
                "parent_asin": src.get("parent_asin"),
                "title": src.get("title"),
                "text": src.get("page_content", ""),
                "price": src.get("price"),
                "average_rating": src.get("average_rating"),
                "score": src.get("score"),
            }
            display_result(result_dict, idx, query, mode=mode_label)

    def _render_web_sources() -> None:
        """Render the Tavily snippet cards beneath the answer."""
        st.caption("Tavily snippets that fed into the answer.")
        for i, snippet in enumerate(result["web_sources"], start=1):
            with st.container(border=True):
                st.markdown(f"**Source {i}**")
                cleaned = re.sub(r"(?m)^#{1,6}\s*", "", snippet)
                st.text(cleaned)

    if has_web:
        col_web, col_prod = st.columns(2)
        with col_web:
            st.subheader("Web Sources (Tavily)")
            _render_web_sources()
        with col_prod:
            st.subheader("Product Sources")
            _render_product_sources()
    else:
        st.subheader("Sources")
        _render_product_sources()


def main():
    """Entry point: configure the Streamlit page and render the Search and RAG tabs."""
    st.set_page_config(page_title="Amazon Beauty Search", layout="wide")
    st.title("Amazon Beauty Product Search")

    retrievers = load_retrievers()

    search_tab, rag_tab = st.tabs(["Search", "RAG"])

    with search_tab:
        _render_search_tab(retrievers)

    with rag_tab:
        _render_rag_tab(retrievers)


if __name__ == "__main__":
    main()
