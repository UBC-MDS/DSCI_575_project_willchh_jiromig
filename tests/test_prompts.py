from langchain_core.documents import Document

from src.prompts import build_context


def _doc(asin, title, text, price=None, rating=None):
    return Document(
        page_content=text,
        metadata={
            "parent_asin": asin,
            "title": title,
            "price": price,
            "average_rating": rating,
        },
    )


def test_build_context_renders_each_doc_with_index_and_metadata():
    docs = [
        _doc("B001", "Vitamin C Serum", "brightening serum", 24.99, 4.5),
        _doc("B002", "Coconut Oil Shampoo", "moisturizing", 12.99, 4.0),
    ]

    out = build_context(docs)

    assert "[1] ASIN: B001" in out
    assert "Title: Vitamin C Serum" in out
    assert "Rating: 4.5/5" in out
    assert "Price: $24.99" in out
    assert "[2] ASIN: B002" in out
    assert "brightening serum" in out
    assert "moisturizing" in out


def test_build_context_handles_empty_list():
    assert build_context([]) == ""


def test_build_context_handles_missing_price_and_rating():
    docs = [_doc("B001", "Mystery Product", "no price/rating fields")]
    out = build_context(docs)
    assert "Price: N/A" in out
    assert "Rating: N/A" in out


def test_build_context_truncates_long_review_text():
    long_text = "x" * 1000
    docs = [_doc("B001", "Long Review", long_text)]
    out = build_context(docs, max_chars=50)
    assert "x" * 50 in out
    assert "x" * 51 not in out


def test_build_context_coerces_string_price_and_rating():
    docs = [
        Document(
            page_content="classic oil",
            metadata={
                "parent_asin": "B011",
                "title": "Coconut Oil",
                "price": "14.50",
                "average_rating": "4.2",
            },
        ),
    ]
    out = build_context(docs)
    assert "Price: $14.50" in out
    assert "Rating: 4.2/5" in out
