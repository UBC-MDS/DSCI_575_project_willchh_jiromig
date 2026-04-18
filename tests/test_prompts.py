from langchain_core.documents import Document

from src.prompts import DEFAULT_PROMPT_NAME, PROMPT_VARIANTS, build_context


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


def test_prompt_variants_has_three_named_templates():
    assert set(PROMPT_VARIANTS) == {"strict_citation", "helpful_shopper", "structured_json"}


def test_default_prompt_name_is_strict_citation():
    assert DEFAULT_PROMPT_NAME == "strict_citation"
    assert DEFAULT_PROMPT_NAME in PROMPT_VARIANTS


def test_each_prompt_variant_renders_context_and_question():
    inputs = {"context": "[1] ASIN: B001 ...", "question": "best vitamin C serum?"}
    for name, template in PROMPT_VARIANTS.items():
        rendered = template.format_messages(**inputs)
        # System message + user message
        assert len(rendered) == 2
        user_text = rendered[1].content
        assert "[1] ASIN: B001" in user_text
        assert "best vitamin C serum?" in user_text


def test_strict_citation_system_message_demands_asin_citations():
    rendered = PROMPT_VARIANTS["strict_citation"].format_messages(context="x", question="y")
    sys_text = rendered[0].content
    assert "ASIN" in sys_text
    assert "ONLY" in sys_text or "only" in sys_text
    assert "[B001]" in sys_text


def test_structured_json_system_message_demands_json_keys():
    rendered = PROMPT_VARIANTS["structured_json"].format_messages(context="x", question="y")
    sys_text = rendered[0].content
    for key in ("recommendation", "reasoning", "asins"):
        assert key in sys_text


def test_helpful_shopper_system_message_mentions_recommendation():
    rendered = PROMPT_VARIANTS["helpful_shopper"].format_messages(context="x", question="y")
    sys_text = rendered[0].content
    assert "recommend" in sys_text.lower()
    assert "price" in sys_text.lower() or "rating" in sys_text.lower()
