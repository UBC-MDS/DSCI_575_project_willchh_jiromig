from langchain_core.documents import Document

from src.prompts import DEFAULT_PROMPT_NAME, PROMPT_VARIANTS, build_context, build_web_context


def _doc(asin, title, text, price=None, rating=None):
    """Build a minimal LangChain Document with the metadata fields used by build_context."""
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
    """build_context emits a numbered block per doc with ASIN, title, rating, and price."""
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
    """An empty doc list produces an empty context string."""
    assert build_context([]) == ""


def test_build_context_handles_missing_price_and_rating():
    """Missing price or rating fields render as 'N/A' instead of raising."""
    docs = [_doc("B001", "Mystery Product", "no price/rating fields")]
    out = build_context(docs)
    assert "Price: N/A" in out
    assert "Rating: N/A" in out


def test_build_context_truncates_long_review_text():
    """Review text is truncated to max_chars to bound prompt size."""
    long_text = "x" * 1000
    docs = [_doc("B001", "Long Review", long_text)]
    out = build_context(docs, max_chars=50)
    assert "x" * 50 in out
    assert "x" * 51 not in out


def test_build_context_coerces_string_price_and_rating():
    """String-typed price and rating metadata are coerced to floats for formatting."""
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
    """PROMPT_VARIANTS exposes the three expected template names."""
    assert set(PROMPT_VARIANTS) == {"strict_citation", "helpful_shopper", "structured_json"}


def test_default_prompt_name_is_strict_citation():
    """The default prompt name points at a registered strict_citation variant."""
    assert DEFAULT_PROMPT_NAME == "strict_citation"
    assert DEFAULT_PROMPT_NAME in PROMPT_VARIANTS


def test_each_prompt_variant_renders_context_and_question():
    """Every prompt variant renders both the retrieval context and the user question."""
    inputs = {
        "context": "[1] ASIN: B001 ...",
        "web_context": "",
        "question": "best vitamin C serum?",
    }
    for name, template in PROMPT_VARIANTS.items():
        rendered = template.format_messages(**inputs)
        # System message + user message
        assert len(rendered) == 2
        user_text = rendered[1].content
        assert "[1] ASIN: B001" in user_text
        assert "best vitamin C serum?" in user_text
        assert "Web Context:" not in user_text


def test_strict_citation_system_message_demands_asin_citations():
    """The strict_citation system prompt mandates ASIN-style citations."""
    rendered = PROMPT_VARIANTS["strict_citation"].format_messages(
        context="x", web_context="", question="y"
    )
    sys_text = rendered[0].content
    assert "ASIN" in sys_text
    assert "ONLY" in sys_text or "only" in sys_text
    assert "[B001]" in sys_text


def test_structured_json_system_message_demands_json_keys():
    """The structured_json system prompt lists the required JSON keys."""
    rendered = PROMPT_VARIANTS["structured_json"].format_messages(
        context="x", web_context="", question="y"
    )
    sys_text = rendered[0].content
    for key in ("recommendation", "reasoning", "asins"):
        assert key in sys_text


def test_helpful_shopper_system_message_mentions_recommendation():
    """The helpful_shopper system prompt asks for a price- or rating-aware recommendation."""
    rendered = PROMPT_VARIANTS["helpful_shopper"].format_messages(
        context="x", web_context="", question="y"
    )
    sys_text = rendered[0].content
    assert "recommend" in sys_text.lower()
    assert "price" in sys_text.lower() or "rating" in sys_text.lower()


def test_build_web_context_returns_empty_string_for_empty_list():
    """An empty snippet list yields an empty web-context string."""
    assert build_web_context([]) == ""


def test_build_web_context_renders_snippets_without_bracket_tags():
    """Web-context snippets are rendered as bullets without bracketed citation tags."""
    out = build_web_context(["alpha", "beta"])
    assert out.startswith("\nWeb Context:\n")
    assert out.endswith("\n")
    assert "alpha" in out
    assert "beta" in out
    assert "[W1]" not in out
    assert "[W2]" not in out


def test_build_web_context_drops_empty_snippets():
    """Empty-string snippets are filtered out before rendering."""
    out = build_web_context(["a", "", "b"])
    assert "a" in out
    assert "b" in out
    lines = [line for line in out.splitlines() if line.startswith("-")]
    assert len(lines) == 2


def test_each_prompt_variant_renders_web_context_when_populated():
    """Every prompt variant propagates a populated web-context block to the user message."""
    populated = "\nWeb Context:\n- fresh snippet\n"
    inputs = {
        "context": "[1] ASIN: B001 review",
        "web_context": populated,
        "question": "what's new in retinol?",
    }
    for name, template in PROMPT_VARIANTS.items():
        rendered = template.format_messages(**inputs)
        user_text = rendered[1].content
        assert "Web Context:" in user_text
        assert "fresh snippet" in user_text
        assert "[1] ASIN: B001 review" in user_text


def test_strict_citation_system_message_forbids_web_bracket_tags():
    """The strict_citation prompt explicitly forbids bracketed tags on web information."""
    rendered = PROMPT_VARIANTS["strict_citation"].format_messages(
        context="x", web_context="", question="y"
    )
    sys_text = rendered[0].content
    assert "[W1]" not in sys_text
    assert "do not attach any bracketed tags to web information" in sys_text


def test_helpful_shopper_system_message_mentions_web_context_usage():
    """The helpful_shopper prompt tells the model how to use the Web Context block."""
    rendered = PROMPT_VARIANTS["helpful_shopper"].format_messages(
        context="x", web_context="", question="y"
    )
    sys_text = rendered[0].content.lower()
    assert "web context" in sys_text
