import gzip
import json

from src.utils import build_text, load_corpus, load_metadata, save_corpus, tokenize


def test_tokenize_basic():
    result = tokenize("Hello World")
    assert result == ["hello", "world"]


def test_tokenize_punctuation():
    result = tokenize("anti-aging cream, 100% organic!")
    assert result == ["anti", "aging", "cream", "100", "organic"]


def test_tokenize_empty_string():
    result = tokenize("")
    assert result == []


def test_tokenize_extra_whitespace():
    result = tokenize("  vitamin   C   serum  ")
    assert result == ["vitamin", "c", "serum"]


def test_tokenize_removes_stopwords():
    result = tokenize("this is a great moisturizer for the skin")
    assert "this" not in result
    assert "is" not in result
    assert "a" not in result
    assert "for" not in result
    assert "the" not in result
    assert "great" in result
    assert "moisturizer" in result
    assert "skin" in result


def test_tokenize_lemmatizes_words():
    result = tokenize("moisturizing creams and lotions")
    assert "moisturizing" in result or "moisturize" in result
    assert "cream" in result  # creams -> cream
    assert "lotion" in result  # lotions -> lotion


def test_tokenize_slash_splitting():
    result = tokenize("shampoo/conditioner combo")
    assert "shampoo" in result
    assert "conditioner" in result
    assert "combo" in result


def test_build_text_all_fields():
    product = {
        "title": "Vitamin C Serum",
        "description": "A brightening serum",
        "features": ["20% vitamin C", "for dark spots"],
    }
    result = build_text(product)
    assert result == "Vitamin C Serum A brightening serum 20% vitamin C for dark spots"


def test_build_text_missing_description():
    product = {
        "title": "Vitamin C Serum",
        "features": ["20% vitamin C"],
    }
    result = build_text(product)
    assert result == "Vitamin C Serum 20% vitamin C"


def test_build_text_empty_features():
    product = {
        "title": "Vitamin C Serum",
        "description": "A brightening serum",
        "features": [],
    }
    result = build_text(product)
    assert result == "Vitamin C Serum A brightening serum"


def test_build_text_title_only():
    product = {"title": "Vitamin C Serum"}
    result = build_text(product)
    assert result == "Vitamin C Serum"


def test_load_metadata(tmp_path):
    """Test loading metadata from a gzipped JSONL file."""
    raw_products = [
        {
            "parent_asin": "B001",
            "title": "Serum",
            "description": "A serum",
            "features": ["vitamin C"],
            "price": 24.99,
            "average_rating": 4.5,
            "images": [{"large": "http://img.jpg"}],
            "extra_field": "ignored",
        },
        {
            "parent_asin": "B002",
            "title": "Shampoo",
            "description": None,
            "features": [],
            "price": None,
            "average_rating": 3.0,
            "images": [],
        },
    ]
    filepath = tmp_path / "meta_test.jsonl.gz"
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        for product in raw_products:
            f.write(json.dumps(product) + "\n")

    corpus = load_metadata(str(filepath))

    assert len(corpus) == 2
    assert corpus[0]["parent_asin"] == "B001"
    assert "Serum" in corpus[0]["text"]
    assert "vitamin C" in corpus[0]["text"]
    assert corpus[0]["price"] == 24.99
    assert corpus[1]["price"] is None
    assert "text" in corpus[1]


def test_save_and_load_corpus_parquet(fake_corpus, tmp_path):
    """Test saving and loading corpus as parquet."""
    filepath = str(tmp_path / "corpus.parquet")
    save_corpus(fake_corpus, filepath)
    loaded = load_corpus(filepath)

    assert len(loaded) == len(fake_corpus)
    assert loaded[0]["parent_asin"] == "B001"
    assert loaded[0]["title"] == "Vitamin C Serum"
    assert loaded[0]["price"] == 24.99
