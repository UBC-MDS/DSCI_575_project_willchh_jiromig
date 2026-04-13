import gzip
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from src.utils import (
    build_indices,
    build_processed_corpus,
    build_text,
    download_raw_data,
    load_corpus,
    load_metadata,
    save_corpus,
    tokenize,
)


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


def test_download_raw_data_creates_parquets(tmp_path):
    """Test that download_raw_data calls DuckDB and creates parquet files."""
    mock_con = MagicMock()
    with patch("src.utils.duckdb.connect", return_value=mock_con):
        download_raw_data(str(tmp_path))

    assert mock_con.execute.call_count == 2
    calls = [str(c) for c in mock_con.execute.call_args_list]
    assert any("meta_All_Beauty" in c for c in calls)
    assert any("reviews_All_Beauty" in c for c in calls)
    mock_con.close.assert_called_once()


def test_download_raw_data_skips_existing(tmp_path):
    """Test that download_raw_data skips files that already exist."""
    (tmp_path / "meta_All_Beauty.parquet").touch()
    (tmp_path / "reviews_All_Beauty.parquet").touch()

    mock_con = MagicMock()
    with patch("src.utils.duckdb.connect", return_value=mock_con):
        download_raw_data(str(tmp_path))

    mock_con.execute.assert_not_called()


def test_build_processed_corpus(tmp_path):
    """Test that build_processed_corpus aggregates reviews and saves corpus."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    meta_df = pd.DataFrame(
        [
            {
                "parent_asin": "B001",
                "title": "Vitamin C Serum",
                "description": "A brightening serum",
                "features": ["20% vitamin C"],
                "price": 24.99,
                "average_rating": 4.5,
            },
            {
                "parent_asin": "B002",
                "title": "Shampoo",
                "description": None,
                "features": [],
                "price": None,
                "average_rating": 3.0,
            },
        ]
    )
    reviews_df = pd.DataFrame(
        [
            {
                "parent_asin": "B001",
                "text": "Great serum!",
                "helpful_vote": 10,
            },
            {
                "parent_asin": "B001",
                "text": "It was okay.",
                "helpful_vote": 2,
            },
            {
                "parent_asin": "B002",
                "text": "Nice shampoo",
                "helpful_vote": 5,
            },
        ]
    )
    meta_df.to_parquet(raw_dir / "meta_All_Beauty.parquet")
    reviews_df.to_parquet(raw_dir / "reviews_All_Beauty.parquet")

    build_processed_corpus(str(raw_dir), str(processed_dir))

    corpus_path = processed_dir / "product_corpus.parquet"
    assert corpus_path.exists()

    corpus_df = pd.read_parquet(corpus_path)
    assert len(corpus_df) == 2
    assert corpus_df.iloc[0]["parent_asin"] == "B001"
    # Should contain the most helpful review (vote=10), not the less helpful one
    assert "Great serum!" in corpus_df.iloc[0]["text"]
    assert "It was okay." not in corpus_df.iloc[0]["text"]
    # Second product should have its review
    assert "Nice shampoo" in corpus_df.iloc[1]["text"]


def test_build_indices(fake_corpus, tmp_path):
    """Test that build_indices builds and saves BM25 and FAISS indices."""
    corpus_path = tmp_path / "corpus.parquet"
    indices_dir = tmp_path / "indices"
    save_corpus(fake_corpus, str(corpus_path))

    build_indices(str(corpus_path), str(indices_dir))

    assert (indices_dir / "bm25_index.pkl").exists()
    assert (indices_dir / "faiss_index" / "index.faiss").exists()
    assert (indices_dir / "faiss_index" / "corpus.pkl").exists()


def test_build_indices_skips_existing(fake_corpus, tmp_path):
    """Test that build_indices skips indices that already exist."""
    corpus_path = tmp_path / "corpus.parquet"
    indices_dir = tmp_path / "indices"
    save_corpus(fake_corpus, str(corpus_path))

    # Create sentinel files so build_indices thinks they already exist
    (indices_dir / "faiss_index").mkdir(parents=True)
    (indices_dir / "bm25_index.pkl").write_bytes(b"fake")

    build_indices(str(corpus_path), str(indices_dir))

    # Files should remain unchanged (not overwritten)
    assert (indices_dir / "bm25_index.pkl").read_bytes() == b"fake"
