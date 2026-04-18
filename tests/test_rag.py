from unittest.mock import MagicMock

import pytest


def test_rag_pipeline_requires_hf_token_when_no_llm_injected(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from src.rag_pipeline import RAGPipeline

    bm25 = MagicMock()
    semantic = MagicMock()

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        RAGPipeline(bm25=bm25, semantic=semantic)
