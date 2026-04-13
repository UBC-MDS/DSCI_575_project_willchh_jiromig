# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - (2026-04-12)

### Added
- Project scaffolding: repo structure, CI, pre-commit hooks, Makefile ([#1](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/1))
- EDA notebook with data loading, sample inspection, and preprocessing documentation ([#15](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/15))
- BM25 keyword-based retriever using rank_bm25 (`src/bm25.py`) ([#15](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/15))
- Semantic retriever using sentence-transformers and FAISS (`src/semantic.py`) ([#16](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/16))
- Hybrid retriever combining BM25 and semantic scores (`src/hybrid.py`) ([#17](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/17))
- Utility module for tokenization, corpus building, and data download (`src/utils.py`)
- Review text enrichment for product corpus ([#19](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/19))
- Streamlit web app with search mode selector, query input, result display, and feedback buttons ([#20](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/20))
- Retrieval evaluation notebook and qualitative discussion comparing BM25 vs semantic across 21 queries ([#18](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/18))
- Ground truth query set with easy, medium, and hard queries (`data/processed/ground_truth.csv`)
- Test suite for BM25 and utils modules (`tests/`)
- README with full developer setup, dataset description, and retrieval method documentation
- Option to download pre-built indices from GitHub Release as an alternative to local index building
- Code of conduct, linting fixes, and final polish ([#22](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/22))

### Fixed
- Added lemmatizer and stopword removal for improved BM25 tokenization
- Resolved dotenv loading for sentence-transformer HuggingFace token
