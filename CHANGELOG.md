# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - (2026-04-22)

### Added
- LLM comparison experiment: `qwen3:1.7b` (1.7B, Ollama local) vs `Meta-Llama-3-8B-Instruct` (8B, HF API) across 5 queries with identical retrieval and prompt ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Final exploration notebook (`notebooks/milestone3_final.ipynb`) with LLM comparison and Tavily web search tool demo (3 queries) ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Final discussion (`results/final_discussion.md`) covering dataset scaling, LLM experiment, tool integration, code quality, and cloud deployment plan ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Cloud deployment plan for AWS grounded in DSCI 525 services: S3 for artifacts, Elastic Beanstalk (load-balanced) for the Streamlit app, HF Inference API by default with SageMaker as the self-hosted alternative ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- `langchain-ollama` dependency for local model inference via Ollama ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Ollama setup instructions in README (macOS, Linux, Windows) ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- `temperature`, `seed`, and `do_sample` arguments on `load_llm` for reproducible generation (`src/rag_pipeline.py`) ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Eval output artifacts for the final milestone: `data/eval_outputs/llm_comparison.json` and `data/eval_outputs/tool_demo.json` ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- `web_search_snippets(query, k)` helper returning a list of Tavily snippets for RAG prompt augmentation (`src/tools.py`)
- `build_web_context` helper and `{web_context}` template variable wired into all RAG prompt variants (`src/prompts.py`)
- `use_web_search` flag on `RAGPipeline.answer` that augments the prompt with Tavily snippets and soft-fails when `TAVILY_API_KEY` is missing or the call raises (`src/rag_pipeline.py`)
- Web-search checkbox in the Streamlit RAG tab rendering retrieved products and web snippets side-by-side (`app/app.py`)
- Sample query dropdowns in the Search and RAG tabs for quick demo inputs (`app/app.py`)
- Test coverage for `web_search_snippets` (happy path, empty results, missing key), `build_web_context`, `RAGPipeline.answer` web-search on/off/exception branches, `top_k` forwarding, and hybrid source cap

### Changed
- Added docstrings to all public functions and classes in `src/` ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Updated `README.md` with Qwen3 comparison model, Ollama in tech stack, final milestone files in repository structure, and the new `data/eval_outputs/` artifacts ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- Pre-built index download instructions in README now point at the `v0.3.0` GitHub Release instead of `v0.1.0` ([#46](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/46))
- `RAGPipeline.answer` forwards `top_k` to the web-search tool and caps hybrid sources at `top_k` (`src/rag_pipeline.py`)
- Dropped `[Wn]` labels from web context and removed the corresponding citation mandate from the `strict_citation` prompt (`src/prompts.py`)
- README: added `web_search` tool to the mermaid workflow diagram and promoted the environment-variables section (`README.md`)
- Added 1-line docstrings to `app/` and `tests/` functions
- `results/final_discussion.md` updated to reflect the Streamlit web-search wiring

### Fixed
- `.gitignore` now excludes local eval artifacts `data/eval_outputs/llm_comparison.json` and `data/eval_outputs/tool_demo.json`

## [v0.2.0] - (2026-04-18)

### Added
- `RAGPipeline` class composing retriever + context + prompt + LLM via LCEL pipes (`src/rag_pipeline.py`) ([#36](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/36))
- `load_llm` helper for HuggingFace Inference API with `HF_TOKEN` guard (`src/rag_pipeline.py`) ([#36](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/36))
- LangChain retriever wrappers: `BM25LCRetriever` and `SemanticLCRetriever` around the Milestone 1 retrievers (`src/retrievers_lc.py`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- `EnsembleRetriever` factory combining BM25 + semantic via RRF, with a `wrap_retriever` dispatch helper (`src/retrievers_lc.py`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- `build_context` helper for RAG prompt assembly from retrieved product docs (`src/prompts.py`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- Three RAG prompt variants — `strict_citation`, `helpful_shopper`, and `structured_json` — exposed via `PROMPT_VARIANTS` (`src/prompts.py`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- Tavily `web_search` tool stub for augmenting RAG with web results (`src/tools.py`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- Streamlit RAG tab with retriever selector, prompt-variant radio, and answer/sources panel ([#39](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/39))
- Exploration notebook with component demos, prompt comparison, and 10-query eval run (`notebooks/milestone2_rag.ipynb`) ([#40](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/40))
- 10-query RAG evaluation set (`data/processed/rag_queries.csv`) ([#40](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/40))
- Qualitative evaluation discussion with model choice, limitations, and improvements (`results/milestone2_discussion.md`) ([#41](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/41))
- `langchain`, `langchain-huggingface`, and `tavily-python` dependencies (`requirements.txt`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- `TAVILY_API_KEY` placeholder in `.env.example` ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- Test suites for RAG pipeline, LangChain retrievers, prompts, and tools (`tests/`) ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35), [#37](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/37), [#38](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/38))
- New env vars: `HF_TOKEN` (required for RAG), `TAVILY_API_KEY` (optional for web search)

### Changed
- `make test` now enforces a 90% line-coverage gate via `--cov-fail-under=90` ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))
- App split into Search (Milestone 1) and RAG (Milestone 2) tabs ([#39](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/39))

### Fixed
- `build_context` coerces string-typed `price` and `rating` fields so real Amazon metadata renders correctly ([#35](https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig/pull/35))


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
