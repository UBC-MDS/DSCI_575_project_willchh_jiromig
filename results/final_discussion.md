# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- **Number of products used:** 112,590 (full Amazon All Beauty category).
- **Changes to sampling strategy:** none. The full category was used from the start. The corpus is built by concatenating `title`, `description`, `features`, and the most helpful review per product into a single text field. DuckDB streams the remote JSONL.gz files directly into local Parquet, avoiding memory pressure on the full 701K reviews. The corpus exceeds the 10,000-product minimum by more than 10x.

### LLM Experiment

**Models compared:**

| Model | Family | Parameters | Provider |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama 3 | 8B | HuggingFace Inference API (remote) |
| `qwen3:1.7b` | Alibaba Qwen 3 | 1.7B | Ollama (local) |

The ~5x parameter gap and the different model families let us compare quality-vs-size tradeoffs and, as a side effect, API vs local inference.

We originally intended to run both models through the HuggingFace Inference API, but `Qwen/Qwen3.5-2B` is not currently hosted by any HF provider for serverless inference (`BadRequestError`). We switched to Ollama to run Qwen3 locally, which also added a practical remote-vs-local comparison angle.

**Experimental setup:**

- Retriever: Hybrid (`EnsembleRetriever` with Reciprocal Rank Fusion over BM25 + FAISS).
- Prompt: `strict_citation` (answer only from context, cite ASINs).
- `top_k = 5`, `temperature = 0`, `seed = 575` for both models (`do_sample=True` on the HF endpoint).
- Identical context and prompt for each query, differing only in which LLM generates the answer.

**Prompt used (`src/prompts.py::PROMPT_VARIANTS["strict_citation"]`):**

```
System: You are an Amazon Beauty product assistant. Answer ONLY using the
provided reviews and metadata. Cite the product ASIN like [B001] for
every claim. If the context is insufficient, reply:
'I don't have enough information.'

User: Context:
{context}

Question: {question}

Answer:
```

The context block passed into `{context}` is built by `build_context` (`src/prompts.py`) and includes, for each retrieved product: ASIN, title, rating (`X.X/5`), price (or `N/A`), and the top review / description truncated to 300 characters.

**Results:**

| # | Query | Llama-3-8B (HF API) | Qwen3 1.7B (Ollama) | Notes |
|---|---|---|---|---|
| 1 | `vitamin C serum for brightening` | Numbered prose list of **7** products with ASINs and short review quotes per product. **Does not surface the `Rating` field.** Minor formatting artifact (`"[B09] is missing for..."`) before one entry. | Bold-header bullet catalog of **4** products — Charlotte Elizabeth (5.0/5), Essano (4.2/5), Trilogy (4.3/5), Eva St. Claire (3.6/5). Closes with a one-line top-pick synthesis. | 3-product overlap (Charlotte Elizabeth, Essano, Trilogy). Split is format + metadata usage; both are well-grounded on ASINs. |
| 2 | `what helps with mild acne and post-acne marks?` | Numbered list of **6** products (tea-tree acne patches cited twice, Tepezcohuite, Tamanu oil, Max Factor green corrector, Burts). Closes with the hedge *"These products may not work for everyone, and individual results may vary."* No ratings surfaced. | **6** products each with a `Rating: X.X/5` field. Surfaces the low **2.9/5** on `B01I05IRII`, but softens it in prose (*"While not the highest-rated, it is noted as a good product..."*). | 3-product overlap (Tepezcohuite, Tamanu, Burts). Llama prose carries more evidence quality nuance; Qwen is easier to scan. |
| 3 | `gentle face wash that won't irritate sensitive skin` | **9** products, numbered list, short paraphrased descriptions. No ratings. | **7** products, bold-header bullets, `Rating: X.X/5` on every entry. | Strong overlap on the sensitive-skin cleanser cluster. Across Q1–Q3 the pattern is now clear: **Qwen uses the rating field on every answer, Llama uses it on none.** |
| 4 | `best skincare routine for oily acne-prone teenage skin under $30` | Recommends a 3-step routine (Rohto Acnes cleanser `B007409E5Q`, salicylic gel `B018STZ1WG`, Tropical Fruits scrub `B09BJK8WKP`). Closes with *"The total cost of these products is under $30."* — **asserts budget compliance with no price evidence in context.** | Builds a fuller routine. Makes a **category error** — suggests *"opt for a gentle, non-oily moisturizer (e.g., `[B019PM5PRQ]` The Best Acne Skin Clearing Face Wash)"*. `B019PM5PRQ` is a face wash, not a moisturizer. Closes with the same unsupported *"All products are under $30"* claim. Does surface the low 1.0/5 rating on `B01L8218Q8` as a reason to avoid it. | **Both models fail the price-constraint hallucination test**, each in its own way. Qwen adds a category-confusion error on top. |
| 5 | `what helps with sun damage on fair skin around the eyes` | **4** products. Correct ASIN attribution throughout (Under Eye Patches `B076J5S1T8`, Coral Safe Aloe `B01I20S3TO`, Josie Maran `B007YPNKOM`, Vertra `B01HMX4YAO`). Explicitly **hedges weak matches** (*"not specifically mentioned to help with sun damage around the eyes"*, *"not a sun damage treatment or specifically used around the eyes"*) and names Vertra as the strongest pick. | **2** products, and **both citations have misattributed ASINs**: Vertra cited as `B01CNFRQPQ` (correct is `B01HMX4YAO`); Under Eye Patches cited as `B07WG9XCH5` (correct is `B076J5S1T8`). Bracket numbers and product names are right, but each is paired with the ASIN from a different row of the context. | Citation integrity failure from Qwen — the single most damaging error class for a RAG system. Cross-referenced against Llama's correct citations over the same context. |

**Key observations:**

1. **Metadata usage is systematically different.** Qwen3 1.7B surfaces the `Rating: X.X/5` field on every answer; Llama-3-8B surfaces it on none. Since both models see the same context with identical rating fields, this looks like a stable stylistic / instruction-following difference between the two models at this scale, not retrieval noise.
2. **Hedging behavior is asymmetric.** Llama is more willing to flag weak matches (Q2 tail caveat; Q5 explicit *"not specifically for"* notes). Qwen tends to write confident-sounding prose even when the underlying evidence is weak (Q2 softens a 2.9/5 rating; Q5 produces invented ASINs).
3. **Both models hallucinate the price constraint in Q4.** Neither model can meet the *"under $30"* budget from the retrieved context alone, but both assert budget compliance anyway. This is a prompt-level and retriever-level problem (prices aren't consistently in context), not a model-specific one.
4. **Qwen has a citation integrity bug.** Q4 shows category confusion (face wash → moisturizer); Q5 shows off-by-one ASIN attribution on both of its citations. These are exactly the kinds of bugs that undermine the value proposition of RAG — the user trusts that `[ASIN]` maps to a specific product, and on Qwen it sometimes doesn't.
5. **Llama is not bug-free.** Q4's *"total cost of these products is under $30"* is a plainly ungrounded quantitative claim, and the minor formatting artifacts on Q1 (`"[B09] is missing for..."`) and Q5 (`"[B09] — Although there is no specific review for a product ASIN: B09"`) show that Llama sometimes leaks its own retrieval-reasoning scratchpad into the answer.
6. **Expanding the eval.** The five queries already surface distinct, reproducible failure modes in both models. Adding more queries would likely produce variations on the same themes rather than new information, so the comparison is written up as-is.

**Which model we chose and why:**

We kept `meta-llama/Meta-Llama-3-8B-Instruct` as the default model in the deployed RAG pipeline. The deciding factor is **citation integrity**: Qwen's Q5 off-by-one ASIN attribution and Q4 category confusion are the kind of errors that silently mislead users, because the prose reads fluently and the product names look plausible. Llama's own failure modes which include occasional ungrounded quantitative claims, formatting leakage — are more visible to users and less likely to produce wrong-product recommendations.

Qwen3 1.7B's structured formatting and uniform rating surfacing would be an excellent fit for the `helpful_shopper` prompt variant, where conversational polish is more important than strict grounding. But the `strict_citation` prompt is the one we ship with, and for that prompt Llama is the safer default.

## Step 2: Additional Feature (Option 2: Tool Integration + Option 3: Scale to ≥ 100K)

We implemented **Option 3 (Scale to ≥ 100K Products)** as the primary feature and **Option 2 (Tool Integration)** as a secondary feature. Option 3 is demonstrated in `notebooks/milestone3_final.ipynb`; Option 2 is demonstrated there and wired end-to-end into the Streamlit RAG tab so users can flip the checkbox and see web snippets augment the RAG answer in real time.

### What We Implemented

#### Option 3 - Scale to ≥ 100K products (primary)

The pipeline processes **112,590 products** from the Amazon All Beauty category, exceeding the 100K threshold. The engineering decisions that made this scale feasible:

1. **DuckDB for ingestion.** Reads remote JSONL.gz files directly via HTTP and converts to local Parquet with ZSTD compression. Avoids loading the full dataset into memory; handles the 701K reviews efficiently.
2. **Batch encoding for FAISS.** `sentence-transformers` encodes all 112K product texts in a single batched `model.encode()` call, producing a `(112590, 384)` float32 matrix.
3. **Persistence.** BM25 index is pickled to `indices/bm25_index.pkl`; FAISS is saved via `faiss.write_index()` to `indices/faiss_index/`. Both load in seconds at app startup via `@st.cache_resource`.
4. **Single-review strategy.** Only the most helpful review per product is included in the corpus, keeping per-document text length manageable while preserving information density.

Code references: `src/utils.py::build_processed_corpus()`, `src/utils.py::build_indices()`.

#### Option 2 - Tool Integration (secondary)

We added a Tavily web-search tool (`src/tools.py`) following the Lecture 8 pattern. The tool is a LangChain `@tool` wrapper around `TavilyClient`:

```python
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for current information about a beauty product..."""
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=max_results)
    return "\n".join([r["content"] for r in results["results"]])
```

Alongside the `@tool`, `src/tools.py` exposes `web_search_snippets(query, max_results) -> list[str]` which is the same Tavily call but returning the snippets as a list so callers can label them individually. The pipeline uses this helper directly so each snippet keeps its boundary for `[W1]`, `[W2]`, … citations.

**End-to-end wiring.** `RAGPipeline.answer(query, use_web_search=False)` now takes a flag. When on, it calls `web_search_snippets(query)`, builds a labeled `Web Context:` block via the new `build_web_context` helper in `src/prompts.py`, and injects it into the prompt through a new `{web_context}` template variable. The `strict_citation` and `helpful_shopper` system prompts were updated to cite web snippets as `[W1]`, `[W2]`, … distinct from product ASIN citations. `structured_json`'s schema was deliberately left untouched. On Tavily failure the pipeline catches the exception, returns a `web_warning` string, and continues with RAG-only context — the answer is still produced from reviews.

**UI.** The Streamlit RAG tab's web-search checkbox is disabled when `TAVILY_API_KEY` is missing; when enabled and toggled on, the tab passes the flag through to `pipeline.answer()`. When web results come back, the tab renders **Web Sources (Tavily)** and **Product Sources** side-by-side (`st.columns(2)`), each web snippet as a bordered card with its `[Wn]` label, and a caption linking the cards back to the citations in the answer. On Tavily failure the tab surfaces the `web_warning` via `st.warning` above the single-column product sources.

**Key results - 3 example queries where the tool was exercised (`notebooks/milestone3_final.ipynb`, cell `2246391a`):**

| # | Query | RAG-only answer | Tavily web search result | Did the tool help? |
|---|---|---|---|---|
| 1 | `Is retinol safe to use with vitamin C serum?` | *"I don't have enough information."* The corpus is product reviews, not dermatological guidance. | Returns practitioner guidance on combining retinol and vitamin C (morning vs. night layering). | **Yes.** The corpus structurally cannot answer ingredient-interaction questions; the tool supplied exactly the missing knowledge type. |
| 2 | `best drugstore sunscreen 2025 dermatologist recommended` | Pivots to dermatologist-recommended sunscreens present in the corpus, but honestly notes it cannot verify the 2025 time stamp (*"does not include specific details about the year 2025"*). | Returns current 2025 picks (EltaMD UV Skin Recovery SPF 40, CeraVe Hydrating Mineral SPF 30, etc.). | **Yes.** The corpus is a static 2023 snapshot; no amount of better retrieval can surface 2025-specific recommendations. |
| 3 | `Johnson & Johnson sunscreen recall or safety issues` | *"I don't have enough information. There are no reviews or metadata related to Johnson & Johnson sunscreen recall or safety issues in the provided list."* | Returns the actual voluntary recall of five NEUTROGENA® and AVEENO® aerosol sunscreen product lines due to detected benzene. | **Yes.** Safety / regulatory news doesn't appear in product reviews; the RAG answer correctly abstains and the tool fills the gap with a concrete, actionable answer. |

**What the three queries together show:**

- The `strict_citation` prompt's explicit *"I don't have enough information"* escape hatch is what makes tool routing practical — it produces a clean signal that the app can use to trigger a web-search fallback instead of a confabulated answer.
- Product-review corpora have systematic blind spots: ingredient interactions, recency-bound questions, and safety / regulatory news. A recommender that covers only product picks will feel frustrating for users who ask the other three question types.
- The tool is **additive, not a replacement**. On our five comparison queries (Q1–Q5 in Step 1) RAG is still the better answer source — the corpus has strong, opinionated, grounded signal for product recommendations. A production system would **route** queries by type: product recommendations → RAG, ingredient / safety / recency → web.

Code references: `src/tools.py` (`@tool` + `web_search_snippets`), `src/prompts.py` (`build_web_context`, `{web_context}` template variable, updated system prompts), `src/rag_pipeline.py` (`RAGPipeline.answer(..., use_web_search=...)` with soft-fail), `app/app.py` (checkbox wiring and side-by-side Web/Product Sources layout), `notebooks/milestone3_final.ipynb` (demo cells), `docs/superpowers/specs/2026-04-22-web-search-streamlit-design.md` (design spec).

## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Rewrote `README.md` with the RAG pipeline architecture diagram (Mermaid), environment-variable documentation, and a repository-structure section reflecting all milestone artifacts.
- Added `CHANGELOG.md` entries for `v0.2.0` (RAG pipeline, LangChain wrappers, Tavily stub, RAG tab, 10-query eval) and `v0.3.0` (Qwen/Llama LLM comparison, Ollama integration, final discussion, cloud deployment plan).
- Setup instructions cover the full workflow: `make setup` → `.env` configuration → `make app`.
- Added Ollama installation instructions for macOS, Linux, and Windows for reproducing the Qwen3 comparison locally.

### Code Quality Changes

- **No hardcoded file paths.** All paths use `pathlib.Path` relative to `BASE_DIR` or `Path(__file__).parent`.
- **No API keys in source.** `HF_TOKEN` and `TAVILY_API_KEY` are loaded via `python-dotenv` from `.env`. `.env` is gitignored; `.env.example` provides placeholder values.
- **Docstrings.** All public functions and classes in `src/` carry at least a one-line docstring. Verified via an AST-based audit over the `src/` tree.
- **Pinned dependencies.** `requirements.txt` pins all dependencies to specific versions, including `langchain-ollama` for the final milestone.
- **Clean repository.** `.gitignore` covers `.claude/`, `local/`, `.env`, `*.pkl`, `*.parquet`, `index.faiss`, `__pycache__/`, and other build artifacts. No large files or ephemeral outputs are committed.
- **Coverage gate.** `make test` enforces `--cov-fail-under=90`; pytest CI fails the build if line coverage in `src/` drops below 90%.

## Step 4: Cloud Deployment Plan

We outline how we plan to deploy the Amazon Beauty RAG system on **AWS**, using the services covered in DSCI 525: **S3** for artifacts, **Elastic Beanstalk** for the app, **SageMaker** for managed model inference, and **IAM** roles to wire them together. All resources live in a single region (e.g., `ca-central-1`) to avoid cross-region data transfer cost.

### 1. Data Storage

Everything we currently persist under `data/` and `indices/` goes to **S3** with a prefix-based layout, following the `model-artifacts/` / `eb-source/` convention from the course.

| Data | S3 location (Hypotheitical) | Notes |
|---|---|---|
| Raw data (JSONL.gz) | `s3://beauty-rag/data/raw/` | Sourced from the HuggingFace McAuley-Lab bucket. An S3 lifecycle rule transitions objects to Glacier after 90 days — raw data is only re-read during full corpus rebuilds. |
| Processed corpus (Parquet) | `s3://beauty-rag/data/processed/product_corpus.parquet` | Bucket versioning enabled so a bad rebuild can be rolled back to the previous object version. |
| FAISS index | `s3://beauty-rag/indices/faiss_index/` | Directory of `index.faiss` + metadata. Pulled to local disk at app startup via `boto3.client("s3").download_file()` and loaded with `FAISS.load_local(...)`. |
| BM25 index (pickle) | `s3://beauty-rag/indices/bm25_index.pkl` | Same download-at-startup pattern as FAISS. |

The existing `src/utils.py` already treats the index paths as read-only at runtime, so the only code change is swapping the local `Path(...)` reads for an S3 download on boot.

### 2. Compute

| Component | Service | Rationale |
|---|---|---|
| Streamlit app | **Elastic Beanstalk** (Python platform, load-balanced environment) | EB is the app-hosting service we used in DSCI 525. We'd deploy a ZIP bundle of `app/`, `src/`, `requirements.txt`, and a `Procfile` of the form `web: streamlit run app/app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true` - the same Procfile pattern used for the Flask/gunicorn spam app, adapted for Streamlit. Instance type: `t3a.medium` (the free-tier `t3a.micro` doesn't have enough RAM to hold the ~500 MB of loaded indices). |
| LLM inference (default) | **HuggingFace Inference API** (external) | Matches what the deployed app already uses; avoids paying for a GPU instance for a class project. `HF_TOKEN` is injected via EB environment variables, mirroring how `ENDPOINT_NAME` was injected in the spam-deploy lab. |
| LLM inference (production alternative) | **Amazon SageMaker endpoint** | The course-taught route for self-hosting a model. We'd package a quantized Llama-3-8B as `model.tar.gz`, upload to `s3://beauty-rag/model-artifacts/`, and deploy with `sk_model.deploy(instance_type="ml.g5.xlarge", initial_instance_count=1, endpoint_name="beauty-rag-llm")`. The Streamlit app invokes it with `boto3.client("sagemaker-runtime").invoke_endpoint(...)` - the same client pattern the Flask spam detector used. This gives lower tail latency and predictable cost once the traffic justifies a dedicated endpoint. |
| Index rebuild | **CloudShell** (one-off) or a small **EC2** job (scheduled) | A short Python script that runs `build_processed_corpus()` → `build_indices()` → S3 upload. CloudShell is fine for the class-project cadence; a scheduled `t3a.medium` EC2 job would be the production version. |

**Concurrency.** Streamlit holds one WebSocket session per connected user, so concurrency is handled at the instance level, not the request level. The EB environment is configured as **load-balanced** (2-4 `t3a.medium` instances, auto-scaling on CPU), with ELB session affinity enabled so each user stays pinned to the same backend for the life of their session. Each instance loads FAISS + BM25 into RAM once at startup and reuses them across all sessions on that host. At significantly larger corpora we would move FAISS to an IVF index with memory-mapped files to cap per-instance RAM.

### 3. Streaming / Updates

DSCI 525 focused on request-response deployment rather than streaming ETL, so the update story is deliberately simple: rebuild on a schedule, ship a new index object to S3, have the app pick it up.

| Concern | Approach |
|---|---|
| New products | On a weekly cadence, re-download the McAuley Lab metadata with the existing DuckDB script, rebuild the corpus and indices, and upload to S3. For the class project this is run manually from **CloudShell** (`aws s3 cp` after running `build_indices()`); for production we'd wrap it in a cron-scheduled EC2 instance so the rebuild is unattended. |
| Pipeline freshness | Each S3 index object exposes a `LastModified` timestamp. The Streamlit app reads it on startup and on a 1-hour timer; if the timestamp has advanced it re-downloads the index and hot-swaps the retriever. This gives eventual consistency within ~1 hour without needing pub/sub infrastructure the course didn't cover. |
| Model updates | If HF reroutes or deprecates the Llama-3 endpoint, or if we switch to SageMaker, we update `HF_MODEL_ID` (or `SAGEMAKER_ENDPOINT_NAME`) in the EB environment variables and redeploy. `src/llm.py::load_llm()` already accepts `model_id` as a parameter, so no code change is required. |
