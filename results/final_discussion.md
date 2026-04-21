# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- **Number of products used:** 112,590 (All Beauty category)
- **Changes to sampling strategy:** None — we used the full category from the start.
  The corpus was built by concatenating title, description, features, and the most helpful review per product into a single text field.
  DuckDB handled remote JSONL.gz ingestion directly to local Parquet, avoiding memory issues with the full 701K reviews.
  This exceeds the 10,000 minimum by over 10x.

### LLM Experiment

**Models compared:**

| Model | Family | Parameters | Provider |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama 3 | 8B | HuggingFace Inference API |
| `Qwen/Qwen3.5-2B` | Alibaba Qwen 3.5 | 2B | HuggingFace Inference API |

**Rationale for comparison:** The 4x parameter difference (8B vs 2B) and different model families (Llama vs Qwen) allow us to isolate the effect of model capacity on RAG answer quality. Both models are accessed through the same HuggingFace Inference API, so the only variable is the model itself.

**Experimental setup:**
- Retriever: Hybrid (EnsembleRetriever with Reciprocal Rank Fusion)
- Prompt: `strict_citation` (answer only from context, cite ASINs)
- top_k: 5
- 5 queries tested, spanning easy_factual, paraphrased_intent, and multi_concept categories

**Prompt used:**
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

**Results:**

<!-- TODO: Fill in after running notebook -->

| # | Query | Llama-3-8B | Qwen3.5-2B | Winner |
|---|---|---|---|---|
| 1 | vitamin C serum for brightening | (fill in) | (fill in) | (fill in) |
| 2 | what helps with mild acne and post-acne marks? | (fill in) | (fill in) | (fill in) |
| 3 | gentle face wash that won't irritate sensitive skin | (fill in) | (fill in) | (fill in) |
| 4 | best skincare routine for oily acne-prone teenage skin under $30 | (fill in) | (fill in) | (fill in) |
| 5 | what helps with sun damage on fair skin around the eyes | (fill in) | (fill in) | (fill in) |

**Key observations:**

<!-- TODO: Fill in after running notebook -->
- (Accuracy comparison)
- (Citation quality comparison)
- (Fluency comparison)
- (Completeness comparison)

**Which model we chose and why:**

<!-- TODO: Fill in after running notebook -->
We retained `meta-llama/Meta-Llama-3-8B-Instruct` as the default because (fill in reasoning).

## Step 2: Additional Feature

### Option 3: Scale to >= 100K Products (Primary)

Our pipeline already processes **112,590 products** from the Amazon All Beauty category, exceeding the 100K threshold. Key engineering decisions that enabled this scale:

1. **DuckDB for ingestion** — reads remote JSONL.gz files directly via HTTP, converts to local Parquet with ZSTD compression. Avoids loading the full dataset into memory.
2. **Batch encoding for FAISS** — `sentence-transformers` encodes all 112K product texts in a single `model.encode()` call with batched processing, producing a (112590, 384) float32 matrix.
3. **Persistence** — BM25 index pickled to `bm25_index.pkl`; FAISS index saved with `faiss.write_index()`. Both load in seconds at app startup via `@st.cache_resource`.
4. **Single-review strategy** — only the most helpful review per product is included, keeping text length manageable while maximizing information density.

**Code reference:** `src/utils.py::build_processed_corpus()` and `src/utils.py::build_indices()`

### Option 2: Tool Integration (Secondary)

We implemented a Tavily web search tool (`src/tools.py`) following the pattern from Lecture 8. The tool uses the `@tool` decorator from LangChain and the `TavilyClient`:

```python
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for current information about a beauty product..."""
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=max_results)
    return "\n".join([r["content"] for r in results["results"]])
```

The tool is exposed in the Streamlit RAG tab as an optional toggle (disabled without `TAVILY_API_KEY`). It fails soft — returning an empty string when the API key is missing.

**3 example queries where the tool was used:**

<!-- TODO: Fill in after running notebook -->

| # | Query | RAG-only answer (summary) | Web search added value? | Why |
|---|---|---|---|---|
| 1 | Is retinol safe to use with vitamin C serum? | (fill in) | (fill in) | (fill in) |
| 2 | best drugstore sunscreen 2025 dermatologist recommended | (fill in) | (fill in) | (fill in) |
| 3 | CeraVe moisturizer recall or safety issues | (fill in) | (fill in) | (fill in) |

**Code reference:** `src/tools.py`, `app/app.py` (RAG tab toggle)

## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Updated `README.md` with the RAG pipeline architecture diagram (Milestone 2), env var documentation, and repository structure reflecting all new files
- Added `CHANGELOG.md` entries for v0.2.0 and v0.3.0 covering all PRs
- Clear setup instructions: `make setup` → add `.env` → `make app`

### Code Quality Changes

- **No hardcoded file paths**: All paths use `pathlib.Path` relative to `BASE_DIR` or `Path(__file__).parent`
- **No API keys in source**: `HF_TOKEN` and `TAVILY_API_KEY` loaded via `python-dotenv` from `.env`; `.env` is gitignored; `.env.example` provides placeholders
- **Docstrings**: All public functions and classes in `src/` have at minimum one-line docstrings
- **Environment file**: `requirements.txt` is up to date with all dependencies pinned
- **Clean repository**: `.gitignore` covers `.claude/`, `local/`, `.env`, `*.pkl`, `*.parquet`, `index.faiss`, `__pycache__/`, and other artifacts

## Step 4: Cloud Deployment Plan

We outline how to deploy the Amazon Beauty RAG system on **AWS**, building on concepts from DSCI 525.

### 1. Data Storage

| Data | Service | Details |
|---|---|---|
| Raw data (JSONL.gz, Parquet) | **S3** | Store in `s3://beauty-rag/data/raw/`. Lifecycle policy moves to Glacier after 90 days since raw data is only needed for reprocessing. |
| Processed corpus (Parquet) | **S3** | `s3://beauty-rag/data/processed/product_corpus.parquet`. Versioned bucket to support rollback. |
| FAISS index | **S3** | `s3://beauty-rag/indices/faiss_index/`. Downloaded to container local disk at startup for fast memory-mapped access. |
| BM25 index (pickle) | **S3** | `s3://beauty-rag/indices/bm25_index.pkl`. Same download-at-startup pattern. |

### 2. Compute

| Component | Service | Details |
|---|---|---|
| Streamlit app | **ECS Fargate** | Containerized (Docker) behind an Application Load Balancer (ALB). Each task runs one Streamlit instance. Auto-scaling group scales from 1 to 4 tasks based on CPU utilization. |
| LLM inference | **HuggingFace Inference API** (external) | We continue using the hosted API rather than self-hosting a GPU instance. This avoids the cost of a `p3.2xlarge` ($3/hr) for a class project. For production, an **Amazon SageMaker endpoint** with a quantized model would provide lower latency and predictable costs. |
| Index building | **EC2 batch job** (or **Lambda** for smaller datasets) | Triggered on-demand or on schedule. Writes updated indices to S3. ECS tasks detect new index versions via S3 object metadata and reload. |

**Concurrency:** Streamlit is single-threaded per session. The ALB distributes users across multiple Fargate tasks. Each task loads indices into memory at startup (~500MB for FAISS + BM25 at 112K products). For larger datasets, we would use FAISS IVF index with memory-mapped files to reduce per-instance RAM.

### 3. Streaming/Updates

| Concern | Approach |
|---|---|
| New products | A scheduled **EventBridge rule** (e.g. weekly) triggers a **Step Functions** workflow: (1) Lambda fetches new reviews from the McAuley Lab source, (2) EC2 batch job rebuilds the corpus and indices, (3) new indices uploaded to S3, (4) ECS tasks detect the update and reload. |
| Pipeline freshness | The S3 index objects carry a version tag. At startup and on a 1-hour timer, each ECS task checks the tag and reloads if it has changed. This gives eventual consistency within 1 hour. |
| Model updates | If the HF Inference API deprecates or updates the Llama-3 endpoint, we swap the `model_id` in the `.env` config and redeploy. No code change needed — `load_llm()` already accepts `model_id` as a parameter. |
