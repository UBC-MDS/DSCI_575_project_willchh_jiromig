# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- **Number of products used:** 112,590 (All Beauty category)
- **Changes to sampling strategy:** None. We used the full category from the start. The corpus was built by concatenating title, description, features, and the most helpful review per product into a single text field. DuckDB handled remote JSONL.gz ingestion directly to local Parquet, avoiding memory issues with the full 701K reviews. This exceeds the 10,000 minimum by over 10x.

### LLM Experiment

**Models compared:**

| Model | Family | Parameters | Provider |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama 3 | 8B | HuggingFace Inference API (remote) |
| `qwen3:1.7b` | Alibaba Qwen 3 | 1.7B | Ollama (local) |

**Why these two models?** We wanted to compare a large API-hosted model against a smaller locally-run model. The 5x parameter gap (8B vs 1.7B) and different model families (Llama vs Qwen) let us examine how model capacity and training background affect RAG answer quality.

We originally planned to use `Qwen/Qwen3.5-2B` via the same HuggingFace Inference API. However, no HF provider currently hosts that model for serverless inference, resulting in a `BadRequestError`. We pivoted to Ollama, which runs the model locally on the machine's GPU (Apple M1 Pro Metal). This also introduced a practical comparison between remote API inference and local inference.

**Experimental setup:**
- Retriever: Hybrid (EnsembleRetriever with Reciprocal Rank Fusion)
- Prompt: `strict_citation` (answer only from context, cite ASINs)
- top_k: 5
- 5 queries tested, spanning easy_factual, paraphrased_intent, and multi_concept categories
- Both models received the exact same retrieved context and prompt for each query

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

| # | Query | Llama-3-8B (API) | Qwen3 1.7B (Local) | Notes |
|---|---|---|---|---|
| 1 | vitamin C serum for brightening | Listed 5 products with ASINs and brief descriptions. Kept responses grounded in context. Included a general disclaimer. | Listed 5 products with ASINs, ratings, and concise bullet descriptions. Added a recommendation section ranking top picks. | Both produced accurate, well-cited answers. Qwen was more structured with ratings inline and a clear recommendation section. |
| 2 | what helps with mild acne and post-acne marks? | Cited 3 products with specific reviewer claims. Conservative in scope. | Cited 6 products with ratings and summarized review content for each. More thorough coverage. | Qwen covered more products from the context. Llama was more selective but each citation was tightly grounded. |
| 3 | gentle face wash for sensitive skin | Listed 9 products with reviewer quotes and specific claims. Very thorough. | Listed 6 products with ratings and key benefit summaries. More concise. | Llama provided the most detailed answer here with many citations. Qwen was more organized but covered fewer products. |
| 4 | skincare routine for oily acne-prone teen skin under $30 | Suggested a 3-step routine (cleanser, exfoliant, spot treatment) with ASINs. Noted prices were unavailable ($nan). | Suggested a structured routine with cleanser/toner/moisturizer steps. Organized by product kits. | Both struggled with the price constraint since most products lack price data. Qwen's routine structure was cleaner, but it incorrectly suggested using a face wash as a toner. |
| 5 | what helps with sun damage on fair skin around the eyes | Responded: "I don't have enough information." | Listed 3 products (sunscreen, facial mask, eye serum) with ASINs and descriptions. | Major difference. Llama followed the strict_citation prompt and refused to answer. Qwen attempted an answer, citing products that were loosely relevant but not specifically about sun damage repair around the eyes. |

**Key observations:**

1. **Citation quality:** Llama-3-8B was more conservative with its citations. When the context was insufficient (Query 5), it refused to answer rather than stretch the available evidence. Qwen3 1.7B was more willing to make connections between the context and the query, even when the match was approximate. This is a tradeoff: Llama has higher precision but lower recall, while Qwen prioritizes completeness.

2. **Answer structure:** Qwen consistently produced better-organized answers with ratings included inline, bold formatting, and clear recommendation sections. Llama's answers read more like natural paragraphs or simple numbered lists. For a product recommendation use case, Qwen's formatting is arguably more useful.

3. **Instruction following:** Llama followed the strict_citation prompt more faithfully. The instruction says to reply "I don't have enough information" when context is insufficient, and Llama did exactly that on Query 5. Qwen ignored this instruction and produced an answer anyway. For applications where grounding and factual accuracy matter, Llama's behavior is preferable.

4. **Hallucination risk:** Qwen's answer to Query 4 incorrectly suggested using a face wash as a toner, which is not something any review stated. Qwen also reused the same ASIN (B01CNFRQPQ) for two different products in Query 5, suggesting it may have confused metadata. Llama did not exhibit these issues.

5. **Completeness vs. caution:** Across all 5 queries, Qwen provided more product recommendations (6 products on average vs. Llama's 4-5). Whether this is an advantage depends on the use case. For browsing and discovery, more suggestions are helpful. For trusted recommendations, fewer but better-grounded citations are safer.

**Which model we chose and why:**

We retained `meta-llama/Meta-Llama-3-8B-Instruct` as the default model. Despite Qwen3 1.7B producing well-structured and more complete answers, Llama's stronger instruction following and lower hallucination risk make it the better choice for a RAG system where grounding in the source material is the primary concern. The strict_citation prompt was designed to prevent the model from going beyond the provided context, and Llama respected that boundary consistently.

That said, Qwen3's formatting and recommendation structure would be a better fit for the `helpful_shopper` prompt variant, where the goal is conversational helpfulness rather than strict grounding.

## Step 2: Additional Feature

We chose **Option 3 (Scale to >= 100K Products)** as our primary feature and **Option 2 (Tool Integration)** as a secondary feature. Both were implemented in earlier milestones.

### Option 3: Scale to >= 100K Products (Primary)

Our pipeline processes **112,590 products** from the Amazon All Beauty category, exceeding the 100K threshold. The engineering decisions that enabled this scale:

1. **DuckDB for ingestion:** Reads remote JSONL.gz files directly via HTTP and converts to local Parquet with ZSTD compression. This avoids loading the full dataset into memory and handles the 701K reviews efficiently.
2. **Batch encoding for FAISS:** `sentence-transformers` encodes all 112K product texts in a single `model.encode()` call with batched processing, producing a (112590, 384) float32 matrix.
3. **Persistence:** BM25 index is pickled to `bm25_index.pkl`; FAISS index is saved with `faiss.write_index()`. Both load in seconds at app startup via Streamlit's `@st.cache_resource`.
4. **Single-review strategy:** Only the most helpful review per product is included, keeping text length manageable while maximizing information density.

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

The tool fails soft when `TAVILY_API_KEY` is not set, returning an empty string. In the Streamlit app, the toggle is disabled without the key.

**3 example queries where the tool was used:**

| # | Query | RAG-only answer | Web search result | Did web search help? |
|---|---|---|---|---|
| 1 | Is retinol safe to use with vitamin C serum? | "I don't have enough information." The corpus contains product reviews, not dermatological advice. | Returned detailed information about using retinol and vitamin C together, including recommendations to use retinol at night and vitamin C in the morning to avoid irritation. | **Yes.** The corpus has no information on ingredient interactions. The web search provided the exact kind of expert guidance the user was looking for. |
| 2 | best drugstore sunscreen 2025 dermatologist recommended | Suggested a few sunscreens from the corpus but could not verify "2025" or "dermatologist recommended" claims. | Returned current 2025 dermatologist recommendations including EltaMD UV Skin Recovery SPF 40 and other products not in our corpus. | **Yes.** The corpus data is static (from 2023). Web search provided up-to-date 2025 recommendations that the corpus simply cannot contain. |
| 3 | CeraVe moisturizer recall or safety issues | "I don't have enough information" about recalls. Found a positive CeraVe bundle review but nothing about safety issues. | Returned an FDA Advisory (2025-0542) warning against an unauthorized CeraVe Vitamin C Serum product, with specific regulatory details. | **Yes.** Safety and recall information changes over time and would never appear in product reviews. The web search surfaced a real FDA warning that is critical for consumer safety. |

**Key takeaway:** The web search tool adds the most value for three types of queries: (1) general knowledge questions about ingredients or skincare science, (2) queries that reference specific time periods or current events, and (3) safety and regulatory questions. These are precisely the areas where a static review corpus falls short.

**Code reference:** `src/tools.py`, `app/app.py` (RAG tab toggle), `notebooks/milestone3_final.ipynb` (demo cells)

## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Updated `README.md` with the RAG pipeline architecture diagram, environment variable documentation, and repository structure reflecting all milestone files
- Added `CHANGELOG.md` entries for v0.2.0 and v0.3.0 covering all PRs
- Setup instructions cover the full workflow: `make setup`, `.env` configuration, `make app`
- Added Ollama installation instructions for macOS, Linux, and Windows

### Code Quality Changes

- **No hardcoded file paths:** All paths use `pathlib.Path` relative to `BASE_DIR` or `Path(__file__).parent`
- **No API keys in source:** `HF_TOKEN` and `TAVILY_API_KEY` are loaded via `python-dotenv` from `.env`. The `.env` file is gitignored. `.env.example` provides placeholder values.
- **Docstrings:** All public functions and classes in `src/` have at minimum a one-line docstring. This was verified programmatically using an AST-based audit.
- **Environment file:** `requirements.txt` is up to date with all dependencies pinned to specific versions, including `langchain-ollama` for the final milestone.
- **Clean repository:** `.gitignore` covers `.claude/`, `local/`, `.env`, `*.pkl`, `*.parquet`, `index.faiss`, `__pycache__/`, and other build artifacts. No large files or temporary outputs are committed.

## Step 4: Cloud Deployment Plan

We outline how to deploy the Amazon Beauty RAG system on **AWS**, building on concepts from DSCI 525.

### 1. Data Storage

| Data | Service | Details |
|---|---|---|
| Raw data (JSONL.gz, Parquet) | **S3** | Store in `s3://beauty-rag/data/raw/`. A lifecycle policy moves files to Glacier after 90 days since raw data is only needed for reprocessing. |
| Processed corpus (Parquet) | **S3** | `s3://beauty-rag/data/processed/product_corpus.parquet`. The bucket is versioned to support rollback if a corpus rebuild introduces errors. |
| FAISS index | **S3** | `s3://beauty-rag/indices/faiss_index/`. Downloaded to the container's local disk at startup for fast memory-mapped access. |
| BM25 index (pickle) | **S3** | `s3://beauty-rag/indices/bm25_index.pkl`. Same download-at-startup pattern as FAISS. |

### 2. Compute

| Component | Service | Details |
|---|---|---|
| Streamlit app | **ECS Fargate** | Containerized with Docker, placed behind an Application Load Balancer (ALB). Each task runs one Streamlit instance. An auto-scaling group scales from 1 to 4 tasks based on CPU utilization. |
| LLM inference | **HuggingFace Inference API** (external) | We continue using the hosted API rather than self-hosting a GPU instance. This avoids the cost of a `p3.2xlarge` (~$3/hr) for a class project. For production, an **Amazon SageMaker endpoint** with a quantized model would provide lower latency and predictable costs. |
| Index building | **EC2 batch job** | Triggered on-demand or on a schedule. The job rebuilds the BM25 and FAISS indices from the latest corpus and writes updated indices to S3. ECS tasks detect new index versions via S3 object metadata and reload. |

**Concurrency:** Streamlit is single-threaded per session. The ALB distributes users across multiple Fargate tasks. Each task loads indices into memory at startup (~500MB for FAISS + BM25 at 112K products). For larger datasets, we would switch to a FAISS IVF index with memory-mapped files to reduce per-instance RAM.

### 3. Streaming/Updates

| Concern | Approach |
|---|---|
| New products | A scheduled **EventBridge rule** (e.g., weekly) triggers a **Step Functions** workflow: (1) Lambda fetches new reviews from the McAuley Lab source, (2) EC2 batch job rebuilds the corpus and indices, (3) new indices are uploaded to S3, (4) ECS tasks detect the update and reload. |
| Pipeline freshness | The S3 index objects carry a version tag. At startup and on a 1-hour timer, each ECS task checks the tag and reloads if it has changed. This provides eventual consistency within 1 hour. |
| Model updates | If the HF Inference API deprecates or updates the Llama-3 endpoint, we swap the `model_id` in the `.env` config and redeploy. No code change is needed since `load_llm()` already accepts `model_id` as a parameter. |
