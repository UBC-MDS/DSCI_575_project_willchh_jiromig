# DSCI 575 — Amazon Beauty Product Search

Hello Ladies and Gentlemen, Are you ready to bring your skin care to the next level ?!?!

We have created a retrieval-based product search system for the Amazon All Beauty dataset.
A Streamlit web app for interactive querying is implemented with three retrieval methods: BM25, semantic search, and hybrid.

**Team:** Jiro Amato(Big 🐐), William Chong(small 🐐)

## Quick Start

HELLO TAs, these condensed steps are just for you. Follow along and you can finish marking in no time. Woohoo!

| Step | Command | What it does |
|------|---------|-------------|
| STEP 1 | `git clone https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig.git && cd DSCI_575_project_willchh_jiromig` | Clone the repo |
| STEP 2 | `conda env create -f environment.yml && conda activate 575-project` | Create and activate the conda environment (Python 3.12 + all dependencies) |
| STEP 3 | `make setup` | Copy `.env.example` to `.env` (no API keys needed for Milestone 1) |
| STEP 4 | Open and Run `notebooks/milestone1_exploration.ipynb` top to bottom | Download data, build corpus, build BM25 and FAISS indices |
| STEP 5 | `make app` | Launch the Streamlit app at `http://localhost:8501` |
| STEP 6 | `make test` | Run the test suite with coverage |
| STEP 7 | `make lint` | Run pre-commit linting hooks |

## Repository Structure

```
DSCI_575_project_willchh_jiromig/
├── app/
│   └── app.py                  # Streamlit web app
├── data/
│   ├── raw/                    # Raw dataset files (gitignored)
│   └── processed/
│       └── ground_truth.csv    # 21 evaluation queries across 3 difficulty tiers
├── indices/                    # Persisted search indices (gitignored)
├── notebooks/
│   ├── milestone1_exploration.ipynb            # EDA, corpus building, index building
│   └── milestone1_retrieval_evaluations.ipynb  # Retrieval evaluation across all methods
├── results/
│   └── milestone1_discussion.md  # Qualitative evaluation write-up
├── src/
│   ├── utils.py                # Data loading, text preprocessing, tokenization
│   ├── bm25.py                 # BM25Retriever class
│   ├── semantic.py             # SemanticRetriever class (FAISS + sentence-transformers)
│   └── hybrid.py               # HybridRetriever class (weighted BM25 + semantic)
├── tests/                      # pytest test suite
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python package dependencies
└── Makefile                    # Common commands (setup, app, test, lint)
```

## Detailed Steps

Below are the steps that anyone can follow to reproduce the full workflow.

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_willchh_jiromig.git
cd DSCI_575_project_willchh_jiromig
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate 575-project
```

This installs Python 3.12 and all dependencies listed in `requirements.txt` (including `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `duckdb`, `streamlit`, etc.).

### 3. Run initial setup

```bash
make setup
```

This copies `.env.example` to `.env`. No API keys are required for Milestone 1.

### 4. Build the corpus and indices

Run the notebook `notebooks/milestone1_exploration.ipynb` from top to bottom. This will:

1. Download the All Beauty metadata from HuggingFace using DuckDB
2. Perform EDA on the dataset
3. Build the product corpus and save it as `data/processed/product_corpus.parquet`
4. Build and persist the BM25 index (`indices/bm25_index.pkl`) and FAISS index (`indices/faiss_index/`)

This step is required before running the app.

### 5. Run the app

```bash
make app
```

Launches the Streamlit app at `http://localhost:8501`. You can search using BM25, Semantic, or Hybrid mode.

### 6. Run tests and linting

```bash
make test
make lint
```

`make test` runs the pytest suite with coverage. `make lint` runs pre-commit formatting hooks.

## Dataset

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — "All Beauty" category

**Size:** ~113,000 products

**Fields used for retrieval:**

| Field | Description |
|-------|-------------|
| `parent_asin` | Unique product identifier |
| `title` | Product name |
| `description` | Product description text |
| `features` | Bullet-point feature list |
| `price` | Product price |
| `average_rating` | Mean user rating (1–5) |

The raw metadata file (`meta_All_Beauty.jsonl.gz`) is downloaded automatically by the notebook via DuckDB from HuggingFace and is not committed to the repository.

## Data Processing

Each product's text fields are concatenated into a single retrieval document:

```
text = title + description + features
```

Products with no text content are excluded. The resulting corpus is saved as a compressed parquet file (`data/processed/product_corpus.parquet`).

**Tokenization (for BM25):**
- Lowercase all text
- Split hyphens and slashes into separate tokens
- Remove punctuation
- Remove English stopwords (via NLTK)
- Lemmatize tokens (via NLTK WordNetLemmatizer)

Semantic search uses raw text directly — the sentence-transformer model handles its own tokenization.

## Retrieval Methods

### BM25 (keyword-based)

Uses the [rank-bm25](https://github.com/dorianbrown/rank_bm25) library (`BM25Okapi`). Queries and documents are tokenized with the same preprocessing pipeline (lowercase, stopword removal, lemmatization). Returns results ranked by BM25 score.

**Strengths:** Exact keyword matching, fast, no model required.

### Semantic Search (embedding-based)

Uses [sentence-transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2`) to encode documents and queries into 384-dimensional vectors. A [FAISS](https://faiss.ai/) inner-product index (`IndexFlatIP`) stores normalized embeddings and retrieves by cosine similarity.

**Strengths:** Understands meaning and intent beyond exact keywords (e.g., "something to protect from sun damage" retrieves sunscreen products).

### Hybrid Search

Combines BM25 and semantic scores using weighted linear combination:

1. Retrieve `top_k * 3` candidates from each method
2. Min-max normalize each score set to [0, 1]
3. Combine: `score = bm25_weight * bm25_score + (1 - bm25_weight) * semantic_score`
4. Re-rank and return top-k

Default weight is 0.5 (equal contribution).

## Running the App

```bash
make app
```

This launches a Streamlit app at `http://localhost:8501` with:

- **Search mode selector:** BM25, Semantic, or Hybrid
- **Query input:** free-text search box
- **Results display:** product title, truncated text, price, star rating, and retrieval score
- **Feedback buttons:** thumbs up/down per result (saved to `data/feedback.csv`)

## Running Tests

```bash
make test
```

Runs the pytest suite with coverage reporting. Tests use a small fixture corpus (10 products) defined in `tests/conftest.py` — no real dataset required.

## Qualitative Evaluation

We evaluated 21 queries across three difficulty tiers (easy, medium, hard) against all three retrieval methods. Results and discussion are in:

- `notebooks/milestone1_retrieval_evaluations.ipynb` — full evaluation runs
- `results/milestone1_discussion.md` — side-by-side comparisons and analysis
