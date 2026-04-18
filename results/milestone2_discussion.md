# Milestone 2: RAG Pipeline Discussion

## Model Choice

We use `meta-llama/Meta-Llama-3-8B-Instruct` via the HuggingFace Inference API
(`langchain_huggingface.HuggingFaceEndpoint` + `ChatHuggingFace`). Rationale:

- **No local compute required** — graders need only an `HF_TOKEN` and the accepted
  Meta Llama 3 license, no GPU.
- **8B parameters** is a strong balance for RAG: good instruction-following without
  needing 16GB+ VRAM that a local model would require.
- The milestone document explicitly lists this exact model in the Option 1 example
  for students without a GPU.

## Prompt Variant Comparison

We tested three system prompts against the query
*"what's a good vitamin C serum for dark spots under $30?"*:

| Variant | Output style | Strengths | Weaknesses |
|---|---|---|---|
| `strict_citation` | Short, ASIN-cited, lists products with ratings | Most faithful to context; cites ASINs for every claim; rarely hallucinates | Can sound terse and mechanical; reads like a product listing rather than advice |
| `helpful_shopper` | Conversational recommendation with price/rating | Most readable and natural; mentions price and rating organically | Occasionally drifts from strict grounding; may embellish beyond what reviews say |
| `structured_json` | JSON object with recommendation, reasoning, asins | Easy to consume programmatically; deterministic schema | Sometimes leaks markdown around the JSON; less human-friendly for display |

## Evaluation: 5 of 10 Queries

Pipeline configuration: Hybrid retriever (EnsembleRetriever with RRF) + `strict_citation` prompt + `top_k=5`.

| # | Query | Category | Accuracy | Completeness | Fluency | Notes |
|---|---|---|---|---|---|---|
| 1 | vitamin C serum for brightening | easy_factual | Yes | Yes | Yes | Listed 5 relevant products with ASINs and ratings. All cited products exist in the corpus. Answer is well-structured and actionable. |
| 4 | what helps with mild acne and post-acne marks? | paraphrased_intent | Yes | Yes | Yes | Recommended relevant products (tamanu oil, tea tree treatments). Correctly cited ASINs. Covered both acne treatment and post-acne marks as requested. |
| 5 | gentle face wash that won't irritate sensitive skin | paraphrased_intent | Yes | Yes | Yes | Identified appropriate gentle/sensitive-skin cleansers. Correctly noted product attributes from reviews. Fluent and well-organized response. |
| 8 | best skincare routine for oily acne-prone teenage skin under $30 | multi_concept | Partial | No | Yes | Attempted to recommend a routine but did not consistently enforce the $30 price constraint. Some suggested products lacked price data in the corpus, making budget verification impossible. |
| 10 | what helps with sun damage on fair skin around the eyes | multi_concept | Partial | No | Yes | Found a relevant eye treatment product but acknowledged the context was insufficient for a comprehensive answer. The query requires domain knowledge beyond what reviews provide. |

## Key Observations

- **Easy factual queries** (1, 2, 3) performed well across all dimensions. When the query closely matches product titles and review language, the hybrid retriever surfaces highly relevant documents and the LLM generates accurate, well-cited answers.
- **Paraphrased intent queries** (4, 5) also performed well. The semantic component of the hybrid retriever successfully matched user intent to product descriptions even when exact keywords differed.
- **Multi-concept queries** (8, 10) were the most challenging. These queries combine multiple constraints (age group + skin type + budget, or skin condition + skin tone + body area) that are difficult to satisfy from a single retrieval pass.

## Limitations

1. **Numeric constraints (e.g. "under $30") are not enforced** — the LLM relies entirely on what surfaces from retrieval. It does not filter results by the price metadata, so it may recommend products without price information or above the stated budget. A structured filtering step before generation would address this.

2. **Single-review context per product is limiting** — each product in the corpus includes only the most helpful review, which may not be representative. Products with mixed reviews may appear more positive or negative than they actually are, and the LLM has no way to assess overall sentiment distribution.

3. **The hybrid retriever uses RRF (Reciprocal Rank Fusion)** on the RAG tab, while the Search tab uses the Milestone 1 min-max weighted hybrid. Different fusion algorithms can rank the same documents differently, which may confuse users comparing results across tabs.

4. **Rate limits on the free HF Inference API** can cause cold starts and sporadic 429 errors during evaluation runs, making reproducibility of exact outputs difficult.

## Suggested Improvements

1. Add structured filtering in the retriever (price ceiling, minimum rating) before passing documents to the LLM, so prompt-level constraints like "under $30" become enforced rather than advisory.

2. Include multiple reviews per product (e.g. top 3 most helpful) to give the LLM a more balanced view of product sentiment, improving accuracy for products with mixed reviews.

3. Explore a cross-encoder re-ranker between retrieval and generation to improve precision on multi-concept queries where the initial retrieval may surface partially-relevant documents.
