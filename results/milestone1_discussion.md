# Milestone 1: Qualitative Evaluation & Discussion

## 4.3 Side-by-Side Comparison: BM25 vs Semantic

We compare BM25 and semantic retrieval on 5 representative queries (2 easy, 2 medium, 1 hard), showing top-5 results per method.

---

### Query 1: "vitamin C serum" [easy]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Organic Vitamin C Serum for Face-Professional (21.47) | PURE VITAMIN C SERUM (0.94) |
| 2 | 24K Vitamin C Serum for Face 2 PACK (21.08) | Vitamin C Plus Serum (0.92) |
| 3 | Renew Vitamin C Serum (20.85) | Vitamin C Serum 1oz (0.87) |
| 4 | Vitamin C Serum 1oz (20.85) | Renew Vitamin C Serum (0.85) |
| 5 | Vitamin C Plus Serum (20.85) | RA Herbals Ultimate Vitamin C Serum (0.78) |

**Comment:** Both methods retrieve highly relevant products. BM25 ranks the longest title first because it contains more keyword repetitions, while semantic ranks "PURE VITAMIN C SERUM" highest for direct meaning alignment. The result sets overlap significantly—this is expected for a query that exactly matches product vocabulary.

---

### Query 2: "sunscreen SPF 50" [easy]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Cerave Sunscreen Bundle SPF 50 (24.09) | Perfectly Plain Collection Sunscreen with SPF (0.85) |
| 2 | Anti-aging sunscreen SPF 50+ Duolys (22.80) | COCOOIL Topical Sunscreen SPF50 (0.84) |
| 3 | Coppertone Kids Clear Sunscreen Lotion SPF (21.98) | Sunscreen + Powder Broad-Spectrum SPF 40 (0.84) |
| 4 | 2 Pack SPF 50 Baby Sunscreen Lotion (21.62) | Anti-aging sunscreen SPF 50+ Duolys (0.82) |
| 5 | Heliocare Ultra SPF 50+ Gel Facial Sunscreen (20.89) | Moisturizing Sunscreen Lotion, SPF 30 (0.81) |

**Comment:** BM25 is more precise here—all top-5 results contain exactly "SPF 50." Semantic retrieval surfaces an SPF 40 product (#3) and an SPF 30 product (#5) because it understands "sunscreen" conceptually but cannot enforce the numeric constraint. This highlights a key limitation of embedding models: they struggle with exact numeric/specification matching.

---

### Query 3: "something to reduce acne scars" [medium]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Crocodile Repair Face Serum for Acne Scars (18.06) | Natural Acne Scar Removal Cream (0.76) |
| 2 | Crocodile Repair Face Serum for Acne Scars (18.06) | Acne Scar Reducing Cream (0.75) |
| 3 | Cutie Academy Crocodile Repair Face Serum (17.56) | New York Laboratories Acne Scar Removal Cream (0.73) |
| 4 | Natural Acne Scar Removal Cream (16.90) | Scar Removal Cream for Women and Men (0.71) |
| 5 | NOTS 28 Remedy Intensive Serum – Anti Acne (16.60) | Acne Scar Cream from New York Biology (0.71) |

**Comment:** BM25 matches on "acne scars" literally but returns duplicate products (Crocodile Repair Serum appears twice). Semantic retrieval returns a more diverse set—all are scar-treatment products. Semantic also ranks a general "Scar Removal Cream" at #4, understanding that scar treatment is relevant even without the word "acne." The word "something" in the query is noise for BM25 but invisible to the embedding model, which focuses on intent.

---

### Query 4: "gentle cleanser for sensitive skin" [medium]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Minimalist 6% Oat Extract Gentle Cleanser (20.54) | EveryShine Rose Mousse Foam Cleanser for Sensitive (0.71) |
| 2 | Tata Harper Refreshing Cleanser (18.94) | Face Foam Cleanser for Sensitive Dry Oily (0.68) |
| 3 | TULA Skin Care So Gentle Cleanser (18.38) | Dermatouch M.E.D. Cleanser for problem skin (0.67) |
| 4 | Men's 2-Step Sensitive Skin Acne Cleanser (18.38) | Serious Skin Care Glycolic Cleanser (0.67) |
| 5 | One Over One - Natural Micellar Cleansing (18.01) | Calming Chamomile Daily Face Cleanser (0.67) |

**Comment:** BM25 anchors on "gentle" and "cleanser" keywords, retrieving products that explicitly use those terms. Semantic retrieval finds products that *imply* gentleness (e.g., "Rose Mousse Foam," "Calming Chamomile") without necessarily using the word "gentle." However, semantic also retrieves "Glycolic Cleanser" (#4)—glycolic acid is typically not gentle, showing the model doesn't fully understand product ingredient implications. BM25's top result (Minimalist Oat Extract) is arguably the best match overall.

---

### Query 5: "best anti-aging routine for dry skin under $25" [hard]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Night Cream For Dry Skin - Intensive Repair (19.44) | 24.7 Skin Care Anti Aging Skin Care Primer (0.70) |
| 2 | Best Eye Cream for Dark Circles Under Eyes (18.90) | Skin Opulent advanced anti-aging cream (0.70) |
| 3 | Best Selling Anti-Aging Skin Care Kits (16.99) | NEW Botanical Anti Aging Anti Wrinkle Cream (0.69) |
| 4 | BEST ANTI-AGING FACE MASK! Strawberry Exfoliating (15.67) | Odacité Dry + Mature Try Me Kit (0.68) |
| 5 | Men's Routine Pheromone Cream (15.57) | True Derma Daily Anti-Aging Serum (0.68) |

**Comment:** Both methods struggle with this multi-faceted query. BM25 matches on individual keywords ("best," "anti-aging," "dry skin," "under") and retrieves an eye cream for "dark circles *under* eyes" at #2—a false positive from the word "under." It also retrieves "Men's Routine Pheromone Cream" at #5, matching on "routine" alone. Semantic retrieval finds more topically relevant anti-aging products but cannot filter by price ($25 constraint). Neither method handles the compound intent (anti-aging + dry skin + budget) well, which is where a RAG pipeline with structured metadata filtering would help.

---

## Where BM25 Fails but Semantic Succeeds

- **Intent-based queries** (e.g., "something to protect from sun damage"): BM25 cannot bridge vocabulary gaps. The word "protect" doesn't match "SPF" or "sunscreen," so BM25 retrieves irrelevant results. Semantic search correctly maps the intent to sun protection products.
- **Paraphrased queries** (e.g., "product to make hair less frizzy"): BM25 needs exact keyword overlap. Semantic search understands that "less frizzy" relates to anti-frizz and smoothing products.
- **Conversational/vague queries** (e.g., "something to keep skin hydrated all day"): The word "something" adds noise for BM25 but is ignored by the embedding model, which focuses on "skin hydrated."

## Where Semantic Fails

- **Exact specifications**: Semantic search retrieves SPF 30 for an "SPF 50" query because the model treats these as semantically similar. BM25 enforces exact number matching.
- **Ingredient-level understanding**: The model doesn't reliably distinguish between gentle and harsh ingredients (e.g., retrieving "Glycolic Cleanser" for a "gentle cleanser" query).
- **Price/budget constraints**: Embedding models encode meaning, not structured attributes like price. "Under $25" is understood conceptually but cannot be enforced.

## Performance by Query Difficulty

| Difficulty | BM25 | Semantic | Hybrid |
|------------|------|----------|--------|
| **Easy** (exact keyword queries) | Strong — keywords match directly | Strong — but may include wrong specs (SPF 30 for SPF 50) | Strong — combines precision of both |
| **Medium** (paraphrased/intent queries) | Mixed — depends on keyword overlap; struggles with "something to…" phrasing | Better — captures intent even without keyword match | Best — semantic handles intent while BM25 boosts exact matches |
| **Hard** (multi-constraint queries) | Weak — matches individual words, produces false positives from "under," "best," "routine" | Moderate — captures theme but ignores structured constraints (price, specs) | Moderate — inherits limitations of both on structured constraints |

## Summary of Insights

1. **BM25 excels at precision** for keyword-specific queries but fails when users paraphrase or describe needs conversationally.
2. **Semantic search excels at recall** for intent-based queries but cannot enforce exact specifications or structured constraints.
3. **Hybrid retrieval provides the best balance** — it surfaces products that score well across both dimensions, reducing the failure modes of each individual method.
4. **Neither method handles multi-constraint queries well** (e.g., combining product type + skin type + budget). This is the gap a RAG pipeline could fill by layering LLM reasoning over retrieved candidates to filter on structured attributes like price and ratings.
5. **Duplicate products in the corpus** (visible in BM25 results for "acne scars") suggest deduplication as a preprocessing improvement.
