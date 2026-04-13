# Milestone 1: Qualitative Evaluation & Discussion

## 4.3 Side-by-Side Comparison: BM25 vs Semantic

We compare BM25 and semantic retrieval on 5 representative queries (2 easy, 2 medium, 1 hard), showing top-5 results per method.

---

### Query 1: "vitamin C serum" [easy]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Organic Vitamin C Serum for Face-Professional Strength (21.65) | PURE VITAMIN C SERUM (0.90) |
| 2 | PREMIUM Vitamin C Serum For Face and Eyes (21.61) | Liz K Super First C Serum Pure Vitamin C 13% (0.73) |
| 3 | 24K Vitamin C Serum for Face 2 PACK (21.41) | Vitamin C Serum for Face - C Booster With Hyaluronic Acid (0.72) |
| 4 | Springs Organic Vitamin C Serum For Your Face (21.36) | Charlotte Elizabeth Organics Advanced Illuminating Vitamin C Serum (0.71) |
| 5 | Mererke_Pretty Vitamin C Serum for Face (21.27) | Collagen SPA Enriching Serum With Vitamin C (0.71) |

**Comment:** Both methods retrieve highly relevant products. BM25 ranks the longest titles first because they contain more keyword repetitions (higher term frequency), while semantic ranks "PURE VITAMIN C SERUM" highest for direct meaning alignment despite its short title. The scores are tightly clustered in both methods, confirming this is an easy query where product vocabulary matches the query directly.

---

### Query 2: "sunscreen SPF 50" [easy]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Cerave Sunscreen Bundle SPF 50 (24.09) | Perfectly Plain Collection Sunscreen with SPF (0.85) |
| 2 | Anti-aging sunscreen SPF 50+ Duolys (22.80) | COCOOIL Topical Sunscreen SPF50 (0.84) |
| 3 | Coppertone Kids Clear Sunscreen Lotion SPF (21.98) | Sunscreen + Powder Broad-Spectrum SPF 40 (0.84) |
| 4 | 2 Pack SPF 50 Baby Sunscreen Lotion (21.62) | Anti-aging sunscreen SPF 50+ Duolys (0.82) |
| 5 | Heliocare Ultra SPF 50+ Gel Facial Sunscreen (20.89) | Moisturizing Sunscreen Lotion, SPF 30 (0.81) |

**Comment:** BM25 is more precise here. All top-5 results contain exactly "SPF 50" or "SPF 50+." Semantic retrieval surfaces an SPF 40 product (#3) and an SPF 30 product (#5) because it understands "sunscreen" conceptually but cannot enforce the numeric constraint. This highlights a key limitation of embedding models: they struggle with exact numeric/specification matching.

---

### Query 3: "something to reduce acne scars" [medium]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Lycopene Skin Care (18.06) | Natural Acne Scar Removal Cream (0.76) |
| 2 | Natural Acne Scar Removal Cream (18.06) | Acne Scar Reducing Cream (0.75) |
| 3 | Cutie Academy Crocodile Repair Face Serum (17.56) | New York Laboratories Acne Scar Removal Cream (0.73) |
| 4 | tarte Mini Timeless Smoothing Primer (16.90) | Scar Removal Cream for Women and Men (0.71) |
| 5 | NOTS 28 Remedy Intensive Serum – Anti Acne (16.60) | Acne Scar Cream from New York Biology (0.71) |

**Comment:** BM25 returns mostly relevant products but is distracted by tangential matches: Lycopene Skin Care at #1 is about reducing fine lines and fighting acne broadly, not specifically scar treatment, and tarte Mini Timeless Smoothing Primer at #4 is a cosmetic primer, not a scar treatment. Semantic retrieval returns a more focused set since five results are scar-treatment products. The word "something" in the query is noise for BM25 but invisible to the embedding model, which focuses on intent.

---

### Query 4: "gentle cleanser for sensitive skin" [medium]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Natural Face Wash for Sensitive Skin - Gentle Anti Aging (20.54) | Medicalia Gentle Cleanser (0.71) |
| 2 | Medicalia Gentle Cleanser (18.94) | Natural Face Wash for Sensitive Skin (0.68) |
| 3 | PHL Naturals Sensitive Skin Cleanser (18.38) | Dermatouch M.E.D. Cleanser for problem skin (0.67) |
| 4 | SELF/ish Men's Face Scrub (18.38) | SELF/ish Men's Face Scrub (0.67) |
| 5 | Calming Chamomile Daily Face Cleanser (18.01) | Calming Chamomile Daily Face Cleanser (0.67) |

**Comment:** BM25 anchors on "gentle," "cleanser," and "sensitive skin" keywords, retrieving products that explicitly use those terms. Semantic retrieval finds largely the same products but reranks them differently. Medicalia Gentle Cleanser scores highest semantically for its direct alignment with the query intent. Both methods surface SELF/ish Men's Face Scrub (#4), an *exfoliating* scrub that is the opposite of "gentle," showing that neither method fully understands product ingredient implications. BM25's top result (Natural Face Wash for Sensitive Skin) is arguably the best match overall.

---

### Query 5: "what helps with sun damage on fair skin" [hard]

| # | BM25 | Semantic |
|---|------|----------|
| 1 | Flawless Finish Foundation (19.44) | Gold Bond Dark Spot Minimizing Cream (0.65) |
| 2 | Josie Maran Self Tanning Cream (18.90) | Ban The Sun After Sun Soothing Gel (0.64) |
| 3 | Best Selling Anti-Aging Skin Care Kits (16.99) | Aloe Soapberry Face Wash - Natural Cleanser For Sun Damage (0.63) |
| 4 | BEST ANTI-AGING FACE MASK! Strawberry Exfoliating (15.67) | Surface Sheer Touch Spray Sunscreen (0.62) |
| 5 | COL-LAB Sun Obsession Sculpting Bronzer (15.57) | Vitamin C Serum for Face - Anti Aging (0.62) |

**Comment:** Both methods struggle significantly with this multi-concept conversational query. BM25 returns largely irrelevant products: Flawless Finish Foundation at #1 matched on "skin" and "fair," Josie Maran Self Tanning Cream at #2 matched on "skin" and tanning-related terms, and COL-LAB Sun Obsession Sculpting Bronzer at #5 is a cosmetic bronzer with no sun-repair function. Semantic does better thematically: Gold Bond Dark Spot Minimizing Cream (#1) and Ban The Sun After Sun Soothing Gel (#2) are at least related to sun damage recovery—but scores are low across the board (0.62–0.65), reflecting the model's difficulty bridging a conversational, multi-concept query to product titles.

---

## Where BM25 Fails but Semantic Succeeds

- **Intent-based queries** (e.g., "something to protect from sun damage"): BM25 cannot bridge vocabulary gaps. The word "protect" doesn't match "SPF" or "sunscreen," so BM25 retrieves irrelevant results. Semantic search correctly maps the intent to sun protection products.
- **Paraphrased queries** (e.g., "product to make hair less frizzy"): BM25 needs exact keyword overlap. Semantic search understands that "less frizzy" relates to anti-frizz and smoothing products.
- **Conversational/vague queries** (e.g., "something to keep skin hydrated all day"): The word "something" adds noise for BM25 but is ignored by the embedding model, which focuses on "skin hydrated."

## Where Semantic Fails

- **Exact specifications**: Semantic search retrieves SPF 30 for an "SPF 50" query because the model treats these as semantically similar. BM25 enforces exact number matching.
- **Ingredient-level understanding**: The model doesn't reliably distinguish between gentle and harsh ingredients (e.g., retrieving an exfoliating scrub for a "gentle cleanser" query).
- **Vague multi-concept queries**: For "what helps with sun damage on fair skin," semantic scores are uniformly low (0.62–0.65), showing the model cannot confidently match a conversational, multi-concept query to product-level descriptions.

## Performance by Query Difficulty

| Difficulty | BM25 | Semantic | Hybrid |
|------------|------|----------|--------|
| **Easy** (exact keyword queries) | Strong - keywords match directly | Strong - but may include wrong specs (SPF 30 for SPF 50) | Strong - combines precision of both |
| **Medium** (paraphrased/intent queries) | Mixed - depends on keyword overlap; distracted by tangential matches | Better - captures intent even without keyword match | Best - semantic handles intent while BM25 boosts exact matches |
| **Hard** (multi-constraint queries) | Weak - matches individual words, produces false positives from "fair," "skin," "sun" | Moderate - captures theme but scores are low and results lack precision | Moderate - inherits limitations of both on structured constraints |

## Summary of Insights

1. **BM25 excels at precision** for keyword-specific queries but fails when users paraphrase or describe needs conversationally.
2. **Semantic search excels at recall** for intent-based queries but cannot enforce exact specifications or structured constraints.
3. **Hybrid retrieval provides the best balance** - it surfaces products that score well across both dimensions, reducing the failure modes of each individual method.
4. **Neither method handles multi-concept conversational queries well** (e.g., combining sun damage + fair skin + remedies). This is the gap a RAG pipeline could fill by layering LLM reasoning over retrieved candidates to filter on structured attributes like price and ratings.
