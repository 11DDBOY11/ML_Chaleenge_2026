# Technical Approach Summary
## Amazon Machine Learning Challenge 2026: Multi-Source Business Entity Resolution

### Team Members
- **Darshan Sakale**
- **Karthik T S**

---

### 1. Executive Summary & Problem Overview
In commercial platforms, business identity data originates from multiple independent sources without unified identifiers. This solution addresses large-scale Entity Resolution across 3 noisy sources (Source 1 reference, Source 2, Source 3) covering open geographic labels (`US`, `India`, `France`, etc.). The objective is to maximize **Macro $F_{0.5}$** score (precision weighted $4\times$ higher than recall) while producing both final match predictions (`matching_results.tsv`) and candidate set pairs (`candidate_pairs.tsv`).

---

### 2. Candidate Generation & Multi-Strategy Inverted Indexing
To scale retrieval across **10.3 million candidate records**, we developed an inverted index architecture applying 7 distinct retrieval strategies:
- **S1 (Exact Normalized Name)**: Primary Unicode category text normalization (`L*`, `M*`, `N*`) preserving Devanagari combining vowel signs (`Mc` matras) and anusvara (`Mn`), supplemented by secondary Latin accent-folding (`Société` $\rightarrow$ `societe`).
- **S2 (Core Business Name)**: Corporate suffix stripping (`Pvt Ltd`, `LLC`, `Corp`, `Inc`).
- **S3 (Token Overlap)**: Jaccard token overlap ($\ge \text{threshold}$).
- **S4 (Address Numbers + First Token)**: Address digits combined with first name token.
- **S5 (Cross-Country Exact Name)**: Matching exact names across differing country labels.
- **S6 (Domain Squashing)**: Concatenation of name tokens without spaces (`slmarine` $\leftrightarrow$ `slmarine.com`).
- **S7 (Distinctive Address Numbers)**: Multi-digit address matching ($\ge 2$ distinct numbers) acting as cross-script fallback for Devanagari vs Latin entity pairs.

**Deterministic Pre-Model Capping (`B_cap200`)**: An additive rank score priority ($S1 > S2 > S4 > S7 > S6 > S3 > S5$) caps candidate sets at **top 200 candidates per S1 entity**. On the validation pilot, this achieved **0.7801 candidate recall** while shrinking full-test candidate file storage from 17.8 GB to **2.07 GB**.

---

### 3. Machine Learning Matcher & Feature Engineering
Candidate pairs from the retrieval engine are passed to a **LightGBM Gradient Boosted Decision Tree** classifier trained on 15 numerical pairwise interaction features:
1. **Name Similarities**: Exact name match, core name match, token Jaccard, token overlap count, token overlap ratio, character 3-gram Jaccard, normalized Levenshtein ratio, prefix token match, domain squashed match.
2. **Address Similarities**: Address token Jaccard, address digit overlap count, address digit Jaccard, 6-digit PIN match boolean.
3. **Contextual Indicators**: Same country boolean, target source indicator (Source 2 vs Source 3).

---

### 4. Decision Threshold Optimization & Validation
- **Data Governance**: 10,000 validation pilot entities were strictly partitioned into **7,000 Train S1 entities** (615,987 candidate pairs: 18,900 positive / 597,087 negative) and **3,000 Held-Out Validation S1 entities** (267,958 candidate pairs). Training labels were strictly kept out of validation evaluation.
- **Tuned Validation Performance**: Grid search over probability thresholds $\tau \in [0.05, 0.95]$ achieved a peak **Tuned Validation Macro $F_{0.5}$ score of 0.7964** at $\tau = 0.50$, with a **Singleton $F_{0.5}$ performance of 0.8301**.

---

### 5. Production Inference Pipeline & Scalability
- **Bounded Memory Execution (< 16 GB RAM)**: Implements SQLite disk-backed entity indexing (`test_entities.db`) and processes test S1 entities in streaming batches of 25,000.
- **Incremental & Resumable Output**: Appends predictions incrementally to `matching_results.tsv` and `candidate_pairs.tsv` with zero row duplication or memory leakage.
