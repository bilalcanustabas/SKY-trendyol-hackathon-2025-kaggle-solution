# 🏆 Team SKY — Trendyol Hackathon 2025 Kaggle Phase 🥉 Solution Write-Up

> **Pipeline:** Feature Engineering -> Learning-to-Rank Model Bilalcan/Kaan -> Ensemble

> **Key Ideas:** Recency-aware histories, leakage-safe joins, session-wise ranking ## TODO: REWRITE HERE

> **Stack:** `polars`, `pandas`, `duckdb`, `catboost`, `numpy`, `pyarrow`

---

## Table of Contents

* [Overview](#overview)
* [Challenges](#challenges)
* [Bilalcan's Solution](#bilalcans-solution)
  * [1) Feature Engineering](#1-feature-engineering)
    * [A) User History](#a-user-history-user_historypy)
    * [B) Content History](#b-content-history-content_historypy)
    * [C) Time History](#c-time-history-time_historypy)
    * [D) Session History](#d-session-history-session_historypy)
    * [E) Decay Features](#e-decay-features-decay_featurespy)
  * [2) Feature Selection](#2-feature-selection)
  * [3) Validation Scheme & LB Behavior](#3-validation-scheme-lb-behavior) ## TODO: WRITE HERE
  * [4) Target Creation & Negative Sampling](#4-target-creation-negative-sampling)
  * [5) Model: CatBoostRanker (YetiRank)](#5-model-catboostranker-yetirank) ## TODO: REWRITE HERE
* [Kaan's Solution](#kaans-solution)
  * [1) FILL HERE](#1-fill-here)
* [Validation & LB Behavior](#validation--lb-behavior) ## TODO: ADD HERE TO CANS SOLUTION
* [Inference Pipeline](#inference-pipeline)
* [Top Features (Qualitative)](#top-features) ## TODO: FILL HERE
* [What Worked, What Didn't?](#what-worked-what-didnt)
* [Improve Steps](#improve-steps)
* [Reproducibility](#reproducibility)
* [Lessons](#lessons)
* [TL;DR](#tldr) ## TODO: REWRITE HERE

---

## Overview

We built a **learning-to-rank** system that blends **short-term session intent** with **longer-term user & content signals**.

**Core choices:**
  * **Rich histories** at user/content/term/time levels using fast columnar tooling (Polars/DuckDB).
  * **Target Weighting** with orders > carts > favs > clicks.
  * **Session-wise ranking** with **CatBoostRanker (YetiRank)** using weighted target label.
---

## Challenges

* **Big Data & Memory Issues**
  Data was quite big for both train/test tables and other tables (100k rows to 100M rows). With these size training/ranking at session granularity with wide, history-heavy features pushed RAM hard. We mitigated by using **Polars lazy scans** and **DuckDB** for on-disk joins/aggregations. We also capped high-cardinality categoricals, pruned unused columns early, and processed data in chunks if needed to keep the working set bounded.

* **Complexity of Feature Engineering**
  We combined user/content/time/session histories, rolling windows, and exponential decays. These were easy to get leaky or inconsistent. To control this, we split logic into **modular builders** (`user_history.py`, `content_history.py`, `time_history.py`, `session_history.py`, `decay_features.py`) with **as-of joins only**, consistent naming conventions, and **idempotent** functions.

* **Target Creation**
  Clicks, favs, carts, and orders carry very different business value and noise levels. We constructed a **weighted ranking target** (order > cart > fav > click) and validated weights. We apply **negative downsampling per session** (keep all positives, cap easy negatives) to stabilize class skew without distorting within-session ordering.

* **High Train Time**
  Session-grouped learning-to-rank with thousands of candidates per session is expensive, training times changed 30m to 2h based on training settings and data. We reduced wall-time by **pre-filtering candidates**, caching intermediate features, and trimming redundant/collinear columns. For CatBoostRanker, we used moderate `depth` and `iterations` to deal with these. We also tested higher depth and iterations but didn't bring that much of improvement on CV.

---

## Bilalcan's Solution

### 1) Feature Engineering

#### a) User History (`user_history.py`)

* Site-wide cumulative & rolling counts: `total_click`, `total_fav`, `total_cart`, `total_order`.
* **Decay-weighted** interaction counts to emphasize recency.
* Behavior ratios (e.g., `cart/click`, `order/click`), plus `avg/max/std/active_session_count`.
* Search-side mirrors (e.g., `term_search_*`, `user_search_*`).

#### b) Content History (`content_history.py`)

* Global popularity metrics and **category-aware** stats (level1/level2/leaf).
* Smoothed review/price signals (Bayesian/Wilson smoothing on low counts).
* Category size priors to stabilize sparse leaves.

#### c) Time History (`time_history.py`)

* Periodic windows (e.g., **24h**, **72h**) with `mean/std/min/max/sum` and ratios.
* Short-term **lags** (e.g., `*_lag1`) for momentum.
* Implemented with as-of joins to avoid leakage.

#### d) Session History (`session_history.py`)

* Session aggregates and `session_candidate_count` (column that shows how many candidates are in that session).
* Session-normalized ratios (sitewide vs search tables).
* Robust utilities that only compute available ratios (schema-safe).

#### e) Decay Features (`decay_features.py`)

* **Session Based History** This feature engineering calculates the `mean, sum, std, min, max` for last N sessions with a decay parameter.
* **Why This Important?** Normally, user session's are important
* **Exponentially decayed** counts with configurable half-life
  (e.g., windows `[3, 6, 12]`; `decay_value = log(0.5)`).
* Applied to both site-wide and search interactions.

### 2) Feature Selection

* **Multi-Collinearity Selection** There were quite a lot collinear feature with each other, so I removed one of the features of collineared with each other, to do that I filtered all correlations that above 0.99 and selected the one with highest correlation with target column then dropped rest of those collineared ones.
* **Fast Feature Selection with LightGBM** After multi-collinearity selection, I fitted lightgbm regression to my training data and sorted features based on feature importance then calculated cumulative improtance to drop features above %99 importance.
* **What is the Final Features?** In the end, I had 455 numerical/binary features and 4 categorical features from ~600 features. You can see these features in `merged_solution.ipynb` at `## Bilalcan's Solution -> ### Modelling` part.

### 3) Validation Scheme & LB Behavior

### 4) Target Creation & Negative Sampling

**Weighted ranking target** blending actions:

```
weighted_target =
  9.0 * ordered
+ 8.0 * added_to_cart
+ 1.8 * added_to_fav
+ 0.5 * clicked
```
* **Negative downsampling**: if a session has >1000 contents, keep **all positives** (contents with non-zero target value) and add negatives (contents with zero target value) until session reaches 1000 contents. This improved my validation score a bit but I didn't see that much of difference on PB, still I used this in my final solution.

### 5) Model: CatBoostRanker (YetiRank)

* **Loss:** `YetiRank`
* **Group:** `session_id` (ranking is per session)
* **Categoricals:** fillna `"unknown"`, pass via `cat_features`
* **Representative params:** `depth=6`, `random_seed=42`, `verbose=100`
  *(Defaults were strong given feature richness.)*

**Why YetiRank?** Strong session-wise pairwise ordering without manual pair mining; handles mixed numeric/categorical features gracefully.
**Is there a better alternative than YetiRank?** Actually YetiRankPairwise gives slightly better results but training time triples because of that we choose YetiRank to try more experiments.

---

## Kaan's Solution

### 1) Feature Engineering

a) **User History**  
- Rolling 24h/72h ratios from sitewide & search logs (`click→order`, `click→cart`, CTR).  
- User–content affinity: cumulative counts + recency gaps (hours since last click/cart/order).  
- Category-level conversion rates (L1/L2/leaf) via time-safe ASOF joins.  

b) **Content History**  
- Popularity priors: all-time totals + 7d/30d velocity ratios.  
- Category-relative ranks (leaf/L2/L1) for competitive positioning.  
- Price & review stats (absolute + relative to category average).  

c) **Search & Term Features**  
- CTR windows for term/content (24h/72h, 3d/7d).  
- Top-term engagement signals for query–item alignment.  

d) **Session Context**  
- In-session ranks for price, rating, review counts.  
- Dispersion metrics (mean, std, min, max) for price & quality signals.  
- Session size (`session_item_count`) and normalised deltas (% difference from session avg).  

e) **Demographic & Affinity**  
- Aggregated age/gender interaction stats (avg age, gender-based conversion rates).  
- Content tenure (days since creation) and CV tag richness.  

**Implementation:**  
Single-shot DuckDB SQL with 20+ CTEs, strict time-safe windows (`… PRECEDING AND 1 MICROSECOND PRECEDING`) and ASOF joins to prevent leakage.

### 2) Target Creation & Tuning Target Weights

**Weighted ranking target** tuned via **Optuna** with 3-fold time-based CV:

```
weighted_target =
  9.0 * ordered
+ 8.0 * added_to_cart
+ 0.1 * added_to_fav
+ 2.2 * clicked
```

*Used Optuna to optimise weights for best AUC on validation splits. Final weights emphasise `order` and `cart` actions while giving moderate importance to `clicks` and minimal to `favs`.*

### 3) Feature Selection

No wrapper-based feature selection method was applied.  
Only removed one feature from each pair with correlation **> 0.98** to reduce redundancy. 17 features are eliminated out of 120, leaving **final 103 features**.

### 4) Model: CatBoostRanker (YetiRank)

* **Loss:** `YetiRank`
* **Group:** `session_id` (ranking is per session)
* **Categoricals:** fillna `"unknown"`, pass via `cat_features`
* **Params:** `iterations=1000`, `learning_rate=0.05`, `depth=6`, `random_seed=42`, `verbose=100`

---

## Validation & LB Behavior

* Time-consistent joins and **session-grouped** evaluation.
* Heavier weights on `order/cart` improved LB while preserving click relevance.
* **Ablations (observed):**

  1. **Decay features** → consistent gains on short sessions.
  2. **Category-aware content priors** → help cold-ish items.
  3. **Session context** (`session_candidate_count`, within-session ratios) → crucial when candidate sets are large.

> Phase-1 focused on public LB; Phase-2 will introduce stricter offline splits for generalization.

---

## Inference Pipeline

1. Build candidates for the session.
2. Join user/content/time/session features (all backward-looking).
3. Fill missing categoricals → `"unknown"`.
4. Predict with CatBoostRanker.
5. Rank within each `session_id` by prediction score.

**Efficiency notes:**
Polars for feature plumbing; DuckDB CTEs for window features. Builders are idempotent for caching and incremental runs.

---

## Top Features

---

## What Worked, What Didn't?

### Worked

* **Session Based History and Decaying**
* **Target Weighting**
* **Ranker Model**
* **Ensemble**

### Didn't Worked or Worked but Worse

* **Wilson Score**
* **4 Classifier Model**
* **Regression Model**
* **Other Algorithms Then Catboost**

---

## Improve Steps

---

## Reproducibility

### Environment

```txt
polars==1.32.2
pandas==2.3.1
numpy==1.26.4
duckdb==1.3.2
catboost==1.2.8
ipykernel==6.30.1
pyarrow==21.0.0
```

### Quickstart

```bash
# 1) Create env (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Explore / run
# - merged_solution.ipynb (includes 1st & 2nd solution sections)
# - feature modules:
#   user_history.py
#   content_history.py
#   time_history.py
#   session_history.py
#   decay_features.py
```

---

## Lessons

* **Query understanding:** normalization/expansion for long-tail intents.
* **Two-stage retrieval:** learnable ANN (embeddings + FAISS/HNSW) before CatBoost.
* **Hard negatives:** popular-but-irrelevant items within same leaf/category.
* **Calibration:** post-ranker score calibration across variable candidate sizes.
* **Modeling:** test listwise losses and distillation from cross-encoders.

---

## TL;DR

A fast, leakage-safe **session ranking** pipeline using Polars/DuckDB features and a single **CatBoostRanker (YetiRank)** trained on a **weighted multi-signal target**. Biggest wins: **decay-weighted histories**, **category-aware content priors**, and **session context features**—all engineered to respect time and keep inference snappy.