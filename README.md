# 🏆 Team SKY — Trendyol Datathon (Phase-1) Solution Write-Up

> **Pipeline:** Learning-to-Rank
> **Key Ideas:** Recency-aware histories, leakage-safe joins, session-wise ranking
> **Stack:** `polars`, `pandas`, `duckdb`, `catboost`, `numpy`, `pyarrow`

---

## Table of Contents

* [Overview](#overview)
* [Data & Leakage Controls](#data--leakage-controls)
* [Feature Engineering](#feature-engineering)
  * [Bilalcan's Features](#a-bilalcans-feature-engineering)
    * [User History](#a-user-history-user_historypy)
    * [Content History](#b-content-history-content_historypy)
    * [Time History](#c-time-history-time_historypy)
    * [Session History](#d-session-history-session_historypy)
    * [Decay Features](#e-decay-features-decay_featurespy)
  * [Kaan's Features](#b-kaans-feature-engineering)
* [Candidate Generation](#candidate-generation)
* [Labeling & Negative Sampling](#labeling--negative-sampling)
* [Model: CatBoostRanker (YetiRank)](#model-catboostranker-yetirank)
* [Validation & LB Behavior](#validation--lb-behavior)
* [Inference Pipeline](#inference-pipeline)
* [Top Features (Qualitative)](#top-features-qualitative)
* [Reproducibility](#reproducibility)
* [Lessons & Next Steps (Phase-2)](#lessons--next-steps-phase2)
* [TL;DR](#tldr)

---

## Overview

We built a **learning-to-rank** system that blends **short-term session intent** with **longer-term user & content signals**.
Core choices:

* **Rich histories** at user/content/term/time levels using fast columnar tooling (Polars/DuckDB).
* **Recency awareness** via exponential-decay features.
* **Session-wise ranking** with **CatBoostRanker (YetiRank)** and a **weighted multi-signal label** that emphasizes orders/carts over softer actions.

---

## Data & Leakage Controls

* **IDs** are hashed; used as categorical keys.
* **Scaling**: Numerical values already in `[0,1]`; no extra scaling required.
* **Missing values**: Numeric aggregates → `0` where additive; categoricals → `"unknown"`.
* **Time alignment**: All histories via **backward as-of joins**; rolling/decay windows only use **past** rows. Same-timestamp label peeking avoided.

---

## Feature Engineering

### (a) Bilalcan's Feature Engineering

#### (a) User History (`user_history.py`)

* Site-wide cumulative & rolling counts: `total_click`, `total_fav`, `total_cart`, `total_order`.
* **Decay-weighted** interaction counts to emphasize recency.
* Behavior ratios (e.g., `cart/click`, `order/click`), plus `avg/max/std/active_session_count`.
* Search-side mirrors (e.g., `term_search_*`, `user_search_*`).

#### (b) Content History (`content_history.py`)

* Global popularity metrics and **category-aware** stats (level1/level2/leaf).
* Smoothed review/price signals (Bayesian/Wilson smoothing on low counts).
* Category size priors to stabilize sparse leaves.

#### (c) Time History (`time_history.py`)

* Periodic windows (e.g., **24h**, **72h**) with `mean/std/min/max/sum` and ratios.
* Short-term **lags** (e.g., `*_lag1`) for momentum.
* Implemented with as-of joins to avoid leakage.

#### (d) Session History (`session_history.py`)

* Session aggregates and `session_candidate_count`.
* Session-normalized ratios (sitewide vs search tables).
* Robust utilities that only compute available ratios (schema-safe).

#### (e) Decay Features (`decay_features.py`)

* **Exponentially decayed** counts with configurable half-life
  (e.g., windows `[3, 6, 12]`; `decay_value = log(0.5)`).
* Applied to both site-wide and search interactions.

---

### (b) Kaan's Feature Engineering

---

## Candidate Generation

* Merge candidates from multiple pools (session, user, content popularity, search terms).
* Join all feature families on `(session_id, user_id_hashed, content_id_hashed, ts_hour)`.
* Add session context via `candidate_counter(df)`.

---

## Labeling & Negative Sampling

**Weighted ranking target** blending actions:

```
weighted_target =
  9.0 * ordered
+ 8.0 * added_to_cart
+ 1.8 * added_to_fav
+ 0.5 * clicked
```

* Auxiliary `new_target = ordered + added_to_cart + added_to_fav + clicked` for filtering.
* **Negative downsampling**: if a session has >1000 zero-target rows, cap negatives at 1000; keep **all positives**.

---

## Model: CatBoostRanker (YetiRank)

* **Loss:** `YetiRank`
* **Group:** `session_id` (ranking is per session)
* **Categoricals:** fillna `"unknown"`, pass via `cat_features`
* **Representative params:** `depth=6`, `random_seed=42`, `verbose=100`
  *(Defaults were strong given feature richness.)*

**Why YetiRank?** Strong session-wise pairwise ordering without manual pair mining; handles mixed numeric/categorical features gracefully.

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

## Top Features (Qualitative)

* **User site-wide rolling/decay** of `click/cart/order` (`avg/sum/std`).
* **Search-term decays** (`total_search_click/impression` with 3/6/12-step windows).
* **Content popularity** + smoothed review/price.
* **Session context** (candidate count & ratios).
* **Time windows** (24h/72h aggregates + `*_lag1`).

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

## Lessons & Next Steps (Phase-2)

* **Query understanding:** normalization/expansion for long-tail intents.
* **Two-stage retrieval:** learnable ANN (embeddings + FAISS/HNSW) before CatBoost.
* **Hard negatives:** popular-but-irrelevant items within same leaf/category.
* **Calibration:** post-ranker score calibration across variable candidate sizes.
* **Modeling:** test listwise losses and distillation from cross-encoders.

---

## TL;DR

A fast, leakage-safe **session ranking** pipeline using Polars/DuckDB features and a single **CatBoostRanker (YetiRank)** trained on a **weighted multi-signal target**. Biggest wins: **decay-weighted histories**, **category-aware content priors**, and **session context features**—all engineered to respect time and keep inference snappy.