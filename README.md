# 🏆 Team SKY — Trendyol Hackathon 2025 Kaggle Phase 🥇 Public / 🥉 Private Solution Write-Up

* **Pipeline:** Feature Engineering -> Learning-to-Rank Model Bilalcan/Kaan -> Ensemble
* **Key Ideas:** Recency-aware histories, leakage-safe joins, session-wise ranking 
* **Stack:** `polars`, `pandas`, `duckdb`, `catboost`, `numpy`, `pyarrow`

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
  * [3) Target Creation & Negative Sampling](#3-target-creation-negative-sampling)
  * [4) Model: CatBoostRanker (YetiRank)](#4-model-catboostranker-yetirank)
* [Kaan's Solution](#kaans-solution)
  * [1) Feature Engineering](#1-feature-engineering)
    * [A) User History](#a-user-history)
    * [B) Content History](#b-content-history)
    * [C) Search & Term Features](#c-search-term-features)
    * [D) Session Context](#d-session-context)
    * [E) Demographic & Affinity](#e-demographic-affinity)
  * [2) Target Creation & Tuning Target Weights](#2-target-creation--tuning-target-weights)
  * [3) Feature Selection](#3-feature-selection)
  * [4) Model: CatBoostRanker (YetiRank)](#4-model-catboostranker-yetirank)
* [Validation Scheme & LB Behavior](#validation-scheme-lb-behavior)
* [Inference Pipeline](#inference-pipeline)
* [Top Features](#top-features)
* [What Worked, What Didn't?](#what-worked-what-didnt)
* [Improve Steps](#improve-steps)
* [Reproducibility](#reproducibility)
* [TL;DR](#tldr)

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
* User metadata informations (e.g., `age`, `join_date`, `gender`)

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

* **Session Based History:** This feature engineering calculates the `mean, sum, std, min, max` for last N sessions with a decay parameter.
* **Why This Important?** Normally, calculating features from last T time of user session's creates a meaning but it fails to capture importance of recent sessions but calculating session based historical features with a decay parameter add a meaning from another level.
* **Exponentially Decayed:** Calculates historical features with last N sessions with decay parameter.
  (e.g., windows `[3, 6, 12]`; `decay_value = log(0.5)`).
* Applied to both site-wide and search interactions.

### 2) Feature Selection

* **Multi-Collinearity Selection:** There were quite a lot collinear feature with each other, so I removed one of the features of collineared with each other, to do that I filtered all correlations that above 0.99 and selected the one with highest correlation with target column then dropped rest of those collineared ones.
* **Fast Feature Selection with LightGBM:** After multi-collinearity selection, I fitted lightgbm regression to my training data and sorted features based on feature importance then calculated cumulative improtance to drop features above %99 importance.
* **What is the Final Features?** In the end, I had 455 numerical/binary features and 4 categorical features from ~600 features. You can see these features in `merged_solution.ipynb` at `## Bilalcan's Solution -> ### Modelling` part.

### 3) Target Creation & Negative Sampling

**Weighted Ranking Target:**

```
weighted_target =
  9.0 * ordered
+ 8.0 * added_to_cart
+ 1.8 * added_to_fav
+ 0.5 * clicked
```
* **Negative Downsampling:** If a session has >1000 contents, keep **all positives** (contents with non-zero target value) and add negatives (contents with zero target value) until session reaches 1000 contents. This improved my validation score a bit but I didn't see that much of difference on PB, still I used this in my final solution.

### 4) Model: CatBoostRanker (YetiRank)

* **Loss:** `YetiRank`
* **Group:** `session_id` (ranking is per session)
* **Categoricals:** fillna `"unknown"`, pass via `cat_features`
* **Params:** `iterations=1000`, `learning_rate=0.05`, `depth=6`, `random_seed=42`, `verbose=100`

**Why YetiRank?** Strong session-wise pairwise ordering without manual pair mining; handles mixed numeric/categorical features gracefully.
**Is there a better alternative than YetiRank?** Actually YetiRankPairwise gives slightly better results but training time triples because of that we choose YetiRank to try more experiments.

---

## Kaan's Solution

### 1) Feature Engineering

**Implementation:**  
Single-shot DuckDB SQL with 20+ CTEs, strict time-safe windows (`… PRECEDING AND 1 MICROSECOND PRECEDING`) and ASOF joins to prevent leakage.

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

## Validation Scheme & LB Behavior

* **Time Based Validation Split:** There are three unique datetimes in train sessions and test sessions only have one unique datetime, since amount of data for each datetime is a lot and distrubutions similar in terms of other columns we **used last datetime of train sessions as validation group**.

* **LB Behavior:** LB and validation scores were quite similar for most of our trials.

---

## Inference Pipeline

1. Join user/content/time/session features (all backward-looking).
2. Fill missing categoricals → `"unknown"`.
3. Predict with CatBoostRanker.
4. Rank within each `session_id` by prediction score.

---

## Top Features

* Content's and User's click/search ratios
* Sitewide search rank of Content
* Content's total decay score in fashion
* User tenure and sign-up age
* Session candidate count
* Content's rank in session based on click/search, click/fav, click/cart, bayesian rating
* Content's sitewide ratio avg for 3d/7d (click/search, click/fav, click/cart, click/order)

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
* **Other Objective Metrics**
* **Ensemble of Same Model with Different Seeds**

---

## Improve Steps

* We didn't used **embeddings of cv tags** (visual labels of items), and it can be added because there is potential to improve.

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

## TL;DR

* Built a **session-aware learning-to-rank** pipeline that blends short-term intent with **user/content/time histories**, all computed via **leakage-safe ASOF joins** in Polars/DuckDB.
* Crafted a **weighted target** (orders > carts > favs > clicks) and light **negative downsampling** per session to stabilize training without distorting within-session order.
* Trained **CatBoostRanker (YetiRank)** on `session_id` groups; moderate depth/iterations for speed–accuracy balance; simple categorical handling (`unknown`).
* Heavy use of **recency/decay features**, rolling windows, and session context; pruned collinear features and (in one variant) used LightGBM for quick importance-based trimming.
* **Validation** used a strict time split (last train timestamp as val); **CV ≈ LB**, giving reliable offline signals.
* **Inference** = join backward-looking features → predict → rank per session; final **ensemble** of two variants nudged gains.
* **Worked:** session-decayed histories, target weighting, CatBoost LTR, small ensemble.
* **Didn’t:** Wilson score, multi-classifier/regression baselines, non-CatBoost LTR/objectives, seed-only ensembles.
* **Next:** integrate **CV-tag/visual embeddings** for further lift.
* **Stack:** `polars`, `duckdb`, `pandas`, `catboost`, `numpy`, `pyarrow`.
* **Result:** 🥇 Public / 🥉 Private in Trendyol Hackathon 2025 Kaggle Phase.