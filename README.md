# Codeforces Problem Difficulty Predictor

Predicting the official Codeforces rating of a problem using only metadata available at contest time — no submission data, no user data, just the problem's tags, contest structure, and position.

```
MAE            → 138.33
Within ±100    → 50.21%
Within ±200    → 76.78%
```

## Why this problem is harder than it looks

Codeforces ratings range from 800 to 3500+, assigned by problemsetters based on how hard a problem *turned out* to be after real contestants attempted it. That's the core difficulty here: the target variable is itself a noisy human judgment, not a physical quantity. Two problems can share every tag, the same contest division, and the same slot, and still differ by 200+ rating points because of how a specific idea was executed. So the goal was never to hit near-zero error — it was to squeeze out every *structural* signal the metadata actually contains before conceding the rest to irreducible noise.

## Data

Two endpoints from the official Codeforces API:

- `problemset.problems` — every problem, its tags, contest ID, and index (A, B, C, ...)
- `problemset.problems` statistics — solve counts per problem
- `contest.list` — contest names, used to recover division (Div. 1/2/3/4, Educational, Global)

Problems without an assigned `rating` were dropped (unrated problems, very new problems, some gym/special contests) — a prediction target has to exist to train on.

## Feature engineering

This is where most of the actual thinking went. Four ideas mattered more than any model choice.

### 1. Division as an ordinal signal
Contest names were parsed into `Div. 1`, `Div. 2`, `Div. 3`, `Div. 4`, `Educational`, `Global`, `Div. 12` (combined rounds), or `Other`. This wasn't one-hot encoded — division has a *real* ordering (Div. 4 problems are systematically easier than Div. 3, which are easier than Div. 2), and it's one of the strongest single predictors of rating. Treating it as ordinal lets the tree split on it naturally instead of learning the ordering from scratch across several dummy columns.

### 2. Position is relative, not absolute
Problem index (A, B, C, ...) was encoded numerically (A=1, B=2, ...), but raw index alone is misleading: problem E in a 5-problem contest is the hardest problem of the round, while problem E in an 8-problem contest is closer to the middle. To capture this, `index_position = index_encoded / max_index_in_contest` was added — the problem's *relative* position within its own contest, computed via a contest-level aggregation (`num_problems`, `max_index`, `contest_avg_solved` per `contestId`).

### 3. Slot statistics — the strongest feature group
For every `(division, position)` pair, the mean/median/std rating was computed **from the training set only** and joined onto both train and test. The intuition: Div. 2 problem C has a historically consistent rating band (~1300–1600), and that band is a much stronger prior than tags alone. This is essentially a hierarchical/target-encoding trick, and it turned out to dominate feature importance. Unseen `(div, index)` combinations in the test set fall back to the global slot average rather than producing NaNs.

### 4. Solve count as a difficulty proxy — but shaped correctly
Solve count is inversely related to difficulty, but its raw distribution spans ~50 to 100k+, so it was log-transformed (`log1p`). More importantly, absolute solve count is contest-size-dependent, so `solved_pct_rank` — the percentile rank of solve count *within that contest* — was added as a second, scale-free version of the same signal.

### 5. Tags: multi-hot, plus interactions, plus a rarity score
Tags are multi-label (a problem can be `dp` + `graphs` + `trees` simultaneously), so `MultiLabelBinarizer` was used instead of one-hot encoding. But a tag alone is a weak signal — `dp` shows up at every difficulty level from 1000 to 3000. What matters is tags *in context*, so for the 12 most common tags, two interaction features were added: `tag × index_encoded` and `tag × div_encoded`. `dp` on problem A of a Div. 4 and `dp` on problem F of a Div. 1 now look completely different to the model.

A `tag_rarity_score` was also computed as a weighted sum of a problem's tags against `(1 - tag_frequency)`, so problems carrying uncommon tags (FFT, flows, meet-in-the-middle — things that almost never show up on easy problems) get pushed toward harder predictions automatically.

## The leakage bug (and why the fix mattered)

The single biggest correctness issue in this project: an early version split train/test **by problem row**, randomly. Since slot statistics and contest aggregates are computed at the *contest* level, this let problems from the same contest leak into both splits — the model was implicitly seeing test-set information through the slot stats. This inflated results by roughly 50 MAE points.

The fix: split by `contestId`, not by row. All problems from a given contest go entirely into train or entirely into test, and slot stats / contest aggregates are computed strictly from the training contests before being joined onto the test set. This is the same principle as time-series or group-based leakage in any ML pipeline — the unit of splitting has to match the unit the leaky features are computed over.

## Why this architecture, specifically

Before landing on "engineered tabular features + LightGBM," a few alternatives were implicitly ruled out, and it's worth writing down why.

**Tabular + gradient boosting over a neural net.** The dataset is a few thousand rows of structured, mixed-type features (a handful of binary tag columns, a few ordinal columns, a few continuous ones). This is exactly the regime where gradient-boosted trees consistently beat neural nets — there's no spatial or sequential structure for a network to exploit, no huge parameter count needed, and no volume of data (millions of rows) that would let a net find patterns trees can't already split on. A neural net here would mostly be re-learning what one-hot/ordinal splits already give a tree for free, at the cost of needing scaling, more tuning, and more data to avoid overfitting.

**LightGBM over XGBoost/CatBoost specifically.** All three would likely perform similarly close. LightGBM was chosen for practical reasons: leaf-wise growth handles the mix of a few dozen sparse binary tag columns and interaction terms efficiently, training is fast enough to iterate quickly in Colab, and native NaN handling meant the unseen-slot fallback (`fillna` to global average) was the only manual NaN handling needed anywhere in the pipeline.

**Feature engineering over letting the model find patterns raw.** This was the actual design bet of the project: rather than handing the model raw tags + raw index + raw contest ID and trusting boosting to find every interaction, domain knowledge was front-loaded into the features themselves — ordinal division, relative index position, slot statistics, tag×context interactions, tag rarity. The reasoning: trees are good at finding threshold splits but bad at discovering *ratios* and *group-relative* quantities (like "position 4 of 8" or "this tag's frequency among training problems") on their own from raw IDs. Encoding those relationships explicitly, using domain understanding of how Codeforces rounds are structured, gave the model a large head start over raw categorical encodings — and it shows up directly in feature importance, where slot stats and the position ratio dominate.

**Group-based split over row-based split.** Already covered above (the leakage bug), but it's part of the architecture, not just a training detail — any feature computed at the contest level (slot stats, contest aggregates) *requires* a contest-level split to be valid. This constrains the problem framing itself: contestId, not problem row, is the correct unit of generalization to test against.

## Model

LightGBM (`LGBMRegressor`), chosen because the feature set is a mix of binary tag columns and continuous engineered features, and trees handle that mix without needing any scaling.

```python
LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
)
```

Early stopping (150 rounds, on a held-out eval set) was used instead of hand-tuning `n_estimators` directly — a low learning rate (0.02) with a high estimator ceiling (3000) lets early stopping find the right point rather than guessing it upfront.

## Results

| Metric | Value |
|---|---|
| MAE | 138.33 |
| Within ±100 | 50.21% |
| Within ±200 | 76.78% |

**Error by difficulty range:**

| Rating bucket | MAE |
|---|---|
| 800–1200 | 120 |
| 1200–1600 | 151 |
| 1600–2000 | 136 |
| 2000–2400 | 143 |
| 2400–2800 | 167 |
| 2800–3500 | 177 |

The pattern here is informative: error is lowest at the easy end (800–1200) and climbs steadily into the 2400+ range. This tracks with intuition about the problem itself, not a model weakness — easy problems are rated fairly mechanically (mostly implementation/math, tightly clustered around known slot bands), while high-difficulty problems are rated based on subtle originality and execution that no amount of tag/position metadata can fully capture. A `dp` problem at 2900 and one at 3200 can look nearly identical in every feature used here.

## What's saved

Everything needed to run inference on a brand-new problem without recomputing the whole pipeline:

- `model.pkl` — trained LightGBM model
- `mlb.pkl` — fitted `MultiLabelBinarizer` for tags
- `slot_stats.csv` — (division, index) → mean/median/std rating lookup
- `tag_freq.csv` — tag frequency table, for rarity scoring
- `feature_columns.pkl` — exact column order the model expects

## Possible extensions

- **Problem statement text**: embedding the actual statement (length, LaTeX density, presence of formal proofs) could capture the "originality" signal that tags/position miss, especially at the 2400+ end where error is worst.
- **Editorial length/complexity** as a difficulty proxy, where available.
- **Author history**: some setters are known for consistently harder/trickier problems at a given nominal rating.
- **Quantile regression / prediction intervals** instead of a point estimate — given that even a well-tuned model has ~140 MAE, communicating a range (e.g. 1500–1750) is arguably more honest than a single number.
