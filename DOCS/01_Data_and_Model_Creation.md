# Document 1 — Data Creation & Model Architecture
## Cricket Analytics | ICC T20 World Cup Ball-by-Ball Prediction System

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Data Sources](#2-data-sources)
3. [Dataset Pipeline — How Data Was Built](#3-dataset-pipeline)
4. [Every Dataset File Explained](#4-every-dataset-file-explained)
5. [Feature Engineering](#5-feature-engineering)
6. [All 8 Models — Input, Output, Purpose](#6-all-8-models)
7. [Why These Outcome Classes](#7-why-these-outcome-classes)
8. [Why Commentary-Only for Phase 2 Training](#8-why-commentary-only-for-phase-2-training)
9. [File Paths Reference](#9-file-paths-reference)
10. [Phase 3 — Narrative & ASR Demo Layer](#10-phase-3--narrative--asr-demo-layer)

---

## 1. Project Goal

Predict the **outcome of the next ball** in an ICC T20 World Cup match, given:
- Who is batting (the batter)
- Who is bowling (the bowler)

The prediction is a **probability distribution** over 7 possible outcomes:

| Class | Meaning |
|---|---|
| `0` | Dot ball — no runs scored |
| `1` | 1 run |
| `2` | 2 runs |
| `3` | 3 runs (rare) |
| `4` | Boundary — 4 runs |
| `6` | Six — 6 runs |
| `W` | Wicket — batter dismissed |

This was built in **two prediction phases** plus a **Phase 3 narrative layer** in the UI:
- **Phase 1** — Statistical and ML models using career/matchup stats
- **Phase 2** — Adds commentary text sentiment to improve predictions
- **Phase 3** — Full-match demo (2024 final) with optional ASR next to ball-by-ball text; does not add a new outcome model

---

## 2. Data Sources

### Primary Source — CricBuzz Internal API (scraped)

Data was collected via the CricBuzz API, which provides ball-by-ball data including commentary text for each delivery.

**Years collected:**
- 2021 — ICC T20 World Cup (UAE) — Series ID: 1267452
- 2022 — ICC T20 World Cup (Australia) — Series ID: 1298423
- 2024 — ICC T20 World Cup (USA + West Indies) — Series ID: 1411166

**Note:** 2016 data was excluded because commentary text was not available in the CricBuzz API for that year.

### Raw data format — per ball JSON
Each ball is stored as a JSON file at:
```
master_data/dataset/raw_commentary_cb/<match_id>_rows.json
```

Each JSON record contains:
- `match_id`, `inning_no`, `over`, `ball_in_over`
- `batter`, `bowler`, `non_striker`
- `runs_batter`, `runs_extras`, `runs_total`
- `is_wicket`, `wicket_kind`, `wicket_player_out`
- `commentary_short` — 1-2 sentence description of the delivery
- `batting_team`, `bowling_team`

---

## 3. Dataset Pipeline

How raw JSON becomes trained models:

```
CricBuzz API
    │
    ▼
Raw JSON files per match
(master_data/dataset/raw_commentary_cb/)
    │
    ├─► master_deliveries.csv          ← all 33,662 balls (no commentary column)
    │   (source/data_creation.ipynb)
    │
    ├─► master_matches.csv             ← 149 match metadata records
    │
    └─► master_deliveries_with_commentary.csv   ← 27,872 balls WITH commentary
        (source/fetch_commentary_all_years.py)       (2021 + 2022 + 2024 only)
            │
            ├─► batting_stats.csv      ← aggregated per-player batting stats
            ├─► bowling_stats.csv      ← aggregated per-player bowling stats
            │
            ├─► [matchup_builder.py]
            │       └─► matchup_stats.csv   ← 7,019 batter-bowler pair stats
            │
            └─► [commentary_parser.py]
                    │
                    └─► [pressure_builder.py]
                            ├─► sentiment_stats.csv   ← 5,546 pair sentiment profiles
                            └─► shot_region_stats.csv  ← 12,780 shot/region records
```

### Then models are trained:

```
batting_stats.csv  ─┐
bowling_stats.csv  ─┤─► Phase 1 Models ─► artifacts/
matchup_stats.csv  ─┘   (XGBoost, RF, LightGBM, LSTM)

master_deliveries_with_commentary.csv ─┐
sentiment_stats.csv ───────────────────┤─► Phase 2 Models ─► artifacts/
                                       └   (Augmented XGBoost, VADER, BiLSTM)
```

---

## 4. Every Dataset File Explained

### 4.1 `master_deliveries.csv`
**Path:** `master_data/dataset/master_deliveries.csv`
**Size:** 33,662 rows × 17 columns

The main ball-by-ball dataset. Every legal delivery bowled in all 149 matches across 2021, 2022, and 2024 T20 World Cups.

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Unique CricBuzz match identifier |
| `inning_no` | int | 1 or 2 — which innings |
| `batting_team` | str | Name of the batting team |
| `bowling_team` | str | Name of the bowling team |
| `over` | int | Over number (0–19) |
| `ball_in_over` | int | Ball within the over (1–6) |
| `batter` | str | Batter's name (CricBuzz format) |
| `bowler` | str | Bowler's name |
| `non_striker` | str | Non-striking batter |
| `runs_batter` | int | Runs scored off the bat (0/1/2/3/4/6) |
| `runs_extras` | int | Extra runs (wides, no-balls, byes) |
| `runs_total` | int | Total runs on the ball |
| `is_wicket` | int | 1 if a wicket fell, 0 otherwise |
| `wicket_player_out` | str | Name of dismissed batter (if any) |
| `wicket_kind` | str | How dismissed: caught/bowled/lbw/etc. |
| `extras_type` | str | Type of extra (wide/no-ball/bye/leg-bye) |
| `year` | int | Tournament year (2021/2022/2024) |

**Important design decisions:**
- Wide deliveries are **excluded** from modeling — they don't count as a "ball faced" by the batter
- Run-outs are **excluded** from wicket modeling — they are not the bowler's doing
- `runs_batter` is clipped at 6 (no 5s or 7s possible in T20)

---

### 4.2 `master_deliveries_with_commentary.csv`
**Path:** `master_data/dataset/master_deliveries_with_commentary.csv`
**Size:** 27,872 rows × 17 columns (same + `commentary_short` + `year`)

Same as `master_deliveries.csv` but with a `commentary_short` column added. Only 2021–2024 data because CricBuzz commentary text was not available for 2016.

**Additional column:**
| Column | Type | Description |
|---|---|---|
| `commentary_short` | str | 1-2 sentence ball description e.g. "Good length delivery, Kohli drives through cover for FOUR!" |

**Coverage stats:**
- 27,872 balls total
- ~99.9% have commentary text
- 123 unique matches across 3 years
- 249 innings total (avg 112 balls per inning)

---

### 4.3 `batting_stats.csv`
**Path:** `master_data/dataset/batting_stats.csv`
**Size:** 450 rows × 12 columns

One row per player. Aggregated batting career stats across all T20 World Cup appearances in the dataset.

| Column | Description |
|---|---|
| `player_name` | Player name |
| `matches` | Number of matches played |
| `innings` | Number of innings batted |
| `not_outs` | Number of times not out |
| `runs` | Total runs scored |
| `balls_faced` | Total balls faced |
| `average` | Batting average (runs / dismissals) |
| `strike_rate` | Runs per 100 balls |
| `hundreds` | Centuries scored |
| `fifties` | Half-centuries scored |
| `fours` | Total boundaries hit |
| `sixes` | Total sixes hit |

**Used to compute model features:**
- `strike_rate` → `bat_sr`
- `average` → `bat_avg`
- (`fours` + `sixes`) / `balls_faced` → `bat_bpct` (boundary percentage)

---

### 4.4 `bowling_stats.csv`
**Path:** `master_data/dataset/bowling_stats.csv`
**Size:** 308 rows × 11 columns

One row per player who bowled. Aggregated bowling stats.

| Column | Description |
|---|---|
| `player_name` | Player name |
| `matches` | Matches played |
| `overs` | Total overs bowled |
| `balls_bowled` | Total balls bowled |
| `runs_conceded` | Total runs given away |
| `wickets` | Total wickets taken |
| `average` | Bowling average (runs per wicket) |
| `economy` | Runs per over |
| `strike_rate` | Balls per wicket |
| `four_wickets` | 4-wicket hauls |
| `five_wickets` | 5-wicket hauls |

**Used to compute model features:**
- `economy` → `bowl_econ`
- `wickets` / `balls_bowled` → `bowl_wktr` (wicket rate per ball)

---

### 4.5 `matchup_stats.csv`
**Path:** `models/phase1/artifacts/matchup_stats.csv`
**Size:** 7,019 rows × 20 columns
**Generated by:** `models/phase1/matchup_builder.py`

One row per batter-bowler pair that has faced each other at least once. This is the key lookup table for the Empirical model.

| Column | Description |
|---|---|
| `batter` | Batter name |
| `bowler` | Bowler name |
| `balls` | Total balls faced in this matchup |
| `runs_scored` | Total runs in this matchup |
| `wickets` | Times batter dismissed by this bowler |
| `cnt_0` to `cnt_6` | Count of each outcome (dot/1/2/3/4/6) |
| `prob_0` to `prob_6` | Probability of each outcome |
| `prob_W` | Wicket probability |
| `expected_runs` | Average runs per ball |
| `strike_rate` | Strike rate in this matchup |

**7,019 unique batter-bowler pairs** across all T20 World Cup matches.

---

### 4.6 `sentiment_stats.csv`
**Path:** `models/phase2/artifacts/sentiment_stats.csv`
**Size:** 5,546 rows × 12 columns
**Generated by:** `models/phase2/pressure_builder.py` (reads commentary_parser output)

One row per batter-bowler pair that has commentary data. Contains sentiment aggregates used as Phase 2 features.

| Column | Description |
|---|---|
| `batter` | Batter name |
| `bowler` | Bowler name |
| `balls_with_commentary` | Number of balls with commentary text |
| `avg_sentiment_score` | Average of sentiment scores (+1 DOMINANT, 0 DEFENSIVE, -0.5 MISTIMED, -1 BEATEN) |
| `boundary_pct` | Fraction of balls resulting in a boundary |
| `dot_pct` | Fraction of balls that were dots |
| `dominant_pct` | Fraction of balls tagged DOMINANT |
| `controlled_pct` | Fraction tagged CONTROLLED |
| `mistimed_pct` | Fraction tagged MISTIMED |
| `beaten_pct` | Fraction tagged BEATEN |
| `defensive_pct` | Fraction tagged DEFENSIVE |
| `pressure_index` | `beaten_pct + mistimed_pct - dominant_pct` (higher = batter under more pressure) |

---

### 4.7 `shot_region_stats.csv`
**Path:** `models/phase2/artifacts/shot_region_stats.csv`
**Size:** 12,780 rows × 6 columns
**Generated by:** `models/phase2/pressure_builder.py`

Records how often each batter (vs a specific bowler) plays each shot type into each region.

| Column | Description |
|---|---|
| `batter` | Batter name |
| `bowler` | Bowler name |
| `shot_type` | drive / pull / cut / sweep / flick / slog / defend / hook |
| `region` | cover / mid-wicket / point / long-on / square leg / etc. |
| `count` | Number of balls matching this shot+region combo |
| `pct_of_balls` | As a fraction of all balls in this matchup |

---

### 4.8 `players_info_teams.csv`
**Path:** `master_data/dataset/players_info_teams.csv`
**Size:** 3,279 rows × 9 columns

Used by the UI to enable team-wise player filtering.

| Column | Description |
|---|---|
| `match_id` | Match identifier |
| `team` | Country/team name (22 teams total) |
| `player_name` | Player name |
| `primary_role` | Batter / Bowler / All-rounder / WK-Batter |
| `is_player_of_match` | Boolean flag |

**22 teams included:** Australia, Bangladesh, England, India, New Zealand, Pakistan, South Africa, Sri Lanka, West Indies, Zimbabwe, and others from qualifying rounds.

---

## 5. Feature Engineering

### Phase 1 Features — 8 features per ball

These are computed at prediction time from the three stats CSVs:

| Feature | Source | Meaning |
|---|---|---|
| `bat_sr` | `batting_stats.strike_rate` | Batter's career strike rate (runs per 100 balls) |
| `bat_avg` | `batting_stats.average` | Batter's career batting average |
| `bat_bpct` | `(fours+sixes)/balls_faced` | Batter's boundary-hitting tendency |
| `bowl_econ` | `bowling_stats.economy` | Bowler's runs conceded per over |
| `bowl_wktr` | `wickets/balls_bowled` | Bowler's wicket-taking rate per ball |
| `mu_balls_log` | `log1p(matchup_stats.balls)` | Log-scaled matchup history (how many balls faced) |
| `mu_sr` | `matchup_stats.strike_rate` | Strike rate specifically in THIS matchup |
| `mu_wktr` | `matchup_stats.prob_W` | Wicket probability specifically in THIS matchup |

**Why log-scale `mu_balls`?** A matchup with 5 balls vs 50 balls is different in reliability, but the raw difference is less important than the log difference (small vs moderate vs large sample).

**Fallback logic for missing data:**
1. If matchup exists with ≥10 balls → use matchup stats directly
2. If matchup exists with 1–9 balls → blend 30% matchup + 70% player overall stats
3. If no matchup at all → use player overall stats only

---

### Phase 2 Additional Features — 6 more features

Added on top of the 8 Phase 1 features for the Augmented XGBoost model:

| Feature | Source | Meaning |
|---|---|---|
| `avg_sentiment_score` | `sentiment_stats` | Average commentary sentiment (+1 to -1) |
| `pressure_index` | `sentiment_stats` | beaten% + mistimed% - dominant% |
| `dominant_pct` | `sentiment_stats` | How often batter is described as dominant |
| `beaten_pct` | `sentiment_stats` | How often batter is beaten or missed |
| `boundary_pct` | `sentiment_stats` | Actual boundary rate from commentary balls |
| `dot_pct` | `sentiment_stats` | Actual dot ball rate from commentary balls |

---

### Commentary Parsing — How Text Becomes Features

`models/phase2/commentary_parser.py` processes each commentary string into 4 structured fields:

**Shot Type** (8 categories, keyword-matched in priority order):
```
defend → sweep → flick → pull → hook → cut → drive → slog
```
Example: "Kohli pulls it over mid-wicket" → shot_type = "pull"

**Region** (11 categories):
```
cover → mid-off → long-off → mid-on → long-on → mid-wicket →
square leg → fine leg → third man → point → slip
```

**Sentiment** (5 labels, scored):
| Label | Score | Example keywords |
|---|---|---|
| BEATEN | -1.0 | "beaten", "beat the bat", "no contact", "missed" |
| DOMINANT | +1.0 | "FOUR", "SIX", "boundary", "smashed", "hammered" |
| MISTIMED | -0.5 | "mistimed", "top edge", "skied", "leading edge" |
| DEFENSIVE | 0.0 | "defended", "blocked", "played out", "leaves" |
| CONTROLLED | +0.5 | "well timed", "placed well", "good running" |

**Priority order:** BEATEN > DOMINANT > MISTIMED > DEFENSIVE > CONTROLLED

---

## 6. All 8 Models

### Summary Table

| # | Model | Phase | Type | Training Data | Features In | Output | Accuracy |
|---|---|---|---|---|---|---|---|
| 1 | Empirical | 1 | Statistical | matchup_stats.csv | matchup frequencies | 7-class probs | — |
| 2 | XGBoost | 1 | ML (Gradient Boosting) | master_deliveries.csv | 8 features | 7-class probs | 44.1% |
| 3 | Random Forest | 1 | ML (Ensemble Trees) | master_deliveries.csv | 8 features | 7-class probs | 31.4% |
| 4 | LightGBM | 1 | ML (Gradient Boosting) | master_deliveries.csv | 8 features | 7-class probs | 46.1% |
| 5 | LSTM | 1 | DL (PyTorch, Sequence) | master_deliveries_with_commentary.csv | 6-ball sequence + 8 features | 7-class probs | 54.6% |
| 6 | Augmented XGBoost | 2 | ML (Gradient Boosting) | master_deliveries_with_commentary.csv | 14 features (8+6 sentiment) | 7-class probs | 48.2% |
| 7 | VADER Sentiment | 2 | Rule-based NLP | None (no training) | commentary text | 5 sentiment labels | 18.8%* |
| 8 | BiLSTM | 2 | DL (PyTorch, Text) | master_deliveries_with_commentary.csv | commentary text | 5 sentiment labels | 96.8%* |

*Accuracy for sentiment models is measured against keyword-rule labels (treated as ground truth).

---

### Model 1: Empirical Predictor
**File:** `models/phase1/empirical_predictor.py`

**How it works:**
Looks up the batter-bowler matchup in `matchup_stats.csv` and returns the observed historical frequency of each outcome directly as probability.

**Input:** Batter name + Bowler name
**Output:** `{0: 0.45, 1: 0.25, 2: 0.08, 3: 0.01, 4: 0.12, 6: 0.05, W: 0.04}`

**Fallback chain:**
1. Direct matchup (≥10 balls) → raw frequencies
2. Small matchup (1–9 balls) → 30% matchup + 70% player overall
3. No matchup → player overall stats only

**Strengths:** Perfectly reflects historical data. Zero training needed.
**Weakness:** Fails for new player pairs. No generalisation.

---

### Model 2: XGBoost (Phase 1)
**File:** `models/phase1/ml_predictor.py`
**Artifact:** `models/phase1/artifacts/xgb_model.pkl`

**Architecture:** XGBClassifier, 300 estimators, max_depth=6, learning_rate=0.1, multiclass:softmax

**Training:** 32,569 balls → 85% train / 15% test (stratified by outcome class)

**Input:** 8 numeric features (bat_sr, bat_avg, bat_bpct, bowl_econ, bowl_wktr, mu_balls_log, mu_sr, mu_wktr)
**Output:** 7-class probability distribution + expected_runs + tendency label

**Accuracy: 44.1%** | **Log-Loss: 1.2581**

---

### Model 3: Random Forest (Phase 1)
**File:** `models/phase1/rf_lgbm_predictor.py` — `RFPredictor` class
**Artifact:** `models/phase1/artifacts/rf_model.pkl`

**Architecture:** RandomForestClassifier, 300 trees, max_depth=12, class_weight="balanced"

Same 8 features as XGBoost. The `class_weight="balanced"` setting improves recall on minority classes (e.g. `6`, `W`) at the expense of overall accuracy.

**Accuracy: 31.4%** | **Log-Loss: 1.5079**

**Why lower accuracy?** Random Forest with `class_weight="balanced"` trades overall accuracy for balanced class recall. It's more conservative on common classes (0, 1 runs) and more willing to predict rare outcomes.

---

### Model 4: LightGBM (Phase 1)
**File:** `models/phase1/rf_lgbm_predictor.py` — `LGBMPredictor` class
**Artifact:** `models/phase1/artifacts/lgbm_model.pkl`

**Architecture:** LGBMClassifier, 500 estimators, learning_rate=0.05, max_depth=8, num_leaves=63, with early stopping (patience=50)

Same 8 features as XGBoost. LightGBM uses histogram-based leaf-wise tree growth which makes it faster than XGBoost and slightly more accurate.

**Accuracy: 46.1%** | **Log-Loss: 1.2359**

**Key feature by importance:** `mu_sr` (matchup strike rate) is consistently the most important feature across all tree-based models.

---

### Model 5: LSTM — Ball Sequence Model (Phase 1, Deep Learning)
**File:** `models/phase1/lstm_predictor.py`
**Artifact:** `models/phase1/artifacts/lstm_model.pt`

**This is the most accurate model at 54.6%.**

**Architecture (PyTorch):**
```
Input A: Sequence of last 6 balls
  → each ball: [outcome/6.0, over/19.0, ball_in_over/5.0]
  → shape: (batch, 6, 3)
  → LSTM(hidden=64) → Dropout(0.3)

Input B: Static features (8 from Phase 1)
  → shape: (batch, 8)

[LSTM output] + [Static features]  →  Concatenate (72,)
  → Dense(64, ReLU) → Dropout(0.2)
  → Dense(7, Softmax)  ← 7 outcome classes
```

**How the sequence is built:** For each ball in the dataset, the previous 6 ball outcomes in the same innings are used as the sequence context. This lets the model learn momentum patterns like:
- After a 6 → bowler may pitch it up → more likely to get wicket or dot
- After 3 dots in a row → batter may be under pressure → wicket more likely

**Training:** 25,452 sequences from 27,872 balls (first 6 balls of each innings have no full history)

**At inference without game context:** A zero-padded sequence (neutral state) is used, making the prediction comparable to other models.

**Accuracy: 54.6%** | **Log-Loss: 1.1227**

---

### Model 6: Augmented XGBoost (Phase 2)
**File:** `models/phase2/augmented_predictor.py`
**Artifact:** `models/phase2/artifacts/aug_xgb_model.pkl`

**14 features** = 8 Phase 1 features + 6 sentiment features from commentary:

```
avg_sentiment_score  — how positive/negative the commentary is for this pair
pressure_index       — beaten% + mistimed% - dominant% (composite pressure)
dominant_pct         — how often commentary calls batter dominant
beaten_pct           — how often batter is beaten or missed
boundary_pct         — actual boundary rate in commentary-tagged balls
dot_pct              — actual dot ball rate in commentary-tagged balls
```

**Training:** Only on balls where commentary exists (27,872 rows), because sentiment features are only available for those balls. A baseline (Phase 1 features only, same data) was also trained for fair comparison.

**Results:**
| Model | Data | Accuracy | Log-Loss |
|---|---|---|---|
| Baseline (8 features, commentary subset) | 27,872 balls | 43.77% | 1.2581 |
| Augmented (14 features) | 27,872 balls | 48.19% | 1.2009 |
| Improvement | — | **+4.43%** | **-0.057** |

The 2nd most important feature (21.67% importance) is `dot_pct`, showing commentary-derived signals materially improve outcome prediction.

---

### Model 7: VADER Sentiment Scorer (Phase 2)
**File:** `models/phase2/commentary_classifier.py` — `VADERSentiment` class

**No training needed.** Uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon, enhanced with cricket-specific word scores:

```python
# Cricket additions to VADER lexicon:
"six": +2.5,  "four": +1.5,  "boundary": +1.5
"beaten": -2.0,  "bowled": -2.0,  "mistimed": -1.0
"dot": -0.5,  "wicket": -1.5
```

**Input:** Commentary text string
**Output:** `{label: "DOMINANT", compound: +0.82, pos: 0.45, neg: 0.0, neu: 0.55}`

Compound score → cricket label:
- ≥ +0.50 → DOMINANT
- +0.10 to +0.50 → CONTROLLED
- -0.10 to +0.10 → DEFENSIVE
- -0.50 to -0.10 → MISTIMED
- < -0.50 → BEATEN

**Accuracy vs keyword rules: 18.8%** — VADER is a general English lexicon not designed for cricket commentary. It struggles because cricket language is domain-specific (e.g. "beaten" meaning "bat missed" vs general negativity).

---

### Model 8: BiLSTM Text Classifier (Phase 2, Deep Learning)
**File:** `models/phase2/commentary_classifier.py` — `CommentaryClassifier` class
**Artifacts:** `models/phase2/artifacts/bilstm_sentiment_model.pt`, `bilstm_tokenizer.pkl`

**Architecture (PyTorch):**
```
Input: Commentary text → tokenise → pad to 30 tokens
  Embedding(vocab=5000, dim=64, padding_idx=0)
  → Bidirectional LSTM(hidden=64)  [→ 128-dim after bi-direction]
  → Mean pooling across all timesteps
  → Dropout(0.3)
  → Dense(32, ReLU)
  → Dense(5, Softmax)  ← 5 sentiment classes
```

**Vocabulary:** 5,000 most common words from 27,842 commentary strings

**Labels used for training:** Keyword-rule labels from `commentary_parser.py` (treated as ground truth)

**Training distribution:**
| Sentiment | Count | % |
|---|---|---|
| CONTROLLED | 11,911 | 52.3% |
| DEFENSIVE | 7,820 | 34.3% |
| BEATEN | 1,623 | 7.1% |
| DOMINANT | 713 | 3.1% |
| MISTIMED | 709 | 3.1% |

**Accuracy vs keyword rules: 96.8%** — The BiLSTM learns the contextual patterns in commentary language that the keyword rules capture, plus additional patterns. It generalises beyond exact keyword matches to synonyms, phrases, and sentence structure.

**Output:** `{bilstm_label: "BEATEN", bilstm_confidence: 0.99, vader_label: "BEATEN", vader_compound: -0.64}`

---

## 7. Why These Outcome Classes

**7 classes: 0, 1, 2, 3, 4, 6, W**

- **Why not 5?** 5 runs is not a valid outcome in T20 cricket under standard rules
- **Why include 3?** Rare but exists (overthrows, all-run threes), included to keep the model honest
- **Why a single Wicket class `W`?** The bowler-responsible wickets are grouped together. Run-outs are excluded because they are not predictable from batter/bowler stats — they depend on fielding and running between wickets.
- **Run-outs excluded:** wicket_kind in `{"run out", "runout", "obstructing the field"}` are excluded from training wicket targets
- **Wides excluded:** Wide deliveries are not legal balls faced by the batter and are not included in any model training

---

## 8. Why Commentary-Only for Phase 2 Training

The Augmented XGBoost (Phase 2) model is trained only on the 27,872 balls that have commentary text (2021–2024), not the full 33,662 balls.

**Reason:** The 6 sentiment features (avg_sentiment_score, pressure_index, etc.) only exist for balls with commentary. Training on all 33,662 balls would require setting sentiment features to 0 for the 5,790 balls without commentary, which would mislead the model into treating "no sentiment data" as "neutral sentiment."

**Fair comparison:** To measure the exact gain from sentiment features, a **baseline model** with only 8 features was also trained on the same 27,872-ball subset. This ensures:
- Baseline: 43.77% accuracy (8 features, commentary subset)
- Augmented: 48.19% accuracy (14 features, same subset)
- Gain: +4.43% — purely from adding sentiment features

---

## 9. File Paths Reference

```
Sports_analytics/
├── master_data/
│   └── dataset/
│       ├── master_deliveries.csv                    ← 33,662 balls, all years
│       ├── master_deliveries_with_commentary.csv    ← 27,872 balls, 2021-2024
│       ├── master_matches.csv                       ← 149 match metadata
│       ├── batting_stats.csv                        ← 450 batters
│       ├── bowling_stats.csv                        ← 308 bowlers
│       ├── players_info_teams.csv                   ← 3,279 player-match records
│       └── raw_commentary_cb/                       ← Raw JSON per match
│
└── models/
    ├── phase1/
    │   ├── empirical_predictor.py
    │   ├── ml_predictor.py         ← XGBoost
    │   ├── rf_lgbm_predictor.py    ← Random Forest + LightGBM
    │   ├── lstm_predictor.py       ← PyTorch LSTM
    │   ├── predictor.py            ← Unified Phase1Predictor (all 5 models)
    │   ├── matchup_builder.py      ← Generates matchup_stats.csv
    │   └── artifacts/
    │       ├── matchup_stats.csv   ← 7,019 batter-bowler pairs
    │       ├── xgb_model.pkl
    │       ├── rf_model.pkl
    │       ├── lgbm_model.pkl
    │       └── lstm_model.pt
    │
    ├── phase2/
    │   ├── commentary_parser.py         ← Text → shot/region/sentiment
    │   ├── pressure_builder.py          ← Generates sentiment_stats.csv
    │   ├── augmented_predictor.py       ← Phase 2 XGBoost (14 features)
    │   ├── shot_predictor.py            ← Shot intelligence lookup
    │   ├── commentary_classifier.py    ← VADER + BiLSTM
    │   └── artifacts/
    │       ├── sentiment_stats.csv      ← 5,546 batter-bowler sentiment profiles
    │       ├── shot_region_stats.csv    ← 12,780 shot-region records
    │       ├── aug_xgb_model.pkl
    │       ├── bilstm_sentiment_model.pt
    │       └── bilstm_tokenizer.pkl
    │
    ├── phase3/
    │   ├── prototype_match.py           ← Builds JSON narrative for 2024 final prototype
    │   ├── download_and_transcribe.py   ← Optional: fetch media + ASR → transcript text
    │   └── artifacts/                   ← prototype JSON, ASR metadata, sample transcripts
    │       ├── prototype_2024_ind_sa_final.json
    │       ├── asr_run_meta.json
    │       └── transcript_1415755_asr.txt   ← example ASR output (sample)
    │
    └── ultimate_model/
        └── app.py                       ← Combined Streamlit UI (6 tabs incl. Phase 3)
```

---

## 10. Phase 3 — Narrative & ASR Demo Layer

Phase 3 is a **presentation and research extension**: it adds a **full-match narrative tab** in the Ultimate Dashboard for one showcase fixture (ICC Men’s T20 World Cup **2024 final — India vs South Africa**). It does **not** introduce a new ball-outcome model; Phases 1 and 2 still supply all `{0,1,2,3,4,6,W}` predictions.

**What it does**

- Bundles **match metadata**, **ICC match-centre / video links**, and **per-ball CricBuzz-style commentary** in a JSON artifact the UI reads.
- Optionally aligns **automatic speech recognition (ASR)** transcript segments with overs/balls for **side-by-side comparison** with written commentary (useful for broadcast archives and multi-modal demos).

**Main code**

| File | Role |
|---|---|
| `models/phase3/prototype_match.py` | Produces `artifacts/prototype_2024_ind_sa_final.json` consumed by Tab 6 |
| `models/phase3/download_and_transcribe.py` | Optional pipeline: download media, run ASR, write `artifacts/transcript_*_asr.txt` |
| `models/phase3/artifacts/` | Versioned JSON + transcript samples; large media files are typically gitignored |

**Relationship to Phases 1–2**

- Phase 1 / 2 **tabs 1–5** are unchanged in purpose: stats, sentiment, augmented outcome, model comparison, leaderboards.
- Phase 3 **Tab 6** answers: *“For this flagship match, what does the full ball-by-ball story look like next to optional ASR text?”*
