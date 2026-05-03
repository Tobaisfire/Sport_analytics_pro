# Document 2 — Phase-wise Model Documentation
## Cricket Analytics | How Each Phase Works, What It Improves, and All Metrics

---

## Table of Contents

1. [Overview — The Three-Phase Approach](#1-overview)
2. [Phase 1 — Ball Outcome Prediction (Stats-Based)](#2-phase-1)
   - Problem Statement
   - Five Models
   - Feature Importance
   - Results and Metrics
   - Limitations of Phase 1
3. [Phase 2 — Commentary Sentiment Enhancement](#3-phase-2)
   - What Phase 1 Was Missing
   - How Commentary Adds Value
   - Commentary Parsing Pipeline
   - Augmented XGBoost Model
   - Sentiment Models (VADER + BiLSTM)
   - Shot Intelligence Feature
   - Results and Metrics
4. [Phase 3 — Narrative Layer (Full-Match ASR Demo)](#4-phase-3--narrative-layer-full-match-asr-demo)
5. [Cross-Phase Comparison — All Models Side by Side](#5-cross-phase-comparison--all-models-side-by-side)
6. [How Each Phase Helped the Other](#6-how-each-phase-helped-the-other)
7. [Key Insights and Findings](#7-key-insights-and-findings)

---

## 1. Overview

The project was designed in **three layers**: two prediction phases (1 and 2) plus a **Phase 3 demo** that does not add a new outcome model.

```
Phase 1                         Phase 2                         Phase 3
───────────────────────────   ───────────────────────────   ───────────────────────────
Uses: Career stats + matchup  Uses: P1 features + commentary Uses: Full-match JSON +
      from scorecards                sentiment, shot/region         ICC links + optional ASR

Models: Empirical, XGBoost,     Models: Augmented XGBoost,      Models: (none for outcomes)
        RF, LightGBM, LSTM            VADER, BiLSTM                  — narrative UI only

Asks: "Outcome distribution   Asks: "What does commentary    Asks: "For the 2024 final,
       for this pair?"               add to that pair?"               what does the story +
                                                                      ASR look like per ball?"
```

**Why separate Phase 1 and Phase 2?**

Phase 1 is built on data that exists for all matches (scorecard stats). Phase 2 requires commentary text which is only available for 2021–2024 matches. Building them separately:
1. Allows a fair controlled comparison (same data, different features)
2. Means Phase 1 still works even without commentary
3. Makes the contribution of sentiment features measurable and clear

**Why Phase 3?**

Phase 3 packages **one flagship match** (2024 IND vs SA final) for **storytelling and multi-modal exploration**: official links, ball-by-ball text, and optional ASR transcripts. It shows how broadcast audio could complement the same commentary NLP pipeline used in Phase 2, without changing the 7-class prediction stack.

---

## 2. Phase 1 — Ball Outcome Prediction

### Problem Statement

Given:
- A **batter** (their career strike rate, average, boundary %, and any historical matchup data)
- A **bowler** (their economy rate, wicket rate, and any historical matchup data)

Predict the **probability distribution** over 7 possible outcomes for the next ball:
`{0, 1, 2, 3, 4, 6, W}`

### The Five Phase 1 Models

#### Model A: Empirical Predictor
**Type:** Pure statistical (no machine learning)

**Logic:**
```
Does matchup exist with ≥10 balls?
    YES → use raw historical frequencies directly
    NO, but 1–9 balls exist → blend 30% matchup + 70% batter overall
    NO matchup at all → use batter overall career stats
```

**Example output for V Kohli vs A Nortje (if 26 balls recorded):**
```json
{
  "method": "empirical",
  "source": "matchup_direct",
  "balls_sample": 26,
  "probs": {"0": 0.3077, "1": 0.3077, "2": 0.0769, "4": 0.2308, "6": 0.0769, "W": 0.0},
  "expected_runs": 1.3077,
  "strike_rate": 130.77,
  "tendency": "Aggressive"
}
```

**Strengths:** Perfectly accurate for well-documented pairs. Interpretable.
**Weaknesses:** Cannot generalise. Fails for new players or new matchups. Small samples are unreliable.

---

#### Model B: XGBoost (Phase 1)
**Type:** Gradient Boosted Decision Trees

**XGBoost parameters:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=7
)
```

**Training data:** 32,569 balls (all years with available data), 85/15 train/test split

**Why XGBoost?**
- Handles tabular data very well
- Robust to irrelevant features
- Built-in regularisation prevents overfitting
- Fast to train
- Industry standard for structured sports prediction tasks

**Feature importance (approximate ranks):**
1. `mu_sr` — matchup strike rate
2. `mu_wktr` — matchup wicket rate
3. `bat_sr` — batter career SR
4. `bat_bpct` — boundary percentage
5. `bowl_econ` — bowler economy
6. `mu_balls_log` — sample size of matchup
7. `bowl_wktr` — bowler wicket rate
8. `bat_avg` — batter average (least important)

**Results:**
| Metric | Value |
|---|---|
| Test accuracy | 44.14% |
| Log-loss | 1.2581 |
| Train balls | 27,683 |
| Test balls | 4,886 |
| Outcome classes | 7 |

**Classification report highlights:**
- 0 runs (dot ball): precision 0.52, recall 0.48 — most common, hardest to predict precisely
- 1 run: precision 0.51, recall 0.55 — second most common
- 4 runs: precision 0.28, recall 0.22 — harder, boundaries are situational
- Wicket: precision 0.44, recall 0.32 — rare class, some signal from matchup data

---

#### Model C: Random Forest
**Type:** Ensemble of Decision Trees

**Parameters:**
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1
)
```

**Key difference from XGBoost:** `class_weight="balanced"` — each class is weighted inversely proportional to its frequency. This means the model puts more effort into predicting rare outcomes (6 runs, wickets) at the cost of overall accuracy.

**Results:**
| Metric | Value |
|---|---|
| Test accuracy | 31.42% |
| Log-loss | 1.5079 |

**Why lower than XGBoost?** The balanced class weighting improves recall for rare outcomes (W, 6) but hurts the much more common outcome (0 runs = 37% of balls). Overall accuracy drops because the model "guesses" rare outcomes more often.

**When Random Forest is useful:** If the use case is specifically about identifying high-wicket-probability or high-boundary-probability situations, RF's balanced approach may be preferred.

---

#### Model D: LightGBM
**Type:** Gradient Boosted Decision Trees (histogram-based, leaf-wise growth)

**Parameters:**
```python
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multiclass",
    early_stopping_rounds=50
)
```

**Key differences from XGBoost:**
- LightGBM grows trees leaf-wise (best leaf first) vs level-wise (XGBoost)
- Faster training (histogram binning vs exact split finding)
- Uses early stopping to prevent overfitting automatically

**Results:**
| Metric | Value |
|---|---|
| Test accuracy | 46.05% |
| Log-loss | 1.2359 |

**Best of the standard ML models.** LightGBM's lower log-loss (1.2359 vs XGBoost 1.2581) means its probability estimates are better calibrated, even though its raw accuracy is similar.

---

#### Model E: LSTM — Ball Sequence Model (Deep Learning)
**Type:** Long Short-Term Memory network (PyTorch) — THE BEST Phase 1 MODEL

**The key insight:** The previous 4 models treat each ball independently. They don't know:
- Whether the batter just hit a six
- Whether this is the 5th dot ball in a row
- Whether the batter is at the start of their innings or in full flow

The LSTM addresses this by learning from the **sequence of last 6 ball outcomes**.

**Architecture:**
```
Input A — Sequence (6 balls):
  Each ball: [outcome/6.0, over_number/19.0, ball_within_over/5.0]
  → Shape: (batch_size, 6, 3)
  → LSTM(input=3, hidden=64)
  → Dropout(0.3)
  → Last hidden state: (batch_size, 64)

Input B — Static features (8):
  [bat_sr, bat_avg, bat_bpct, bowl_econ, bowl_wktr, mu_balls_log, mu_sr, mu_wktr]
  → Shape: (batch_size, 8)

Concatenate A + B → (batch_size, 72)
  → Dense(64, ReLU)
  → Dropout(0.2)
  → Dense(7, Softmax)
  → 7-class outcome probability
```

**Training data:** `master_deliveries_with_commentary.csv` (has `over` and `ball_in_over` columns)
**Sequences built:** 25,452 rolling 6-ball windows from 249 innings
**Training:** 20 epochs, Adam optimizer (lr=0.001), early stopping (patience=4)

**Loss progression:**
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.76 | 1.53 |
| 5 | 1.45 | 1.41 |
| 10 | 1.40 | 1.35 |
| 15 | 1.26 | 1.21 |
| 20 | 1.16 | 1.12 |

**Results:**
| Metric | Value |
|---|---|
| Test accuracy | **54.64%** |
| Log-loss | **1.1227** |

**Why LSTM is the best?** It captures momentum and context. After a sequence like `[4, 0, 1, 6, 0, 1]`, the LSTM recognises the batter is scoring freely and adjusts boundary/wicket probabilities accordingly — something a static feature vector cannot capture.

**At inference without context:** Uses zero-padded sequence (neutral game state), giving predictions comparable to other Phase 1 models.

---

### Phase 1 — All Model Comparison Table

| Model | Accuracy | Log-Loss | Data Size | Generalises? | Context-aware? |
|---|---|---|---|---|---|
| Empirical | — (no test) | — | Matchup only | No | No |
| Random Forest | 31.4% | 1.508 | 32,569 balls | Yes | No |
| XGBoost | 44.1% | 1.258 | 32,569 balls | Yes | No |
| LightGBM | 46.1% | 1.236 | 32,569 balls | Yes | No |
| **LSTM** | **54.6%** | **1.123** | 25,452 seqs | Yes | **Yes** |

### Phase 1 Limitations

**What Phase 1 cannot tell us:**
1. **Pressure and dominance** — Is the batter in rhythm or struggling today?
2. **Shot tendencies** — Where does the batter tend to hit vs this bowler?
3. **Match situation** — Is the batter trying to slog or consolidate?
4. **Current form** — Stats are career averages, not recent form signals

These gaps are exactly what Phase 2 addresses using commentary text.

---

## 3. Phase 2 — Commentary Sentiment Enhancement

### What Phase 1 Was Missing

Phase 1 uses aggregate career statistics. If V Kohli's career strike rate is 137, that's what the model uses — regardless of whether he just hit 3 sixes in a row or is playing cautiously at the start of an innings.

**Commentary text captures the in-match, in-over, in-spell context** that stats cannot:
- "Beaten outside off stump, no contact" → batter struggling with this bowler
- "Kohli drives beautifully through the covers for FOUR!" → batter in dominant form
- "Mistimed pull shot, skied to mid-on but safe" → batter not timing well

### How Commentary Adds Value

The Phase 2 approach converts the observation "this batter is historically 25% dominant and 15% beaten against this bowler" into a **sentiment feature** that captures their relationship quality.

**Example — two different pairs with same career SR=130:**

| Pair | Commentary profile | Pressure Index | Dominant % | Beaten % |
|---|---|---|---|---|
| Batter A vs Bowler X | "Driven beautifully", "Pulls to the fence" | -0.12 (low pressure) | 18% | 6% |
| Batter B vs Bowler Y | "Beaten outside off", "Mistimed slog" | +0.22 (high pressure) | 4% | 20% |

Same strike rate, completely different underlying story. Phase 2 captures this difference.

---

### Commentary Parsing Pipeline

**Step 1: `commentary_parser.py` — Text → Structured Fields**

Every commentary string is processed through keyword-matching rules:

```
Input:  "Kohli drives beautifully through the covers for FOUR!"

Output:
  shot_type       = "drive"
  region          = "cover"
  sentiment       = "DOMINANT"
  sentiment_score = +1.0
```

**Shot detection** (8 types, in priority order):
```
defend → sweep → flick → pull → hook → cut → drive → slog
```
Why priority order? Some commentary uses multiple words. "slog sweep" should map to "sweep" not "slog" — so sweep is checked before slog.

**Region detection** (11 regions):
```
cover, mid-off, long-off, mid-on, long-on, mid-wicket,
square leg, fine leg, third man, point, slip
```
Multi-word regions checked before sub-patterns (e.g., "square leg" before "leg").

**Sentiment detection** (5 labels, strict priority):
```
Priority 1 — BEATEN:     beaten/missed/no contact/beat the bat    → score -1.0
Priority 2 — DOMINANT:   FOUR/SIX/boundary/smashed/hammered      → score +1.0
Priority 3 — MISTIMED:   mistimed/top edge/skied/leading edge     → score -0.5
Priority 4 — DEFENSIVE:  defended/blocked/played out/leaves       → score 0.0
Priority 5 — CONTROLLED: (everything else that has content)       → score +0.5
```

Why BEATEN has highest priority? A ball can say "driven for FOUR but it was a thick edge" — the thick edge (MISTIMED/BEATEN signal) overrides the boundary.

**Step 2: `pressure_builder.py` — Aggregation**

Groups parsed data by `(batter, bowler)` pair and computes:

```python
# Per pair:
avg_sentiment_score = mean(sentiment_score)   # weighted average of all ball scores
dominant_pct        = count(DOMINANT) / total_balls
controlled_pct      = count(CONTROLLED) / total_balls
mistimed_pct        = count(MISTIMED) / total_balls
beaten_pct          = count(BEATEN) / total_balls
defensive_pct       = count(DEFENSIVE) / total_balls
boundary_pct        = count(actual boundaries) / total_balls  # from runs data
dot_pct             = count(actual dots) / total_balls
pressure_index      = beaten_pct + mistimed_pct - dominant_pct
```

**Output:** 5,546 batter-bowler pairs with full sentiment profiles (saved to `sentiment_stats.csv`)

---

### Augmented XGBoost Model

**14 features = 8 Phase 1 + 6 sentiment:**

```
Phase 1 (8):    bat_sr, bat_avg, bat_bpct, bowl_econ, bowl_wktr,
                mu_balls_log, mu_sr, mu_wktr

Phase 2 (6):    avg_sentiment_score, pressure_index,
                dominant_pct, beaten_pct, boundary_pct, dot_pct
```

**Training comparison (same data, same split, only features differ):**

| | Baseline (8 feat.) | Augmented (14 feat.) | Gain |
|---|---|---|---|
| Training data | 27,872 commentary balls | Same | — |
| Train/Test split | 85% / 15% | Same | — |
| Accuracy | 43.77% | **48.19%** | **+4.43%** |
| Log-Loss | 1.2581 | **1.2009** | **-0.057** |

**Feature importance in Augmented model (top 5):**
1. `mu_sr` — matchup strike rate (27.3%)
2. `dot_pct` — dot ball rate from commentary (21.7%) ← **2nd most important!**
3. `bat_sr` — batter career SR (12.1%)
4. `avg_sentiment_score` — overall sentiment (9.8%)
5. `mu_wktr` — matchup wicket rate (8.4%)

The fact that `dot_pct` is the 2nd most important feature confirms that commentary-derived signals carry real predictive power beyond what scorecards alone can provide.

---

### Sentiment Models — VADER vs BiLSTM

**Why two sentiment models?**

The keyword rules in `commentary_parser.py` are the "ground truth" labels. To see if a general NLP model (VADER) or a learned model (BiLSTM) can match or beat these rules:

#### VADER Sentiment
**Accuracy vs keyword rules: 18.84%**

VADER was designed for social media and general English text. Cricket commentary uses domain-specific language:
- "beaten" in cricket means "bat missed the ball" — VADER treats this as generally negative but doesn't map well to BEATEN category
- "four" meaning "boundary" — VADER doesn't know this is extremely positive
- "dot" meaning "no run" — general English treat it neutrally

After adding cricket-specific words to the VADER lexicon (six: +2.5, beaten: -2.0, etc.), accuracy improved but still only reached 18.8%. This shows that rule-based general NLP is insufficient for sports domain text.

#### BiLSTM Text Classifier
**Accuracy vs keyword rules: 96.78%**

The BiLSTM was trained directly on the 22,776 commentary strings that had keyword-rule labels. It learned:
- Word patterns that indicate each sentiment class
- Context effects (the full sentence, not just individual words)
- Cricket-specific vocabulary from scratch

**Class-level performance:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BEATEN | 0.94 | 0.91 | 0.93 | 244 |
| CONTROLLED | 0.98 | 0.99 | 0.99 | 1,787 |
| DEFENSIVE | 0.98 | 0.98 | 0.98 | 1,173 |
| DOMINANT | 0.83 | 0.88 | 0.85 | 107 |
| MISTIMED | 0.77 | 0.68 | 0.72 | 106 |
| **Overall** | | | **0.97** | 3,417 |

The hardest classes are DOMINANT and MISTIMED — both rare and sometimes described with ambiguous language in commentary.

**Key insight:** The BiLSTM goes beyond the keyword rules. For example, "tucks it off the pads for a single" would not match any CONTROLLED keywords but the BiLSTM correctly classifies it as CONTROLLED based on learned patterns.

---

### Shot Intelligence Feature

Beyond the outcome prediction models, Phase 2 adds the **Shot Predictor** — a lookup-based system that tells analysts:

> "When batter X faces bowler Y, they play 38% drives to cover, 22% pulls to mid-wicket, and 15% sweeps. They are BEATEN 12% of the time."

**How it works:**
1. `shot_region_stats.csv` is pre-computed for all 5,546 batter-bowler pairs
2. `shot_predictor.py` loads this and provides lookup functions
3. Fallback to batter-overall stats if the specific matchup has fewer than 10 commentary balls

**Output from `ShotPredictor.predict("V Kohli", "A Nortje")`:**
```python
{
  "batter": "V Kohli",
  "bowler": "A Nortje",
  "balls_sample": 26,
  "source": "matchup_direct",
  "shot_types": [
    {"shot_type": "drive", "count": 8, "pct": 0.308},
    {"shot_type": "defend", "count": 5, "pct": 0.192},
    {"shot_type": "pull", "count": 4, "pct": 0.154},
  ],
  "regions": [
    {"region": "cover", "count": 7, "pct": 0.269},
    {"region": "mid-wicket", "count": 4, "pct": 0.154},
  ],
  "sentiment": {
    "avg_sentiment_score": 0.23,
    "dominant_pct": 0.154,
    "beaten_pct": 0.077,
    "pressure_index": -0.077
  }
}
```

---

### Pressure Rankings

Phase 2 also generates **league tables** of:

**Most pressured batters:** Ranked by `avg_pressure_index` (higher = more pressure across all bowlers they faced)

**Bowlers who create most pressure:** Ranked by average pressure index they impose on batters they bowl to

These rankings are derived from `sentiment_stats.csv` and are displayed in the Leaderboards tab of the UI.

---

### Phase 2 Results Summary

| Component | What it does | Key metric |
|---|---|---|
| Commentary Parser | Text → structured fields | 22,776 balls parsed (98.2% of commentary rows) |
| Pressure Builder | Aggregates sentiment per pair | 5,546 unique batter-bowler profiles |
| Augmented XGBoost | 14-feature outcome prediction | 48.19% accuracy (+4.43% over baseline) |
| Shot Predictor | Shot type + region intelligence | 12,780 shot-region combinations tracked |
| VADER Sentiment | General NLP baseline | 18.84% vs keyword rules |
| BiLSTM Classifier | Learned cricket sentiment | 96.78% vs keyword rules |

---

## 4. Phase 3 — Narrative Layer (Full-Match ASR Demo)

### What Phase 3 Is (and Is Not)

**Is:** A **dashboard tab** and **small code folder** (`models/phase3/`) that serve a structured **full-match narrative** for the ICC Men’s T20 World Cup **2024 final (India vs South Africa)**, including ICC match-centre / video buttons and **per-ball commentary** with optional **ASR transcript** snippets for comparison.

**Is not:** A new trained predictor for `{0,1,2,3,4,6,W}`. All outcome and sentiment metrics in Tabs 1–5 still come from Phases 1 and 2.

### Artifacts and Scripts

| Component | Role |
|---|---|
| `prototype_match.py` | Builds `artifacts/prototype_2024_ind_sa_final.json` (balls, text, links) for the UI |
| `download_and_transcribe.py` | Optional: acquire broadcast media and run ASR → `artifacts/transcript_*_asr.txt` |
| `artifacts/*.json` | Versioned narrative payload read by `ultimate_model/app.py` Tab 6 |

### How Phase 3 Relates to Phase 2

Phase 2 already parses **short CricBuzz-style strings** per ball for sentiment and shot intelligence. Phase 3 shows those ideas at **full-match scale** and adds a path for **spoken commentary** (ASR) alongside the same ball index — useful for demos, archival video, and future work on audio+text fusion (without claiming a new accuracy number for ball outcome).

---

## 5. Cross-Phase Comparison — All Models Side by Side

### Outcome Prediction Models

| Model | Phase | Training Data | Features | Accuracy | Log-Loss | Key Advantage |
|---|---|---|---|---|---|---|
| Empirical | 1 | matchup_stats.csv | Historical freq. | N/A | N/A | Perfect for known pairs |
| Random Forest | 1 | 32,569 balls | 8 | 31.4% | 1.508 | Balanced class prediction |
| XGBoost | 1 | 32,569 balls | 8 | 44.1% | 1.258 | Fast, robust baseline |
| LightGBM | 1 | 32,569 balls | 8 | 46.1% | 1.236 | Better calibrated probs |
| **LSTM (DL)** | **1** | **25,452 seqs** | **seq+8** | **54.6%** | **1.123** | **Sequence context** |
| Augmented XGBoost | 2 | 27,872 balls | 14 | 48.2% | 1.201 | Commentary signals |

### Sentiment Models

| Model | Type | Accuracy vs Keywords | Strengths | Weaknesses |
|---|---|---|---|---|
| Keyword Rules | Rule-based | Ground truth | Cricket-specific, interpretable | Misses synonyms |
| VADER | Rule-based NLP | 18.8% | No training needed | Not cricket-aware |
| BiLSTM | Deep Learning | 96.8% | Generalises, handles context | Needs training data |

---

## 6. How Each Phase Helped the Other

**Phase 1 → Phase 2:**

Phase 1's 8 features are the base for the Phase 2 Augmented model. Without Phase 1's feature engineering (matchup_stats, batting/bowling stats), Phase 2 would need to re-derive all batter/bowler context from scratch.

Phase 1's XGBoost model is included as a **comparison baseline** inside Phase 2's training — the augmented model trains a copy with only Phase 1 features on the same data, so the gain is measurable.

**Phase 2 → Phase 1:**

Phase 2 established which sentiment signals are informative (dot_pct being 21.7% feature importance suggests that how a bowler builds pressure in the commentary matters). This validates that Phase 1 was missing real signal.

The LSTM in Phase 1 uses `master_deliveries_with_commentary.csv` (which has `over` and `ball_in_over` columns, available because commentary data processing added them). So Phase 2's data pipeline directly enabled the LSTM architecture.

**Phases 1–2 → Phase 3:**

Phase 3 reuses the **same batter/bowler selectors and model stack** as the rest of the Ultimate Dashboard. It adds **contextual storytelling** (one complete match) so stakeholders see prediction features (Phases 1–2) and **full narrative + optional ASR** in one place.

---

## 7. Key Insights and Findings

**Insight 1: The sequence matters more than the average.**

The LSTM achieves 54.6% accuracy — 8 percentage points above the next best Phase 1 model (LightGBM at 46.1%). The 6-ball window of context contains more predictive information than career statistics alone. Cricket is inherently sequential — momentum is real.

**Insight 2: Commentary text captures signal that scorecards miss.**

`dot_pct` (derived purely from commentary parsing) is the 2nd most important feature in the Augmented XGBoost, beating several traditional cricket statistics like `bat_avg` and `bowl_wktr`. The words used to describe deliveries carry real predictive information about future outcomes.

**Insight 3: VADER is insufficient for domain-specific text.**

18.8% accuracy shows that off-the-shelf NLP tools fail on cricket commentary. A domain-trained model (BiLSTM at 96.8%) is necessary. This validates the decision to use keyword rules as a first pass and then learn from them.

**Insight 4: LightGBM is the best tree-based model.**

LightGBM achieves lower log-loss (1.236) than XGBoost (1.258) with the same 8 features. Better probability calibration matters for downstream use cases like expected runs calculation.

**Insight 5: Random Forest's balanced weights are a design choice, not a bug.**

Random Forest's lower accuracy (31.4%) is because `class_weight="balanced"` was set deliberately. If you care more about correctly predicting a wicket or six (even at the cost of overall accuracy), RF is the right choice. If you want best overall accuracy, use LightGBM or LSTM.

**Insight 6: Pressure Index is a novel metric.**

`pressure_index = beaten_pct + mistimed_pct - dominant_pct` is a composite measure derived entirely from commentary. A batter with pressure_index = +0.25 is in a difficult matchup (more beaten/mistimed than dominant). A batter with pressure_index = -0.15 is in a comfortable matchup (more dominant than being troubled). This metric does not exist in traditional cricket statistics and is a unique contribution of the Phase 2 approach.
