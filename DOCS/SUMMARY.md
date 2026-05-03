# Cricket Analytics Project — Master Summary
### What everything is, what it does, where it lives

---

## What This Project Does (One Line)

**Predicts the probability distribution of the next ball in an ICC T20 World Cup match (dot / 1 run / 2 runs / 3 runs / 4 runs / 6 runs / wicket) given a batter and a bowler — using 5 ML/DL models plus commentary text sentiment analysis.**

---

## Project Structure At a Glance

```
Sports_analytics/
├── master_data/dataset/         ← All raw and processed data (CSVs + JSON)
├── models/
│   ├── phase1/                  ← Phase 1: Stats-based outcome models
│   ├── phase2/                  ← Phase 2: Commentary sentiment + augmented model
│   ├── phase3/                  ← Phase 3: 2024 final narrative JSON + optional ASR artifacts
│   └── ultimate_model/          ← Combined Streamlit dashboard app (6 tabs)
├── source/                      ← Data fetch / scraping scripts
└── DOCS/                        ← All documentation
    ├── 01_Data_and_Model_Creation.md
    ├── 02_Phase_Model_Documentation.md
    ├── 03_UI_Guide_with_Screenshots.md
    ├── 04_Presentation_Guide.md
    ├── SUMMARY.md               ← this file
    ├── SUMMARY.txt
    ├── screenshots/             ← 8 live app screenshots (overview, sidebar, tabs 1–6)
    └── docx/                    ← Word versions of all docs
```

---

## Data — What Exists and Where

| File | Location | Size | What it is |
|---|---|---|---|
| `master_deliveries.csv` | `master_data/dataset/` | 33,662 rows | Every legal delivery, all years, no commentary |
| `master_deliveries_with_commentary.csv` | `master_data/dataset/` | 27,872 rows | Same + `commentary_short` text, 2021–2024 only |
| `master_matches.csv` | `master_data/dataset/` | 149 rows | Match metadata (date, teams, venue, winner) |
| `batting_stats.csv` | `master_data/dataset/` | 450 rows | Career batting stats per player |
| `bowling_stats.csv` | `master_data/dataset/` | 308 rows | Career bowling stats per player |
| `players_info_teams.csv` | `master_data/dataset/` | 3,279 rows | Player-to-team mapping (22 teams, used for UI filters) |
| `matchup_stats.csv` | `models/phase1/artifacts/` | 7,019 rows | Head-to-head batter-bowler pair stats |
| `sentiment_stats.csv` | `models/phase2/artifacts/` | 5,546 rows | Commentary sentiment profile per batter-bowler pair |
| `shot_region_stats.csv` | `models/phase2/artifacts/` | 12,780 rows | Shot type + region frequency per batter-bowler pair |
| Raw JSON files | `master_data/dataset/raw_commentary_cb/` | ~150 files | One JSON per match, raw CricBuzz API response |

**Data source:** CricBuzz internal API — ICC T20 World Cup 2021 (UAE), 2022 (Australia), 2024 (USA/West Indies). 2016 excluded — no commentary available.

---

## The Three Phases — Big Picture

```
PHASE 1                              PHASE 2                              PHASE 3
─────────────────────────────────    ──────────────────────────────────    ─────────────────────────────
"Predict from scorecards"            "Predict + commentary NLP"           "Full-match narrative + ASR demo"

Data:  master_deliveries.csv         Data:  commentary-augmented rows      Data:  prototype JSON + transcripts
       batting/bowling stats                 sentiment_stats / shot CSVs          (2024 IND vs SA final)

Models: Empirical                    Models: Augmented XGBoost, VADER,     Models: (none — UI narrative layer)
        XGBoost     44.1%                    BiLSTM
        Random Forest 31.4%
        LightGBM    46.1%
        LSTM (DL)   54.6%   ← BEST   *sentiment accuracy vs keyword rules

Output: 7-class probability          Output: same 7-class + sentiment      Output: links, ball text, optional ASR
        {0,1,2,3,4,6,W}
```

---

## All 8 Models — What Each Does

### Outcome Prediction Models (Phase 1)

| Model | File | Input | Output | Accuracy | Key fact |
|---|---|---|---|---|---|
| **Empirical** | `empirical_predictor.py` | matchup CSV lookup | 7-class probs | No test | Direct historical frequencies. Falls back to blended/overall if not enough data |
| **XGBoost P1** | `ml_predictor.py` | 8 features | 7-class probs | 44.1% | Original Phase 1 model. 300 trees, max_depth=6 |
| **Random Forest** | `rf_lgbm_predictor.py` | 8 features | 7-class probs | 31.4% | Balanced class weights — predicts rare outcomes more |
| **LightGBM** | `rf_lgbm_predictor.py` | 8 features | 7-class probs | 46.1% | Best tree-based model. Lowest log-loss (1.236) |
| **LSTM (DL)** | `lstm_predictor.py` | 6-ball sequence + 8 features | 7-class probs | **54.6%** | BEST MODEL. PyTorch LSTM. Reads last 6 ball outcomes as context |

**The 8 Phase 1 features:**
`bat_sr` (strike rate), `bat_avg` (average), `bat_bpct` (boundary%), `bowl_econ` (economy), `bowl_wktr` (wicket rate), `mu_balls_log` (matchup sample size), `mu_sr` (matchup SR), `mu_wktr` (matchup wicket prob)

### Augmented Outcome Model (Phase 2)

| Model | File | Input | Output | Accuracy | Key fact |
|---|---|---|---|---|---|
| **Augmented XGBoost** | `augmented_predictor.py` | 14 features (8 P1 + 6 sentiment) | 7-class probs | 48.2% | +4.43% over same-data baseline. `dot_pct` is 2nd most important feature |

**6 extra Phase 2 features:** `avg_sentiment_score`, `pressure_index`, `dominant_pct`, `beaten_pct`, `boundary_pct`, `dot_pct`

### Sentiment Models (Phase 2)

| Model | File | Input | Output | Accuracy | Key fact |
|---|---|---|---|---|---|
| **Keyword Rules** | `commentary_parser.py` | Commentary text | 5 sentiment labels | Ground truth | Rule-based. 80 patterns. Instant. Cricket-specific |
| **VADER** | `commentary_classifier.py` | Commentary text | 5 sentiment labels | 18.8% | General NLP. No training needed. Fails on cricket domain |
| **BiLSTM (DL)** | `commentary_classifier.py` | Commentary text | 5 sentiment labels | **96.8%** | PyTorch BiLSTM. Trained on 22K commentary strings. Best |

**5 Sentiment Labels:** `DOMINANT (+1.0)`, `CONTROLLED (+0.5)`, `DEFENSIVE (0.0)`, `MISTIMED (-0.5)`, `BEATEN (-1.0)`

**Pressure Index** = `beaten_pct + mistimed_pct - dominant_pct` (novel metric, higher = more pressure on batter)

---

## Model Artifacts — What's Saved and Where

| File | Model | Location |
|---|---|---|
| `xgb_model.pkl` + `label_encoder.pkl` | XGBoost P1 | `models/phase1/artifacts/` |
| `rf_model.pkl` + `rf_label_encoder.pkl` | Random Forest | `models/phase1/artifacts/` |
| `lgbm_model.pkl` + `lgbm_label_encoder.pkl` | LightGBM | `models/phase1/artifacts/` |
| `lstm_model.pt` + `lstm_label_encoder.pkl` | LSTM (PyTorch) | `models/phase1/artifacts/` |
| `aug_xgb_model.pkl` + `aug_label_encoder.pkl` | Augmented XGBoost | `models/phase2/artifacts/` |
| `bilstm_sentiment_model.pt` + `bilstm_tokenizer.pkl` | BiLSTM | `models/phase2/artifacts/` |

---

## The UI — 6 Tabs, What Each Shows

**Run:** `streamlit run app.py` from `models/ultimate_model/`  
**URL:** `http://localhost:8501` (default Streamlit port; your terminal may show another)

| Tab | What it shows |
|---|---|
| **Tab 1 — Phase 1: Outcome Prediction** | Grouped bar chart of all selected models' probability distributions. Metric cards for expected runs per model. Full probability table with gradient shading |
| **Tab 2 — Phase 2: Shot Intelligence** | Shot type % (drive/pull/cut etc.), region % (cover/mid-wicket etc.), sentiment donut chart, Pressure Index. Switches between Keyword Rules / VADER / BiLSTM via sidebar |
| **Tab 3 — Phase 2: Augmented Outcome** | Direct P1 XGBoost vs P2 Augmented comparison. Sentiment context message. 3-way chart (Empirical + XGB + Augmented) |
| **Tab 4 — Model Comparison** | Accuracy + log-loss table for all models. Expected Runs bar chart per model for current selection. Sentiment model comparison table |
| **Tab 5 — Leaderboards & Rankings** | Top bowlers vs selected batter (by wicket prob). Top batters vs selected bowler (by exp runs). Pressure index rankings (batters and bowlers) |
| **Tab 6 — Phase 3: Narrative (2024 final)** | ICC match-centre / video links, ball-by-ball written commentary vs optional ASR transcript segments (`models/phase3/artifacts/`) |

**Sidebar controls:**
- Batting Team → filters batter dropdown (22 teams)
- Bowling Team → filters bowler dropdown (independent)
- Model checkboxes → toggle each of 6 models on/off in Tab 1
- Sentiment radio → Keyword Rules / VADER / BiLSTM for Tab 2
- Analyze button → refresh all results

---

## All Scripts — What Each Does

| Script | Location | What it does |
|---|---|---|
| `matchup_builder.py` | `models/phase1/` | Reads deliveries CSV, computes per-pair stats → saves `matchup_stats.csv` |
| `empirical_predictor.py` | `models/phase1/` | Frequency-based prediction using matchup_stats.csv |
| `ml_predictor.py` | `models/phase1/` | XGBoost training + prediction |
| `rf_lgbm_predictor.py` | `models/phase1/` | Random Forest + LightGBM training + prediction |
| `lstm_predictor.py` | `models/phase1/` | PyTorch LSTM training + prediction (6-ball sequence) |
| `predictor.py` | `models/phase1/` | Unified wrapper — `Phase1Predictor` exposes all 5 Phase 1 models via `predict(method=...)` |
| `commentary_parser.py` | `models/phase2/` | Keyword rules: commentary text → shot_type, region, sentiment, sentiment_score |
| `pressure_builder.py` | `models/phase2/` | Aggregates parsed commentary → `sentiment_stats.csv` + `shot_region_stats.csv` |
| `augmented_predictor.py` | `models/phase2/` | XGBoost with 14 features (8 P1 + 6 sentiment). Trains baseline + augmented for comparison |
| `shot_predictor.py` | `models/phase2/` | Lookup-based shot/region/sentiment profile per batter-bowler pair |
| `commentary_classifier.py` | `models/phase2/` | VADER scorer + PyTorch BiLSTM text classifier for sentiment |
| `app.py` | `models/phase1/` | Phase 1 standalone Streamlit app |
| `app.py` | `models/phase2/` | Phase 2 standalone Streamlit app |
| `prototype_match.py` | `models/phase3/` | Builds prototype JSON for Tab 6 (2024 final) |
| `download_and_transcribe.py` | `models/phase3/` | Optional media download + ASR → transcript files |
| `app.py` | `models/ultimate_model/` | Combined Ultimate Dashboard (6 tabs, all models + Phase 3) |
| `fetch_commentary_all_years.py` | `source/` | Fetches commentary JSON from CricBuzz API for 2021/2022/2024 |
| `data_creation.ipynb` | `source/` | Notebook: raw JSON → master_deliveries.csv, batting/bowling stats |

---

## Key Results — Numbers to Remember

| What | Number |
|---|---|
| Total deliveries | 33,662 |
| Deliveries with commentary | 27,872 |
| Matches | 149 |
| Teams | 22 |
| Batters | 450 |
| Bowlers | 308 |
| Batter-bowler pairs | 7,019 |
| Pairs with sentiment profiles | 5,546 |
| **Best outcome model accuracy** | **54.6% (LSTM)** |
| Augmented XGBoost accuracy | 48.2% |
| Phase 1→Phase 2 gain | +4.43% |
| VADER accuracy vs keywords | 18.8% |
| **BiLSTM accuracy vs keywords** | **96.8%** |
| Most important Phase 2 feature | `dot_pct` (21.7% importance) |

---

## Documentation — What Each Doc Covers

| Doc | File | Who should read it | What's inside |
|---|---|---|---|
| **Doc 1** | `01_Data_and_Model_Creation.md` | Anyone wanting to understand the data | How each CSV was built, every column explained, all 8 models with I/O details, data pipeline diagram |
| **Doc 2** | `02_Phase_Model_Documentation.md` | Technical reviewer / lead | Deep-dive on each model, training parameters, accuracy tables, Phase 1 vs 2 vs 3, 6 key insights |
| **Doc 3** | `03_UI_Guide_with_Screenshots.md` | Anyone using the app | Every tab, chart, number, and dropdown explained with screenshots |
| **Doc 4** | `04_Presentation_Guide.md` | The presenter | Ready-to-say scripts, Q&A answers, cheat sheet card, 10-minute demo flow |
| **SUMMARY.md** | this file | Anyone wanting a quick overview | Everything in one place |

---

## How the Phases Connect

```
Phase 1 gives Phase 2:
  • 8 base features (bat_sr, bat_avg, etc.) — Phase 2 augments these
  • XGBoost as baseline comparison inside augmented_predictor.py

Phase 2 gives Phase 1:
  • master_deliveries_with_commentary.csv has over + ball_in_over columns
    that made the LSTM sequence model possible
  • Validates that Phase 1 was missing real signal (dot_pct = 21.7% importance)

Phases 1–2 give Phase 3:
  • Same selectors and model stack in the UI; Phase 3 adds JSON + optional ASR
    for one showcase final (no new outcome model)

All feed into:
  • ultimate_model/app.py — unified dashboard (6 tabs)
```

---

## One-Line Answers to Common Questions

**"What problem does this solve?"**  
Predict next-ball outcome distribution for any T20 World Cup batter vs bowler pair.

**"What's the best model?"**  
LSTM at 54.6% for outcome prediction. BiLSTM at 96.8% for sentiment.

**"Why two prediction phases?"**  
Commentary text only exists for 2021–2024. Phase 1 works on all data; Phase 2 adds commentary signals where available.

**"What is Phase 3?"**  
A narrative / ASR demo tab for the 2024 final — full ball-by-ball text and optional broadcast transcripts — not an eighth outcome model.

**"What is Pressure Index?"**  
`beaten_pct + mistimed_pct - dominant_pct` — a novel metric derived from commentary. Higher = bowler is genuinely troubling the batter.

**"Why is Random Forest accuracy so low?"**  
By design — `class_weight="balanced"` was set to improve recall on rare outcomes (wickets, sixes) at the cost of overall accuracy.

**"What's the most surprising finding?"**  
`dot_pct` from commentary is the 2nd most important feature in the augmented model — meaning how a bowler builds pressure in commentary language predicts future outcomes better than most traditional cricket statistics.
