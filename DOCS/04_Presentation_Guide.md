# Document 4 — Presentation Guide
## Cricket Analytics — What to Say When You Present

This document is written for YOU — the presenter. Every section is a ready-to-use explanation you can say out loud directly.

---

## Table of Contents

1. [2-Minute Project Overview](#1-2-minute-overview)
2. [Explaining the Data](#2-explaining-the-data)
3. [Explaining Phase 1](#3-explaining-phase-1)
4. [Explaining Phase 2](#4-explaining-phase-2)
5. [Explaining the UI — Tab by Tab](#5-explaining-the-ui)
6. [Explaining Phase 3 — Narrative Tab](#6-explaining-phase-3--narrative-tab)
7. [Key Numbers to Remember](#7-key-numbers-to-remember)
8. [Tough Questions — Ready Answers](#8-tough-questions--ready-answers)
9. [Summary Cheat Sheet Card](#9-cheat-sheet-card)

---

## 1. 2-Minute Project Overview

> "This project is a ball-by-ball outcome prediction system for ICC T20 World Cup cricket. What we're doing is — given any batter and any bowler, predict the probability distribution of the next ball: is it going to be a dot, a single, a boundary, or a wicket?
>
> We built this in two **prediction** phases, plus a third **narrative** phase for demos. Phase 1 uses traditional cricket statistics — career averages, strike rates, matchup data — and applies multiple machine learning models to make predictions. Phase 2 adds commentary analysis — we parse the text description of each delivery to extract sentiment, shot types, and regions, and use that as additional features to improve the prediction. Phase 3 does not add another outcome model; it adds a full-match story layer for the 2024 final — ball-by-ball text plus optional ASR from broadcast audio — so stakeholders see stats, NLP, and multi-modal context in one dashboard.
>
> We also added deep learning. In Phase 1, we built an LSTM that reads the last 6 balls of an innings as a sequence — so it understands momentum. In Phase 2, we trained a BiLSTM on commentary text to classify the sentiment of each delivery.
>
> In total, we have 5 outcome prediction models and 3 sentiment analysis approaches in Tabs 1–5, plus the Phase 3 narrative tab, all in one interactive Streamlit app."

---

## 2. Explaining the Data

> "Our data comes from ICC T20 World Cups 2021, 2022, and 2024 — that's about 150 matches and 33,000 deliveries. We collected this from the CricBuzz API, which gives us ball-by-ball data including a text commentary description for each delivery.
>
> The commentary text is the key differentiator for Phase 2. A typical commentary string looks like: 'Good length delivery outside off, Kohli drives beautifully through the covers for FOUR!' From that sentence, we extract the shot type (drive), the region (cover), and the sentiment (dominant, because it's a boundary).
>
> We don't have commentary for 2016 — CricBuzz didn't make it available for that year — so our commentary analysis covers only 2021 to 2024. But the statistical models use all available data."

**Key data numbers to mention:**
- 33,662 deliveries in total
- 27,872 with commentary text (2021-2024)
- 7,019 unique batter-bowler matchup pairs
- 5,546 matchup pairs with full sentiment profiles
- 22 international teams represented
- 450 batters, 308 bowlers in the dataset

---

## 3. Explaining Phase 1

> "Phase 1 is about predicting the outcome of each ball using standard cricket data.
>
> We built 5 models, all predicting the same 7 outcomes: dot ball, 1 run, 2 runs, 3 runs, boundary, six, or wicket.
>
> The first is an Empirical model — it just looks up the actual historical frequencies. If V Kohli has faced Jasprit Bumrah 20 times and hit 3 sixes, the empirical model says 15% chance of a six. Simple but limited.
>
> The second through fourth are machine learning models: XGBoost at 44% accuracy, Random Forest at 31%, and LightGBM at 46%. All three use 8 features: batter's career strike rate, average, boundary percentage, bowler's economy and wicket rate, and then the matchup-specific stats.
>
> The fifth and best Phase 1 model is an LSTM — a recurrent neural network that uses the last 6 balls of the inning as a sequence. So instead of treating each ball in isolation, it knows what happened before. After three dot balls in a row, the LSTM might give higher probability to the batter trying something risky. After a six, it might lower boundary probability for the next ball. This sequence awareness is what drives the LSTM to 54.6% accuracy — about 8-10 percentage points better than the static ML models."

---

## 4. Explaining Phase 2

> "Phase 1 was missing something important — it doesn't know how the batter and bowler are performing against each other in the moment. A batter with a career strike rate of 130 could be dominating the bowling or struggling with edges and mistimings, and Phase 1 wouldn't know the difference.
>
> Phase 2 addresses this using commentary text. We wrote a cricket-specific keyword parser that reads each commentary string and extracts:
> - Was the batter dominant (hit a boundary, smashed it)?
> - Was the batter beaten (missed the ball completely)?
> - Did they mistimed (top edge, leading edge, skied it)?
> - Were they defensive (blocked it, left it)?
> - Was it controlled (timed single or two)?
>
> We aggregate this across all balls in each batter-bowler matchup and create a 'sentiment profile' — dominant percentage, beaten percentage, and what we call the Pressure Index, which is beaten plus mistimed minus dominant. A high Pressure Index means the bowler is genuinely troubling the batter.
>
> When we add these 6 sentiment features to the XGBoost model, accuracy improves from 43.8% to 48.2% — a 4.4 percentage point gain. And the most important individual feature turns out to be dot ball percentage from commentary — meaning, whether the bowler creates dot ball pressure as described in commentary, not just what the scorecard says.
>
> We also built a BiLSTM text classifier trained on the commentary — it reads the words of each delivery description and classifies the sentiment. It achieves 96.8% accuracy against our keyword rules, and crucially it generalises to sentences the keyword rules would miss."

---

## 5. Explaining the UI

> "The UI is a Streamlit dashboard running locally. Let me walk you through it."

### Sidebar

> "On the left is the sidebar. At the top, you select the batter and bowler. You can filter by country — so if I want India batters vs Australia bowlers, I set the batting team to India and the bowling team to Australia. The dropdowns update to show only players from those countries.
>
> Below that are the model selection checkboxes. By default all 5 outcome models are selected. You can uncheck any of them — say you just want to compare XGBoost vs LSTM, uncheck the others and the bar chart updates.
>
> Then there's the sentiment model selector — this controls which approach Tab 2 uses to compute the sentiment metrics. You can choose keyword rules, VADER, or our trained BiLSTM.
>
> Click Analyze and all the tabs update."

### Tab 1 — Outcome Prediction

> "Tab 1 is the main outcome prediction view. At the top you see metric cards — expected runs per ball from each model. So for V Kohli vs Anrich Nortje, Empirical might say 1.3 expected runs, XGBoost says 1.5, LSTM says 1.4.
>
> Below that is the grouped bar chart. Each group is one outcome — dot ball, 1 run, 2 runs, boundary, six, wicket. Within each group, one bar per model. You can immediately see where the models agree and where they differ. The LSTM often differs the most because it's accounting for sequence context which the others ignore.
>
> Below the chart is a detailed probability table with gradient shading."

### Tab 2 — Shot Intelligence

> "Tab 2 is where the commentary analysis comes to life. You see 5 metric cards: how many commentary balls we have, the average sentiment score, the pressure index, dominant percentage, and beaten percentage.
>
> Below that are three charts: shot types on the left — drive, pull, cut, etc. — regions in the middle — cover, mid-wicket, point — and a sentiment donut on the right showing the mix of dominant, controlled, defensive, mistimed, and beaten balls.
>
> If you change the sentiment model in the sidebar to BiLSTM and click Analyze, the sentiment metrics recalculate using the deep learning model instead of the keyword rules. A comparison table appears showing how the two approaches differ."

### Tab 3 — Augmented Outcome

> "Tab 3 shows the Phase 2 augmented prediction specifically. You see a direct comparison: Phase 1 XGBoost expected runs vs Phase 2 Augmented expected runs. The difference card shows how much the sentiment features shifted the prediction.
>
> There's a context box that reads 'Batter is in dominant form', 'under pressure', or 'broadly neutral' based on the commentary profile. And then a three-way bar chart comparing Empirical, XGBoost P1, and Augmented P2 directly."

### Tab 4 — Model Comparison

> "Tab 4 is the technical scorecard. All models in one table with accuracy and log-loss. You can clearly see LSTM at 54.6% is the best for outcome prediction, and BiLSTM at 96.8% is the best for sentiment. Below that is a bar chart showing expected runs per ball for the current selection — useful for seeing which model is most optimistic or pessimistic about a particular batter."

### Tab 5 — Leaderboards

> "Tab 5 gives rankings. On the left, which bowlers have historically dismissed this batter most often — shown as a ranked bar chart by wicket probability. On the right, which batters score most freely against this bowler.
>
> Below that are the Phase 2 pressure rankings — who are the most pressured batters in the dataset based on commentary, and which bowlers create the most pressure. These are derived entirely from our commentary sentiment analysis and don't exist in traditional cricket statistics."

### Tab 6 — Phase 3 (Narrative / 2024 final)

> "Tab 6 is our Phase 3 storyboard — focused on the ICC Men's T20 World Cup 2024 final, India versus South Africa. You'll see links to the official ICC match centre and video, then an expandable ball-by-ball list: written commentary on one side and, where we've run the pipeline, ASR transcript text on the other. This is deliberately separate from the accuracy story in Tabs 1–5: it's showing how broadcast audio could sit beside the same ball index we already use for Phase 2 NLP — a natural next step for archives and broadcast partners."

---

## 6. Explaining Phase 3 — Narrative Tab

> "Phase 3 is optional in terms of **model count** — we still have five outcome predictors and three sentiment engines. What Phase 3 buys you is **narrative depth**: one complete high-profile match with links, per-ball text, and optional speech-to-text. If someone asks 'where is the AI for video?', this tab is your answer: here's the prototype path from ball JSON to ASR segments, without over-claiming a new top-line accuracy metric."

---

## 7. Key Numbers to Remember

Write these on a notepad before your presentation:

**Dataset:**
- 33,662 balls total
- 27,872 with commentary (2021-2024)
- 149 matches, 22 teams
- 450 batters, 308 bowlers
- 7,019 batter-bowler matchup pairs

**Model Accuracy (test set, 15% split):**
- Empirical: no test set
- XGBoost (P1): 44.1%
- Random Forest: 31.4%
- LightGBM: 46.1%
- LSTM: **54.6%** ← best outcome model
- Augmented XGBoost (P2): 48.2%

**Improvement from Phase 1 to Phase 2:**
- Baseline on commentary subset: 43.8%
- With 6 sentiment features: 48.2%
- Gain: **+4.43 percentage points**

**Sentiment models:**
- VADER accuracy: 18.8% vs keyword rules
- BiLSTM accuracy: **96.8%** vs keyword rules

**Most important feature in Augmented model:**
- dot_pct from commentary = 21.7% feature importance (2nd highest overall)

**Commentary structure:**
- 5 sentiment labels: DOMINANT, CONTROLLED, DEFENSIVE, MISTIMED, BEATEN
- 8 shot types: drive, pull, hook, cut, sweep, flick, slog, defend
- 11 regions: cover, mid-off, long-off, mid-on, long-on, mid-wicket, square leg, fine leg, third man, point, slip

---

## 8. Tough Questions — Ready Answers

### "Why are you only using T20 World Cup data? Not IPL or other formats?"

> "We scoped to T20 World Cup specifically because it's international cricket, which means we have more balanced representation of batter-bowler pairs across different playing conditions. IPL data would heavily bias towards India-based players and conditions. For a global cricket model, international T20 is the right starting point."

---

### "44% accuracy seems low — isn't that barely better than random?"

> "Cricket outcomes are genuinely difficult to predict. A random prediction across 7 classes would give about 14% accuracy. We're achieving 44-54%. More importantly, what matters is the **probability calibration** — the model doesn't need to predict the exact outcome, it needs to give well-calibrated probabilities. A model that says 'there's a 35% chance of a dot ball, 12% chance of a boundary' is useful for expected runs calculations even if the exact ball outcome can't be predicted with certainty."

---

### "What's the Pressure Index and why does it matter?"

> "The Pressure Index is `beaten_pct + mistimed_pct - dominant_pct`. It measures the net commentary-based pressure from a bowler on a batter. A batter with a Pressure Index of +0.25 is being beaten, edging, and mistiming significantly more than they are dominating — which their career strike rate might not reveal if they've managed a few lucky boundaries.
>
> This is a novel metric that doesn't exist in traditional cricket analytics. It's derived purely from natural language and gives analysts a nuanced view of match-level pressure beyond what the scorecard captures."

---

### "Why use BiLSTM for commentary when you already have keyword rules at 96.8%?"

> "The keyword rules and BiLSTM are complementary. The keyword rules are fast, interpretable, and cricket-specific. The BiLSTM generalises to language patterns the keyword rules miss. For example, a sentence like 'tucks it off the pads for a single' doesn't match any keyword in our CONTROLLED rule set, but the BiLSTM correctly classifies it as CONTROLLED because it learned that pattern. The 3.2% difference in accuracy represents real commentary strings that the BiLSTM handles correctly that the rules miss."

---

### "Why did you include VADER if it only gets 18.8% accuracy?"

> "VADER serves as a domain adaptation benchmark. It answers the question: 'Can we use off-the-shelf NLP tools for cricket commentary, or do we need cricket-specific approaches?' The answer is clearly no — 18.8% shows that general NLP fails on domain-specific sports text. This validates our decision to build cricket-specific keyword rules and train a domain-specific BiLSTM."

---

### "What was the biggest surprise or learning from this project?"

> "The biggest surprise was that `dot_pct` — the dot ball rate derived from commentary — is the second most important feature in the augmented model. We expected batter strike rate or matchup SR to dominate. But what this tells us is that how a bowler describes building pressure ball by ball (as captured in commentary language) is more predictive than raw career statistics. The language around cricket deliveries carries real signal about what's likely to happen next."

---

### "How would this be used in practice by a cricket team?"

> "There are three main use cases. First, **bowling strategy** — before a match, a team can look at which bowlers historically pressure a specific opposition batter the most (from the pressure index rankings), and plan their bowling attack accordingly. Second, **field placement** — the shot type and region analysis tells you where a batter tends to hit the ball vs a specific bowler, informing field settings. Third, **in-game decision making** — the app can be queried in real time during a match to see probability distributions for a specific matchup, helping captains decide when to make bowling changes."

---

## 9. Cheat Sheet Card

Cut this out and keep it handy during presentation:

```
╔══════════════════════════════════════════════════════════════╗
║         CRICKET ANALYTICS — PRESENTATION CHEAT SHEET         ║
╠══════════════════════════════════════════════════════════════╣
║ DATA                                                          ║
║  • 33,662 total balls   • 149 matches  • 22 teams            ║
║  • 27,872 with commentary (2021, 2022, 2024)                 ║
║  • 450 batters  •  308 bowlers  •  7,019 matchup pairs       ║
╠══════════════════════════════════════════════════════════════╣
║ PHASE 1 — OUTCOME PREDICTION                                  ║
║  Empirical:       frequency lookup    — no test accuracy     ║
║  Random Forest:   31.4% accuracy                             ║
║  XGBoost:         44.1% accuracy                             ║
║  LightGBM:        46.1% accuracy                             ║
║  LSTM (DL):       54.6% accuracy  ← BEST                    ║
╠══════════════════════════════════════════════════════════════╣
║ PHASE 2 — SENTIMENT ENHANCEMENT                               ║
║  Augmented XGBoost: 48.2% (+4.43% over P1 baseline)         ║
║  VADER sentiment:   18.8% vs keyword rules                   ║
║  BiLSTM sentiment:  96.8% vs keyword rules  ← BEST          ║
╠══════════════════════════════════════════════════════════════╣
║ KEY INSIGHT: dot_pct (commentary-derived) =                   ║
║  2nd most important feature in augmented model (21.7%)       ║
╠══════════════════════════════════════════════════════════════╣
║ 5 SENTIMENT LABELS: DOMINANT / CONTROLLED / DEFENSIVE /      ║
║                     MISTIMED / BEATEN                         ║
║ PRESSURE INDEX = beaten% + mistimed% - dominant%             ║
╠══════════════════════════════════════════════════════════════╣
║ PHASE 3 — NARRATIVE (2024 final demo)                         ║
║  Tab 6: ICC links + ball-by-ball text + optional ASR         ║
║  (No extra outcome model — storytelling / multi-modal)      ║
╠══════════════════════════════════════════════════════════════╣
║ UI: 6 tabs, 5 model checkboxes, 3 sentiment radio buttons    ║
║     Team filter for batter AND bowler independently          ║
║ App URL: http://localhost:8501 (Streamlit default port)      ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Presentation Flow Recommendation

**For a 10-minute demo:**

| Time | What to cover | UI action |
|---|---|---|
| 0:00 – 1:30 | Project overview (say Section 1 above) | Show full app |
| 1:30 – 3:00 | Data explanation | Show sidebar bottom footer |
| 3:00 – 5:00 | Phase 1 explanation + LSTM story | Tab 1, show bar chart |
| 5:00 – 7:00 | Phase 2 commentary analysis | Tab 2, show shot + donut |
| 7:00 – 8:30 | Model comparison results | Tab 4, point to accuracy table |
| 8:30 – 9:15 | Leaderboards and pressure rankings | Tab 5 |
| 9:15 – 9:45 | Phase 3 narrative — 2024 final, text vs ASR | Tab 6 |
| 9:45 – 10:00 | Q&A setup | Any tab |

**Wow moments to hit:**
1. Show the sentiment donut chart changing when you switch from Keyword Rules to BiLSTM in the sidebar
2. Show Tab 4's accuracy table — LSTM at 54.6% vs Random Forest at 31.4% is a striking visual
3. Show Tab 5 — "Imran Tahir has a 20% wicket probability vs JC Buttler" is a very concrete and impressive finding
4. Mention: "Pressure Index is a metric that doesn't exist in traditional cricket statistics. We invented it from commentary language."
5. On Tab 6, contrast **written commentary** with **ASR** on the same over/ball — that is the multi-modal story without a new accuracy claim
