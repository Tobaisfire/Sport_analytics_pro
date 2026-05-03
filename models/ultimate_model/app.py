"""
app.py  —  Ultimate Cricket Analytics Dashboard  |  Phase 1 + Phase 2 + Phase 3
-------------------------------------------------------------------------------
5 ML/DL outcome models + 3 sentiment analysis approaches in one app.

Run from the ultimate_model/ folder:
    streamlit run app.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
P1_DIR    = os.path.join(BASE_DIR, "..", "phase1")
P2_DIR    = os.path.join(BASE_DIR, "..", "phase2")
P3_DIR    = os.path.join(BASE_DIR, "..", "phase3")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")

sys.path.insert(0, P1_DIR)
sys.path.insert(0, P2_DIR)
sys.path.insert(0, P3_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Analytics — Ultimate Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Widen main (right) content — default "wide" still caps width (~1280px)
st.markdown(
    """
    <style>
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: min(96vw, 1800px);
            padding-left: 2.25rem;
            padding-right: 2.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── constants ─────────────────────────────────────────────────────────────────
OUTCOMES       = ["0", "1", "2", "3", "4", "6", "W"]
OUTCOME_LABELS = {
    "0": "0 runs", "1": "1 run",  "2": "2 runs",
    "3": "3 runs", "4": "4 runs", "6": "6 runs", "W": "Wicket",
}

# colors per model
MODEL_META = {
    "empirical":  ("Empirical",        "#1976D2"),
    "xgboost":    ("XGBoost (P1)",     "#F57C00"),
    "rf":         ("Random Forest",    "#388E3C"),
    "lgbm":       ("LightGBM",         "#7B1FA2"),
    "lstm":       ("LSTM (DL)",        "#E53935"),
    "augmented":  ("Augmented (P2)",   "#00ACC1"),
    "actual":     ("Actual (hist.)",   "#546E7A"),
}

SHOT_COLOURS = {
    "drive": "#1976D2", "pull": "#E64A19", "flick": "#7B1FA2",
    "cut":   "#00796B", "sweep": "#F57C00", "slog":  "#C62828",
    "defend":"#37474F", "hook": "#6D4C41",
}
REGION_COLOURS = {
    "cover":       "#1565C0", "mid-wicket":  "#AD1457",
    "point":       "#2E7D32", "long-on":     "#BF360C",
    "square leg":  "#4527A0", "long-off":    "#00695C",
    "mid-off":     "#6A1B9A", "mid-on":      "#0277BD",
    "fine leg":    "#558B2F", "third man":   "#FF8F00",
    "slip":        "#4E342E",
}
DEFAULT_CLR = "#90CAF9"

SOURCE_BADGE = {
    "matchup_direct":          ("Direct matchup",                     "#2e7d32"),
    "blended":                 ("Blended (30% matchup + 70% overall)","#e65100"),
    "batter_overall_fallback": ("Fallback — batter overall",          "#b71c1c"),
    "batter_overall":          ("Batter overall",                     "#1565c0"),
    "ml_model":                ("ML model",                           "#4a148c"),
}

PLAYERS_CSV  = os.path.join(DATASET, "players_info_teams.csv")
MATCHUP_CSV  = os.path.join(P1_DIR,  "artifacts", "matchup_stats.csv")
SENT_CSV     = os.path.join(P2_DIR,  "artifacts", "sentiment_stats.csv")
SHOT_CSV     = os.path.join(P2_DIR,  "artifacts", "shot_region_stats.csv")

# Phase 3 — official ICC **match** highlights (not the short YouTube “Epic Montage”)
PHASE3_ICC_MATCH_HIGHLIGHTS_URL = (
    "https://www.icc-cricket.com/videos/"
    "india-script-stunning-title-win-match-highlights-sa-v-ind-t20wc-2024-final"
)
PHASE3_ICC_EXTENDED_HIGHLIGHTS_URL = (
    "https://www.icc-cricket.com/tournaments/t20cricketworldcup/videos/"
    "india-win-their-second-t20-world-cup-title-extended-highlights-sa-v-ind-t20wc-2024-final"
)
PHASE3_ASR_META_JSON = os.path.join(BASE_DIR, "..", "phase3", "artifacts", "asr_run_meta.json")


def phase3_asr_pipeline_source_url() -> str | None:
    """YouTube (or other) URL recorded when ASR was run — for transparency only."""
    if not os.path.isfile(PHASE3_ASR_META_JSON):
        return None
    try:
        with open(PHASE3_ASR_META_JSON, encoding="utf-8") as f:
            m = json.load(f)
        u = m.get("source_url")
        return str(u).strip() if u else None
    except (json.JSONDecodeError, OSError):
        return None


# accuracy summary (from training runs)
MODEL_ACCURACY = {
    "empirical": None,
    "xgboost":   0.4414,
    "rf":        0.3142,
    "lgbm":      0.4605,
    "lstm":      0.5464,
    "augmented": 0.4819,
}
MODEL_LOGLOSS = {
    "xgboost":   1.2581,
    "rf":        1.5079,
    "lgbm":      1.2359,
    "lstm":      1.1227,
    "augmented": 1.2009,
}

# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Phase 1 predictor (all models)...")
def load_p1():
    from predictor import Phase1Predictor
    return Phase1Predictor()


@st.cache_resource(show_spinner="Loading Phase 2 augmented predictor...")
def load_aug():
    from augmented_predictor import AugmentedPredictor
    aug = AugmentedPredictor()
    try:
        aug.load()
        return aug
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading Phase 2 shot predictor...")
def load_shot():
    from shot_predictor import ShotPredictor
    return ShotPredictor().load()


@st.cache_resource(show_spinner="Loading BiLSTM commentary classifier...")
def load_bilstm():
    try:
        from commentary_classifier import CommentaryClassifier
        clf = CommentaryClassifier()
        clf.load()
        return clf
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        from commentary_classifier import VADERSentiment
        return VADERSentiment()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_matchup() -> pd.DataFrame:
    return pd.read_csv(MATCHUP_CSV)


@st.cache_data(show_spinner=False)
def load_sentiment() -> pd.DataFrame:
    return pd.read_csv(SENT_CSV)


@st.cache_data(show_spinner=False)
def load_deliveries_for_sentiment() -> pd.DataFrame:
    """Load commentary rows with keyword-parsed sentiment (for Phase 2 toggle)."""
    p = os.path.join(DATASET, "master_deliveries_with_commentary.csv")
    df = pd.read_csv(p, low_memory=False)
    df = df[df["commentary_short"].notna()].copy()
    try:
        import sys as _sys
        _sys.path.insert(0, P2_DIR)
        from commentary_parser import parse_dataframe
        df = parse_dataframe(df, text_col="commentary_short")
    except Exception:
        df["sentiment"] = None
    return df


@st.cache_data(show_spinner=False)
def load_phase3_prototype_bundle() -> dict:
    """
    One 2024 final: full-match ball commentary profile vs simulated highlight ASR text.
    Uses the same keyword sentiment rules as Phase 2 for both sides.
    """
    try:
        from prototype_match import build_phase3_comparison, load_prototype
        from commentary_parser import parse_one

        proto = load_prototype()
        df = load_deliveries_for_sentiment()
        return build_phase3_comparison(df, proto, parse_one)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def build_team_map() -> dict[str, set]:
    pi = pd.read_csv(PLAYERS_CSV)
    team_map: dict[str, set] = {}
    for _, row in pi.iterrows():
        team_map.setdefault(str(row["team"]), set()).add(str(row["player_name"]))
    return team_map


@st.cache_data(show_spinner=False)
def all_teams() -> list[str]:
    return sorted(pd.read_csv(PLAYERS_CSV)["team"].unique().tolist())


@st.cache_data(show_spinner=False)
def p1_all_batters(_matchup_df: pd.DataFrame) -> list[str]:
    return sorted(_matchup_df["batter"].unique().tolist())


@st.cache_data(show_spinner=False)
def p1_all_bowlers(_matchup_df: pd.DataFrame) -> list[str]:
    return sorted(_matchup_df["bowler"].unique().tolist())


@st.cache_data(show_spinner=False)
def p2_all_batters(_sent_df: pd.DataFrame) -> list[str]:
    return sorted(_sent_df["batter"].unique().tolist())


@st.cache_data(show_spinner=False)
def p2_all_bowlers(_sent_df: pd.DataFrame) -> list[str]:
    return sorted(_sent_df["bowler"].unique().tolist())


# ── chart helpers ─────────────────────────────────────────────────────────────

def _dark(fig, ax):
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def outcome_bar_chart(model_probs: dict[str, dict],
                      title: str = "Outcome Probabilities") -> plt.Figure:
    """
    Generic grouped bar chart for any set of models.

    model_probs : { model_key: { outcome: probability } }
    """
    series, names, clrs = [], [], []
    for key, probs in model_probs.items():
        if not probs:
            continue
        label, clr = MODEL_META.get(key, (key, DEFAULT_CLR))
        series.append([probs.get(o, 0) * 100 for o in OUTCOMES])
        names.append(label)
        clrs.append(clr)

    if not series:
        fig, ax = plt.subplots(figsize=(8, 3))
        _dark(fig, ax)
        ax.text(0.5, 0.5, "No data", ha="center", color="white", transform=ax.transAxes)
        return fig

    n = len(series)
    x = np.arange(len(OUTCOMES))
    w = min(0.8 / n, 0.18)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _dark(fig, ax)

    bars_list = []
    for i, (vals, name, clr) in enumerate(zip(series, names, clrs)):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=clr,
                      alpha=0.9, edgecolor="none")
        bars_list.append(bars)

    for bars in bars_list:
        for bar in bars:
            h = bar.get_height()
            if h >= 3.0 and n <= 4:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=6.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], color="white", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Probability (%)", color="white")
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8,
              ncol=min(n, 5))
    plt.tight_layout()
    return fig


def horiz_bar(labels, values, title, colours=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, max(2.2, len(labels) * 0.52)))
    _dark(fig, ax)
    clrs = colours or [DEFAULT_CLR] * len(labels)
    bars = ax.barh(labels, values, color=clrs, edgecolor="none", alpha=0.92)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, pad=8, color="white")
    ax.set_xticks([]); ax.set_xticklabels([])
    plt.tight_layout()
    return fig


def sentiment_compare_bar(p_ball: dict, p_narr: dict, title: str) -> plt.Figure:
    """Grouped bars: ball-by-ball commentary vs highlight narrative (percent mix)."""
    labels = ["Dominant", "Controlled", "Defensive", "Mistimed", "Beaten"]
    keys = ["dominant_pct", "controlled_pct", "defensive_pct",
            "mistimed_pct", "beaten_pct"]
    b1 = [float(p_ball.get(k, 0)) * 100 for k in keys]
    b2 = [float(p_narr.get(k, 0)) * 100 for k in keys]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _dark(fig, ax)
    ax.bar(x - w / 2, b1, w, label="Ball commentary (every delivery)", color="#1976D2", alpha=0.92)
    ax.bar(x + w / 2, b2, w, label="Highlight narrative (video→audio→text)", color="#E53935", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylabel("% of labelled snippets", color="white")
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    plt.tight_layout()
    return fig


def sentiment_donut(pcts: dict) -> plt.Figure | None:
    order = [("dominant_pct", "Dominant", "#43A047"),
             ("controlled_pct", "Controlled", "#1976D2"),
             ("defensive_pct", "Defensive", "#546E7A"),
             ("mistimed_pct",  "Mistimed",  "#FFA000"),
             ("beaten_pct",    "Beaten",    "#E53935")]
    sizes, clrs, lbls = [], [], []
    for key, label, clr in order:
        v = float(pcts.get(key, 0)) * 100
        if v > 0.1:
            sizes.append(v); clrs.append(clr)
            lbls.append(f"{label}\n{v:.1f}%")
    if not sizes:
        return None
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    wedges, _ = ax.pie(sizes, colors=clrs, startangle=90,
                       wedgeprops=dict(width=0.5, edgecolor="#0e1117"))
    ax.legend(wedges, lbls, loc="lower center", ncol=2, fontsize=6.5,
              facecolor="#1e1e2e", labelcolor="white",
              bbox_to_anchor=(0.5, -0.38))
    ax.set_title("Sentiment Mix", color="white", fontsize=9, pad=8)
    plt.tight_layout()
    return fig


def leaderboard_horiz(df: pd.DataFrame, x_col: str, y_col: str,
                      title: str, cmap_name: str = "Blues") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(df) * 0.44)))
    _dark(fig, ax)
    vals   = df[x_col].values
    labels = df[y_col].astype(str).values
    norm   = plt.Normalize(vals.min() - 0.001, vals.max() + 0.001)
    colours = [plt.get_cmap(cmap_name)(norm(v)) for v in vals]
    bars = ax.barh(labels, vals, color=colours, edgecolor="none", alpha=0.92)
    for bar, val in zip(bars, vals):
        ax.text(val + abs(vals.max()) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_xticks([]); ax.set_xticklabels([])
    plt.tight_layout()
    return fig


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:3px 9px;'
        f'border-radius:11px;font-size:0.80em;font-weight:600">{text}</span>'
    )


# ── player list filtering ─────────────────────────────────────────────────────

def filter_players(all_players: list[str],
                   team_map: dict[str, set],
                   selected_team: str) -> list[str]:
    if selected_team == "All Teams":
        return all_players
    team_set = team_map.get(selected_team, set())
    filtered = [p for p in all_players if p in team_set]
    return filtered if filtered else all_players


# ── sentiment analysis helpers ────────────────────────────────────────────────

SENT_LABEL_COLORS = {
    "DOMINANT":   "#43A047",
    "CONTROLLED": "#1976D2",
    "DEFENSIVE":  "#546E7A",
    "MISTIMED":   "#FFA000",
    "BEATEN":     "#E53935",
}

def _compute_sentiment_distribution(
    batter: str,
    bowler: str | None,
    delivery_df: pd.DataFrame,
    model: str,
    vader=None,
    bilstm=None,
) -> dict:
    """
    Compute sentiment distribution for a batter (vs optional bowler)
    using the chosen sentiment model.

    Returns dict with keys: label_counts, pcts, total
    """
    df = delivery_df.copy()
    if bowler:
        df = df[(df["batter"] == batter) & (df["bowler"] == bowler)]
    else:
        df = df[df["batter"] == batter]

    if df.empty:
        return {}

    texts = df["commentary_short"].fillna("").tolist()

    if model == "Keyword Rules":
        labels = df["sentiment"].dropna().tolist()
        if not labels:
            return {}
    elif model == "VADER":
        if vader is None:
            return {}
        labels = [vader.score(t)["label"] for t in texts]
    else:  # BiLSTM
        if bilstm is None:
            return {}
        result_df = bilstm.predict_batch(texts, batch_size=256)
        labels = result_df["bilstm_label"].tolist()

    from collections import Counter
    counts = Counter(labels)
    total  = sum(counts.values())
    pcts   = {k: v / total for k, v in counts.items()} if total else {}
    return {"label_counts": counts, "pcts": pcts, "total": total}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # load predictors & data
    p1_pred   = load_p1()
    aug_pred  = load_aug()
    shot_pred = load_shot()
    bilstm_clf = load_bilstm()
    vader_clf  = load_vader()
    matchup_df = load_matchup()
    sent_df    = load_sentiment()
    team_map   = build_team_map()
    teams      = all_teams()

    p1_batters  = p1_all_batters(matchup_df)
    p1_bowlers  = p1_all_bowlers(matchup_df)
    p2_batters_ = p2_all_batters(sent_df)
    p2_bowlers_ = p2_all_bowlers(sent_df)

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Cricket Analytics")
        st.markdown("##### Phase 1 + Phase 2 + Phase 3 — Ultimate Dashboard")
        st.markdown("---")

        all_bat_combined  = sorted(set(p1_batters) | set(p2_batters_))
        all_bowl_combined = sorted(set(p1_bowlers) | set(p2_bowlers_))

        st.markdown("**Batter**")
        bat_team_sel = st.selectbox(
            "Batting Team",
            options=["All Teams"] + teams,
            index=0, key="bat_team",
            help="Filter the batter list to players from this team.",
        )
        filtered_bats = filter_players(all_bat_combined, team_map, bat_team_sel)
        default_bat   = "JC Buttler"
        if default_bat not in filtered_bats:
            default_bat = filtered_bats[0] if filtered_bats else all_bat_combined[0]
        batter = st.selectbox(
            "Select Batter",
            options=filtered_bats,
            index=filtered_bats.index(default_bat) if default_bat in filtered_bats else 0,
        )

        st.markdown("---")

        st.markdown("**Bowler**")
        bowl_team_sel = st.selectbox(
            "Bowling Team",
            options=["All Teams"] + teams,
            index=0, key="bowl_team",
            help="Filter the bowler list to players from this team.",
        )
        filtered_bowls = filter_players(all_bowl_combined, team_map, bowl_team_sel)
        bowler_opts    = ["(Any / Overall)"] + filtered_bowls
        bowler_sel     = st.selectbox("Select Bowler  (optional)", options=bowler_opts, index=0)
        bowler         = None if bowler_sel == "(Any / Overall)" else bowler_sel

        st.markdown("---")

        # ── Model selector ────────────────────────────────────────────────────
        st.markdown("**Outcome Models**")
        sel_empirical = st.checkbox("Empirical",      value=True)
        sel_xgboost   = st.checkbox("XGBoost (P1)",   value=True)
        sel_rf        = st.checkbox("Random Forest",  value=True)
        sel_lgbm      = st.checkbox("LightGBM",       value=True)
        sel_lstm      = st.checkbox("LSTM (DL)",      value=True)
        sel_augmented = st.checkbox("Augmented (P2)", value=(aug_pred is not None))

        selected_p1_methods = [m for m, s in [
            ("empirical", sel_empirical), ("xgboost", sel_xgboost),
            ("rf", sel_rf), ("lgbm", sel_lgbm), ("lstm", sel_lstm),
        ] if s]

        st.markdown("---")

        # ── Sentiment model selector ──────────────────────────────────────────
        st.markdown("**Sentiment Model (Tab 2)**")
        sentiment_model_choice = st.radio(
            "Sentiment approach",
            options=["Keyword Rules", "VADER", "BiLSTM (DL)"],
            index=0,
            help="Choose how commentary sentiment is computed in the Shot Intelligence tab.",
        )

        st.markdown("---")
        analyze = st.button("Analyze", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown(
            "<small>"
            f"Batting team: <b>{bat_team_sel}</b> &rarr; {len(filtered_bats):,} batters<br>"
            f"Bowling team: <b>{bowl_team_sel}</b> &rarr; {len(filtered_bowls):,} bowlers<br><br>"
            "Data: ICC T20 World Cup<br>"
            "2021 &bull; 2022 &bull; 2024<br>"
            "27,871 balls &bull; 99.9% commentary<br>"
            "<b>Phase 3</b>: prototype — 2024 Final (IND vs SA)"
            "</small>",
            unsafe_allow_html=True,
        )

    # ── page title ────────────────────────────────────────────────────────────
    st.title("Cricket Analytics — Ultimate Dashboard")
    st.caption(
        "**5 ML/DL models** for outcome prediction (Empirical, XGBoost, Random Forest, "
        "LightGBM, LSTM) + **3 sentiment approaches** (Keyword Rules, VADER, BiLSTM) "
        "powered by Phase 1 & Phase 2 commentary analysis. "
        "**Phase 3** (prototype): **2024 T20 WC Final** (IND vs SA, match **1415755**) — full ball commentary "
        "vs simulated highlight-reel transcript."
    )

    # ── run analysis ──────────────────────────────────────────────────────────
    need = (
        analyze
        or "ult_batter" not in st.session_state
        or st.session_state.get("ult_batter") != batter
        or st.session_state.get("ult_bowler") != bowler
        or st.session_state.get("ult_methods") != selected_p1_methods
    )
    if need:
        st.session_state.ult_batter  = batter
        st.session_state.ult_bowler  = bowler
        st.session_state.ult_methods = selected_p1_methods

        with st.spinner("Running all models..."):
            # Phase 1 — all selected models
            try:
                cmp = p1_pred.compare(batter, bowler, methods=selected_p1_methods)
                st.session_state.ult_cmp = cmp
            except Exception as e:
                st.session_state.ult_cmp = {"error": str(e)}

            # individual model results for metrics
            st.session_state.ult_results = {}
            for m in selected_p1_methods:
                try:
                    st.session_state.ult_results[m] = p1_pred.predict(batter, bowler, method=m)
                except Exception as exc:
                    st.session_state.ult_results[m] = {"error": str(exc)}

            # Phase 2 shot
            st.session_state.ult_shot = shot_pred.predict(batter, bowler, top_n=6)

            # Phase 2 augmented
            if aug_pred:
                st.session_state.ult_aug = aug_pred.predict(batter, bowler)
            else:
                st.session_state.ult_aug = None

    cmp_r     = st.session_state.get("ult_cmp", {})
    results_r = st.session_state.get("ult_results", {})
    shot_r    = st.session_state.get("ult_shot", {})
    aug_r     = st.session_state.get("ult_aug")

    bowler_label = bowler if bowler else "(any bowler)"

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Phase 1 — Outcome Prediction",
        "Phase 2 — Shot Intelligence",
        "Phase 2 — Augmented Outcome",
        "Model Comparison",
        "Leaderboards & Rankings",
        "Phase 3 — Narrative prototype (2024)",
    ])

    # ═══════════════════════════════════ TAB 1 ════════════════════════════════
    with tab1:
        st.subheader(f"Outcome Prediction: {batter}  vs  {bowler_label}")
        st.caption(f"Models selected: {', '.join(MODEL_META[m][0] for m in selected_p1_methods)}")

        if "error" in cmp_r:
            st.error(f"Phase 1 error: {cmp_r['error']}")
        else:
            emp_r = results_r.get("empirical", {})
            src   = emp_r.get("source", "")
            info  = SOURCE_BADGE.get(src, (src, "#555"))
            st.markdown(f"Data source: {_badge(info[0], info[1])}", unsafe_allow_html=True)
            st.markdown("")

            # actual matchup probs
            actual_probs = None
            if bowler:
                mask = ((matchup_df["batter"] == batter) & (matchup_df["bowler"] == bowler))
                row  = matchup_df[mask]
                if not row.empty:
                    r = row.iloc[0]
                    actual_probs = {
                        "0": float(r["prob_0"]), "1": float(r["prob_1"]),
                        "2": float(r["prob_2"]), "3": float(r["prob_3"]),
                        "4": float(r["prob_4"]), "6": float(r["prob_6"]),
                        "W": float(r["prob_W"]),
                    }

            # build model_probs dict for chart
            model_probs: dict[str, dict] = {}
            for m in selected_p1_methods:
                r = results_r.get(m, {})
                if r and "probs" in r:
                    model_probs[m] = r["probs"]
            if actual_probs:
                model_probs["actual"] = actual_probs

            # metric cards
            er_vals = cmp_r.get("expected_runs", {})
            cols    = st.columns(min(len(selected_p1_methods) + 1, 5))
            cols[0].metric("Balls on Record", str(cmp_r.get("balls_sample", 0)))
            for i, m in enumerate(selected_p1_methods[:4], start=1):
                er = er_vals.get(m)
                label, _ = MODEL_META.get(m, (m, ""))
                cols[i].metric(f"Exp Runs ({label})", f"{er:.3f}" if er is not None else "n/a")

            st.markdown("---")

            fig = outcome_bar_chart(model_probs, title=f"Outcome Probabilities — {batter} vs {bowler_label}")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("---")

            # probability table
            rows_tbl = []
            for o in OUTCOMES:
                row_d = {"Outcome": OUTCOME_LABELS[o]}
                for m in selected_p1_methods:
                    r = results_r.get(m, {})
                    p = float(r.get("probs", {}).get(o, 0)) * 100
                    label, _ = MODEL_META.get(m, (m, ""))
                    row_d[label] = round(p, 1)
                if actual_probs:
                    row_d["Actual %"] = round(actual_probs.get(o, 0) * 100, 1)
                rows_tbl.append(row_d)

            df_tbl   = pd.DataFrame(rows_tbl)
            num_cols = [c for c in df_tbl.columns if c != "Outcome"]
            fmt_dict = {c: "{:.1f}" for c in num_cols}
            st.dataframe(
                df_tbl.style.format(fmt_dict)
                      .background_gradient(subset=num_cols, cmap="Blues", vmin=0, vmax=55),
                hide_index=True, use_container_width=True,
            )

    # ═══════════════════════════════════ TAB 2 ════════════════════════════════
    with tab2:
        st.subheader(f"Phase 2 Shot Intelligence: {batter}  vs  {bowler_label}")

        # Sentiment model selector (mirrors sidebar)
        sent_choice = sentiment_model_choice

        if sent_choice == "Keyword Rules":
            sent_badge_lbl, sent_badge_clr = "Keyword Rules", "#2e7d32"
        elif sent_choice == "VADER":
            sent_badge_lbl, sent_badge_clr = "VADER Sentiment", "#e65100"
        else:
            sent_badge_lbl, sent_badge_clr = "BiLSTM (DL)", "#7b1fa2"

        st.markdown(f"Sentiment model: {_badge(sent_badge_lbl, sent_badge_clr)}",
                    unsafe_allow_html=True)

        if sent_choice != "Keyword Rules":
            # live computation from commentary text
            with st.spinner(f"Running {sent_choice} sentiment analysis..."):
                delivery_df = load_deliveries_for_sentiment()
                sent_dist   = _compute_sentiment_distribution(
                    batter, bowler, delivery_df,
                    model=sent_choice,
                    vader=vader_clf,
                    bilstm=bilstm_clf,
                )
        else:
            sent_dist = {}   # use precomputed shot_r below

        st.markdown("")

        if "error" in shot_r:
            st.warning(shot_r["error"])
        else:
            # use keyword-based sentiment from shot_predictor for shot/region,
            # override sentiment distribution if another model was selected
            src_clr  = "#2e7d32" if shot_r["source"] == "matchup_direct" else "#e65100"
            src_lbl  = ("Direct pair data" if shot_r["source"] == "matchup_direct"
                        else "Batter-overall fallback")
            st.markdown(f"Commentary source: {_badge(src_lbl, src_clr)}", unsafe_allow_html=True)

            sent = shot_r.get("sentiment") or {}

            # if alternate sentiment model was chosen, override pcts
            if sent_dist and sent_dist.get("pcts"):
                pcts = sent_dist["pcts"]
                # remap to field names expected by sentiment_donut
                sent_override = {
                    "dominant_pct":   pcts.get("DOMINANT",   0),
                    "controlled_pct": pcts.get("CONTROLLED", 0),
                    "defensive_pct":  pcts.get("DEFENSIVE",  0),
                    "mistimed_pct":   pcts.get("MISTIMED",   0),
                    "beaten_pct":     pcts.get("BEATEN",     0),
                    "avg_sentiment_score": sent.get("avg_sentiment_score", 0),
                    "pressure_index":      pcts.get("BEATEN", 0) + pcts.get("MISTIMED", 0) - pcts.get("DOMINANT", 0),
                    "boundary_pct":        sent.get("boundary_pct", 0),
                    "dot_pct":             sent.get("dot_pct", 0),
                }
                display_sent  = sent_override
                n_balls_shown = sent_dist.get("total", shot_r["balls_sample"])
            else:
                display_sent  = sent
                n_balls_shown = shot_r["balls_sample"]

            # metric cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Commentary Balls", n_balls_shown)
            c2.metric("Avg Sentiment",
                      f"{display_sent.get('avg_sentiment_score', 0):+.3f}",
                      help="-1 (beaten) to +1 (dominant)")
            pi = display_sent.get("pressure_index", 0)
            c3.metric("Pressure Index", f"{pi:+.3f}")
            c4.metric("Dominant %",  f"{display_sent.get('dominant_pct', 0)*100:.1f}%")
            c5.metric("Beaten %",    f"{display_sent.get('beaten_pct', 0)*100:.1f}%")

            st.markdown("---")
            col_s, col_r, col_d = st.columns([2, 2, 1.3])

            with col_s:
                shots = [s for s in shot_r["shot_types"] if s["shot_type"] != "unknown"]
                if shots:
                    labels  = [s["shot_type"].title() for s in shots]
                    values  = [s["pct"] * 100 for s in shots]
                    colours = [SHOT_COLOURS.get(s["shot_type"], DEFAULT_CLR) for s in shots]
                    fig = horiz_bar(labels, values, f"Shot Types — {batter}", colours)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                else:
                    st.info("No recognised shot type data.")

            with col_r:
                regions = [r for r in shot_r["regions"] if r["region"] != "unknown"]
                if regions:
                    labels  = [r["region"].title() for r in regions]
                    values  = [r["pct"] * 100 for r in regions]
                    colours = [REGION_COLOURS.get(r["region"], DEFAULT_CLR) for r in regions]
                    fig = horiz_bar(labels, values, f"Regions — {batter}", colours)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                else:
                    st.info("No recognised region data.")

            with col_d:
                if display_sent:
                    fig = sentiment_donut(display_sent)
                    if fig:
                        st.pyplot(fig, use_container_width=True); plt.close(fig)

            # detailed sentiment breakdown
            if display_sent:
                st.markdown("---")
                st.markdown("#### Full Sentiment Breakdown")
                tbl = pd.DataFrame({
                    "Signal": ["Dominant", "Controlled", "Defensive",
                               "Mistimed", "Beaten",
                               "Boundary Hit", "Dot Ball", "Pressure Index"],
                    "Value": [
                        f"{display_sent.get('dominant_pct',   0)*100:.1f}%",
                        f"{display_sent.get('controlled_pct', 0)*100:.1f}%",
                        f"{display_sent.get('defensive_pct',  0)*100:.1f}%",
                        f"{display_sent.get('mistimed_pct',   0)*100:.1f}%",
                        f"{display_sent.get('beaten_pct',     0)*100:.1f}%",
                        f"{display_sent.get('boundary_pct',   0)*100:.1f}%",
                        f"{display_sent.get('dot_pct',        0)*100:.1f}%",
                        f"{display_sent.get('pressure_index', 0):+.4f}",
                    ],
                    "Meaning": [
                        "Boundaries & big shots — batter in control",
                        "Timed singles/doubles — batter comfortable",
                        "Blocked/played out — safe but passive",
                        "Top/leading edges — not timing well",
                        "Beaten / missed completely",
                        "Actual boundary rate in commentary balls",
                        "Actual dot ball rate",
                        "beaten%+mistimed%-dominant% (higher=more pressure)",
                    ],
                })
                st.dataframe(tbl, hide_index=True, use_container_width=True)

            # Sentiment model comparison (if not keyword rules)
            if sent_choice != "Keyword Rules" and sent_dist:
                st.markdown("---")
                st.markdown("#### Sentiment Model Comparison")
                # keyword baseline
                kw_dist = {}
                if sent and "dominant_pct" in sent:
                    kw_dist = {
                        "DOMINANT":   sent.get("dominant_pct", 0),
                        "CONTROLLED": sent.get("controlled_pct", 0),
                        "DEFENSIVE":  sent.get("defensive_pct", 0),
                        "MISTIMED":   sent.get("mistimed_pct", 0),
                        "BEATEN":     sent.get("beaten_pct", 0),
                    }
                model_labels = ["DOMINANT","CONTROLLED","DEFENSIVE","MISTIMED","BEATEN"]
                cmp_rows = []
                for lbl in model_labels:
                    row = {"Sentiment": lbl}
                    if kw_dist:
                        row["Keyword Rules %"] = round(kw_dist.get(lbl, 0) * 100, 1)
                    row[f"{sent_choice} %"] = round(
                        sent_dist["pcts"].get(lbl, 0) * 100, 1
                    )
                    cmp_rows.append(row)
                df_cmp = pd.DataFrame(cmp_rows)
                num_c  = [c for c in df_cmp.columns if c != "Sentiment"]
                st.dataframe(
                    df_cmp.style.format({c: "{:.1f}" for c in num_c})
                          .background_gradient(subset=num_c, cmap="Blues"),
                    hide_index=True, use_container_width=True,
                )

    # ═══════════════════════════════════ TAB 3 ════════════════════════════════
    with tab3:
        st.subheader(f"Phase 2 Augmented Outcome: {batter}  vs  {bowler_label}")

        if aug_r is None:
            st.warning(
                "Augmented model not found. "
                "Run `python augmented_predictor.py` from `models/phase2/` first."
            )
        else:
            emp_r  = results_r.get("empirical", {})
            xgb_r  = results_r.get("xgboost",   {})
            emp_probs = emp_r.get("probs", {})
            xgb_probs = xgb_r.get("probs", {})
            aug_probs = aug_r.get("probs", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Exp Runs (P1 XGBoost)",  f"{xgb_r.get('expected_runs', 0):.3f}" if xgb_r else "n/a")
            c2.metric("Exp Runs (P2 Augmented)", f"{aug_r['expected_runs']:.3f}")
            if xgb_r and xgb_r.get("expected_runs") is not None:
                diff = aug_r["expected_runs"] - xgb_r.get("expected_runs", 0)
                c3.metric("Difference", f"{diff:+.3f}", delta=f"{diff:+.3f}", delta_color="normal")
            else:
                c3.metric("Pressure Index", f"{aug_r.get('pressure_index', 0):+.3f}")
            c4.metric("Tendency", aug_r.get("tendency", "—"))

            pi = aug_r.get("pressure_index", 0)
            sc = aug_r.get("avg_sentiment_score", 0)
            dom  = aug_r.get("dominant_pct", 0)
            beat = aug_r.get("beaten_pct", 0)
            if sc > 0.3 and dom > 0.05:
                ctx = f"Commentary signals: **{batter}** is in **dominant** form vs {bowler_label}."
            elif pi > 0.05 and beat > 0.05:
                ctx = f"Commentary signals: **{batter}** is **under pressure** vs {bowler_label}."
            else:
                ctx = f"Sentiment profile for **{batter}** vs {bowler_label} is broadly neutral."
            st.info(ctx)

            st.markdown("---")
            aug_model_probs = {}
            if emp_probs: aug_model_probs["empirical"] = emp_probs
            if xgb_probs: aug_model_probs["xgboost"]  = xgb_probs
            if aug_probs: aug_model_probs["augmented"] = aug_probs
            fig = outcome_bar_chart(aug_model_probs,
                                    title="Empirical vs XGBoost vs Augmented")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.markdown("#### Full Probability Comparison")
            rows_c = []
            for o in OUTCOMES:
                ep = emp_probs.get(o, 0) * 100
                xp = xgb_probs.get(o, 0) * 100 if xgb_probs else 0.0
                ap = aug_probs.get(o, 0) * 100
                rows_c.append({
                    "Outcome":         OUTCOME_LABELS[o],
                    "Empirical %":     round(ep, 1),
                    "P1 XGBoost %":    round(xp, 1),
                    "P2 Augmented %":  round(ap, 1),
                    "P2 vs P1 diff":   round(ap - xp, 1),
                })
            df_c = pd.DataFrame(rows_c)
            st.dataframe(
                df_c.style.format({
                    "Empirical %":    "{:.1f}",
                    "P1 XGBoost %":   "{:.1f}",
                    "P2 Augmented %": "{:.1f}",
                    "P2 vs P1 diff":  "{:+.1f}",
                })
                .background_gradient(subset=["P2 Augmented %"],
                                     cmap="Purples", vmin=0, vmax=60)
                .applymap(
                    lambda v: "color:#e53935" if isinstance(v, float) and v < 0
                    else "color:#43a047" if isinstance(v, float) and v > 0 else "",
                    subset=["P2 vs P1 diff"],
                ),
                hide_index=True, use_container_width=True,
            )

    # ═══════════════════════════════════ TAB 4 ════════════════════════════════
    with tab4:
        st.subheader("Model Performance Comparison")
        st.caption("Accuracy and Log-Loss on held-out test sets (15% split)")

        # accuracy table
        acc_rows = []
        for m in ["empirical", "xgboost", "rf", "lgbm", "lstm", "augmented"]:
            name, clr = MODEL_META.get(m, (m, ""))
            acc  = MODEL_ACCURACY.get(m)
            ll   = MODEL_LOGLOSS.get(m)
            acc_rows.append({
                "Model":    name,
                "Type":     ("Rule-based" if m == "empirical"
                             else "DL (LSTM)" if m in ("lstm",)
                             else "DL (BiLSTM)" if m == "augmented"
                             else "ML"),
                "Accuracy": f"{acc:.2%}" if acc else "—",
                "Log-Loss": f"{ll:.4f}" if ll else "—",
                "Notes":    ("Frequency-based, no training" if m == "empirical"
                             else "Ball-sequence context, 25K sequences" if m == "lstm"
                             else "+6 commentary sentiment features" if m == "augmented"
                             else "Same 8 Phase-1 features"),
            })
        df_acc = pd.DataFrame(acc_rows)
        st.dataframe(df_acc, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Expected Runs Comparison for Current Selection")
        st.markdown(f"**{batter}** vs **{bowler_label}**")

        er_rows = []
        for m in selected_p1_methods:
            r = results_r.get(m, {})
            er = r.get("expected_runs")
            name, _ = MODEL_META.get(m, (m, ""))
            er_rows.append({"Model": name, "Expected Runs": er if er is not None else float("nan")})
        if aug_r:
            er_rows.append({"Model": "Augmented (P2)", "Expected Runs": aug_r.get("expected_runs")})

        df_er = pd.DataFrame(er_rows).dropna()
        if not df_er.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            _dark(fig2, ax2)
            colors = [MODEL_META.get(
                m, (m, DEFAULT_CLR)
            )[1] for m in selected_p1_methods]
            if aug_r: colors.append(MODEL_META["augmented"][1])
            ax2.bar(df_er["Model"], df_er["Expected Runs"],
                    color=colors[:len(df_er)], alpha=0.9, edgecolor="none")
            for i, (_, row) in enumerate(df_er.iterrows()):
                ax2.text(i, row["Expected Runs"] + 0.02,
                         f"{row['Expected Runs']:.3f}",
                         ha="center", va="bottom", fontsize=9, color="white")
            ax2.set_ylabel("Expected Runs / Ball", color="white")
            ax2.set_title("Expected Runs by Model", color="white", pad=8)
            plt.xticks(rotation=15, color="white")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close(fig2)

        st.markdown("---")
        st.markdown("#### Sentiment Approach Comparison")
        sent_rows = [
            {"Model":        "Keyword Rules",
             "Type":         "Rule-based",
             "Accuracy":     "Ground truth (baseline)",
             "Vocab":        "~80 patterns",
             "Speed":        "Instant",
             "Key advantage":"Cricket-specific, interpretable"},
            {"Model":        "VADER",
             "Type":         "Rule-based NLP",
             "Accuracy":     "18.8% vs keyword rules",
             "Vocab":        "General English lexicon",
             "Speed":        "Instant",
             "Key advantage":"No training, standard NLP"},
            {"Model":        "BiLSTM (DL)",
             "Type":         "Deep Learning",
             "Accuracy":     "96.8% vs keyword rules",
             "Vocab":        "5,000 commentary words",
             "Speed":        "~1s per batch",
             "Key advantage":"Generalises beyond keyword list"},
        ]
        st.dataframe(pd.DataFrame(sent_rows), hide_index=True, use_container_width=True)

    # ═══════════════════════════════════ TAB 5 ════════════════════════════════
    with tab5:
        st.subheader("Leaderboards & Pressure Rankings")
        st.markdown(
            f"Batter filter: **{bat_team_sel}** &nbsp;|&nbsp; "
            f"Bowler filter: **{bowl_team_sel}**"
        )
        st.markdown("---")

        st.markdown("### Phase 1 — Batter vs Bowlers")
        col_lb1, col_lb2 = st.columns(2)

        with col_lb1:
            st.markdown(f"##### Top bowlers vs **{batter}**")
            st.caption("Ranked by wicket probability (min 5 balls)")
            lb_bowl = p1_pred.leaderboard_bowlers(batter, n=10)
            if bowl_team_sel != "All Teams" and not lb_bowl.empty:
                team_set  = team_map.get(bowl_team_sel, set())
                lb_bowl_f = lb_bowl[lb_bowl["bowler"].isin(team_set)]
                if not lb_bowl_f.empty:
                    lb_bowl = lb_bowl_f
            if lb_bowl.empty:
                st.info("Not enough data.")
            else:
                fig = leaderboard_horiz(lb_bowl, "prob_W", "bowler",
                                        f"Wicket Prob vs {batter}", cmap_name="Reds")
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.dataframe(
                    lb_bowl.rename(columns={
                        "bowler": "Bowler", "balls": "Balls",
                        "prob_W": "Wicket Prob", "expected_runs": "Exp Runs",
                        "strike_rate": "SR",
                    }).style.format({
                        "Wicket Prob": "{:.3f}", "Exp Runs": "{:.3f}", "SR": "{:.1f}",
                    }).background_gradient(subset=["Wicket Prob"], cmap="Reds"),
                    hide_index=True, use_container_width=True,
                )

        with col_lb2:
            if bowler:
                st.markdown(f"##### Top batters vs **{bowler}**")
                st.caption("Ranked by expected runs per ball (min 5 balls)")
                lb_bat = p1_pred.leaderboard_batters(bowler, n=10)
                if bat_team_sel != "All Teams" and not lb_bat.empty:
                    team_set = team_map.get(bat_team_sel, set())
                    lb_bat_f = lb_bat[lb_bat["batter"].isin(team_set)]
                    if not lb_bat_f.empty:
                        lb_bat = lb_bat_f
                if lb_bat.empty:
                    st.info("Not enough data.")
                else:
                    fig = leaderboard_horiz(lb_bat, "expected_runs", "batter",
                                            f"Exp Runs vs {bowler}", cmap_name="Blues")
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                    st.dataframe(
                        lb_bat.rename(columns={
                            "batter": "Batter", "balls": "Balls",
                            "expected_runs": "Exp Runs", "strike_rate": "SR",
                            "prob_W": "Wicket Prob",
                        }).style.format({
                            "Exp Runs": "{:.3f}", "SR": "{:.1f}", "Wicket Prob": "{:.3f}",
                        }).background_gradient(subset=["Exp Runs"], cmap="Blues"),
                        hide_index=True, use_container_width=True,
                    )
            else:
                st.info("Select a specific bowler in the sidebar to see the batter leaderboard.")

        st.markdown("---")
        st.markdown("### Phase 2 — Pressure Rankings")
        st.caption("Based on commentary sentiment (2021-2024). Min 20 commentary balls.")

        col_p1, col_p2 = st.columns(2)

        def _filter_pressure_df(df: pd.DataFrame, name_col: str,
                                 team_filter: str) -> pd.DataFrame:
            if team_filter == "All Teams" or df.empty:
                return df
            team_set = team_map.get(team_filter, set())
            filtered = df[df[name_col].isin(team_set)]
            return filtered if not filtered.empty else df

        with col_p1:
            st.markdown("##### Most pressured batters")
            top_bat_p = shot_pred.top_batters_under_pressure(n=10, min_balls=20)
            top_bat_p = _filter_pressure_df(top_bat_p, "batter", bat_team_sel)
            if top_bat_p.empty:
                st.info("Not enough data.")
            else:
                fig = leaderboard_horiz(top_bat_p, "avg_pressure_index", "batter",
                                        "Pressure Index (Batters)", cmap_name="RdYlGn_r")
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.dataframe(
                    top_bat_p[["batter", "avg_pressure_index",
                               "beaten_pct", "dominant_pct", "total_balls"]]
                    .rename(columns={
                        "batter":             "Batter",
                        "avg_pressure_index": "Pressure Idx",
                        "beaten_pct":         "Beaten %",
                        "dominant_pct":       "Dominant %",
                        "total_balls":        "Balls",
                    }).style.format({
                        "Pressure Idx": "{:+.4f}",
                        "Beaten %":     "{:.2%}",
                        "Dominant %":   "{:.2%}",
                    }).background_gradient(subset=["Pressure Idx"], cmap="RdYlGn_r"),
                    hide_index=True, use_container_width=True,
                )

        with col_p2:
            st.markdown("##### Bowlers creating most pressure")
            top_bowl_p = shot_pred.top_bowlers_by_pressure_created(n=10, min_balls=20)
            top_bowl_p = _filter_pressure_df(top_bowl_p, "bowler", bowl_team_sel)
            if top_bowl_p.empty:
                st.info("Not enough data.")
            else:
                fig = leaderboard_horiz(top_bowl_p, "avg_pressure_index", "bowler",
                                        "Pressure Index (Bowlers)", cmap_name="RdYlGn_r")
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.dataframe(
                    top_bowl_p[["bowler", "avg_pressure_index",
                                "beaten_pct", "dominant_pct", "total_balls"]]
                    .rename(columns={
                        "bowler":             "Bowler",
                        "avg_pressure_index": "Pressure Idx",
                        "beaten_pct":         "Beaten %",
                        "dominant_pct":       "Dominant %",
                        "total_balls":        "Balls",
                    }).style.format({
                        "Pressure Idx": "{:+.4f}",
                        "Beaten %":     "{:.2%}",
                        "Dominant %":   "{:.2%}",
                    }).background_gradient(subset=["Pressure Idx"], cmap="RdYlGn_r"),
                    hide_index=True, use_container_width=True,
                )

    # ═══════════════════════════════════ TAB 6 — Phase 3 prototype ════════════
    with tab6:
        st.subheader("Phase 3 — Match narrative prototype (2024 only)")
        st.caption(
            "Portfolio demo: **one match** — ICC T20 World Cup **2024 Final** "
            "(India vs South Africa, `match_id` **1415755**). Compare **every ball’s CricBuzz commentary** in our dataset "
            "with a **simulated highlight transcript** (what you would get from "
            "**video → audio → speech-to-text**)."
        )

        p3 = load_phase3_prototype_bundle()
        if p3.get("error"):
            st.error(f"Could not load Phase 3 prototype: {p3['error']}")
        else:
            meta = p3.get("meta") or {}
            st.info(meta.get("disclaimer", ""))
            if meta.get("video_reference_note"):
                st.caption(meta["video_reference_note"])
            if meta.get("transcript_source") == "asr_file":
                st.success(
                    "Loaded **real ASR transcript** from `artifacts/transcript_1415755_asr.txt` "
                    "(video → audio → Whisper). See `artifacts/asr_run_meta.json` for run details."
                )

            h_btn1, h_btn2 = st.columns(2)
            with h_btn1:
                st.link_button(
                    "Watch match highlights (ICC, new tab)",
                    PHASE3_ICC_MATCH_HIGHLIGHTS_URL,
                    help="Official ICC **Match Highlights** — IND vs SA, T20 World Cup 2024 Final (full package on ICC.tv).",
                    type="primary",
                    use_container_width=True,
                )
            with h_btn2:
                st.link_button(
                    "Watch extended highlights (ICC, new tab)",
                    PHASE3_ICC_EXTENDED_HIGHLIGHTS_URL,
                    help="Official ICC **Extended Highlights** — same final, longer cut.",
                    type="secondary",
                    use_container_width=True,
                )
            st.caption(
                "These links open **ICC’s match / extended highlight** players — not the short "
                "“Epic Montage” YouTube edit. Use them to compare ball-by-ball data with real broadcast-style highlights."
            )
            _asr_src = phase3_asr_pipeline_source_url()
            if _asr_src:
                st.caption(
                    f"**ASR transcript source** (audio you ran through Whisper): `{_asr_src}` — "
                    "re-run `models/phase3/download_and_transcribe.py --url …` if you want the text aligned to a different clip."
                )

            h_cols = st.columns(4)
            h_cols[0].metric("Match ID", str(meta.get("match_id", "—")))
            h_cols[1].metric("Fixture", f"{meta.get('team1', '')} vs {meta.get('team2', '')}")
            h_cols[2].metric("Result", str(meta.get("result_summary", "—"))[:24])
            h_cols[3].metric("Venue", str(meta.get("venue", "—"))[:28])

            ball = p3.get("ball_commentary") or {}
            narr = p3.get("highlight_narrative") or {}
            if not ball:
                st.warning("No ball-by-ball rows found for this match_id in the commentary CSV.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Balls in dataset", f"{ball.get('balls_total', 0):,}")
                m2.metric("Wickets (dataset)", str(ball.get("wickets", "—")))
                m3.metric(
                    "Pressure (balls)",
                    f"{ball.get('pressure_index', 0):+.3f}",
                    help="beaten + mistimed − dominant (keyword rules)",
                )
                m4.metric(
                    "Pressure (narrative)",
                    f"{narr.get('pressure_index', 0):+.3f}",
                    help="Same formula on ASR-style sentences",
                )

                st.markdown("---")
                st.markdown("#### Sentiment mix: micro (every ball) vs macro (highlights)")
                st.markdown(
                    "Both bars use the **same Phase 2 keyword rules** so the comparison is fair. "
                    "Highlights **over-sample** exciting moments (boundaries, wickets, tension), "
                    "so you expect **more DOMINANT / BEATEN / MISTIMED** in the red bars than in the blue."
                )

                pb = ball.get("display_pcts") or {}
                pn = narr.get("display_pcts") or {}
                fig_c = sentiment_compare_bar(
                    pb, pn,
                    f"{meta.get('event', 'Final')} — keyword sentiment profile",
                )
                st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)

                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.markdown("**Ball commentary** (full match)")
                    fd = sentiment_donut(pb)
                    if fd:
                        st.pyplot(fd, use_container_width=True)
                        plt.close(fd)
                    st.caption(
                        f"Avg sentiment score: **{ball.get('avg_sentiment_score', 0):+.3f}** · "
                        f"Labelled balls: **{ball.get('balls_labeled', 0):,}** / {ball.get('balls_total', 0):,}"
                    )
                with c_d2:
                    st.markdown("**Highlight narrative** (prototype transcript)")
                    fd2 = sentiment_donut(pn)
                    if fd2:
                        st.pyplot(fd2, use_container_width=True)
                        plt.close(fd2)
                    st.caption(
                        f"Avg sentiment score: **{narr.get('avg_sentiment_score', 0):+.3f}** · "
                        f"ASR-style sentences: **{narr.get('segment_count', 0)}** · "
                        f"Keyword hits: **{narr.get('labeled_chunks', 0)}**"
                    )

                st.markdown("---")
                with st.expander("Read the prototype transcript (simulated ASR)", expanded=False):
                    st.text(p3.get("transcript_full") or p3.get("transcript_preview", ""))

                st.markdown(
                    "**Takeaway for your report:** Phase 3 text is **complementary** — it reflects "
                    "how a broadcast **story** frames the match, while Phase 2 ball text reflects "
                    "**what actually happened each delivery**. The gap between blue and red bars "
                    "is the whole point of the prototype."
                )


if __name__ == "__main__":
    main()
