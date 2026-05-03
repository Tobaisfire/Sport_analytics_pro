"""
app.py  —  Phase 2 Commentary Sentiment & Shot Intelligence  |  Streamlit UI
-----------------------------------------------------------------------------
Run from the phase2/ folder:
    streamlit run app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phase 2 — Shot Intelligence & Sentiment",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(BASE_DIR, "artifacts")
SENTIMENT_CSV  = os.path.join(ARTIFACTS, "sentiment_stats.csv")
SHOT_CSV       = os.path.join(ARTIFACTS, "shot_region_stats.csv")

# Phase 1 paths (for comparison)
P1_DIR       = os.path.join(BASE_DIR, "..", "phase1")
P1_MATCHUP   = os.path.join(P1_DIR, "artifacts", "matchup_stats.csv")

# colour palette (dark-mode friendly)
COLOURS = {
    "drive":       "#1976D2",
    "pull":        "#E64A19",
    "flick":       "#7B1FA2",
    "cut":         "#00796B",
    "sweep":       "#F57C00",
    "slog":        "#C62828",
    "defend":      "#37474F",
    "hook":        "#6D4C41",
    "cover":       "#1565C0",
    "mid-wicket":  "#AD1457",
    "point":       "#2E7D32",
    "long-on":     "#BF360C",
    "square leg":  "#4527A0",
    "long-off":    "#00695C",
    "mid-off":     "#6A1B9A",
    "mid-on":      "#0277BD",
    "fine leg":    "#558B2F",
    "third man":   "#FF8F00",
    "slip":        "#4E342E",
}
DEFAULT_COLOUR  = "#90CAF9"
DOMINANT_COLOUR = "#43A047"
BEATEN_COLOUR   = "#E53935"

OUTCOME_LABELS = {
    "0": "0 runs", "1": "1 run", "2": "2 runs",
    "3": "3 runs", "4": "4 runs", "6": "6 runs", "W": "Wicket",
}
OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]

# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Phase 2 models...")
def load_shot_predictor():
    from shot_predictor import ShotPredictor
    return ShotPredictor().load()


@st.cache_resource(show_spinner="Loading augmented predictor...")
def load_aug_predictor():
    from augmented_predictor import AugmentedPredictor
    aug = AugmentedPredictor()
    try:
        aug.load()
        return aug
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading Phase 1 XGBoost...")
def load_p1_predictor():
    try:
        p1_path = os.path.join(BASE_DIR, "..", "phase1")
        sys.path.insert(0, p1_path)
        from predictor import Phase1Predictor
        return Phase1Predictor()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sentiment_df() -> pd.DataFrame:
    return pd.read_csv(SENTIMENT_CSV)


@st.cache_data(show_spinner=False)
def batter_list() -> list[str]:
    df = pd.read_csv(SENTIMENT_CSV)
    return sorted(df["batter"].unique().tolist())


@st.cache_data(show_spinner=False)
def bowler_list() -> list[str]:
    df = pd.read_csv(SENTIMENT_CSV)
    return sorted(df["bowler"].unique().tolist())

# ── chart helpers ─────────────────────────────────────────────────────────────

def _ax_dark(fig, ax):
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def horizontal_bar_chart(labels: list, values: list, title: str,
                         colours: list | None = None,
                         fmt: str = "{:.1f}%") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.55)))
    _ax_dark(fig, ax)

    if colours is None:
        colours = [DEFAULT_COLOUR] * len(labels)

    bars = ax.barh(labels, values, color=colours, edgecolor="none", alpha=0.92)
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center", color="white", fontsize=8,
        )

    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, pad=8, color="white")
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def sentiment_donut(pcts: dict) -> plt.Figure:
    """Donut chart of sentiment distribution."""
    labels_map = {
        "dominant_pct":  "Dominant",
        "controlled_pct": "Controlled",
        "defensive_pct": "Defensive",
        "mistimed_pct":  "Mistimed",
        "beaten_pct":    "Beaten",
    }
    palette = {
        "Dominant":   "#43A047",
        "Controlled": "#1976D2",
        "Defensive":  "#546E7A",
        "Mistimed":   "#FFA000",
        "Beaten":     "#E53935",
    }
    sizes, clrs, lbls = [], [], []
    for key, label in labels_map.items():
        v = float(pcts.get(key, 0)) * 100
        if v > 0.1:
            sizes.append(v)
            clrs.append(palette[label])
            lbls.append(f"{label}\n{v:.1f}%")

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    wedges, _ = ax.pie(
        sizes, colors=clrs, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="#0e1117"),
    )
    ax.legend(
        wedges, lbls,
        loc="lower center", ncol=2, fontsize=7,
        facecolor="#1e1e2e", labelcolor="white",
        bbox_to_anchor=(0.5, -0.35),
    )
    ax.set_title("Sentiment Distribution", color="white", fontsize=10, pad=10)
    plt.tight_layout()
    return fig


def outcome_comparison_chart(p1_probs: dict, aug_probs: dict) -> plt.Figure:
    outcomes  = OUTCOMES
    labels    = [OUTCOME_LABELS[o] for o in outcomes]
    p1_vals   = [p1_probs.get(o, 0) * 100  for o in outcomes]
    aug_vals  = [aug_probs.get(o, 0) * 100 for o in outcomes]

    x = np.arange(len(outcomes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _ax_dark(fig, ax)

    b1 = ax.bar(x - w / 2, p1_vals,  w, label="Phase 1 XGBoost",
                color="#F57C00", alpha=0.9, edgecolor="none")
    b2 = ax.bar(x + w / 2, aug_vals, w, label="Phase 2 Augmented",
                color="#7B1FA2", alpha=0.9, edgecolor="none")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h >= 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=7, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Probability (%)", color="white")
    ax.set_title("Outcome Probabilities: Phase 1 vs Phase 2", color="white",
                 fontsize=12, pad=10)
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def leaderboard_chart(df: pd.DataFrame, x_col: str, y_col: str,
                      title: str, cmap_col: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.45)))
    _ax_dark(fig, ax)

    vals   = df[x_col].values
    labels = df[y_col].astype(str).values

    # colour gradient: higher value → stronger colour
    norm   = plt.Normalize(vals.min() - 0.01, vals.max() + 0.01)
    cmap   = plt.cm.RdYlGn_r if "pressure" in x_col else plt.cm.Blues
    colours = [cmap(norm(v)) for v in vals]

    bars = ax.barh(labels, vals, color=colours, edgecolor="none", alpha=0.92)
    for bar, val in zip(bars, vals):
        ax.text(
            val + abs(vals.max()) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}" if "pressure" in x_col else f"{val:.3f}",
            va="center", color="white", fontsize=8,
        )

    ax.invert_yaxis()
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.82em;font-weight:600">{text}</span>'
    )


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    sp        = load_shot_predictor()
    aug_pred  = load_aug_predictor()
    p1_pred   = load_p1_predictor()
    sent_df   = load_sentiment_df()
    all_bats  = batter_list()
    all_bowls = bowler_list()

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Phase 2 — Shot Intelligence")
        st.markdown("---")

        default_bat = "JC Buttler" if "JC Buttler" in all_bats else all_bats[0]
        batter = st.selectbox(
            "Select Batter",
            options=all_bats,
            index=all_bats.index(default_bat),
        )

        bowler_options = ["(Any / Overall)"] + all_bowls
        bowler_sel = st.selectbox(
            "Select Bowler  (optional)",
            options=bowler_options,
            index=0,
        )
        bowler = None if bowler_sel == "(Any / Overall)" else bowler_sel

        analyze_clicked = st.button("Analyze", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown(
            "<small>"
            "Data: ICC T20 World Cup 2021, 2022, 2024<br>"
            f"27,871 balls &bull; {len(all_bats)} batters &bull; {len(all_bowls)} bowlers<br>"
            "Commentary coverage: 99.9%<br>"
            "Sentiment tagged: 81.8%"
            "</small>",
            unsafe_allow_html=True,
        )

    # ── header ────────────────────────────────────────────────────────────────
    st.title("Phase 2 — Commentary Sentiment & Shot Intelligence")
    st.caption(
        "Uses ball-by-ball commentary to extract **shot types**, **regions**, and "
        "**sentiment signals** (dominant/beaten/controlled). "
        "Augmented XGBoost adds these signals to improve outcome prediction."
    )

    # ── run analysis on demand or on selection change ─────────────────────────
    need_refresh = (
        analyze_clicked
        or "p2_last_batter" not in st.session_state
        or st.session_state.p2_last_batter != batter
        or st.session_state.p2_last_bowler != bowler
    )
    if need_refresh:
        st.session_state.p2_last_batter = batter
        st.session_state.p2_last_bowler = bowler
        with st.spinner("Analyzing..."):
            st.session_state.p2_shot_result = sp.predict(batter, bowler, top_n=6)
            if aug_pred:
                st.session_state.p2_aug_result = aug_pred.predict(batter, bowler)
            else:
                st.session_state.p2_aug_result = None
            if p1_pred:
                st.session_state.p2_p1_result = p1_pred.predict(batter, bowler, method="xgboost")
            else:
                st.session_state.p2_p1_result = None

    shot_r = st.session_state.p2_shot_result
    aug_r  = st.session_state.p2_aug_result
    p1_r   = st.session_state.p2_p1_result

    if "error" in shot_r:
        st.error(shot_r["error"])
        return

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Shot Intelligence",
        "Augmented Outcome vs Phase 1",
        "Pressure Rankings",
    ])

    # ═══════════════════════════════ TAB 1 ════════════════════════════════════
    with tab1:
        bowler_label = bowler if bowler else "(any bowler)"
        st.subheader(f"{batter}  vs  {bowler_label}")

        src_color = "#2e7d32" if shot_r["source"] == "matchup_direct" else "#e65100"
        src_label = (
            "Direct matchup data" if shot_r["source"] == "matchup_direct"
            else "Batter-overall fallback (no/few pair balls)"
        )
        st.markdown(
            f"Data source: {_badge(src_label, src_color)}",
            unsafe_allow_html=True,
        )
        st.markdown("")

        sent = shot_r.get("sentiment") or {}

        # ── metric cards ──────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Commentary Balls",  shot_r["balls_sample"])
        c2.metric(
            "Avg Sentiment Score",
            f"{sent.get('avg_sentiment_score', 0):+.3f}",
            help="Range -1 (beaten) to +1 (dominant). Positive = batter in control.",
        )
        pi = sent.get("pressure_index", 0)
        c3.metric(
            "Pressure Index",
            f"{pi:+.3f}",
            help="beaten% + mistimed% - dominant%. Higher = batter under more pressure.",
        )
        c4.metric(
            "Dominant %",
            f"{sent.get('dominant_pct', 0)*100:.1f}%",
            help="% of balls tagged as dominant (boundaries, big shots).",
        )
        c5.metric(
            "Beaten %",
            f"{sent.get('beaten_pct', 0)*100:.1f}%",
            help="% of balls where batter was beaten / missed.",
        )

        st.markdown("---")

        # ── shot types + regions charts ───────────────────────────────────────
        col_shot, col_reg, col_donut = st.columns([2, 2, 1.4])

        with col_shot:
            shots = [s for s in shot_r["shot_types"] if s["shot_type"] != "unknown"]
            if shots:
                labels  = [s["shot_type"].title() for s in shots]
                values  = [s["pct"] * 100 for s in shots]
                colours = [COLOURS.get(s["shot_type"], DEFAULT_COLOUR) for s in shots]
                fig = horizontal_bar_chart(
                    labels, values,
                    title=f"Shot Types — {batter}",
                    colours=colours,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No recognised shot type data available.")

        with col_reg:
            regions = [r for r in shot_r["regions"] if r["region"] != "unknown"]
            if regions:
                labels  = [r["region"].title() for r in regions]
                values  = [r["pct"] * 100 for r in regions]
                colours = [COLOURS.get(r["region"], DEFAULT_COLOUR) for r in regions]
                fig = horizontal_bar_chart(
                    labels, values,
                    title=f"Regions — {batter}",
                    colours=colours,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No recognised region data available.")

        with col_donut:
            if sent:
                fig = sentiment_donut(sent)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        # ── full sentiment breakdown table ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Sentiment Breakdown")
        if sent:
            table_data = {
                "Signal":    ["Dominant", "Controlled", "Defensive", "Mistimed", "Beaten",
                              "Boundary Hit", "Dot Ball", "Pressure Index"],
                "Value %":   [
                    f"{sent.get('dominant_pct',   0) * 100:.1f}%",
                    f"{sent.get('controlled_pct', 0) * 100:.1f}%",
                    f"{sent.get('defensive_pct',  0) * 100:.1f}%",
                    f"{sent.get('mistimed_pct',   0) * 100:.1f}%",
                    f"{sent.get('beaten_pct',     0) * 100:.1f}%",
                    f"{sent.get('boundary_pct',   0) * 100:.1f}%",
                    f"{sent.get('dot_pct',        0) * 100:.1f}%",
                    f"{sent.get('pressure_index', 0):+.4f}",
                ],
                "Meaning": [
                    "Big shots (4s, 6s, boundaries) — batter dominates",
                    "Well-timed singles/doubles — batter in control",
                    "Blocked or played out — passive but safe",
                    "Top/leading edges — batter not timing well",
                    "Beaten / missed — bowler winning the contest",
                    "% of actual boundary outcomes in commentary",
                    "% of dot balls in commentary",
                    "beaten% + mistimed% − dominant% (higher = more pressure)",
                ],
            }
            st.dataframe(
                pd.DataFrame(table_data),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("How Phase 2 sentiment is extracted"):
            st.markdown("""
**Commentary parsing** — each ball's commentary text is scanned for keyword patterns:

| Category | Example keywords |
|---|---|
| **Shot type** | *driven*, *pulled*, *swept*, *cut*, *flicked*, *slog*, *defended* |
| **Region** | *cover*, *mid-wicket*, *long-on*, *square leg*, *point* |
| **DOMINANT** | *six, four, boundary, hammered, smashed, maximum* |
| **BEATEN** | *beaten, missed, doesn't connect, beat the bat* |
| **MISTIMED** | *mistimed, top edge, leading edge, skied* |
| **DEFENSIVE** | *defended, blocked, back down the pitch, kept out* |
| **CONTROLLED** | *placed, eased, nudged, guided, worked* |

**Pressure Index** = `beaten% + mistimed% − dominant%`
A positive pressure index means the bowler is winning the battle.
Data source: ICC T20 World Cup 2021, 2022, 2024 (27,871 deliveries, 99.9% commentary coverage).
""")

    # ═══════════════════════════════ TAB 2 ════════════════════════════════════
    with tab2:
        st.subheader("Augmented Outcome Prediction vs Phase 1 XGBoost")

        model_ready = aug_r is not None
        p1_ready    = p1_r is not None

        if not model_ready:
            st.warning(
                "Augmented model not found. "
                "Run `python augmented_predictor.py` from the phase2/ folder first."
            )
        else:
            # metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Phase 1 Exp. Runs",   f"{p1_r['expected_runs']:.3f}"  if p1_ready else "n/a")
            c2.metric("Phase 2 Exp. Runs",   f"{aug_r['expected_runs']:.3f}")
            if p1_ready:
                diff = aug_r["expected_runs"] - p1_r["expected_runs"]
                c3.metric("Difference",      f"{diff:+.3f}",
                          delta=f"{diff:+.3f}", delta_color="normal")
            else:
                c3.metric("Phase 2 Pressure", f"{aug_r['pressure_index']:+.3f}")
            c4.metric("Tendency",             aug_r.get("tendency", "—"))

            # sentiment context sentence
            pi = aug_r.get("pressure_index", 0)
            sc = aug_r.get("avg_sentiment_score", 0)
            dom = aug_r.get("dominant_pct", 0)
            beat = aug_r.get("beaten_pct", 0)
            if sc > 0.3 and dom > 0.05:
                ctx = f"Commentary signals suggest **{batter}** is in dominant form vs {bowler_label}."
            elif pi > 0.05 and beat > 0.05:
                ctx = f"Commentary signals suggest **{batter}** is under pressure vs {bowler_label}."
            else:
                ctx = f"Sentiment profile for **{batter}** vs {bowler_label} is broadly neutral."
            st.info(ctx)

            st.markdown("---")

            # side-by-side comparison chart
            if p1_ready:
                p1_probs  = p1_r.get("probs", {})
                aug_probs = aug_r.get("probs", {})
                fig = outcome_comparison_chart(p1_probs, aug_probs)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # comparison table
                st.markdown("#### Probability Comparison")
                rows = []
                for o in OUTCOMES:
                    p1v  = p1_probs.get(o, 0) * 100
                    augv = aug_probs.get(o, 0) * 100
                    rows.append({
                        "Outcome":       OUTCOME_LABELS[o],
                        "Phase 1 %":     round(p1v, 1),
                        "Phase 2 %":     round(augv, 1),
                        "Diff (P2-P1)":  round(augv - p1v, 1),
                    })
                df_cmp = pd.DataFrame(rows)
                st.dataframe(
                    df_cmp.style
                    .format({
                        "Phase 1 %":    "{:.1f}",
                        "Phase 2 %":    "{:.1f}",
                        "Diff (P2-P1)": "{:+.1f}",
                    })
                    .background_gradient(subset=["Phase 1 %", "Phase 2 %"],
                                         cmap="Purples", vmin=0, vmax=60)
                    .map(
                        lambda v: "color:#e53935" if isinstance(v, float) and v < 0
                        else "color:#43a047" if isinstance(v, float) and v > 0 else "",
                        subset=["Diff (P2-P1)"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                # Phase 1 predictor not loaded — just show Phase 2 probs
                st.markdown("#### Phase 2 Outcome Probabilities")
                aug_probs = aug_r.get("probs", {})
                rows = [{"Outcome": OUTCOME_LABELS[o],
                         "Phase 2 %": round(aug_probs.get(o, 0) * 100, 1)}
                        for o in OUTCOMES]
                st.dataframe(
                    pd.DataFrame(rows).style.format({"Phase 2 %": "{:.1f}"}),
                    hide_index=True, use_container_width=True,
                )

            with st.expander("Why Phase 2 is different from Phase 1"):
                st.markdown("""
**Phase 1** uses only numerical features:
- Batter strike rate, average, boundary%
- Bowler economy, wicket rate
- Head-to-head matchup stats

**Phase 2 (Augmented)** adds 6 commentary-derived features:
- `avg_sentiment_score` — overall tone of how this batter plays vs this bowler
- `pressure_index` — beaten% + mistimed% − dominant%
- `dominant_pct` — how often the batter plays dominant shots
- `beaten_pct` — how often the batter is beaten by the bowler
- `boundary_pct` — actual boundary rate in commentary balls
- `dot_pct` — actual dot ball rate in commentary balls

**Training comparison (Phase 2 test set):**

| Model | Accuracy | Log-Loss |
|---|---|---|
| Baseline (Phase 1 features only) | 43.77% | 1.2581 |
| **Augmented (+ sentiment)** | **48.19%** | **1.2009** |
| Improvement | **+4.43%** | **-0.057** |

`dot_pct` is the **2nd most important** feature (21.67%), showing that
commentary-derived pressure signals materially improve outcome prediction.
""")

    # ═══════════════════════════════ TAB 3 ════════════════════════════════════
    with tab3:
        st.subheader("Pressure Rankings")
        st.caption("Based on aggregated sentiment across all commentary (2021–2024). "
                   "Min 20 commentary balls.")

        col_bat, col_bowl = st.columns(2)

        with col_bat:
            st.markdown("#### Batters Most Under Pressure")
            st.caption("Ranked by avg pressure index (higher = more beaten/mistimed vs bowlers)")
            top_bat = sp.top_batters_under_pressure(n=10, min_balls=20)
            if top_bat.empty:
                st.info("Not enough data.")
            else:
                fig = leaderboard_chart(
                    top_bat, x_col="avg_pressure_index", y_col="batter",
                    title="Avg Pressure Index (Batters)",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(
                    top_bat[["batter", "avg_pressure_index", "beaten_pct",
                              "dominant_pct", "dot_pct", "total_balls"]]
                    .rename(columns={
                        "batter":              "Batter",
                        "avg_pressure_index":  "Pressure Idx",
                        "beaten_pct":          "Beaten %",
                        "dominant_pct":        "Dominant %",
                        "dot_pct":             "Dot %",
                        "total_balls":         "Balls",
                    })
                    .style.format({
                        "Pressure Idx": "{:+.4f}",
                        "Beaten %":     "{:.2%}",
                        "Dominant %":   "{:.2%}",
                        "Dot %":        "{:.2%}",
                    })
                    .background_gradient(subset=["Pressure Idx"], cmap="RdYlGn_r"),
                    hide_index=True, use_container_width=True,
                )

        with col_bowl:
            st.markdown("#### Bowlers Creating Most Pressure")
            st.caption("Ranked by avg pressure index they impose across all batters")
            top_bowl = sp.top_bowlers_by_pressure_created(n=10, min_balls=20)
            if top_bowl.empty:
                st.info("Not enough data.")
            else:
                fig = leaderboard_chart(
                    top_bowl, x_col="avg_pressure_index", y_col="bowler",
                    title="Avg Pressure Index (Bowlers)",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(
                    top_bowl[["bowler", "avg_pressure_index", "beaten_pct",
                               "dominant_pct", "dot_pct", "total_balls"]]
                    .rename(columns={
                        "bowler":              "Bowler",
                        "avg_pressure_index":  "Pressure Idx",
                        "beaten_pct":          "Beaten %",
                        "dominant_pct":        "Dominant %",
                        "dot_pct":             "Dot %",
                        "total_balls":         "Balls",
                    })
                    .style.format({
                        "Pressure Idx": "{:+.4f}",
                        "Beaten %":     "{:.2%}",
                        "Dominant %":   "{:.2%}",
                        "Dot %":        "{:.2%}",
                    })
                    .background_gradient(subset=["Pressure Idx"], cmap="RdYlGn_r"),
                    hide_index=True, use_container_width=True,
                )

        st.markdown("---")
        st.markdown("#### Full Sentiment Stats Table")
        with st.expander("Show all batter-bowler pairs"):
            cols_show = [
                "batter", "bowler", "balls_with_commentary",
                "avg_sentiment_score", "pressure_index",
                "dominant_pct", "beaten_pct", "controlled_pct",
                "boundary_pct", "dot_pct",
            ]
            disp = sent_df[cols_show].sort_values(
                "balls_with_commentary", ascending=False
            )
            st.dataframe(
                disp.style.format({
                    c: "{:.3f}" for c in cols_show
                    if c not in ("batter", "bowler", "balls_with_commentary")
                }).background_gradient(subset=["pressure_index"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )


if __name__ == "__main__":
    main()
