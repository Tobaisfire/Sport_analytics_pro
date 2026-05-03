"""
app.py  —  Phase 1 Cricket Prediction  |  Streamlit UI
------------------------------------------------------
Run from the phase1/ folder:
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
    page_title="Phase 1 — Cricket Outcome Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(BASE_DIR, "artifacts")
MATCHUP_CSV  = os.path.join(ARTIFACTS, "matchup_stats.csv")

OUTCOMES     = ["0", "1", "2", "3", "4", "6", "W"]
OUTCOME_LABELS = {
    "0": "0 runs", "1": "1 run", "2": "2 runs",
    "3": "3 runs", "4": "4 runs", "6": "6 runs", "W": "Wicket",
}

SOURCE_BADGE = {
    "matchup_direct":          ("Direct matchup",          "#2e7d32"),  # green
    "blended":                 ("Blended (30% matchup + 70% overall)", "#e65100"),  # orange
    "batter_overall_fallback": ("Fallback — batter overall (no history)", "#b71c1c"),  # red
    "batter_overall":          ("Batter overall (no bowler selected)",    "#1565c0"),  # blue
    "ml_model":                ("XGBoost ML model",        "#4a148c"),
}

# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_predictor():
    from predictor import Phase1Predictor
    return Phase1Predictor()


@st.cache_data(show_spinner=False)
def load_matchup() -> pd.DataFrame:
    return pd.read_csv(MATCHUP_CSV)


@st.cache_data(show_spinner=False)
def get_batter_list(matchup_df: pd.DataFrame) -> list:
    return sorted(matchup_df["batter"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_bowler_list(matchup_df: pd.DataFrame) -> list:
    return sorted(matchup_df["bowler"].unique().tolist())


# ── helpers ───────────────────────────────────────────────────────────────────

def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.82em;font-weight:600">{text}</span>'
    )


def prob_bar_chart(emp_probs: dict, xgb_probs: dict,
                   actual_probs: dict | None = None) -> plt.Figure:
    outcomes = OUTCOMES
    labels   = [OUTCOME_LABELS[o] for o in outcomes]
    emp_vals = [emp_probs.get(o, 0) * 100 for o in outcomes]
    xgb_vals = [xgb_probs.get(o, 0) * 100 for o in outcomes]

    has_actual = actual_probs is not None
    n_groups   = 3 if has_actual else 2
    x = np.arange(len(outcomes))
    w = 0.28 if has_actual else 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    offset = -(n_groups - 1) * w / 2
    b1 = ax.bar(x + offset,       emp_vals, w, label="Empirical",
                color="#1976D2", alpha=0.9, edgecolor="none")
    b2 = ax.bar(x + offset + w,   xgb_vals, w, label="XGBoost",
                color="#F57C00", alpha=0.9, edgecolor="none")
    if has_actual:
        act_vals = [actual_probs.get(o, 0) * 100 for o in outcomes]
        ax.bar(x + offset + 2 * w, act_vals, w, label="Actual (historical)",
               color="#388E3C", alpha=0.9, edgecolor="none")

    # value labels on bars
    for bars in ([b1, b2] + ([ax.containers[2]] if has_actual else [])):
        for bar in bars:
            h = bar.get_height()
            if h >= 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=7, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(colors="white")
    ax.set_ylabel("Probability (%)", color="white")
    ax.set_title("Outcome Probabilities", color="white", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.yaxis.label.set_color("white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def leaderboard_chart(df: pd.DataFrame, x_col: str,
                      y_col: str, title: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    vals   = df[x_col].values
    labels = df[y_col].astype(str).values
    colors = [color if v == vals.max() else "#90CAF9" for v in vals]

    bars = ax.barh(labels, vals, color=colors, edgecolor="none", alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(val + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlabel(x_col.replace("_", " ").title(), color="white")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.xaxis.label.set_color("white")
    plt.tight_layout()
    return fig


# ── app ───────────────────────────────────────────────────────────────────────

def main():
    # load data + models
    matchup_df  = load_matchup()
    predictor   = load_predictor()
    all_batters = get_batter_list(matchup_df)
    all_bowlers = get_bowler_list(matchup_df)

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏏 Phase 1 Predictor")
        st.markdown("---")

        batter = st.selectbox(
            "Select Batter",
            options=all_batters,
            index=all_batters.index("MN Samuels") if "MN Samuels" in all_batters else 0,
            help="Type to search by name"
        )

        bowler_options = ["(Any / Overall)"] + all_bowlers
        bowler_sel = st.selectbox(
            "Select Bowler  (optional)",
            options=bowler_options,
            index=0,
            help="Leave as 'Any / Overall' to see batter-only stats"
        )
        bowler = None if bowler_sel == "(Any / Overall)" else bowler_sel

        predict_clicked = st.button("Predict", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown(
            "<small>Data: ICC T20 World Cup (all editions)<br>"
            "33,662 balls · 450 batters · 308 bowlers</small>",
            unsafe_allow_html=True
        )

    # ── header ────────────────────────────────────────────────────────────────
    st.title("Phase 1 — Cricket Ball Outcome Predictor")
    st.caption(
        "Predicts the probability of each ball outcome (0, 1, 2, 3, 4, 6, Wicket) "
        "using two approaches: Empirical statistics and XGBoost ML."
    )

    # Run prediction on button click OR first load
    if predict_clicked or "last_batter" not in st.session_state:
        st.session_state.last_batter  = batter
        st.session_state.last_bowler  = bowler
        with st.spinner("Running predictions…"):
            st.session_state.cmp    = predictor.compare(batter, bowler)
            st.session_state.emp_r  = predictor.predict(batter, bowler, method="empirical")
            st.session_state.xgb_r  = predictor.predict(batter, bowler, method="xgboost")
    else:
        # Re-run if selection changed
        if (st.session_state.last_batter != batter or
                st.session_state.last_bowler != bowler):
            st.session_state.last_batter = batter
            st.session_state.last_bowler = bowler
            with st.spinner("Running predictions…"):
                st.session_state.cmp   = predictor.compare(batter, bowler)
                st.session_state.emp_r = predictor.predict(batter, bowler, method="empirical")
                st.session_state.xgb_r = predictor.predict(batter, bowler, method="xgboost")

    cmp   = st.session_state.cmp
    emp_r = st.session_state.emp_r
    xgb_r = st.session_state.xgb_r

    if "error" in cmp:
        st.error(cmp["error"])
        return

    emp_probs = emp_r.get("probs", {})
    xgb_probs = xgb_r.get("probs", {})

    # actual probs from matchup_stats (if pair exists)
    actual_probs = None
    if bowler:
        mask = (matchup_df["batter"] == batter) & (matchup_df["bowler"] == bowler)
        row  = matchup_df[mask]
        if not row.empty:
            r = row.iloc[0]
            actual_probs = {
                "0": float(r["prob_0"]), "1": float(r["prob_1"]),
                "2": float(r["prob_2"]), "3": float(r["prob_3"]),
                "4": float(r["prob_4"]), "6": float(r["prob_6"]),
                "W": float(r["prob_W"]),
            }

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Head-to-Head Prediction",
        "Leaderboard",
        "Explore Stats",
    ])

    # ═══════════════════════════════ TAB 1 ═══════════════════════════════════
    with tab1:
        bowler_label = bowler if bowler else "(any bowler)"
        st.subheader(f"{batter}  vs  {bowler_label}")

        # source badge
        src  = emp_r.get("source", "")
        info = SOURCE_BADGE.get(src, (src, "#555"))
        st.markdown(f"Data source: {badge(info[0], info[1])}", unsafe_allow_html=True)
        st.markdown("")

        # metric cards row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Balls on record",        str(cmp["balls_sample"]))
        col2.metric("Exp. Runs (Empirical)",  f"{cmp['expected_runs']['empirical']:.3f}")
        col3.metric("Exp. Runs (XGBoost)",    f"{cmp['expected_runs']['xgboost']:.3f}")
        col4.metric("Tendency",               emp_r.get("tendency", "—"))

        st.markdown("---")

        # bar chart
        fig = prob_bar_chart(emp_probs, xgb_probs, actual_probs)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")

        # raw probability table
        rows = []
        for o in OUTCOMES:
            ep  = emp_probs.get(o, 0) * 100
            xp  = xgb_probs.get(o, 0) * 100
            act = actual_probs.get(o, None) * 100 if actual_probs else None
            rows.append({
                "Outcome":       OUTCOME_LABELS[o],
                "Empirical %":   round(ep, 1),
                "XGBoost %":     round(xp, 1),
                "Diff (XGB-Emp)": round(xp - ep, 1),
                **({"Actual %": round(act, 1)} if act is not None else {}),
            })
        df_table = pd.DataFrame(rows)

        fmt_dict = {
            "Empirical %":    "{:.1f}",
            "XGBoost %":      "{:.1f}",
            "Diff (XGB-Emp)": "{:+.1f}",
        }
        if actual_probs:
            fmt_dict["Actual %"] = "{:.1f}"

        st.dataframe(
            df_table.style
            .format(fmt_dict)
            .background_gradient(subset=["Empirical %", "XGBoost %"],
                                  cmap="Blues", vmin=0, vmax=50)
            .applymap(lambda v: "color: #e53935" if isinstance(v, float) and v < 0
                      else "color: #43a047" if isinstance(v, float) and v > 0 else "",
                      subset=["Diff (XGB-Emp)"]),
            use_container_width=True,
            hide_index=True,
        )

        # interpretation note
        with st.expander("How to read this"):
            st.markdown("""
**Empirical** — pure historical frequencies: out of all balls this batter faced this bowler,
how many resulted in each outcome?

**XGBoost** — an ML model trained on all 33k+ deliveries, using batter/bowler career stats
and head-to-head features. It generalises beyond the observed sample.

**Actual** — shown only when ≥5 balls of head-to-head history exist in the dataset
(same as Empirical for direct-matchup pairs).

**Source badge:**
- **Direct matchup** — ≥10 balls recorded between this pair (most reliable)
- **Blended** — <10 balls: 30% matchup + 70% batter overall
- **Fallback** — no matchup at all, using batter's career distribution
""")

    # ═══════════════════════════════ TAB 2 ═══════════════════════════════════
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"#### Top 10 bowlers vs **{batter}**")
            st.caption("Ranked by wicket probability (min. 5 balls)")
            lb_bowl = predictor.leaderboard_bowlers(batter, n=10)
            if lb_bowl.empty:
                st.info("Not enough data for this batter.")
            else:
                fig2 = leaderboard_chart(
                    lb_bowl, x_col="prob_W", y_col="bowler",
                    title=f"Wicket Prob. vs {batter}",
                    color="#D32F2F"
                )
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
                st.dataframe(
                    lb_bowl.rename(columns={
                        "bowler": "Bowler", "balls": "Balls",
                        "prob_W": "Wicket Prob", "expected_runs": "Exp Runs",
                        "strike_rate": "SR"
                    }).style.format({
                        "Wicket Prob": "{:.3f}",
                        "Exp Runs":    "{:.3f}",
                        "SR":          "{:.1f}",
                    }).background_gradient(subset=["Wicket Prob"], cmap="Reds"),
                    hide_index=True, use_container_width=True,
                )

        with col_b:
            if bowler:
                st.markdown(f"#### Top 10 batters vs **{bowler}**")
                st.caption("Ranked by expected runs per ball (min. 5 balls)")
                lb_bat = predictor.leaderboard_batters(bowler, n=10)
                if lb_bat.empty:
                    st.info("Not enough data for this bowler.")
                else:
                    fig3 = leaderboard_chart(
                        lb_bat, x_col="expected_runs", y_col="batter",
                        title=f"Exp. Runs vs {bowler}",
                        color="#1565C0"
                    )
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)
                    st.dataframe(
                        lb_bat.rename(columns={
                            "batter": "Batter", "balls": "Balls",
                            "expected_runs": "Exp Runs", "strike_rate": "SR",
                            "prob_W": "Wicket Prob"
                        }).style.format({
                            "Exp Runs":    "{:.3f}",
                            "SR":          "{:.1f}",
                            "Wicket Prob": "{:.3f}",
                        }).background_gradient(subset=["Exp Runs"], cmap="Blues"),
                        hide_index=True, use_container_width=True,
                    )
            else:
                st.info("Select a specific bowler in the sidebar to see the batter leaderboard.")

    # ═══════════════════════════════ TAB 3 ═══════════════════════════════════
    with tab3:
        st.markdown("#### Raw Matchup Data")

        if bowler:
            mask = (matchup_df["batter"] == batter) & (matchup_df["bowler"] == bowler)
            row  = matchup_df[mask]
            if row.empty:
                st.warning(f"No balls recorded between **{batter}** and **{bowler}**.")
            else:
                r = row.iloc[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Balls",   int(r["balls"]))
                c2.metric("Runs Scored",   int(r["runs_scored"]))
                c3.metric("Wickets",       int(r["wickets"]))
                c4.metric("Strike Rate",   f"{r['strike_rate']:.1f}")
                c5.metric("Exp. Runs",     f"{r['expected_runs']:.3f}")

                st.markdown("**Per-outcome breakdown:**")
                breakdown = pd.DataFrame({
                    "Outcome":    ["0", "1", "2", "3", "4", "6", "W"],
                    "Count":      [int(r["cnt_0"]), int(r["cnt_1"]), int(r["cnt_2"]),
                                   int(r["cnt_3"]), int(r["cnt_4"]), int(r["cnt_6"]),
                                   int(r["wickets"])],
                    "Probability":[r["prob_0"], r["prob_1"], r["prob_2"],
                                   r["prob_3"], r["prob_4"], r["prob_6"], r["prob_W"]],
                })
                st.dataframe(
                    breakdown.style.format({"Probability": "{:.3f}"}),
                    hide_index=True, use_container_width=True
                )
        else:
            st.info("Select a bowler in the sidebar to inspect the raw matchup row.")

        st.markdown("---")
        st.markdown(f"#### All matchups for **{batter}**")
        batter_matchups = matchup_df[matchup_df["batter"] == batter].sort_values(
            "balls", ascending=False
        )
        st.caption(f"{len(batter_matchups)} bowlers faced in dataset")
        st.dataframe(
            batter_matchups[[
                "bowler", "balls", "runs_scored", "wickets",
                "expected_runs", "strike_rate", "prob_W"
            ]].rename(columns={
                "bowler": "Bowler", "balls": "Balls",
                "runs_scored": "Runs", "wickets": "Wickets",
                "expected_runs": "Exp Runs", "strike_rate": "SR",
                "prob_W": "Wicket Prob"
            }).style.format({
                "Exp Runs":    "{:.3f}",
                "SR":          "{:.1f}",
                "Wicket Prob": "{:.3f}",
            }).background_gradient(subset=["SR"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
