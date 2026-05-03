"""
matchup_builder.py
------------------
Builds the batter-bowler matchup statistics table from master_deliveries.csv
and saves it to artifacts/matchup_stats.csv.

Run directly:
    python matchup_builder.py

Columns in output:
    batter, bowler, balls, runs_scored,
    cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_6, wickets,
    prob_0, prob_1, prob_2, prob_3, prob_4, prob_6, prob_W,
    expected_runs, strike_rate
"""

import os
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, "..", "..")
DATASET    = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS  = os.path.join(BASE_DIR, "artifacts")

DELIVERIES_PATH = os.path.join(DATASET, "master_deliveries.csv")
OUTPUT_PATH     = os.path.join(ARTIFACTS, "matchup_stats.csv")

# Run-out kinds should not credit the bowler with a wicket
RUNOUT_KINDS = {"run out", "runout", "obstructing the field"}


def build_matchup_table(deliveries_path: str = DELIVERIES_PATH) -> pd.DataFrame:
    """
    Read ball-by-ball data and compute per-(batter, bowler) matchup statistics.
    Returns a DataFrame with one row per unique pair.
    """
    print(f"Loading {deliveries_path} ...")
    df = pd.read_csv(deliveries_path, low_memory=False)
    print(f"  Loaded {len(df):,} delivery rows")

    # Normalise: batter-faced balls exclude wides (no ball faced by batter)
    # extras_type == 'wides' means no ball is faced
    df["extras_type"] = df["extras_type"].fillna("").str.strip().str.lower()
    df_faced = df[df["extras_type"] != "wides"].copy()

    # Outcome column: runs scored by batter (0/1/2/3/4/6)
    df_faced["outcome"] = df_faced["runs_batter"].clip(upper=6).astype(int)

    # Bowler wicket: is_wicket AND wicket is NOT a run-out
    df_faced["wicket_kind_clean"] = df_faced["wicket_kind"].fillna("").str.strip().str.lower()
    df_faced["bowler_wicket"] = (
        (df_faced["is_wicket"] == 1) &
        (~df_faced["wicket_kind_clean"].isin(RUNOUT_KINDS))
    ).astype(int)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    grp = df_faced.groupby(["batter", "bowler"])

    # Basic counts
    agg = grp.agg(
        balls        = ("outcome", "count"),
        runs_scored  = ("runs_batter", "sum"),
        wickets      = ("bowler_wicket", "sum"),
    ).reset_index()

    # Per-outcome counts using pivot for efficiency (no deprecation warning)
    outcome_counts = (
        df_faced.groupby(["batter", "bowler", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for val, label in [(0, "cnt_0"), (1, "cnt_1"), (2, "cnt_2"),
                       (3, "cnt_3"), (4, "cnt_4"), (6, "cnt_6")]:
        col = val
        if col in outcome_counts.columns:
            agg[label] = agg.merge(
                outcome_counts[["batter", "bowler", col]],
                on=["batter", "bowler"], how="left"
            )[col].fillna(0).astype(int).values
        else:
            agg[label] = 0

    # Probabilities
    for val, label in [("cnt_0", "prob_0"), ("cnt_1", "prob_1"),
                       ("cnt_2", "prob_2"), ("cnt_3", "prob_3"),
                       ("cnt_4", "prob_4"), ("cnt_6", "prob_6")]:
        agg[label] = agg[val] / agg["balls"]

    agg["prob_W"] = agg["wickets"] / agg["balls"]

    # Expected runs and strike rate
    agg["expected_runs"] = (
        agg["prob_1"] * 1 +
        agg["prob_2"] * 2 +
        agg["prob_3"] * 3 +
        agg["prob_4"] * 4 +
        agg["prob_6"] * 6
    )
    agg["strike_rate"] = (agg["runs_scored"] / agg["balls"] * 100).round(2)

    # Round probabilities
    prob_cols = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_6", "prob_W"]
    agg[prob_cols] = agg[prob_cols].round(4)
    agg["expected_runs"] = agg["expected_runs"].round(4)

    print(f"  Matchup pairs: {len(agg):,}")
    print(f"  Pairs with >= 10 balls: {(agg['balls'] >= 10).sum():,}")
    return agg


def save(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    matchup_df = build_matchup_table()
    save(matchup_df)
    print("\nSample (top 5 pairs by balls):")
    print(matchup_df.nlargest(5, "balls")[
        ["batter", "bowler", "balls", "expected_runs", "strike_rate", "prob_W"]
    ].to_string(index=False))
