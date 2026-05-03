"""
pressure_builder.py
-------------------
Reads master_deliveries_with_commentary.csv, parses commentary, and aggregates
sentiment + shot/region frequencies into two artifact CSVs:

  artifacts/sentiment_stats.csv    — one row per (batter, bowler)
  artifacts/shot_region_stats.csv  — one row per (batter, bowler, shot_type, region)

Run directly:
    python pressure_builder.py
"""

import os
import sys
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

COMMENTARY_PATH  = os.path.join(DATASET, "master_deliveries_with_commentary.csv")
SENTIMENT_OUT    = os.path.join(ARTIFACTS, "sentiment_stats.csv")
SHOT_REGION_OUT  = os.path.join(ARTIFACTS, "shot_region_stats.csv")

sys.path.insert(0, BASE_DIR)
from commentary_parser import parse_dataframe

SENTIMENT_LABELS = ["DOMINANT", "CONTROLLED", "MISTIMED", "BEATEN", "DEFENSIVE"]


def build(commentary_path: str = COMMENTARY_PATH,
          sentiment_out:   str = SENTIMENT_OUT,
          shot_region_out: str = SHOT_REGION_OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both artifact CSVs from the commentary dataset.

    Returns (sentiment_stats_df, shot_region_stats_df).
    """
    print("Loading commentary data ...")
    df = pd.read_csv(commentary_path, low_memory=False)
    total     = len(df)
    with_comm = int(df["commentary_short"].notna().sum())
    print(f"  {total:,} total rows | {with_comm:,} with commentary "
          f"({with_comm / total * 100:.1f}%)")

    print("Parsing commentary ...")
    df = parse_dataframe(df, text_col="commentary_short")

    df_sent = df[df["sentiment"].notna()].copy()
    print(f"  {len(df_sent):,} rows received a sentiment tag "
          f"({len(df_sent) / with_comm * 100:.1f}% of commentary rows)")

    # ── per-ball helper columns ────────────────────────────────────────────────
    df_sent["is_boundary"] = df_sent["runs_batter"].isin([4, 6]).astype(int)
    df_sent["is_dot"]      = (df_sent["runs_batter"] == 0).astype(int)

    # binary flag per sentiment label
    for label in SENTIMENT_LABELS:
        df_sent[f"is_{label.lower()}"] = (df_sent["sentiment"] == label).astype(int)

    # ── sentiment stats ────────────────────────────────────────────────────────
    print("Aggregating sentiment stats per (batter, bowler) ...")
    grp = df_sent.groupby(["batter", "bowler"])

    sentiment_stats = grp.agg(
        balls_with_commentary = ("sentiment_score", "count"),
        avg_sentiment_score   = ("sentiment_score", "mean"),
        boundary_pct          = ("is_boundary", "mean"),
        dot_pct               = ("is_dot", "mean"),
        dominant_pct          = ("is_dominant", "mean"),
        controlled_pct        = ("is_controlled", "mean"),
        mistimed_pct          = ("is_mistimed", "mean"),
        beaten_pct            = ("is_beaten", "mean"),
        defensive_pct         = ("is_defensive", "mean"),
    ).reset_index()

    # pressure_index: higher → batter is under more pressure vs this bowler
    sentiment_stats["pressure_index"] = (
        sentiment_stats["beaten_pct"] +
        sentiment_stats["mistimed_pct"] -
        sentiment_stats["dominant_pct"]
    )

    float_cols = [c for c in sentiment_stats.columns
                  if c not in ("batter", "bowler", "balls_with_commentary")]
    sentiment_stats[float_cols] = sentiment_stats[float_cols].round(4)

    # ── shot / region stats ────────────────────────────────────────────────────
    print("Aggregating shot/region stats per (batter, bowler, shot_type, region) ...")
    df_shot = df_sent[
        df_sent["shot_type"].notna() | df_sent["region"].notna()
    ].copy()
    df_shot["shot_type"] = df_shot["shot_type"].fillna("unknown")
    df_shot["region"]    = df_shot["region"].fillna("unknown")

    shot_grp = (
        df_shot
        .groupby(["batter", "bowler", "shot_type", "region"])
        .size()
        .reset_index(name="count")
    )

    # add pct relative to commentary balls for each pair
    pair_totals = (
        df_sent
        .groupby(["batter", "bowler"])["sentiment_score"]
        .count()
        .reset_index(name="total_commentary_balls")
    )
    shot_grp = shot_grp.merge(pair_totals, on=["batter", "bowler"], how="left")
    shot_grp["pct_of_balls"] = (
        shot_grp["count"] / shot_grp["total_commentary_balls"]
    ).round(4)
    shot_grp = shot_grp.drop(columns=["total_commentary_balls"])
    shot_grp = shot_grp.sort_values(
        ["batter", "bowler", "count"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS, exist_ok=True)
    sentiment_stats.to_csv(sentiment_out, index=False)
    shot_grp.to_csv(shot_region_out, index=False)

    print(f"\nSaved: {sentiment_out}")
    print(f"  {len(sentiment_stats):,} (batter, bowler) pairs")
    print(f"Saved: {shot_region_out}")
    print(f"  {len(shot_grp):,} (batter, bowler, shot, region) rows")

    # ── summary ───────────────────────────────────────────────────────────────
    sep = "-" * 50
    print(f"\n{sep}")
    print("Sentiment distribution (overall)")
    print(sep)
    total_tagged = len(df_sent)
    for label in SENTIMENT_LABELS:
        cnt = int(df_sent[f"is_{label.lower()}"].sum())
        print(f"  {label:<12} {cnt:>6,}  ({cnt / total_tagged * 100:.1f}%)")

    print(f"\n{sep}")
    print("Top shot types")
    print(sep)
    shot_totals = (
        df_sent[df_sent["shot_type"].notna()]
        .groupby("shot_type").size()
        .sort_values(ascending=False)
    )
    for shot, cnt in shot_totals.head(10).items():
        print(f"  {shot:<12} {cnt:>6,}")

    print(f"\n{sep}")
    print("Top regions")
    print(sep)
    region_totals = (
        df_sent[df_sent["region"].notna()]
        .groupby("region").size()
        .sort_values(ascending=False)
    )
    for region, cnt in region_totals.head(10).items():
        print(f"  {region:<16} {cnt:>6,}")

    print(f"\n{sep}")
    print("Top 10 most-pressured batters (avg pressure_index, min 20 balls)")
    print(sep)
    batter_pressure = (
        sentiment_stats[sentiment_stats["balls_with_commentary"] >= 20]
        .groupby("batter")["pressure_index"]
        .mean()
        .nlargest(10)
    )
    for batter, pi in batter_pressure.items():
        print(f"  {batter:<25} {pi:+.4f}")

    return sentiment_stats, shot_grp


if __name__ == "__main__":
    build()
