"""
shot_predictor.py
-----------------
Frequency-based predictor for shot types, regions, and sentiment for any
batter (optionally vs a specific bowler) using commentary-derived stats.

Fallback logic:
  - >= MIN_BALLS_PAIR commentary balls for the (batter, bowler) pair
      → use pair stats directly
  - < MIN_BALLS_PAIR or no pair data
      → aggregate across ALL bowlers for this batter (batter-overall fallback)
  - no batter data at all
      → return error

Usage:
    sp = ShotPredictor().load()
    result = sp.predict("V Kohli", "A Nortje")
    result = sp.predict("JC Buttler")         # batter-overall
    top    = sp.top_batters_under_pressure()
    top    = sp.top_bowlers_under_pressure()
"""

import os
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

SHOT_REGION_PATH = os.path.join(ARTIFACTS, "shot_region_stats.csv")
SENTIMENT_PATH   = os.path.join(ARTIFACTS, "sentiment_stats.csv")

MIN_BALLS_PAIR = 10   # minimum commentary balls to use direct pair stats


class ShotPredictor:
    """
    Predict shot types and regions for a batter (optionally vs a bowler)
    using aggregated commentary statistics.
    """

    def __init__(self):
        self._shot_df: pd.DataFrame | None = None
        self._sent_df: pd.DataFrame | None = None
        self._loaded  = False

    def load(self,
             shot_region_path: str = SHOT_REGION_PATH,
             sentiment_path:   str = SENTIMENT_PATH) -> "ShotPredictor":
        if not os.path.exists(shot_region_path):
            raise FileNotFoundError(
                f"Shot/region stats not found at {shot_region_path}. "
                "Run pressure_builder.py first."
            )
        self._shot_df = pd.read_csv(shot_region_path)
        if os.path.exists(sentiment_path):
            self._sent_df = pd.read_csv(sentiment_path)
        self._loaded = True
        return self

    # ── internal helpers ──────────────────────────────────────────────────────

    def _shot_rows_for_pair(self, batter: str,
                            bowler: str | None) -> tuple[pd.DataFrame, str]:
        """
        Return (rows, source) where source is 'matchup_direct' or
        'batter_overall_fallback'.
        """
        df = self._shot_df
        if bowler:
            mask = (df["batter"] == batter) & (df["bowler"] == bowler)
            rows = df[mask]
            if rows["count"].sum() >= MIN_BALLS_PAIR:
                return rows, "matchup_direct"

        # batter-level aggregate across all bowlers
        rows = df[df["batter"] == batter]
        return rows, "batter_overall_fallback"

    def _sentiment_for_pair(self, batter: str,
                            bowler: str | None) -> tuple[pd.Series | None, str]:
        """
        Return (sentiment_series, source).
        """
        if self._sent_df is None:
            return None, "no_data"

        df = self._sent_df
        if bowler:
            mask = (df["batter"] == batter) & (df["bowler"] == bowler)
            rows = df[mask]
            if not rows.empty:
                return rows.iloc[0], "matchup_direct"

        # aggregate batter across all bowlers
        rows = df[df["batter"] == batter]
        if rows.empty:
            return None, "no_data"

        agg = pd.Series({
            "avg_sentiment_score": rows["avg_sentiment_score"].mean(),
            "pressure_index":      rows["pressure_index"].mean(),
            "dominant_pct":        rows["dominant_pct"].mean(),
            "controlled_pct":      rows["controlled_pct"].mean(),
            "mistimed_pct":        rows["mistimed_pct"].mean(),
            "beaten_pct":          rows["beaten_pct"].mean(),
            "defensive_pct":       rows["defensive_pct"].mean(),
            "boundary_pct":        rows["boundary_pct"].mean(),
            "dot_pct":             rows["dot_pct"].mean(),
            "balls_with_commentary": rows["balls_with_commentary"].sum(),
        })
        return agg, "batter_overall_fallback"

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, batter: str, bowler: str | None = None,
                top_n: int = 3) -> dict:
        """
        Return top shot types, top regions, and full sentiment profile
        for a batter (optionally vs a specific bowler).

        Returns
        -------
        dict with keys:
          source, balls_sample,
          shot_types   : [{"shot_type", "count", "pct"}, ...]
          regions      : [{"region", "count", "pct"}, ...]
          sentiment    : { avg_sentiment_score, pressure_index,
                           dominant_pct, controlled_pct, mistimed_pct,
                           beaten_pct, defensive_pct, boundary_pct,
                           dot_pct, balls_with_commentary }
        """
        if not self._loaded:
            self.load()

        shot_rows, source = self._shot_rows_for_pair(batter, bowler)

        if shot_rows.empty:
            return {
                "error": f"No commentary data found for batter '{batter}'.",
                "source": "no_data",
            }

        total_balls = int(shot_rows["count"].sum())

        # ── shot types ────────────────────────────────────────────────────────
        shot_agg = (
            shot_rows
            .groupby("shot_type")["count"]
            .sum()
            .reset_index()
        )
        shot_agg["pct"] = (shot_agg["count"] / total_balls).round(4)
        shot_agg = (
            shot_agg
            .sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        # ── regions ───────────────────────────────────────────────────────────
        region_agg = (
            shot_rows
            .groupby("region")["count"]
            .sum()
            .reset_index()
        )
        region_agg["pct"] = (region_agg["count"] / total_balls).round(4)
        region_agg = (
            region_agg
            .sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        # ── sentiment profile ─────────────────────────────────────────────────
        sent_row, _ = self._sentiment_for_pair(batter, bowler)
        sentiment = None
        if sent_row is not None:
            sentiment = {
                "avg_sentiment_score":   round(float(sent_row.get("avg_sentiment_score", 0)),   4),
                "pressure_index":        round(float(sent_row.get("pressure_index",      0)),   4),
                "dominant_pct":          round(float(sent_row.get("dominant_pct",        0)),   4),
                "controlled_pct":        round(float(sent_row.get("controlled_pct",      0)),   4),
                "mistimed_pct":          round(float(sent_row.get("mistimed_pct",        0)),   4),
                "beaten_pct":            round(float(sent_row.get("beaten_pct",          0)),   4),
                "defensive_pct":         round(float(sent_row.get("defensive_pct",       0)),   4),
                "boundary_pct":          round(float(sent_row.get("boundary_pct",        0)),   4),
                "dot_pct":               round(float(sent_row.get("dot_pct",             0)),   4),
                "balls_with_commentary": int(sent_row.get("balls_with_commentary",       0)),
            }

        return {
            "source":       source,
            "balls_sample": total_balls,
            "shot_types":   shot_agg[["shot_type", "count", "pct"]].to_dict("records"),
            "regions":      region_agg[["region", "count", "pct"]].to_dict("records"),
            "sentiment":    sentiment,
        }

    def all_batters(self) -> list[str]:
        """Return sorted list of all batters with commentary data."""
        if not self._loaded:
            self.load()
        return sorted(self._shot_df["batter"].unique().tolist())

    def all_bowlers(self) -> list[str]:
        """Return sorted list of all bowlers with commentary data."""
        if not self._loaded:
            self.load()
        return sorted(self._shot_df["bowler"].unique().tolist())

    def top_batters_under_pressure(self,
                                   n: int = 10,
                                   min_balls: int = 20) -> pd.DataFrame:
        """
        Return the n batters with the highest average pressure_index.
        Higher pressure_index = bowlers dominate them more (more beaten/mistimed,
        fewer dominant shots).
        """
        if not self._loaded:
            self.load()
        if self._sent_df is None:
            return pd.DataFrame()

        df = self._sent_df[self._sent_df["balls_with_commentary"] >= min_balls].copy()
        agg = (
            df.groupby("batter")
            .agg(
                avg_pressure_index  = ("pressure_index",      "mean"),
                avg_sentiment_score = ("avg_sentiment_score", "mean"),
                dominant_pct        = ("dominant_pct",        "mean"),
                beaten_pct          = ("beaten_pct",          "mean"),
                mistimed_pct        = ("mistimed_pct",        "mean"),
                boundary_pct        = ("boundary_pct",        "mean"),
                dot_pct             = ("dot_pct",             "mean"),
                total_balls         = ("balls_with_commentary", "sum"),
            )
            .reset_index()
        )
        float_cols = agg.select_dtypes("float").columns
        agg[float_cols] = agg[float_cols].round(4)
        return agg.nlargest(n, "avg_pressure_index").reset_index(drop=True)

    def top_bowlers_by_pressure_created(self,
                                        n: int = 10,
                                        min_balls: int = 20) -> pd.DataFrame:
        """
        Return the n bowlers who create the most pressure (highest mean
        pressure_index across all batters they face).
        """
        if not self._loaded:
            self.load()
        if self._sent_df is None:
            return pd.DataFrame()

        df = self._sent_df[self._sent_df["balls_with_commentary"] >= min_balls].copy()
        agg = (
            df.groupby("bowler")
            .agg(
                avg_pressure_index  = ("pressure_index",      "mean"),
                avg_sentiment_score = ("avg_sentiment_score", "mean"),
                beaten_pct          = ("beaten_pct",          "mean"),
                dominant_pct        = ("dominant_pct",        "mean"),
                dot_pct             = ("dot_pct",             "mean"),
                total_balls         = ("balls_with_commentary", "sum"),
            )
            .reset_index()
        )
        float_cols = agg.select_dtypes("float").columns
        agg[float_cols] = agg[float_cols].round(4)
        return agg.nlargest(n, "avg_pressure_index").reset_index(drop=True)

    def year_sentiment_trend(self, batter: str) -> pd.DataFrame | None:
        """
        Return per-year sentiment profile for a batter (requires 'year' column
        in the underlying commentary CSV; computed lazily here via sentiment_stats
        if possible — returns None if year info is not available).
        """
        if not self._loaded:
            self.load()
        if self._sent_df is None or "year" not in self._sent_df.columns:
            return None
        rows = self._sent_df[self._sent_df["batter"] == batter]
        if rows.empty:
            return None
        return (
            rows.groupby("year")
            .agg(
                avg_sentiment_score = ("avg_sentiment_score", "mean"),
                pressure_index      = ("pressure_index",      "mean"),
                dominant_pct        = ("dominant_pct",        "mean"),
                beaten_pct          = ("beaten_pct",          "mean"),
                balls               = ("balls_with_commentary", "sum"),
            )
            .reset_index()
            .round(4)
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sp = ShotPredictor().load()
    print(f"Batters: {len(sp.all_batters())}   Bowlers: {len(sp.all_bowlers())}")

    for batter, bowler in [
        ("V Kohli",    "A Nortje"),
        ("JC Buttler", None),
        ("RG Sharma",  "Shaheen Shah Afridi"),
    ]:
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        r = sp.predict(batter, bowler)
        if "error" in r:
            print(f"\n  {label}: {r['error']}")
            continue
        print(f"\n{label}  [source: {r['source']}  |  balls: {r['balls_sample']}]")
        print("  Shot types:")
        for s in r["shot_types"]:
            print(f"    {s['shot_type']:<12} {s['count']:>4}  ({s['pct']*100:.1f}%)")
        print("  Regions:")
        for reg in r["regions"]:
            print(f"    {reg['region']:<16} {reg['count']:>4}  ({reg['pct']*100:.1f}%)")
        if r["sentiment"]:
            sent = r["sentiment"]
            print(f"  Sentiment — pressure_index: {sent['pressure_index']:+.4f}  "
                  f"dominant: {sent['dominant_pct']*100:.1f}%  "
                  f"beaten: {sent['beaten_pct']*100:.1f}%  "
                  f"controlled: {sent['controlled_pct']*100:.1f}%")

    print("\nTop 5 batters under pressure:")
    print(sp.top_batters_under_pressure(n=5).to_string(index=False))

    print("\nTop 5 pressure-creating bowlers:")
    print(sp.top_bowlers_by_pressure_created(n=5).to_string(index=False))
