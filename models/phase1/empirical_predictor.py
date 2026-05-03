"""
empirical_predictor.py
----------------------
Option A: Pure statistical prediction from observed ball-by-ball frequencies.

Fallback logic:
  - >= 10 balls in matchup  →  use direct empirical probabilities
  - 1–9 balls in matchup    →  blend 30% matchup + 70% player overall
  - no matchup at all       →  use batter overall stats
"""

import os
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

MATCHUP_PATH = os.path.join(ARTIFACTS, "matchup_stats.csv")
BATTING_PATH = os.path.join(DATASET, "batting_stats.csv")
DELIVERIES_PATH = os.path.join(DATASET, "master_deliveries.csv")

MIN_BALLS_FULL    = 10    # use matchup directly above this
BLEND_WEIGHT_MATCHUP = 0.30   # when < MIN_BALLS_FULL

OUTCOME_LABELS = ["0", "1", "2", "3", "4", "6", "W"]
RUNOUT_KINDS   = {"run out", "runout", "obstructing the field"}


def _tendency(strike_rate: float) -> str:
    if strike_rate >= 140:
        return "Aggressive"
    if strike_rate >= 100:
        return "Balanced"
    return "Defensive"


def _overall_probs_from_batting(bat_row: pd.Series, balls_overall: float) -> dict:
    """
    Derive approximate outcome probabilities from batting_stats aggregate row.
    Uses: runs, balls_faced, fours, sixes, average (for wicket rate).
    """
    if balls_overall <= 0:
        return {k: 1 / len(OUTCOME_LABELS) for k in OUTCOME_LABELS}

    fours  = float(bat_row.get("fours", 0) or 0)
    sixes  = float(bat_row.get("sixes", 0) or 0)
    runs   = float(bat_row.get("runs", 0) or 0)
    balls  = float(bat_row.get("balls_faced", balls_overall) or balls_overall)
    avg    = float(bat_row.get("average", 20) or 20)

    p4 = fours / balls
    p6 = sixes / balls
    # Wicket rate: 1 dismissal per `average` runs, scaled by SR
    p_W = (1 / avg) * (runs / balls) if avg > 0 else 0.04
    p_W = min(p_W, 0.15)

    # 2s and 3s: rough estimates based on residual
    p2 = 0.06
    p3 = 0.01
    # 1s: fill from remaining
    used = p4 + p6 + p_W + p2 + p3
    p1 = max(0.0, 0.25 - (used - 0.25) * 0.3)
    p0 = max(0.0, 1.0 - p4 - p6 - p_W - p2 - p3 - p1)

    total = p0 + p1 + p2 + p3 + p4 + p6 + p_W
    probs = {
        "0": round(p0 / total, 4),
        "1": round(p1 / total, 4),
        "2": round(p2 / total, 4),
        "3": round(p3 / total, 4),
        "4": round(p4 / total, 4),
        "6": round(p6 / total, 4),
        "W": round(p_W / total, 4),
    }
    return probs


class EmpiricalPredictor:
    """
    Predict ball outcomes for a batter (optionally vs a specific bowler)
    using observed frequency distributions from matchup_stats.csv.
    """

    def __init__(self):
        self.matchup_df: pd.DataFrame | None = None
        self.batting_df: pd.DataFrame | None = None
        self._batter_overall: dict = {}   # batter -> overall probs dict
        self._loaded = False

    def load(self,
             matchup_path: str = MATCHUP_PATH,
             batting_path: str = BATTING_PATH,
             deliveries_path: str = DELIVERIES_PATH) -> "EmpiricalPredictor":
        self.matchup_df = pd.read_csv(matchup_path)
        self.batting_df = pd.read_csv(batting_path)
        # Build overall batter probs from deliveries (more accurate than batting_stats)
        self._build_overall_batter_probs(deliveries_path)
        self._loaded = True
        return self

    def _build_overall_batter_probs(self, deliveries_path: str) -> None:
        """Pre-compute per-batter overall outcome probabilities from raw deliveries."""
        df = pd.read_csv(deliveries_path, low_memory=False)
        df["extras_type"] = df["extras_type"].fillna("").str.strip().str.lower()
        df = df[df["extras_type"] != "wides"].copy()
        df["outcome"] = df["runs_batter"].clip(upper=6).astype(int)
        df["wicket_kind_clean"] = df["wicket_kind"].fillna("").str.strip().str.lower()
        df["bowler_wicket"] = (
            (df["is_wicket"] == 1) &
            (~df["wicket_kind_clean"].isin(RUNOUT_KINDS))
        ).astype(int)

        for batter, grp in df.groupby("batter"):
            balls = len(grp)
            if balls == 0:
                continue
            cnts = grp["outcome"].value_counts().to_dict()
            wkts = grp["bowler_wicket"].sum()
            probs = {}
            for v, lbl in [(0,"0"),(1,"1"),(2,"2"),(3,"3"),(4,"4"),(6,"6")]:
                probs[lbl] = round(cnts.get(v, 0) / balls, 4)
            probs["W"] = round(wkts / balls, 4)
            sr = round(grp["runs_batter"].sum() / balls * 100, 2)
            self._batter_overall[batter] = {
                "probs": probs,
                "expected_runs": round(
                    sum(int(k) * v for k, v in probs.items() if k != "W"), 4
                ),
                "strike_rate": sr,
                "balls": balls,
            }

    def _get_matchup_row(self, batter: str, bowler: str) -> pd.Series | None:
        if self.matchup_df is None:
            return None
        mask = (
            (self.matchup_df["batter"] == batter) &
            (self.matchup_df["bowler"] == bowler)
        )
        rows = self.matchup_df[mask]
        return rows.iloc[0] if len(rows) > 0 else None

    def _probs_from_row(self, row: pd.Series) -> dict:
        return {
            "0": round(float(row["prob_0"]), 4),
            "1": round(float(row["prob_1"]), 4),
            "2": round(float(row["prob_2"]), 4),
            "3": round(float(row["prob_3"]), 4),
            "4": round(float(row["prob_4"]), 4),
            "6": round(float(row["prob_6"]), 4),
            "W": round(float(row["prob_W"]), 4),
        }

    def _blend(self, matchup_probs: dict, overall_probs: dict, w_matchup: float) -> dict:
        w_overall = 1 - w_matchup
        blended = {}
        for k in OUTCOME_LABELS:
            blended[k] = round(
                w_matchup * matchup_probs.get(k, 0) +
                w_overall  * overall_probs.get(k, 0),
                4
            )
        return blended

    def predict(self, batter: str, bowler: str | None = None) -> dict:
        """
        Predict ball outcomes for a batter, optionally vs a specific bowler.

        Returns
        -------
        dict with keys:
            method, balls_sample, source, probs, expected_runs,
            strike_rate, tendency
        """
        if not self._loaded:
            self.load()

        overall = self._batter_overall.get(batter)

        # ── No bowler specified: use batter overall ───────────────────────────
        if bowler is None:
            if overall is None:
                return {"error": f"Batter '{batter}' not found in dataset."}
            return {
                "method": "empirical",
                "source": "batter_overall",
                "balls_sample": overall["balls"],
                "probs": overall["probs"],
                "expected_runs": overall["expected_runs"],
                "strike_rate": overall["strike_rate"],
                "tendency": _tendency(overall["strike_rate"]),
            }

        # ── Bowler specified ──────────────────────────────────────────────────
        row = self._get_matchup_row(batter, bowler)

        if row is None:
            # No matchup data at all → fall back to batter overall
            if overall is None:
                return {"error": f"Batter '{batter}' not found in dataset."}
            return {
                "method": "empirical",
                "source": "batter_overall_fallback",
                "balls_sample": overall["balls"],
                "note": f"No balls between {batter} and {bowler} on record.",
                "probs": overall["probs"],
                "expected_runs": overall["expected_runs"],
                "strike_rate": overall["strike_rate"],
                "tendency": _tendency(overall["strike_rate"]),
            }

        balls       = int(row["balls"])
        matchup_probs = self._probs_from_row(row)
        sr_matchup  = float(row["strike_rate"])

        if balls >= MIN_BALLS_FULL:
            # Enough data → use matchup directly
            probs  = matchup_probs
            source = "matchup_direct"
            er     = float(row["expected_runs"])
            sr     = sr_matchup
        else:
            # Too few balls → blend with overall
            if overall:
                probs = self._blend(matchup_probs, overall["probs"], BLEND_WEIGHT_MATCHUP)
                sr    = round(
                    BLEND_WEIGHT_MATCHUP * sr_matchup +
                    (1 - BLEND_WEIGHT_MATCHUP) * overall["strike_rate"],
                    2
                )
            else:
                probs = matchup_probs
                sr    = sr_matchup
            er     = round(sum(int(k) * v for k, v in probs.items() if k != "W"), 4)
            source = "blended"

        return {
            "method": "empirical",
            "source": source,
            "balls_sample": balls,
            "probs": probs,
            "expected_runs": round(er, 4),
            "strike_rate": round(sr, 2),
            "tendency": _tendency(sr),
        }

    def top_bowlers_vs_batter(self, batter: str, n: int = 10) -> pd.DataFrame:
        """Return the n bowlers with highest wicket probability vs this batter."""
        if self.matchup_df is None:
            self.load()
        sub = self.matchup_df[
            (self.matchup_df["batter"] == batter) & (self.matchup_df["balls"] >= 5)
        ].copy()
        return sub.nlargest(n, "prob_W")[
            ["bowler", "balls", "prob_W", "expected_runs", "strike_rate"]
        ].reset_index(drop=True)

    def top_batters_vs_bowler(self, bowler: str, n: int = 10) -> pd.DataFrame:
        """Return the n batters with highest expected runs vs this bowler."""
        if self.matchup_df is None:
            self.load()
        sub = self.matchup_df[
            (self.matchup_df["bowler"] == bowler) & (self.matchup_df["balls"] >= 5)
        ].copy()
        return sub.nlargest(n, "expected_runs")[
            ["batter", "balls", "expected_runs", "strike_rate", "prob_W"]
        ].reset_index(drop=True)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pred = EmpiricalPredictor().load()

    tests = [
        ("Rohit Sharma", "JC Archer"),
        ("Virat Kohli", None),
        ("MN Samuels", "CJ Jordan"),
    ]
    for batter, bowler in tests:
        result = pred.predict(batter, bowler)
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        print(f"\n{label}")
        print(f"  Source  : {result.get('source','?')}")
        print(f"  Balls   : {result.get('balls_sample','?')}")
        print(f"  Probs   : {result.get('probs', {})}")
        print(f"  Exp runs: {result.get('expected_runs','?')}")
        print(f"  SR      : {result.get('strike_rate','?')}")
        print(f"  Tendency: {result.get('tendency','?')}")
