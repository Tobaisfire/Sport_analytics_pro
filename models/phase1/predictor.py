"""
predictor.py
------------
Phase1Predictor — unified interface wrapping all 5 outcome models:
  empirical  — frequency-based probabilities from observed matchup data
  xgboost    — XGBoost classifier (Phase 1 original)
  rf         — Random Forest classifier
  lgbm       — LightGBM classifier
  lstm       — PyTorch LSTM using 6-ball rolling sequence context

Usage:
    from predictor import Phase1Predictor

    p   = Phase1Predictor()
    cmp = p.compare("RG Sharma", "JC Archer")       # all 5 models
    r   = p.predict("RG Sharma", method="lstm")     # single model
    r   = p.predict("RG Sharma", method="rf")
"""

import os
from empirical_predictor import EmpiricalPredictor
from ml_predictor         import MLPredictor

OUTCOME_LABELS = ["0", "1", "2", "3", "4", "6", "W"]

_ML_METHODS = ("xgboost", "rf", "lgbm", "lstm")


class Phase1Predictor:
    """
    Unified Phase 1 prediction interface for all 5 models.

    Parameters
    ----------
    auto_train : bool
        If True (default False), train a missing model on first use.
    """

    def __init__(self, auto_train: bool = False):
        self.empirical    = EmpiricalPredictor()
        self.ml           = MLPredictor()          # XGBoost
        self._rf          = None                   # lazy-loaded
        self._lgbm        = None
        self._lstm        = None

        self._emp_loaded  = False
        self._ml_loaded   = False
        self._rf_loaded   = False
        self._lgbm_loaded = False
        self._lstm_loaded = False
        self._auto_train  = auto_train

    # ── lazy loading helpers ──────────────────────────────────────────────────

    def _ensure_loaded(self, method: str = "both") -> None:
        needs = {method} if method != "both" else {"empirical"} | set(_ML_METHODS)

        if "empirical" in needs and not self._emp_loaded:
            self.empirical.load()
            self._emp_loaded = True

        if "xgboost" in needs and not self._ml_loaded:
            from ml_predictor import MODEL_PATH
            if not os.path.exists(MODEL_PATH):
                if self._auto_train:
                    self.ml.train()
                else:
                    raise FileNotFoundError(
                        f"XGBoost model not found at {MODEL_PATH}. "
                        "Run MLPredictor().train() first."
                    )
            else:
                self.ml.load()
            self._ml_loaded = True

        if "rf" in needs and not self._rf_loaded:
            from rf_lgbm_predictor import RFPredictor, RF_MODEL_PATH
            if not os.path.exists(RF_MODEL_PATH):
                if self._auto_train:
                    self._rf = RFPredictor(); self._rf.train()
                else:
                    raise FileNotFoundError(
                        f"RF model not found at {RF_MODEL_PATH}. "
                        "Run RFPredictor().train() first."
                    )
            else:
                self._rf = RFPredictor(); self._rf.load()
            self._rf_loaded = True

        if "lgbm" in needs and not self._lgbm_loaded:
            from rf_lgbm_predictor import LGBMPredictor, LGBM_MODEL_PATH
            if not os.path.exists(LGBM_MODEL_PATH):
                if self._auto_train:
                    self._lgbm = LGBMPredictor(); self._lgbm.train()
                else:
                    raise FileNotFoundError(
                        f"LGBM model not found at {LGBM_MODEL_PATH}. "
                        "Run LGBMPredictor().train() first."
                    )
            else:
                self._lgbm = LGBMPredictor(); self._lgbm.load()
            self._lgbm_loaded = True

        if "lstm" in needs and not self._lstm_loaded:
            from lstm_predictor import LSTMOutcomePredictor, LSTM_MODEL_PATH
            if not os.path.exists(LSTM_MODEL_PATH):
                if self._auto_train:
                    self._lstm = LSTMOutcomePredictor(); self._lstm.train()
                else:
                    raise FileNotFoundError(
                        f"LSTM model not found at {LSTM_MODEL_PATH}. "
                        "Run LSTMOutcomePredictor().train() first."
                    )
            else:
                self._lstm = LSTMOutcomePredictor(); self._lstm.load()
            self._lstm_loaded = True

    def _get_model(self, method: str):
        """Return the predictor object for a given method string."""
        if method == "empirical": return self.empirical
        if method == "xgboost":  return self.ml
        if method == "rf":       return self._rf
        if method == "lgbm":     return self._lgbm
        if method == "lstm":     return self._lstm
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose from: empirical, xgboost, rf, lgbm, lstm")

    # ── public API ───────────────────────────────────────────────────────────

    def predict(self,
                batter: str,
                bowler: str | None = None,
                method: str = "xgboost",
                recent_sequence: list | None = None) -> dict:
        """
        Predict ball outcome for a (batter, bowler) pair.

        method : "empirical" | "xgboost" | "rf" | "lgbm" | "lstm"

        recent_sequence : optional list of last N outcome ints (for LSTM only).
        """
        self._ensure_loaded(method)
        model = self._get_model(method)
        if method == "lstm" and recent_sequence is not None:
            return model.predict(batter, bowler, recent_sequence)
        return model.predict(batter, bowler)

    def compare(self, batter: str, bowler: str | None = None,
                methods: list | None = None) -> dict:
        """
        Side-by-side comparison of all (or selected) models.

        methods : list of method names to include (default: all 5).

        Returns
        -------
        dict with keys:
            batter, bowler, balls_sample,
            outcomes : list of dicts — one per outcome label,
                       with a key per model showing pct,
            expected_runs : {model_name: value, ...},
            tendency      : {model_name: value, ...}
        """
        all_methods = ["empirical", "xgboost", "rf", "lgbm", "lstm"]
        use_methods = methods if methods else all_methods

        results = {}
        for m in use_methods:
            try:
                self._ensure_loaded(m)
                results[m] = self._get_model(m).predict(batter, bowler)
            except Exception as exc:
                results[m] = {"error": str(exc), "probs": {}, "expected_runs": None}

        emp_result = results.get("empirical", {})
        if "error" in emp_result and all(
            "error" in v for v in results.values()
        ):
            return {"error": "All models failed"}

        outcomes = []
        for lbl in OUTCOME_LABELS:
            row = {"outcome": lbl}
            for m in use_methods:
                p = float(results[m].get("probs", {}).get(lbl, 0))
                row[f"{m}_pct"] = round(p * 100, 2)
            outcomes.append(row)

        return {
            "batter":        batter,
            "bowler":        bowler or "(any)",
            "balls_sample":  emp_result.get("balls_sample", 0),
            "outcomes":      outcomes,
            "expected_runs": {m: results[m].get("expected_runs") for m in use_methods},
            "tendency":      {m: results[m].get("tendency")      for m in use_methods},
            "sources":       {m: results[m].get("source")        for m in use_methods},
        }

    # ── leaderboard helpers ───────────────────────────────────────────────────

    def leaderboard_bowlers(self, batter: str, n: int = 10):
        """Top n bowlers vs this batter by wicket probability (empirical)."""
        self._ensure_loaded("empirical")
        return self.empirical.top_bowlers_vs_batter(batter, n)

    def leaderboard_batters(self, bowler: str, n: int = 10):
        """Top n batters vs this bowler by expected runs (empirical)."""
        self._ensure_loaded("empirical")
        return self.empirical.top_batters_vs_bowler(bowler, n)

    @property
    def all_batters(self) -> list:
        self._ensure_loaded("empirical")
        return self.empirical.all_batters

    @property
    def all_bowlers(self) -> list:
        self._ensure_loaded("empirical")
        return self.empirical.all_bowlers


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    p = Phase1Predictor()

    print("=== 5-Model Comparison ===")
    cmp = p.compare("MN Samuels", "CJ Jordan")
    print(f"Batter: {cmp['batter']}  Bowler: {cmp['bowler']}")
    print("Expected runs per model:")
    for model, er in cmp["expected_runs"].items():
        print(f"  {model:>10}: {er}")

    print("\nLeaderboard — best bowlers vs MN Samuels:")
    print(p.leaderboard_bowlers("MN Samuels").to_string(index=False))
