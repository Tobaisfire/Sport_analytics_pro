"""
ml_predictor.py
---------------
XGBoost ML prediction model for ball outcomes.

Features (per ball, from master_deliveries + batting/bowling stats):
  - batter overall SR, average, boundary %
  - bowler economy, wicket rate
  - matchup balls (log-scaled), matchup SR, matchup wicket rate

Target: 7-class outcome → 0, 1, 2, 3, 4, 6, W

Usage:
    ml = MLPredictor()
    ml.train()                           # trains and saves artifacts/xgb_model.pkl
    result = ml.predict("RG Sharma", "JC Archer")

    ml2 = MLPredictor()
    result = ml2.predict("RG Sharma")   # loads saved model
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] xgboost not installed. Run: pip install xgboost")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

DELIVERIES_PATH  = os.path.join(DATASET, "master_deliveries.csv")
BATTING_PATH     = os.path.join(DATASET, "batting_stats.csv")
BOWLING_PATH     = os.path.join(DATASET, "bowling_stats.csv")
MATCHUP_PATH     = os.path.join(ARTIFACTS, "matchup_stats.csv")
MODEL_PATH       = os.path.join(ARTIFACTS, "xgb_model.pkl")
ENCODER_PATH     = os.path.join(ARTIFACTS, "label_encoder.pkl")

RUNOUT_KINDS = {"run out", "runout", "obstructing the field"}
OUTCOME_MAP  = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 6: "6", 99: "W"}
OUTCOME_CLASSES = ["0", "1", "2", "3", "4", "6", "W"]


def _tendency(sr: float) -> str:
    if sr >= 140:
        return "Aggressive"
    if sr >= 100:
        return "Balanced"
    return "Defensive"


class MLPredictor:
    """XGBoost classifier for ball-outcome prediction."""

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.le: LabelEncoder | None = None
        self._batting_stats: dict  = {}
        self._bowling_stats: dict  = {}
        self._matchup_stats: dict  = {}
        self._global_defaults: dict = {}
        self._loaded = False

    # ─────────────────────── data loading helpers ────────────────────────────

    def _load_stats(self,
                    batting_path=BATTING_PATH,
                    bowling_path=BOWLING_PATH,
                    matchup_path=MATCHUP_PATH) -> None:
        bat = pd.read_csv(batting_path)
        bowl = pd.read_csv(bowling_path)
        matchup = pd.read_csv(matchup_path)

        # batting stats: keyed by player name
        for _, row in bat.iterrows():
            name = row.get("player") or row.get("batter") or row.iloc[0]
            balls_faced = float(row.get("balls_faced", 0) or 0)
            runs        = float(row.get("runs", 0) or 0)
            fours       = float(row.get("fours", 0) or 0)
            sixes       = float(row.get("sixes", 0) or 0)
            avg         = float(row.get("average", 20) or 20)
            sr          = float(row.get("strike_rate", 100) or 100)
            boundary_pct = (fours + sixes) / balls_faced if balls_faced > 0 else 0.15
            self._batting_stats[str(name)] = {
                "sr": sr,
                "average": avg,
                "boundary_pct": boundary_pct,
            }

        # bowling stats: keyed by player name
        for _, row in bowl.iterrows():
            name = row.get("player") or row.get("bowler") or row.iloc[0]
            econ = float(row.get("economy", 7.5) or 7.5)
            wkts = float(row.get("wickets", 0) or 0)
            balls_bowled = float(row.get("balls_bowled", 1) or 1)
            wicket_rate = wkts / balls_bowled if balls_bowled > 0 else 0.04
            self._bowling_stats[str(name)] = {
                "economy": econ,
                "wicket_rate": wicket_rate,
            }

        # matchup stats: keyed by (batter, bowler)
        for _, row in matchup.iterrows():
            key = (str(row["batter"]), str(row["bowler"]))
            self._matchup_stats[key] = {
                "balls": float(row["balls"]),
                "strike_rate": float(row["strike_rate"]),
                "prob_W": float(row["prob_W"]),
            }

        # global defaults (dataset-wide medians)
        self._global_defaults = {
            "bat_sr": float(bat.get("strike_rate", pd.Series([120])).median()),
            "bat_avg": float(bat.get("average", pd.Series([20])).median()),
            "bat_bpct": 0.20,
            "bowl_econ": float(bowl.get("economy", pd.Series([7.5])).median()),
            "bowl_wktr": 0.04,
        }

    def _build_features(self, batter: str, bowler: str | None) -> np.ndarray:
        """Return a (1, 8) feature vector for a single (batter, bowler) pair."""
        bd = self._batting_stats.get(batter, {})
        bld = self._bowling_stats.get(bowler, {}) if bowler else {}
        mu  = self._matchup_stats.get((batter, bowler), {}) if bowler else {}

        bat_sr     = bd.get("sr",           self._global_defaults["bat_sr"])
        bat_avg    = bd.get("average",      self._global_defaults["bat_avg"])
        bat_bpct   = bd.get("boundary_pct", self._global_defaults["bat_bpct"])
        bowl_econ  = bld.get("economy",     self._global_defaults["bowl_econ"])
        bowl_wktr  = bld.get("wicket_rate", self._global_defaults["bowl_wktr"])
        mu_balls   = np.log1p(mu.get("balls", 0))
        mu_sr      = mu.get("strike_rate",  bat_sr)
        mu_wktr    = mu.get("prob_W",       bowl_wktr)

        return np.array([[bat_sr, bat_avg, bat_bpct,
                          bowl_econ, bowl_wktr,
                          mu_balls, mu_sr, mu_wktr]], dtype=np.float32)

    # ──────────────────────────── training ───────────────────────────────────

    def train(self,
              deliveries_path=DELIVERIES_PATH,
              batting_path=BATTING_PATH,
              bowling_path=BOWLING_PATH,
              matchup_path=MATCHUP_PATH,
              model_path=MODEL_PATH,
              encoder_path=ENCODER_PATH,
              test_size=0.15) -> dict:
        """
        Build feature matrix from all deliveries, train XGBClassifier,
        report accuracy, and save model + encoder to artifacts/.
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost is required. Install with: pip install xgboost")

        print("Loading stats ...")
        self._load_stats(batting_path, bowling_path, matchup_path)

        print("Loading deliveries ...")
        df = pd.read_csv(deliveries_path, low_memory=False)
        df["extras_type"] = df["extras_type"].fillna("").str.strip().str.lower()
        df = df[df["extras_type"] != "wides"].copy()

        df["wicket_kind_clean"] = df["wicket_kind"].fillna("").str.strip().str.lower()
        df["bowler_wicket"] = (
            (df["is_wicket"] == 1) &
            (~df["wicket_kind_clean"].isin(RUNOUT_KINDS))
        ).astype(int)

        # outcome label
        df["outcome_val"] = df["runs_batter"].clip(upper=6).fillna(0).astype(int)
        df.loc[df["bowler_wicket"] == 1, "outcome_val"] = 99
        df["outcome_label"] = df["outcome_val"].map(OUTCOME_MAP)
        # drop any rows that didn't resolve to a known label
        df = df[df["outcome_label"].notna()].copy()

        print(f"  {len(df):,} balls; class distribution:")
        print(df["outcome_label"].value_counts().to_string())

        # Build feature matrix row-by-row
        print("Building features ...")
        rows = []
        for _, row in df.iterrows():
            batter = str(row["batter"])
            bowler = str(row["bowler"]) if pd.notna(row.get("bowler")) else None
            feat = self._build_features(batter, bowler).flatten()
            rows.append(feat)

        X = np.array(rows, dtype=np.float32)
        le = LabelEncoder()
        y = le.fit_transform(df["outcome_label"])
        print(f"  Classes: {list(le.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        print("Training XGBoost ...")
        n_classes = len(le.classes_)
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50)

        y_pred = clf.predict(X_test)
        class_names = [str(c) for c in le.classes_]
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        acc = report.get("accuracy", 0)
        print(f"\n  Test Accuracy : {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Feature importances
        feat_names = ["bat_sr", "bat_avg", "bat_bpct",
                      "bowl_econ", "bowl_wktr",
                      "mu_balls_log", "mu_sr", "mu_wktr"]
        imp = dict(zip(feat_names, clf.feature_importances_.tolist()))
        print("  Feature importances:", {k: round(v, 4) for k, v in imp.items()})

        # Save
        os.makedirs(ARTIFACTS, exist_ok=True)
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        print(f"\nSaved model : {model_path}")
        print(f"Saved encoder: {encoder_path}")

        self.model = clf
        self.le    = le
        self._loaded = True

        return {"accuracy": acc, "feature_importances": imp, "classes": list(le.classes_)}

    # ──────────────────────────── prediction ─────────────────────────────────

    def load(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH,
             batting_path=BATTING_PATH, bowling_path=BOWLING_PATH,
             matchup_path=MATCHUP_PATH) -> "MLPredictor":
        """Load a previously trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run MLPredictor().train() first."
            )
        self.model = joblib.load(model_path)
        self.le    = joblib.load(encoder_path)
        self._load_stats(batting_path, bowling_path, matchup_path)
        self._loaded = True
        return self

    def predict(self, batter: str, bowler: str | None = None) -> dict:
        """
        Predict ball outcomes for (batter, bowler).

        Returns dict with same schema as EmpiricalPredictor.predict().
        """
        if not self._loaded:
            self.load()

        X = self._build_features(batter, bowler)
        probs_raw = self.model.predict_proba(X)[0]
        classes   = list(self.le.classes_)

        probs = {cls: round(float(p), 4) for cls, p in zip(classes, probs_raw)}

        expected_runs = round(
            sum(int(k) * v for k, v in probs.items() if k != "W"), 4
        )

        mu = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        balls_sample = int(mu.get("balls", 0))

        bd = self._batting_stats.get(batter, {})
        sr = round(float(bd.get("sr", self._global_defaults["bat_sr"])), 2)

        return {
            "method": "xgboost",
            "source": "ml_model",
            "balls_sample": balls_sample,
            "probs": probs,
            "expected_runs": expected_runs,
            "strike_rate": sr,
            "tendency": _tendency(sr),
        }

    def feature_importances(self) -> dict:
        """Return feature importance dict (requires model loaded/trained)."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded yet.")
        feat_names = ["bat_sr", "bat_avg", "bat_bpct",
                      "bowl_econ", "bowl_wktr",
                      "mu_balls_log", "mu_sr", "mu_wktr"]
        return dict(zip(feat_names, self.model.feature_importances_.tolist()))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Install xgboost if missing
    import subprocess, sys
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("Installing xgboost ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])

    ml = MLPredictor()
    results = ml.train()
    print(f"\nTraining complete. Accuracy: {results['accuracy']:.4f}")

    print("\nSample predictions:")
    for batter, bowler in [("RG Sharma", "JC Archer"), ("V Kohli", None),
                            ("MN Samuels", "CJ Jordan")]:
        r = ml.predict(batter, bowler)
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        print(f"\n  {label}")
        print(f"    Probs  : {r['probs']}")
        print(f"    ExpRuns: {r['expected_runs']}  Tendency: {r['tendency']}")
