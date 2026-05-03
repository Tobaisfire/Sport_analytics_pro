"""
rf_lgbm_predictor.py
--------------------
Random Forest and LightGBM classifiers for ball-outcome prediction.

Both use the same 8 Phase-1 features as XGBoost:
  bat_sr, bat_avg, bat_bpct, bowl_econ, bowl_wktr,
  mu_balls_log, mu_sr, mu_wktr

Returns the same dict schema as MLPredictor so they are
drop-in alternatives in the unified Phase1Predictor.

Artifacts:
  artifacts/rf_model.pkl
  artifacts/lgbm_model.pkl
  artifacts/rf_label_encoder.pkl    (shared LabelEncoder)
  artifacts/lgbm_label_encoder.pkl
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARNING] lightgbm not installed. Run: pip install lightgbm")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

DELIVERIES_PATH = os.path.join(DATASET, "master_deliveries.csv")
BATTING_PATH    = os.path.join(DATASET, "batting_stats.csv")
BOWLING_PATH    = os.path.join(DATASET, "bowling_stats.csv")
MATCHUP_PATH    = os.path.join(ARTIFACTS, "matchup_stats.csv")

RF_MODEL_PATH    = os.path.join(ARTIFACTS, "rf_model.pkl")
LGBM_MODEL_PATH  = os.path.join(ARTIFACTS, "lgbm_model.pkl")
RF_ENC_PATH      = os.path.join(ARTIFACTS, "rf_label_encoder.pkl")
LGBM_ENC_PATH    = os.path.join(ARTIFACTS, "lgbm_label_encoder.pkl")

RUNOUT_KINDS = {"run out", "runout", "obstructing the field"}
OUTCOME_MAP  = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 6: "6", 99: "W"}
FEATURE_NAMES = ["bat_sr", "bat_avg", "bat_bpct",
                 "bowl_econ", "bowl_wktr",
                 "mu_balls_log", "mu_sr", "mu_wktr"]


def _tendency(sr: float) -> str:
    if sr >= 140: return "Aggressive"
    if sr >= 100: return "Balanced"
    return "Defensive"


class _BasePredictor:
    """Shared data-loading and feature-building logic."""

    def __init__(self):
        self.model = None
        self.le    = None
        self._batting_stats:   dict = {}
        self._bowling_stats:   dict = {}
        self._matchup_stats:   dict = {}
        self._global_defaults: dict = {}
        self._loaded = False

    def _load_stats(self, batting_path=BATTING_PATH,
                    bowling_path=BOWLING_PATH,
                    matchup_path=MATCHUP_PATH) -> None:
        bat     = pd.read_csv(batting_path)
        bowl    = pd.read_csv(bowling_path)
        matchup = pd.read_csv(matchup_path)

        for _, row in bat.iterrows():
            name = row.get("player") or row.get("batter") or row.iloc[0]
            bf   = float(row.get("balls_faced", 0) or 0)
            f4   = float(row.get("fours",  0) or 0)
            f6   = float(row.get("sixes",  0) or 0)
            avg  = float(row.get("average",       20)  or 20)
            sr   = float(row.get("strike_rate",   100) or 100)
            bp   = (f4 + f6) / bf if bf > 0 else 0.15
            self._batting_stats[str(name)] = {
                "sr": sr, "average": avg, "boundary_pct": bp
            }

        for _, row in bowl.iterrows():
            name = row.get("player") or row.get("bowler") or row.iloc[0]
            econ = float(row.get("economy",    7.5) or 7.5)
            wkts = float(row.get("wickets",    0)   or 0)
            bb   = float(row.get("balls_bowled", 1) or 1)
            self._bowling_stats[str(name)] = {
                "economy":     econ,
                "wicket_rate": wkts / bb if bb > 0 else 0.04,
            }

        for _, row in matchup.iterrows():
            key = (str(row["batter"]), str(row["bowler"]))
            self._matchup_stats[key] = {
                "balls":       float(row["balls"]),
                "strike_rate": float(row["strike_rate"]),
                "prob_W":      float(row["prob_W"]),
            }

        self._global_defaults = {
            "bat_sr":    float(bat.get("strike_rate", pd.Series([120])).median()),
            "bat_avg":   float(bat.get("average",     pd.Series([20])).median()),
            "bat_bpct":  0.20,
            "bowl_econ": float(bowl.get("economy",    pd.Series([7.5])).median()),
            "bowl_wktr": 0.04,
        }

    def _build_features(self, batter: str,
                        bowler: str | None) -> np.ndarray:
        bd  = self._batting_stats.get(batter, {})
        bld = self._bowling_stats.get(bowler, {}) if bowler else {}
        mu  = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        dfl = self._global_defaults

        return np.array([[
            bd.get("sr",           dfl["bat_sr"]),
            bd.get("average",      dfl["bat_avg"]),
            bd.get("boundary_pct", dfl["bat_bpct"]),
            bld.get("economy",     dfl["bowl_econ"]),
            bld.get("wicket_rate", dfl["bowl_wktr"]),
            np.log1p(mu.get("balls", 0)),
            mu.get("strike_rate",  bd.get("sr", dfl["bat_sr"])),
            mu.get("prob_W",       bld.get("wicket_rate", dfl["bowl_wktr"])),
        ]], dtype=np.float32)

    def _load_deliveries(self, deliveries_path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
        print("  Loading deliveries ...")
        df = pd.read_csv(deliveries_path, low_memory=False)
        df["extras_type"]      = df["extras_type"].fillna("").str.strip().str.lower()
        df = df[df["extras_type"] != "wides"].copy()
        df["wicket_kind_clean"] = df["wicket_kind"].fillna("").str.strip().str.lower()
        df["bowler_wicket"] = (
            (df["is_wicket"] == 1) &
            (~df["wicket_kind_clean"].isin(RUNOUT_KINDS))
        ).astype(int)
        df["outcome_val"]   = df["runs_batter"].clip(upper=6).fillna(0).astype(int)
        df.loc[df["bowler_wicket"] == 1, "outcome_val"] = 99
        df["outcome_label"] = df["outcome_val"].map(OUTCOME_MAP)
        df = df[df["outcome_label"].notna()].copy()
        print(f"  {len(df):,} balls")

        rows = []
        for _, row in df.iterrows():
            batter = str(row["batter"])
            bowler = str(row["bowler"]) if pd.notna(row.get("bowler")) else None
            rows.append(self._build_features(batter, bowler).flatten())

        X  = np.array(rows, dtype=np.float32)
        # Replace any NaN / inf with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = ~np.isfinite(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        le = LabelEncoder()
        y  = le.fit_transform(df["outcome_label"])
        return X, y, le

    def predict(self, batter: str, bowler: str | None = None) -> dict:
        if not self._loaded:
            self.load()

        X         = self._build_features(batter, bowler)
        probs_raw = self.model.predict_proba(X)[0]
        classes   = list(self.le.classes_)
        probs     = {cls: round(float(p), 4) for cls, p in zip(classes, probs_raw)}
        er        = round(sum(int(k) * v for k, v in probs.items() if k != "W"), 4)
        mu        = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        bd        = self._batting_stats.get(batter, {})
        dfl       = self._global_defaults
        sr        = round(float(bd.get("sr", dfl["bat_sr"])), 2)

        return {
            "method":        self.__class__.__name__,
            "source":        "ml_model",
            "balls_sample":  int(mu.get("balls", 0)),
            "probs":         probs,
            "expected_runs": er,
            "strike_rate":   sr,
            "tendency":      _tendency(sr),
        }


# ═══════════════════════════════════════════════════════════════════════════════
class RFPredictor(_BasePredictor):
    """Random Forest classifier for ball-outcome prediction."""

    def train(self,
              deliveries_path=DELIVERIES_PATH,
              batting_path=BATTING_PATH,
              bowling_path=BOWLING_PATH,
              matchup_path=MATCHUP_PATH,
              model_path=RF_MODEL_PATH,
              encoder_path=RF_ENC_PATH,
              test_size=0.15) -> dict:

        print("=== Random Forest ===")
        print("Loading stats ...")
        self._load_stats(batting_path, bowling_path, matchup_path)

        X, y, le = self._load_deliveries(deliveries_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        print("Training Random Forest ...")
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        ll      = log_loss(y_test, y_proba)
        class_names = [str(c) for c in le.classes_]

        print(f"  Accuracy: {acc:.4f}  |  Log-Loss: {ll:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        imp = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
        print("  Feature importances:", {k: round(v, 4) for k, v in
              sorted(imp.items(), key=lambda x: -x[1])})

        os.makedirs(ARTIFACTS, exist_ok=True)
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        print(f"Saved: {model_path}")

        self.model = clf; self.le = le; self._loaded = True
        return {"model": "RandomForest", "accuracy": acc, "log_loss": ll,
                "feature_importances": imp, "classes": class_names}

    def load(self,
             model_path=RF_MODEL_PATH, encoder_path=RF_ENC_PATH,
             batting_path=BATTING_PATH, bowling_path=BOWLING_PATH,
             matchup_path=MATCHUP_PATH) -> "RFPredictor":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RF model not found at {model_path}. Run RFPredictor().train() first."
            )
        self.model = joblib.load(model_path)
        self.le    = joblib.load(encoder_path)
        self._load_stats(batting_path, bowling_path, matchup_path)
        self._loaded = True
        return self


# ═══════════════════════════════════════════════════════════════════════════════
class LGBMPredictor(_BasePredictor):
    """LightGBM classifier for ball-outcome prediction."""

    def train(self,
              deliveries_path=DELIVERIES_PATH,
              batting_path=BATTING_PATH,
              bowling_path=BOWLING_PATH,
              matchup_path=MATCHUP_PATH,
              model_path=LGBM_MODEL_PATH,
              encoder_path=LGBM_ENC_PATH,
              test_size=0.15) -> dict:

        if not LGBM_AVAILABLE:
            raise RuntimeError("Install LightGBM: pip install lightgbm")

        print("=== LightGBM ===")
        print("Loading stats ...")
        self._load_stats(batting_path, bowling_path, matchup_path)

        X, y, le = self._load_deliveries(deliveries_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        print("Training LightGBM ...")
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=len(le.classes_),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
        )

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        ll      = log_loss(y_test, y_proba)
        class_names = [str(c) for c in le.classes_]

        print(f"  Accuracy: {acc:.4f}  |  Log-Loss: {ll:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        imp = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
        print("  Feature importances:", {k: round(v, 4) for k, v in
              sorted(imp.items(), key=lambda x: -x[1])})

        os.makedirs(ARTIFACTS, exist_ok=True)
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
        print(f"Saved: {model_path}")

        self.model = clf; self.le = le; self._loaded = True
        return {"model": "LightGBM", "accuracy": acc, "log_loss": ll,
                "feature_importances": imp, "classes": class_names}

    def load(self,
             model_path=LGBM_MODEL_PATH, encoder_path=LGBM_ENC_PATH,
             batting_path=BATTING_PATH, bowling_path=BOWLING_PATH,
             matchup_path=MATCHUP_PATH) -> "LGBMPredictor":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LGBM model not found at {model_path}. Run LGBMPredictor().train() first."
            )
        self.model = joblib.load(model_path)
        self.le    = joblib.load(encoder_path)
        self._load_stats(batting_path, bowling_path, matchup_path)
        self._loaded = True
        return self


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training RF ...")
    rf = RFPredictor()
    r_rf = rf.train()

    print("\nTraining LGBM ...")
    lgbm = LGBMPredictor()
    r_lg = lgbm.train()

    print("\n=== Comparison ===")
    print(f"  Random Forest:  Acc={r_rf['accuracy']:.4f}  LL={r_rf['log_loss']:.4f}")
    print(f"  LightGBM:       Acc={r_lg['accuracy']:.4f}  LL={r_lg['log_loss']:.4f}")

    print("\nSample predictions (RF):")
    for batter, bowler in [("V Kohli", "A Nortje"), ("RG Sharma", None)]:
        r = rf.predict(batter, bowler)
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        print(f"  {label}: exp_runs={r['expected_runs']}  tendency={r['tendency']}")
