"""
augmented_predictor.py
----------------------
XGBoost outcome predictor augmented with Phase 2 sentiment features.

Feature set (14 total):
  Phase 1 (8): bat_sr, bat_avg, bat_bpct, bowl_econ, bowl_wktr,
               mu_balls_log, mu_sr, mu_wktr
  Phase 2 (6): avg_sentiment_score, pressure_index,
               dominant_pct, beaten_pct, boundary_pct, dot_pct

Training data: master_deliveries_with_commentary.csv (commentary rows only,
               so the training distribution matches where sentiment features exist).

During train() a baseline (Phase-1-only features) is also trained on the same
split, so you can see the exact accuracy gain from adding sentiment features.

Outputs:
  artifacts/aug_xgb_model.pkl
  artifacts/aug_label_encoder.pkl

Usage:
    aug = AugmentedPredictor()
    aug.train()                              # train + save
    result = aug.predict("V Kohli", "A Nortje")

    aug2 = AugmentedPredictor()
    result = aug2.predict("V Kohli", "A Nortje")   # auto-loads saved model
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(BASE_DIR, "..", "..")
DATASET      = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS    = os.path.join(BASE_DIR, "artifacts")
P1_ARTIFACTS = os.path.join(ROOT_DIR, "models", "phase1", "artifacts")

COMMENTARY_PATH = os.path.join(DATASET, "master_deliveries_with_commentary.csv")
BATTING_PATH    = os.path.join(DATASET, "batting_stats.csv")
BOWLING_PATH    = os.path.join(DATASET, "bowling_stats.csv")
P1_MATCHUP_PATH = os.path.join(P1_ARTIFACTS, "matchup_stats.csv")
SENTIMENT_PATH  = os.path.join(ARTIFACTS, "sentiment_stats.csv")
MODEL_PATH      = os.path.join(ARTIFACTS, "aug_xgb_model.pkl")
ENCODER_PATH    = os.path.join(ARTIFACTS, "aug_label_encoder.pkl")

RUNOUT_KINDS  = {"run out", "runout", "obstructing the field"}
OUTCOME_MAP   = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 6: "6", 99: "W"}

# Feature names in exact order used in _build_features()
FEATURE_NAMES = [
    # Phase 1 features
    "bat_sr", "bat_avg", "bat_bpct",
    "bowl_econ", "bowl_wktr",
    "mu_balls_log", "mu_sr", "mu_wktr",
    # Phase 2 sentiment features
    "avg_sentiment_score", "pressure_index",
    "dominant_pct", "beaten_pct",
    "boundary_pct", "dot_pct",
]
N_P1_FEATURES = 8    # first 8 are Phase 1 features (for baseline comparison)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] xgboost not installed. Run: pip install xgboost")


def _tendency(sr: float) -> str:
    if sr >= 140:
        return "Aggressive"
    if sr >= 100:
        return "Balanced"
    return "Defensive"


class AugmentedPredictor:
    """XGBoost outcome predictor using Phase 1 stats + Phase 2 sentiment features."""

    def __init__(self):
        self.model = None
        self.le    = None
        self._batting_stats:   dict = {}
        self._bowling_stats:   dict = {}
        self._matchup_stats:   dict = {}
        self._sentiment_stats: dict = {}
        self._global_defaults: dict = {}
        self._loaded = False

    # ──────────────────────────── data loading ───────────────────────────────

    def _load_stats(self,
                    batting_path=BATTING_PATH,
                    bowling_path=BOWLING_PATH,
                    matchup_path=P1_MATCHUP_PATH,
                    sentiment_path=SENTIMENT_PATH) -> None:

        bat     = pd.read_csv(batting_path)
        bowl    = pd.read_csv(bowling_path)
        matchup = pd.read_csv(matchup_path)

        for _, row in bat.iterrows():
            name   = row.get("player") or row.get("batter") or row.iloc[0]
            bf     = float(row.get("balls_faced", 0) or 0)
            fours  = float(row.get("fours", 0) or 0)
            sixes  = float(row.get("sixes", 0) or 0)
            avg    = float(row.get("average", 20) or 20)
            sr     = float(row.get("strike_rate", 100) or 100)
            bp     = (fours + sixes) / bf if bf > 0 else 0.15
            self._batting_stats[str(name)] = {
                "sr": sr, "average": avg, "boundary_pct": bp
            }

        for _, row in bowl.iterrows():
            name = row.get("player") or row.get("bowler") or row.iloc[0]
            econ = float(row.get("economy", 7.5) or 7.5)
            wkts = float(row.get("wickets", 0) or 0)
            bb   = float(row.get("balls_bowled", 1) or 1)
            self._bowling_stats[str(name)] = {
                "economy":     econ,
                "wicket_rate": wkts / bb if bb > 0 else 0.04,
            }

        for _, row in matchup.iterrows():
            key = (str(row["batter"]), str(row["bowler"]))
            self._matchup_stats[key] = {
                "balls":        float(row["balls"]),
                "strike_rate":  float(row["strike_rate"]),
                "prob_W":       float(row["prob_W"]),
            }

        # global fallback defaults (dataset medians)
        self._global_defaults = {
            "bat_sr":    float(bat.get("strike_rate", pd.Series([120])).median()),
            "bat_avg":   float(bat.get("average",     pd.Series([20])).median()),
            "bat_bpct":  0.20,
            "bowl_econ": float(bowl.get("economy",    pd.Series([7.5])).median()),
            "bowl_wktr": 0.04,
            # sentiment defaults (updated below if file exists)
            "avg_sentiment_score": 0.0,
            "pressure_index":      0.0,
            "dominant_pct":        0.03,
            "beaten_pct":          0.07,
            "boundary_pct":        0.15,
            "dot_pct":             0.35,
        }

        # sentiment stats
        if os.path.exists(sentiment_path):
            sent = pd.read_csv(sentiment_path)
            for _, row in sent.iterrows():
                key = (str(row["batter"]), str(row["bowler"]))
                self._sentiment_stats[key] = {
                    "avg_sentiment_score": float(row.get("avg_sentiment_score", 0) or 0),
                    "pressure_index":      float(row.get("pressure_index",      0) or 0),
                    "dominant_pct":        float(row.get("dominant_pct",        0) or 0),
                    "beaten_pct":          float(row.get("beaten_pct",          0) or 0),
                    "boundary_pct":        float(row.get("boundary_pct",        0) or 0),
                    "dot_pct":             float(row.get("dot_pct",             0) or 0),
                }
            self._global_defaults.update({
                "avg_sentiment_score": float(sent["avg_sentiment_score"].median()),
                "pressure_index":      float(sent["pressure_index"].median()),
                "dominant_pct":        float(sent["dominant_pct"].median()),
                "beaten_pct":          float(sent["beaten_pct"].median()),
                "boundary_pct":        float(sent["boundary_pct"].median()),
                "dot_pct":             float(sent["dot_pct"].median()),
            })
        else:
            print(f"[WARNING] sentiment_stats not found at {sentiment_path}. "
                  "Run pressure_builder.py first.")

    def _build_features(self, batter: str, bowler: str | None) -> np.ndarray:
        """Return a (1, 14) feature vector for a (batter, bowler) pair."""
        bd  = self._batting_stats.get(batter, {})
        bld = self._bowling_stats.get(bowler, {}) if bowler else {}
        mu  = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        sd  = self._sentiment_stats.get((batter, bowler), {}) if bowler else {}
        dfl = self._global_defaults

        return np.array([[
            # Phase 1 features
            bd.get("sr",           dfl["bat_sr"]),
            bd.get("average",      dfl["bat_avg"]),
            bd.get("boundary_pct", dfl["bat_bpct"]),
            bld.get("economy",     dfl["bowl_econ"]),
            bld.get("wicket_rate", dfl["bowl_wktr"]),
            np.log1p(mu.get("balls", 0)),
            mu.get("strike_rate",  bd.get("sr", dfl["bat_sr"])),
            mu.get("prob_W",       bld.get("wicket_rate", dfl["bowl_wktr"])),
            # Phase 2 sentiment features
            sd.get("avg_sentiment_score", dfl["avg_sentiment_score"]),
            sd.get("pressure_index",      dfl["pressure_index"]),
            sd.get("dominant_pct",        dfl["dominant_pct"]),
            sd.get("beaten_pct",          dfl["beaten_pct"]),
            sd.get("boundary_pct",        dfl["boundary_pct"]),
            sd.get("dot_pct",             dfl["dot_pct"]),
        ]], dtype=np.float32)

    # ─────────────────────────────── training ────────────────────────────────

    def train(self,
              commentary_path=COMMENTARY_PATH,
              batting_path=BATTING_PATH,
              bowling_path=BOWLING_PATH,
              matchup_path=P1_MATCHUP_PATH,
              sentiment_path=SENTIMENT_PATH,
              model_path=MODEL_PATH,
              encoder_path=ENCODER_PATH,
              test_size=0.15) -> dict:
        """
        Train augmented XGBoost model and compare vs Phase-1-only baseline.
        Saves aug_xgb_model.pkl and aug_label_encoder.pkl.

        Returns dict with accuracy, log_loss, feature_importances, comparison.
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("Install xgboost: pip install xgboost")

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, log_loss, accuracy_score

        print("Loading stats ...")
        self._load_stats(batting_path, bowling_path, matchup_path, sentiment_path)

        print("Loading commentary deliveries ...")
        df = pd.read_csv(commentary_path, low_memory=False)
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

        print(f"  {len(df):,} balls in training set")
        print("  Class distribution:\n" +
              df["outcome_label"].value_counts().to_string())

        print("Building feature matrix ...")
        rows = []
        for _, row in df.iterrows():
            batter = str(row["batter"])
            bowler = str(row["bowler"]) if pd.notna(row.get("bowler")) else None
            feat   = self._build_features(batter, bowler).flatten()
            rows.append(feat)

        X  = np.array(rows, dtype=np.float32)
        le = LabelEncoder()
        y  = le.fit_transform(df["outcome_label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        n_classes   = len(le.classes_)
        class_names = [str(c) for c in le.classes_]

        def _make_clf(n_feat=None):
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=n_classes,
                tree_method="hist",
                random_state=42,
                verbosity=0,
            )
            if n_feat:
                return clf, X_train[:, :n_feat], X_test[:, :n_feat]
            return clf, X_train, X_test

        # ── Baseline: Phase 1 features only ──────────────────────────────────
        print("\nTraining BASELINE (Phase 1 features only, 8 features) ...")
        base_clf, Xb_train, Xb_test = _make_clf(N_P1_FEATURES)
        base_clf.fit(Xb_train, y_train,
                     eval_set=[(Xb_test, y_test)], verbose=False)
        base_pred  = base_clf.predict(Xb_test)
        base_proba = base_clf.predict_proba(Xb_test)
        base_acc   = accuracy_score(y_test, base_pred)
        base_ll    = log_loss(y_test, base_proba)
        print(f"  Baseline Accuracy : {base_acc:.4f}  |  Log-Loss: {base_ll:.4f}")

        # ── Augmented: all 14 features ────────────────────────────────────────
        print("\nTraining AUGMENTED (Phase 1 + Phase 2 sentiment, 14 features) ...")
        aug_clf, Xa_train, Xa_test = _make_clf()
        aug_clf.fit(Xa_train, y_train,
                    eval_set=[(Xa_test, y_test)], verbose=50)
        aug_pred  = aug_clf.predict(Xa_test)
        aug_proba = aug_clf.predict_proba(Xa_test)
        aug_acc   = accuracy_score(y_test, aug_pred)
        aug_ll    = log_loss(y_test, aug_proba)

        print(f"\n  Augmented Accuracy : {aug_acc:.4f}  |  Log-Loss: {aug_ll:.4f}")
        print(classification_report(y_test, aug_pred, target_names=class_names))

        # ── Comparison table ──────────────────────────────────────────────────
        print("  Model comparison on same test split:")
        print(f"  {'Model':<30} {'Accuracy':>10} {'Log-Loss':>10} {'Acc gain':>10}")
        print("  " + "-" * 60)
        print(f"  {'Baseline (Phase 1 features)':<30} {base_acc:>10.4f} {base_ll:>10.4f} {'(base)':>10}")
        gain_acc = aug_acc - base_acc
        gain_ll  = aug_ll  - base_ll
        print(f"  {'Augmented (+sentiment)':<30} {aug_acc:>10.4f} {aug_ll:>10.4f} {gain_acc:>+10.4f}")
        print(f"\n  Sentiment features shifted accuracy by {gain_acc*100:+.2f}%  "
              f"and log-loss by {gain_ll:+.4f}")

        # ── Feature importances ───────────────────────────────────────────────
        imp = dict(zip(FEATURE_NAMES, aug_clf.feature_importances_.tolist()))
        print("\n  Feature importances (augmented model):")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            bar = "#" * int(v * 200)
            print(f"    {k:<30} {v:.4f}  {bar}")

        # ── Save ──────────────────────────────────────────────────────────────
        os.makedirs(ARTIFACTS, exist_ok=True)
        joblib.dump(aug_clf, model_path)
        joblib.dump(le,      encoder_path)
        print(f"\nSaved: {model_path}")
        print(f"Saved: {encoder_path}")

        self.model   = aug_clf
        self.le      = le
        self._loaded = True

        return {
            "accuracy":   aug_acc,
            "log_loss":   aug_ll,
            "comparison": {
                "baseline_accuracy": base_acc,
                "baseline_log_loss": base_ll,
                "aug_accuracy":      aug_acc,
                "aug_log_loss":      aug_ll,
                "accuracy_gain":     gain_acc,
                "log_loss_delta":    gain_ll,
            },
            "feature_importances": imp,
            "classes": class_names,
        }

    # ─────────────────────────────── inference ───────────────────────────────

    def load(self,
             model_path=MODEL_PATH, encoder_path=ENCODER_PATH,
             batting_path=BATTING_PATH, bowling_path=BOWLING_PATH,
             matchup_path=P1_MATCHUP_PATH,
             sentiment_path=SENTIMENT_PATH) -> "AugmentedPredictor":
        """Load a previously trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run AugmentedPredictor().train() first."
            )
        self.model = joblib.load(model_path)
        self.le    = joblib.load(encoder_path)
        self._load_stats(batting_path, bowling_path, matchup_path, sentiment_path)
        self._loaded = True
        return self

    def predict(self, batter: str, bowler: str | None = None) -> dict:
        """
        Predict ball outcomes for (batter, bowler) using the augmented model.

        Returns dict compatible with Phase 1 predictor schema, plus extra
        sentiment context keys.
        """
        if not self._loaded:
            self.load()

        X         = self._build_features(batter, bowler)
        probs_raw = self.model.predict_proba(X)[0]
        classes   = list(self.le.classes_)
        probs     = {cls: round(float(p), 4) for cls, p in zip(classes, probs_raw)}

        expected_runs = round(
            sum(int(k) * v for k, v in probs.items() if k != "W"), 4
        )

        mu  = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        sd  = self._sentiment_stats.get((batter, bowler), {}) if bowler else {}
        bd  = self._batting_stats.get(batter, {})
        dfl = self._global_defaults
        sr  = round(float(bd.get("sr", dfl["bat_sr"])), 2)

        return {
            "method":       "augmented_xgboost",
            "source":       "aug_ml_model",
            "balls_sample": int(mu.get("balls", 0)),
            "probs":        probs,
            "expected_runs":         expected_runs,
            "strike_rate":           sr,
            "tendency":              _tendency(sr),
            # sentiment context
            "avg_sentiment_score":   round(sd.get("avg_sentiment_score", dfl["avg_sentiment_score"]), 4),
            "pressure_index":        round(sd.get("pressure_index",      dfl["pressure_index"]),      4),
            "dominant_pct":          round(sd.get("dominant_pct",        dfl["dominant_pct"]),        4),
            "beaten_pct":            round(sd.get("beaten_pct",          dfl["beaten_pct"]),          4),
            "boundary_pct":          round(sd.get("boundary_pct",        dfl["boundary_pct"]),        4),
            "dot_pct":               round(sd.get("dot_pct",             dfl["dot_pct"]),             4),
        }

    def feature_importances(self) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded yet.")
        return dict(zip(FEATURE_NAMES, self.model.feature_importances_.tolist()))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    aug = AugmentedPredictor()
    results = aug.train()
    print(f"\nDone. Augmented accuracy: {results['accuracy']:.4f}")

    print("\nSample predictions:")
    for batter, bowler in [
        ("V Kohli",    "A Nortje"),
        ("RG Sharma",  None),
        ("JC Buttler", "Shaheen Shah Afridi"),
    ]:
        r = aug.predict(batter, bowler)
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        print(f"\n  {label}")
        print(f"    Probs      : {r['probs']}")
        print(f"    Exp Runs   : {r['expected_runs']}  SR: {r['strike_rate']}")
        print(f"    Pressure   : {r['pressure_index']}  "
              f"Dominant%: {r['dominant_pct']}  Beaten%: {r['beaten_pct']}")
