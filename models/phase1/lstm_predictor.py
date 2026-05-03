"""
lstm_predictor.py
-----------------
LSTM sequence model for ball-outcome prediction (PyTorch).

Architecture:
  Input A — sequence of last SEQ_LEN balls in the inning
             shape (SEQ_LEN, 3): [outcome_normalised, over/19, ball_in_over/5]
  Input B — static batter/bowler features
             shape (8,): same 8 features as XGBoost

  LSTM(input=3, hidden=64)(A)
    → last hidden state (64,) + Dropout(0.3)
    → Concatenate with B  → (72,)
    → Linear(72 → 64) + ReLU + Dropout(0.2)
    → Linear(64 → 7)  → LogSoftmax

Training:
  Builds rolling windows from master_deliveries_with_commentary.csv
  (over + ball columns needed; this file has them).
  Each sample = (last SEQ_LEN balls, static features) → next outcome.

At inference:
  If no recent_sequence supplied, uses zero-padded sequence
  (neutral game context), making predictions comparable to other models.

Artifacts:
  artifacts/lstm_model.pt
  artifacts/lstm_label_encoder.pkl
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

# use the commentary CSV — it has over + ball_in_over columns
DELIVERIES_PATH = os.path.join(DATASET, "master_deliveries_with_commentary.csv")
BATTING_PATH    = os.path.join(DATASET, "batting_stats.csv")
BOWLING_PATH    = os.path.join(DATASET, "bowling_stats.csv")
MATCHUP_PATH    = os.path.join(ARTIFACTS, "matchup_stats.csv")

LSTM_MODEL_PATH = os.path.join(ARTIFACTS, "lstm_model.pt")
LSTM_ENC_PATH   = os.path.join(ARTIFACTS, "lstm_label_encoder.pkl")

RUNOUT_KINDS = {"run out", "runout", "obstructing the field"}
OUTCOME_MAP  = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 6: "6", 99: "W"}

SEQ_LEN = 6       # number of past balls used as context


def _tendency(sr: float) -> str:
    if sr >= 140: return "Aggressive"
    if sr >= 100: return "Balanced"
    return "Defensive"


class LSTMOutcomePredictor:
    """
    LSTM that uses the rolling window of the last SEQ_LEN ball outcomes
    (+ over context) alongside static batter/bowler features.
    """

    def __init__(self):
        self.model = None
        self.le    = None
        self._batting_stats:   dict = {}
        self._bowling_stats:   dict = {}
        self._matchup_stats:   dict = {}
        self._global_defaults: dict = {}
        self._loaded = False

    # ── stats loading ─────────────────────────────────────────────────────────

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

    def _static_features(self, batter: str, bowler: str | None) -> np.ndarray:
        bd  = self._batting_stats.get(batter, {})
        bld = self._bowling_stats.get(bowler, {}) if bowler else {}
        mu  = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        dfl = self._global_defaults
        return np.array([
            bd.get("sr",           dfl["bat_sr"]),
            bd.get("average",      dfl["bat_avg"]),
            bd.get("boundary_pct", dfl["bat_bpct"]),
            bld.get("economy",     dfl["bowl_econ"]),
            bld.get("wicket_rate", dfl["bowl_wktr"]),
            np.log1p(mu.get("balls", 0)),
            mu.get("strike_rate",  bd.get("sr", dfl["bat_sr"])),
            mu.get("prob_W",       bld.get("wicket_rate", dfl["bowl_wktr"])),
        ], dtype=np.float32)

    # ── sequence builder ──────────────────────────────────────────────────────

    def _build_sequences(self, df: pd.DataFrame,
                         outcome_int: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (X_seq, X_static, y) from the sorted deliveries DataFrame.

        X_seq  : (N, SEQ_LEN, 3)  — [outcome/6, over/19, ball/5]
        X_static: (N, 8)
        y      : (N,)
        """
        seq_list, static_list, y_list = [], [], []

        for (match_id, inning_no), group in df.groupby(
                ["match_id", "inning_no"], sort=False):
            group = group.reset_index(drop=True)
            outcomes  = outcome_int[group.index]
            overs     = group["over"].values
            balls     = group["ball_in_over"].values
            batters   = group["batter"].astype(str).values
            bowlers   = group["bowler"].astype(str).values

            for i in range(SEQ_LEN, len(group)):
                # sequence of last SEQ_LEN balls
                window = np.zeros((SEQ_LEN, 3), dtype=np.float32)
                for j, idx in enumerate(range(i - SEQ_LEN, i)):
                    oc   = float(outcomes[idx]) / 6.0        # normalise 0-1
                    ov   = float(overs[idx]) / 19.0
                    bl   = float(balls[idx]) / 5.0
                    window[j] = [oc, ov, bl]

                static = self._static_features(batters[i], bowlers[i])
                seq_list.append(window)
                static_list.append(static)
                y_list.append(outcomes[i])

        return (np.array(seq_list,  dtype=np.float32),
                np.array(static_list, dtype=np.float32),
                np.array(y_list,    dtype=np.int32))

    # ── PyTorch model definition ──────────────────────────────────────────────

    def _build_torch_model(self, n_classes: int):
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, seq_feat=3, static_feat=8,
                         hidden=64, n_cls=7):
                super().__init__()
                self.lstm    = nn.LSTM(seq_feat, hidden,
                                       batch_first=True)
                self.drop1   = nn.Dropout(0.3)
                self.fc1     = nn.Linear(hidden + static_feat, 64)
                self.relu    = nn.ReLU()
                self.drop2   = nn.Dropout(0.2)
                self.fc2     = nn.Linear(64, n_cls)

            def forward(self, seq, static):
                _, (h, _) = self.lstm(seq)
                h = h.squeeze(0)           # (batch, hidden)
                h = self.drop1(h)
                x = torch.cat([h, static], dim=1)
                x = self.relu(self.fc1(x))
                x = self.drop2(x)
                return self.fc2(x)         # raw logits

        return LSTMNet(n_cls=n_classes)

    # ── training ──────────────────────────────────────────────────────────────

    def train(self,
              deliveries_path=DELIVERIES_PATH,
              batting_path=BATTING_PATH,
              bowling_path=BOWLING_PATH,
              matchup_path=MATCHUP_PATH,
              model_path=LSTM_MODEL_PATH,
              encoder_path=LSTM_ENC_PATH,
              epochs=20,
              batch_size=512,
              test_size=0.15) -> dict:

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, log_loss, classification_report

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== LSTM Ball-Sequence Model (PyTorch, device={device}) ===")
        print("Loading stats ...")
        self._load_stats(batting_path, bowling_path, matchup_path)

        print("Loading deliveries ...")
        df = pd.read_csv(deliveries_path, low_memory=False)
        df["extras_type"]       = df["extras_type"].fillna("").str.strip().str.lower()
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

        le = LabelEncoder()
        df["outcome_enc"] = le.fit_transform(df["outcome_label"])
        df = df.sort_values(
            ["match_id", "inning_no", "over", "ball_in_over"]
        ).reset_index(drop=True)
        outcome_int = df["outcome_enc"].values

        print(f"  {len(df):,} balls across {df['match_id'].nunique()} matches")
        print("Building rolling sequences ...")
        X_seq, X_static, y = self._build_sequences(df, outcome_int)
        print(f"  {len(X_seq):,} training samples (SEQ_LEN={SEQ_LEN})")

        col_med  = np.nanmedian(X_static, axis=0)
        nan_mask = ~np.isfinite(X_static)
        X_static[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

        idx = np.arange(len(y))
        tr_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=42, stratify=y
        )
        Xseq_tr  = torch.tensor(X_seq[tr_idx],    dtype=torch.float32)
        Xstat_tr = torch.tensor(X_static[tr_idx], dtype=torch.float32)
        y_tr_t   = torch.tensor(y[tr_idx],        dtype=torch.long)
        Xseq_te  = torch.tensor(X_seq[te_idx],    dtype=torch.float32)
        Xstat_te = torch.tensor(X_static[te_idx], dtype=torch.float32)
        y_te     = y[te_idx]

        print(f"  Train: {len(y_tr_t):,}  |  Test: {len(y_te):,}")

        n_classes = len(le.classes_)
        model     = self._build_torch_model(n_classes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=2, factor=0.5, verbose=False
        )

        train_ds  = TensorDataset(Xseq_tr, Xstat_tr, y_tr_t)
        loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state    = None
        patience_cnt  = 0
        PATIENCE      = 4

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for seq_b, stat_b, y_b in loader:
                seq_b  = seq_b.to(device)
                stat_b = stat_b.to(device)
                y_b    = y_b.to(device)
                optimiser.zero_grad()
                logits = model(seq_b, stat_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(y_b)

            avg_loss = total_loss / len(y_tr_t)

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(Xseq_te.to(device), Xstat_te.to(device))
                val_loss   = criterion(val_logits, torch.tensor(y_te, dtype=torch.long).to(device)).item()
            scheduler.step(val_loss)

            print(f"  Epoch {epoch:>2}/{epochs}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        # evaluate
        model.eval()
        with torch.no_grad():
            logits   = model(Xseq_te.to(device), Xstat_te.to(device)).cpu()
            y_proba  = torch.softmax(logits, dim=1).numpy()
            y_pred   = np.argmax(y_proba, axis=1)

        acc = accuracy_score(y_te, y_pred)
        ll  = log_loss(y_te, y_proba, labels=list(range(n_classes)))
        class_names = [str(c) for c in le.classes_]
        print(f"\n  LSTM Accuracy: {acc:.4f}  |  Log-Loss: {ll:.4f}")
        present_labels = sorted(set(y_te))
        present_names  = [class_names[i] for i in present_labels]
        print(classification_report(y_te, y_pred,
                                    labels=present_labels,
                                    target_names=present_names))

        os.makedirs(ARTIFACTS, exist_ok=True)
        torch.save({"model_state": best_state or model.state_dict(),
                    "n_classes":   n_classes}, model_path)
        joblib.dump(le, encoder_path)
        print(f"Saved: {model_path}")
        print(f"Saved: {encoder_path}")

        self.model   = model
        self.le      = le
        self._loaded = True
        return {"model": "LSTM", "accuracy": acc,
                "log_loss": ll, "classes": class_names}

    # ── inference ─────────────────────────────────────────────────────────────

    def load(self,
             model_path=LSTM_MODEL_PATH, encoder_path=LSTM_ENC_PATH,
             batting_path=BATTING_PATH, bowling_path=BOWLING_PATH,
             matchup_path=MATCHUP_PATH) -> "LSTMOutcomePredictor":
        import torch

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LSTM model not found at {model_path}. "
                "Run LSTMOutcomePredictor().train() first."
            )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        n_classes  = checkpoint["n_classes"]
        net        = self._build_torch_model(n_classes)
        net.load_state_dict(checkpoint["model_state"])
        net.eval()
        self.model   = net
        self.le      = joblib.load(encoder_path)
        self._load_stats(batting_path, bowling_path, matchup_path)
        self._loaded = True
        return self

    def predict(self, batter: str, bowler: str | None = None,
                recent_sequence: list | None = None) -> dict:
        """
        Predict next ball outcome.

        recent_sequence : list of ints (up to SEQ_LEN) — last ball outcomes
                          (runs 0-6 or 99 for wicket).  None = neutral context.
        """
        import torch

        if not self._loaded:
            self.load()

        seq = np.zeros((1, SEQ_LEN, 3), dtype=np.float32)
        if recent_sequence:
            trimmed = recent_sequence[-SEQ_LEN:]
            for j, val in enumerate(trimmed):
                seq[0, SEQ_LEN - len(trimmed) + j, 0] = float(val) / 6.0

        static = self._static_features(batter, bowler).reshape(1, -1)
        static[~np.isfinite(static)] = 0.0

        with torch.no_grad():
            logits    = self.model(
                torch.tensor(seq,    dtype=torch.float32),
                torch.tensor(static, dtype=torch.float32),
            )
            probs_raw = torch.softmax(logits, dim=1).numpy()[0]

        classes = list(self.le.classes_)
        probs   = {cls: round(float(p), 4) for cls, p in zip(classes, probs_raw)}
        er      = round(sum(int(k) * v for k, v in probs.items() if k != "W"), 4)
        mu      = self._matchup_stats.get((batter, bowler), {}) if bowler else {}
        bd      = self._batting_stats.get(batter, {})
        dfl     = self._global_defaults
        sr      = round(float(bd.get("sr", dfl["bat_sr"])), 2)
        note    = "neutral" if not recent_sequence else f"{len(recent_sequence)}-ball ctx"

        return {
            "method":        "lstm",
            "source":        f"lstm_model ({note})",
            "balls_sample":  int(mu.get("balls", 0)),
            "probs":         probs,
            "expected_runs": er,
            "strike_rate":   sr,
            "tendency":      _tendency(sr),
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lstm = LSTMOutcomePredictor()
    result = lstm.train(epochs=20)
    print(f"\nDone. LSTM Accuracy: {result['accuracy']:.4f}")

    print("\nSample predictions:")
    for batter, bowler in [("V Kohli", "A Nortje"), ("JC Buttler", None)]:
        r = lstm.predict(batter, bowler)
        label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
        print(f"  {label}: exp_runs={r['expected_runs']}  tendency={r['tendency']}")

    # with recent context
    r = lstm.predict("V Kohli", "A Nortje", recent_sequence=[4, 0, 1, 6, 0, 1])
    print(f"  V Kohli (after 4,0,1,6,0,1): exp_runs={r['expected_runs']}")
