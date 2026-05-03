"""
commentary_classifier.py
------------------------
Two complementary sentiment approaches for cricket commentary text.

1. VADER Sentiment Scorer (rule-based NLP)
   - Applies SentimentIntensityAnalyzer to each commentary string
   - Maps compound score to our 5 cricket-specific sentiment labels
   - Instant — no training needed
   - Useful as a baseline against keyword rules and the learned model

2. BiLSTM Text Classifier (PyTorch)
   - Tokenises raw commentary text
   - Embedding(vocab, 64) → Bidirectional LSTM(64) → Dense → Softmax
   - Trained on keyword-rule labels (from commentary_parser) as pseudo-ground-truth
   - "Learns" patterns beyond the keyword list (word co-occurrences, context)

Usage
-----
  # VADER (no training)
  from commentary_classifier import VADERSentiment
  v = VADERSentiment()
  print(v.score("Kohli drives beautifully through the covers for FOUR!"))

  # BiLSTM (train once)
  from commentary_classifier import CommentaryClassifier
  clf = CommentaryClassifier()
  clf.train()
  print(clf.predict_sentiment("beaten outside off stump, no contact"))

Artifacts:
  artifacts/bilstm_sentiment_model.pt
  artifacts/bilstm_tokenizer.pkl
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..", "..")
DATASET   = os.path.join(ROOT_DIR, "master_data", "dataset")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

DELIVERIES_PATH = os.path.join(DATASET, "master_deliveries_with_commentary.csv")
BILSTM_MODEL_PATH = os.path.join(ARTIFACTS, "bilstm_sentiment_model.pt")
BILSTM_TOK_PATH   = os.path.join(ARTIFACTS, "bilstm_tokenizer.pkl")

# ── Sentiment label map ────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["BEATEN", "CONTROLLED", "DEFENSIVE", "DOMINANT", "MISTIMED"]
LABEL2IDX = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
IDX2LABEL  = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

# VADER compound → cricket sentiment
VADER_COMPOUND_MAP = [
    (0.50,  "DOMINANT"),
    (0.10,  "CONTROLLED"),
    (-0.10, "DEFENSIVE"),
    (-0.50, "MISTIMED"),
]  # below -0.5 → BEATEN


def _vader_to_label(compound: float) -> str:
    for threshold, label in VADER_COMPOUND_MAP:
        if compound >= threshold:
            return label
    return "BEATEN"


# ═══════════════════════════════════════════════════════════════════════════════
class VADERSentiment:
    """Rule-based sentiment using VADER lexicon (no training)."""

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._sia = SentimentIntensityAnalyzer()
        # Add cricket-specific words to VADER lexicon
        self._sia.lexicon.update({
            "four":    1.5,
            "six":     2.5,
            "boundary": 1.5,
            "cracked":  1.0,
            "smashed":  1.5,
            "driven":   0.8,
            "pulled":   0.8,
            "beaten":  -2.0,
            "bowled":  -2.0,
            "caught":  -1.5,
            "lbw":     -1.5,
            "wicket":  -1.5,
            "mistimed": -1.0,
            "miscued":  -1.0,
            "skied":   -0.8,
            "dot":     -0.5,
            "miss":    -1.0,
        })

    def score(self, text: str) -> dict:
        """Return VADER scores + mapped cricket sentiment label."""
        scores  = self._sia.polarity_scores(str(text))
        label   = _vader_to_label(scores["compound"])
        return {
            "label":    label,
            "compound": round(scores["compound"], 4),
            "pos":      round(scores["pos"], 4),
            "neg":      round(scores["neg"], 4),
            "neu":      round(scores["neu"], 4),
        }

    def score_series(self, texts: pd.Series) -> pd.DataFrame:
        rows = [self.score(t) for t in texts.fillna("")]
        return pd.DataFrame(rows, index=texts.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Simple word-level tokeniser (no external deps)

class _Tokenizer:
    def __init__(self, max_vocab=5000, max_len=30):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self._word2idx = {"<PAD>": 0, "<UNK>": 1}
        self._built    = False

    def _clean(self, text: str) -> list[str]:
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    def fit(self, texts: list[str]) -> "_Tokenizer":
        from collections import Counter
        counts = Counter(w for t in texts for w in self._clean(t))
        vocab  = [w for w, _ in counts.most_common(self.max_vocab - 2)]
        for w in vocab:
            if w not in self._word2idx:
                self._word2idx[w] = len(self._word2idx)
        self._built = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        result = []
        for t in texts:
            ids = [self._word2idx.get(w, 1) for w in self._clean(t)]
            if len(ids) < self.max_len:
                ids = [0] * (self.max_len - len(ids)) + ids
            else:
                ids = ids[-self.max_len:]
            result.append(ids)
        return np.array(result, dtype=np.int32)

    def vocab_size(self) -> int:
        return len(self._word2idx)


# ═══════════════════════════════════════════════════════════════════════════════
class CommentaryClassifier:
    """BiLSTM sentiment classifier trained on keyword-rule labels."""

    def __init__(self):
        self._model   = None
        self._tok     = None
        self._vader   = VADERSentiment()
        self._loaded  = False

    def _build_model(self, vocab_size: int, n_classes: int):
        import torch.nn as nn

        class BiLSTMClassifier(nn.Module):
            def __init__(self, v_size, embed_dim=64, hidden=64, n_cls=5, dropout=0.3):
                super().__init__()
                self.embed   = nn.Embedding(v_size, embed_dim, padding_idx=0)
                self.bilstm  = nn.LSTM(embed_dim, hidden, batch_first=True,
                                       bidirectional=True)
                self.drop    = nn.Dropout(dropout)
                self.fc1     = nn.Linear(hidden * 2, 32)
                self.relu    = nn.ReLU()
                self.fc2     = nn.Linear(32, n_cls)

            def forward(self, x):
                e        = self.embed(x)          # (B, L, embed)
                out, _   = self.bilstm(e)         # (B, L, hidden*2)
                pooled   = out.mean(dim=1)        # mean pooling
                pooled   = self.drop(pooled)
                h        = self.relu(self.fc1(pooled))
                return self.fc2(h)

        return BiLSTMClassifier(vocab_size, n_cls=n_classes)

    # ── training ──────────────────────────────────────────────────────────────

    def train(self,
              deliveries_path=DELIVERIES_PATH,
              model_path=BILSTM_MODEL_PATH,
              tok_path=BILSTM_TOK_PATH,
              epochs=15,
              batch_size=256,
              test_size=0.15) -> dict:

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        # import local commentary_parser
        import sys
        sys.path.insert(0, BASE_DIR)
        from commentary_parser import parse_dataframe

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== BiLSTM Commentary Classifier (device={device}) ===")

        print("Loading data ...")
        df = pd.read_csv(deliveries_path, low_memory=False)
        df = df[df["commentary_short"].notna()].copy()
        print(f"  {len(df):,} commentary rows")

        print("Parsing sentiment labels (keyword rules) ...")
        df = parse_dataframe(df, text_col="commentary_short")
        df = df[df["sentiment"].notna()].copy()
        print(f"  {len(df):,} labelled rows")
        print("  Distribution:\n", df["sentiment"].value_counts().to_string())

        texts  = df["commentary_short"].fillna("").tolist()
        labels = df["sentiment"].map(LABEL2IDX).values

        # VADER comparison
        print("\nComputing VADER labels for comparison ...")
        vader_labels = [_vader_to_label(
            self._vader._sia.polarity_scores(t)["compound"]
        ) for t in texts]
        vader_enc = np.array([LABEL2IDX.get(l, 0) for l in vader_labels])
        vader_acc = accuracy_score(labels, vader_enc)
        print(f"  VADER accuracy vs keyword rules: {vader_acc:.4f}")

        # tokenise
        print("\nTokenising commentary ...")
        tok = _Tokenizer(max_vocab=5000, max_len=30)
        tok.fit(texts)
        X = tok.transform(texts)
        print(f"  Vocab size: {tok.vocab_size():,}")

        # split
        idx = np.arange(len(labels))
        tr_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=42, stratify=labels
        )
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        print(f"  Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

        n_classes = len(SENTIMENT_LABELS)
        model     = self._build_model(tok.vocab_size(), n_classes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=2, factor=0.5
        )

        X_tr_t = torch.tensor(X_tr, dtype=torch.long)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_te_t = torch.tensor(X_te, dtype=torch.long)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state    = None
        patience_cnt  = 0
        PATIENCE      = 4

        print(f"\nTraining {epochs} epochs ...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                optimiser.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(yb)

            avg_loss = total_loss / len(y_tr)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_te_t.to(device))
                val_loss   = criterion(
                    val_logits,
                    torch.tensor(y_te, dtype=torch.long).to(device)
                ).item()
            scheduler.step(val_loss)
            print(f"  Epoch {epoch:>2}/{epochs}  train={avg_loss:.4f}  val={val_loss:.4f}")

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

        model.eval()
        with torch.no_grad():
            logits  = model(X_te_t.to(device)).cpu()
            y_proba = torch.softmax(logits, dim=1).numpy()
            y_pred  = np.argmax(y_proba, axis=1)

        acc = accuracy_score(y_te, y_pred)
        print(f"\n  BiLSTM Accuracy vs keyword rules: {acc:.4f}")
        print(classification_report(
            y_te, y_pred,
            target_names=SENTIMENT_LABELS,
            labels=list(range(n_classes)),
            zero_division=0,
        ))

        # comparison summary
        print("\n" + "=" * 50)
        print("SENTIMENT MODEL COMPARISON")
        print("=" * 50)
        print(f"  Keyword Rules  (baseline) :  N/A  (ground truth)")
        print(f"  VADER                     : {vader_acc:.4f}")
        print(f"  BiLSTM (learned)          : {acc:.4f}")
        print("=" * 50)

        os.makedirs(ARTIFACTS, exist_ok=True)
        torch.save({
            "model_state": best_state or model.state_dict(),
            "vocab_size":  tok.vocab_size(),
            "n_classes":   n_classes,
        }, model_path)
        joblib.dump(tok, tok_path)
        print(f"Saved: {model_path}")
        print(f"Saved: {tok_path}")

        self._model  = model
        self._tok    = tok
        self._loaded = True

        return {
            "bilstm_accuracy": acc,
            "vader_accuracy":  vader_acc,
            "n_classes":       n_classes,
            "vocab_size":      tok.vocab_size(),
        }

    # ── inference ─────────────────────────────────────────────────────────────

    def load(self, model_path=BILSTM_MODEL_PATH,
             tok_path=BILSTM_TOK_PATH) -> "CommentaryClassifier":
        import torch

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BiLSTM model not found at {model_path}. "
                "Run CommentaryClassifier().train() first."
            )
        self._tok = joblib.load(tok_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        net = self._build_model(checkpoint["vocab_size"], checkpoint["n_classes"])
        net.load_state_dict(checkpoint["model_state"])
        net.eval()
        self._model  = net
        self._loaded = True
        return self

    def predict_sentiment(self, text: str) -> dict:
        """Predict sentiment for a single commentary string."""
        import torch

        if not self._loaded:
            self.load()

        x    = torch.tensor(self._tok.transform([text]), dtype=torch.long)
        with torch.no_grad():
            logits = self._model(x)
            probs  = torch.softmax(logits, dim=1).numpy()[0]

        idx   = int(np.argmax(probs))
        label = IDX2LABEL[idx]

        # also compute VADER for comparison
        vader_result = self._vader.score(text)

        return {
            "bilstm_label":      label,
            "bilstm_confidence": round(float(probs[idx]), 4),
            "bilstm_probs":      {IDX2LABEL[i]: round(float(p), 4)
                                  for i, p in enumerate(probs)},
            "vader_label":       vader_result["label"],
            "vader_compound":    vader_result["compound"],
            "text":              text,
        }

    def predict_batch(self, texts: list[str],
                      batch_size: int = 512) -> pd.DataFrame:
        """Predict sentiment for a list of commentary strings."""
        import torch

        if not self._loaded:
            self.load()

        all_preds = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            X     = torch.tensor(self._tok.transform(chunk), dtype=torch.long)
            with torch.no_grad():
                probs = torch.softmax(self._model(X), dim=1).numpy()
            for text, p in zip(chunk, probs):
                idx   = int(np.argmax(p))
                all_preds.append({
                    "text":             text,
                    "bilstm_label":     IDX2LABEL[idx],
                    "bilstm_conf":      round(float(p[idx]), 4),
                    "vader_label":      self._vader.score(text)["label"],
                })
        return pd.DataFrame(all_preds)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. VADER — instant
    print("--- VADER (no training needed) ---")
    vader = VADERSentiment()
    examples = [
        "Kohli drives beautifully through the covers for FOUR!",
        "Beaten outside off stump, no contact whatsoever",
        "Dot ball, pushed back to the bowler",
        "Mistimed pull shot, skied to mid-on and caught!",
        "Nudged off the pads for a single",
    ]
    for ex in examples:
        r = vader.score(ex)
        print(f"  [{r['label']:>10}] (cmp={r['compound']:+.2f}) {ex[:60]}")

    # 2. BiLSTM — train
    print("\n--- BiLSTM Training ---")
    clf    = CommentaryClassifier()
    result = clf.train(epochs=15)
    print(f"\nBiLSTM accuracy: {result['bilstm_accuracy']:.4f}")
    print(f"VADER accuracy:  {result['vader_accuracy']:.4f}")

    print("\n--- BiLSTM Predictions ---")
    for ex in examples:
        r = clf.predict_sentiment(ex)
        print(f"  BiLSTM=[{r['bilstm_label']:>10}] conf={r['bilstm_confidence']:.2f}  "
              f"VADER=[{r['vader_label']:>10}]  {ex[:55]}")
