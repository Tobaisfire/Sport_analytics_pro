"""
Phase 3 prototype — one 2024 match, highlight-style narrative vs ball commentary.

The JSON transcript simulates what you would get from:
  match highlights video → extract audio → speech-to-text (e.g. Whisper).

Same keyword sentiment rules as Phase 2 (commentary_parser) are applied to
narrative sentences for a fair apples-to-apples demo.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import pandas as pd

P3_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(P3_DIR, "artifacts", "prototype_2024_ind_sa_final.json")

_SENT_MAP = {
    "DOMINANT": "dominant_pct",
    "CONTROLLED": "controlled_pct",
    "DEFENSIVE": "defensive_pct",
    "MISTIMED": "mistimed_pct",
    "BEATEN": "beaten_pct",
}


def load_prototype(path: str | None = None) -> dict[str, Any]:
    p = path or DEFAULT_JSON
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    mid = int(data["match_id"])
    asr_txt = os.path.join(P3_DIR, "artifacts", f"transcript_{mid}_asr.txt")
    if os.path.isfile(asr_txt):
        with open(asr_txt, encoding="utf-8") as tf:
            data["highlight_transcript"] = tf.read().strip()
        data["transcript_source"] = "asr_file"
    else:
        data.setdefault("transcript_source", "json_embedded")
    return data


def _split_narrative_segments(transcript: str, min_len: int = 25) -> list[str]:
    """Split highlight prose into chunks parse_one can use."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", transcript.strip())
    out: list[str] = []
    for s in parts:
        s = s.strip()
        if len(s) >= min_len:
            out.append(s)
    return out


def narrative_sentiment_profile(transcript: str, parse_one) -> dict[str, Any]:
    segments = _split_narrative_segments(transcript)
    counts: Counter[str] = Counter()
    scores: list[float] = []
    for seg in segments:
        r = parse_one(seg)
        lab = r.get("sentiment")
        sc = r.get("sentiment_score")
        if lab:
            counts[lab] += 1
        if sc is not None:
            scores.append(float(sc))
    total = sum(counts.values())
    pcts = {lab: counts[lab] / total for lab in counts} if total else {}
    display = {v: 0.0 for v in _SENT_MAP.values()}
    for lab, pct in pcts.items():
        key = _SENT_MAP.get(lab)
        if key:
            display[key] = pct
    pi = (
        display["beaten_pct"] + display["mistimed_pct"] - display["dominant_pct"]
    )
    return {
        "segment_count": len(segments),
        "labeled_chunks": total,
        "label_counts": dict(counts),
        "pcts_raw": pcts,
        "display_pcts": display,
        "avg_sentiment_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "pressure_index": float(pi),
    }


def ball_commentary_match_profile(df: pd.DataFrame, match_id: int) -> dict[str, Any]:
    """Aggregate Phase-2-style sentiment over all balls in one match."""
    m = df[df["match_id"] == match_id].copy()
    if m.empty:
        return {}
    if "sentiment" not in m.columns:
        return {}
    valid = m[m["sentiment"].notna()].copy()
    counts = valid["sentiment"].value_counts().to_dict()
    total = int(valid["sentiment"].notna().sum())
    display = {v: 0.0 for v in _SENT_MAP.values()}
    if total:
        for lab, n in counts.items():
            key = _SENT_MAP.get(str(lab))
            if key:
                display[key] = float(n) / total
    sc = valid["sentiment_score"].dropna()
    pi = display["beaten_pct"] + display["mistimed_pct"] - display["dominant_pct"]
    return {
        "balls_total": len(m),
        "balls_labeled": total,
        "runs_total": int(m["runs_total"].sum()) if "runs_total" in m.columns else None,
        "wickets": int(m["is_wicket"].sum()) if "is_wicket" in m.columns else None,
        "boundaries_4": int((m["runs_batter"] == 4).sum()) if "runs_batter" in m.columns else None,
        "boundaries_6": int((m["runs_batter"] == 6).sum()) if "runs_batter" in m.columns else None,
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "display_pcts": display,
        "avg_sentiment_score": float(sc.mean()) if len(sc) else 0.0,
        "pressure_index": float(pi),
    }


def build_phase3_comparison(
    delivery_df: pd.DataFrame,
    prototype: dict[str, Any],
    parse_one,
) -> dict[str, Any]:
    mid = int(prototype["match_id"])
    ball = ball_commentary_match_profile(delivery_df, mid)
    narr = narrative_sentiment_profile(prototype["highlight_transcript"], parse_one)
    full_txt = prototype["highlight_transcript"]
    return {
        "meta": {k: prototype[k] for k in prototype if k != "highlight_transcript"},
        "ball_commentary": ball,
        "highlight_narrative": narr,
        "transcript_full": full_txt,
        "transcript_preview": full_txt[:1200] + ("…" if len(full_txt) > 1200 else ""),
    }
