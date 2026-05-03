"""
commentary_parser.py
--------------------
Parses cricket commentary text into structured fields per ball:
  - shot_type  : drive / pull / hook / cut / sweep / flick / slog / defend / None
  - region     : cover / mid-off / long-off / mid-on / long-on / mid-wicket /
                 square leg / fine leg / third man / point / slip / None
  - sentiment  : DOMINANT / CONTROLLED / MISTIMED / BEATEN / DEFENSIVE / None
  - sentiment_score : +1.0 / +0.5 / -0.5 / -1.0 / 0.0 / None

Priority for sentiment detection:
  BEATEN > DOMINANT > MISTIMED > DEFENSIVE > CONTROLLED
"""

import re
import pandas as pd

# ── Shot keyword lookup ────────────────────────────────────────────────────────
# Ordered by specificity — first match wins.
SHOT_PATTERNS: list[tuple[str, list[str]]] = [
    ("defend",  ["defended", "defence", "defending", "back down the pitch",
                 "blocked", "blocks", "back and across", "pushed back"]),
    ("sweep",   ["slog sweep", "reverse sweep", "swept", "sweep", "sweeping"]),
    ("flick",   ["flicked", "flick", "whipped", "whippy", "whips", "wrists it",
                 "wrist", "glanced", "glance", "glances"]),
    ("pull",    ["pulled to", "pulled down", "pulls to", "pulled away",
                 "pull shot", "pull to", "pull down", "pull"]),
    ("hook",    ["hooked", "hook", "hooks"]),
    ("cut",     ["square cut", "late cut", "cuts away", "cuts through", "cutting",
                 "cut away", "cut through", "cut to", "cuts to", "cut"]),
    ("drive",   ["cover drive", "on drive", "off drive", "straight drive",
                 "driven well", "driven to", "driven through", "drives",
                 "drove", "driven", "drive to", "drive through"]),
    ("slog",    ["slog", "heave", "heaved", "chip over", "chipped over",
                 "chipped", "chip", "lofted over", "hoick", "hacked",
                 "clubbed", "swipe", "swiped", "big shot"]),
]

# ── Region keyword lookup ──────────────────────────────────────────────────────
# Multi-word regions must come before single-word sub-patterns.
REGION_PATTERNS: list[tuple[str, list[str]]] = [
    ("cover",       ["extra cover", "mid-off/cover", "cover drive", "cover region",
                     "cover point", "through covers", "covers", "cover"]),
    ("mid-off",     ["mid-off", "mid off"]),
    ("long-off",    ["long-off", "long off"]),
    ("mid-on",      ["mid-on", "mid on"]),
    ("long-on",     ["long-on", "long on"]),
    ("mid-wicket",  ["mid-wicket", "mid wicket", "midwicket"]),
    ("square leg",  ["deep square", "backward square", "square leg", "square-leg"]),
    ("fine leg",    ["fine leg", "fine-leg", "backward fine"]),
    ("third man",   ["third man", "third-man"]),
    ("point",       ["backward point", "square point", "point"]),
    ("slip",        ["slip cordon", "slip", "slips"]),
]

# ── Sentiment rules (priority: BEATEN > DOMINANT > MISTIMED > DEFENSIVE > CONTROLLED) ─
SENTIMENT_RULES: list[tuple[str, float, list[str]]] = [
    ("BEATEN", -1.0, [
        "beaten", "beat the bat", "beats the bat", "no contact",
        "doesn't connect", "does not connect", "missed", "misses",
        "through the gate", "big appeal", "massive appeal",
        "could have been out", "goes past", "passes the bat",
    ]),
    ("DOMINANT", +1.0, [
        " six", "sixes", "six runs", "over the ropes", "into the stands",
        "maximum", "clears the ropes", " four", "four runs", "boundary",
        "hammered", "smashed", "blasted", "tonked", "bludgeoned",
        "creamed", "pummelled", "crunched",
    ]),
    ("MISTIMED", -0.5, [
        "mistimed", "mistimes", "miscued", "miscues", "leading edge",
        "top edge", "top-edge", "bottom edge", "thick edge",
        "skied", "skies", "popped up", "dollied",
        "inside edge onto pad", "off the edge",
        "tickle", "tickled", "tamely hit", "feathered",
    ]),
    ("DEFENSIVE", 0.0, [
        "defended", "blocked", "blocks", "quietly", "back down the pitch",
        "kept out", "watchfully", "forward defense", "backfoot defence",
        "played out", "sees it off", "leaves", "left alone",
        "good leave", "shoulders arms",
    ]),
    ("CONTROLLED", +0.5, [
        "placed", "eased", "nudged", "guided", "timed well", "clipped",
        "tapped", "worked", "steered", "pushed through", "comfortably",
        "sensibly", "well placed", "along the ground", "elegant",
        "classical", "nicely timed", "well timed",
    ]),
]


# ── Core parse function ────────────────────────────────────────────────────────

def parse_one(text: str) -> dict:
    """
    Parse a single commentary string.

    Returns
    -------
    dict with keys: shot_type, region, sentiment, sentiment_score
    All values are None if the text is empty or unrecognisable.
    """
    result: dict = {
        "shot_type":      None,
        "region":         None,
        "sentiment":      None,
        "sentiment_score": None,
    }

    if not isinstance(text, str) or not text.strip():
        return result

    t = text.lower()

    # ── shot type ─────────────────────────────────────────────────────────────
    for shot, keywords in SHOT_PATTERNS:
        if any(kw in t for kw in keywords):
            result["shot_type"] = shot
            break

    # ── region ────────────────────────────────────────────────────────────────
    for region, keywords in REGION_PATTERNS:
        if any(kw in t for kw in keywords):
            result["region"] = region
            break

    # ── sentiment ─────────────────────────────────────────────────────────────
    for label, score, keywords in SENTIMENT_RULES:
        if any(kw in t for kw in keywords):
            result["sentiment"]       = label
            result["sentiment_score"] = score
            break

    # ── fallback sentiment from outcome words ─────────────────────────────────
    if result["sentiment"] is None:
        if re.search(r"\bno run\b", t) or "dot ball" in t:
            result["sentiment"]       = "DEFENSIVE"
            result["sentiment_score"] = 0.0
        elif re.search(r"\b(1|2|3) run", t) or "single" in t or "couple" in t:
            result["sentiment"]       = "CONTROLLED"
            result["sentiment_score"] = 0.5

    return result


def parse_dataframe(df: pd.DataFrame,
                    text_col: str = "commentary_short") -> pd.DataFrame:
    """
    Apply parse_one to every row of df[text_col].

    Returns a copy of df with four new columns appended:
      shot_type, region, sentiment, sentiment_score
    """
    parsed = df[text_col].apply(parse_one).apply(pd.Series)
    out = df.reset_index(drop=True).copy()
    out["shot_type"]       = parsed["shot_type"].values
    out["region"]          = parsed["region"].values
    out["sentiment"]       = parsed["sentiment"].values
    out["sentiment_score"] = parsed["sentiment_score"].values
    return out


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLES = [
        "Bilal Khan to Tony Ura, no run, slight inswing, quietly defended back down the pitch",
        "Nortje to Gous, six, Short, Gous top-edges the pull shot into the stands! Maximum!",
        "Jordan to Pollard, no run, Pollard beaten on the flick, inside edge onto pad",
        "Saifuddin to Rajapaksa, 1 run, slower ball, Rajapaksa mistimes the slog through midwicket",
        "Dockrell to Williamson, 2 runs, pitched up outside off, driven wide of long off for a couple",
        "Woakes to Asalanka, 1 run, full, leg-lined, whipped down to deep square leg",
        "M Theekshana to Bavuma, 1 run, stays back and drags this away with a whippy pull down towards deep mid-wicket",
        "Raza to Shadab Khan, 1 run, tossed up outside off, Shadab Khan eases the drive to long-off",
        "Ferguson to Mark Adair, 1 run, on a good length, backs away and hacks the drive to long-on",
        "Campher to Kusal Mendis, 1 run, off-cutter dug in short, pulled hard and straight to deep square leg",
    ]

    print(f"{'Commentary':<70} {'Shot':<10} {'Region':<14} {'Sentiment':<12} {'Score'}")
    print("-" * 120)
    for s in SAMPLES:
        r = parse_one(s)
        print(
            f"{s[:68]:<70} "
            f"{str(r['shot_type']):<10} "
            f"{str(r['region']):<14} "
            f"{str(r['sentiment']):<12} "
            f"{str(r['sentiment_score'])}"
        )
