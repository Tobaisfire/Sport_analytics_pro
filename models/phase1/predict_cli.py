"""
predict_cli.py
--------------
Command-line tool for Phase 1 ball-outcome predictions.

Usage examples:
    python predict_cli.py "MN Samuels" "CJ Jordan"
    python predict_cli.py "V Kohli"                          # batter only
    python predict_cli.py "RG Sharma" --method empirical
    python predict_cli.py "RG Sharma" --leaderboard          # best bowlers vs batter
    python predict_cli.py --bowler "JJ Bumrah" --leaderboard # best batters vs bowler
"""

import argparse
import sys
import os

# Allow imports from same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import Phase1Predictor

OUTCOME_DISPLAY = {
    "0": "0 runs  ",
    "1": "1 run   ",
    "2": "2 runs  ",
    "3": "3 runs  ",
    "4": "4 runs  ",
    "6": "6 runs  ",
    "W": "Wicket  ",
}

DIVIDER = "-" * 55


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_comparison(cmp: dict, show_bars: bool = False) -> None:
    batter = cmp["batter"]
    bowler = cmp["bowler"]
    balls  = cmp["balls_sample"]
    emp_src = cmp["sources"]["empirical"]
    emp_label = {
        "matchup_direct":         "direct matchup",
        "blended":                "blended (matchup + overall)",
        "batter_overall_fallback":"batter overall (no matchup)",
        "batter_overall":         "batter overall",
    }.get(emp_src, emp_src)

    print()
    print(DIVIDER)
    print(f"  Batter : {batter}")
    print(f"  Bowler : {bowler}")
    print(f"  Balls  : {balls}  ({emp_label})")
    print(DIVIDER)

    header = f"{'Outcome':<10}  {'Empirical':>10}  {'XGBoost':>10}  {'Diff':>8}"
    print(header)
    print("-" * len(header))

    for row in cmp["outcomes"]:
        lbl = OUTCOME_DISPLAY.get(row["outcome"], row["outcome"])
        ep  = row["empirical_pct"]
        xp  = row["xgboost_pct"]
        diff = row["diff"]
        diff_str = f"{diff:+.1f}%"
        line = f"{lbl:<10}  {ep:>9.1f}%  {xp:>9.1f}%  {diff_str:>8}"
        if show_bars:
            line += f"  {_bar(max(ep, xp))}"
        print(line)

    print(DIVIDER)
    er = cmp["expected_runs"]
    td = cmp["tendency"]
    print(f"  Expected runs  Empirical: {er['empirical']:.3f}   XGBoost: {er['xgboost']:.3f}")
    print(f"  Tendency       Empirical: {td['empirical']:<12}  XGBoost: {td['xgboost']}")
    print(DIVIDER)
    print()


def _print_single(result: dict, batter: str, bowler: str | None, method: str) -> None:
    if "error" in result:
        print(f"[Error] {result['error']}")
        return

    label = f"{batter} vs {bowler}" if bowler else f"{batter} (overall)"
    probs = result.get("probs", {})
    er    = result.get("expected_runs", "?")
    sr    = result.get("strike_rate", "?")
    tend  = result.get("tendency", "?")
    src   = result.get("source", "?")
    balls = result.get("balls_sample", "?")

    print()
    print(DIVIDER)
    print(f"  Method : {method}")
    print(f"  Pair   : {label}")
    print(f"  Balls  : {balls}  ({src})")
    print(DIVIDER)
    for lbl in ["0", "1", "2", "3", "4", "6", "W"]:
        p = float(probs.get(lbl, 0))
        disp = OUTCOME_DISPLAY.get(lbl, lbl)
        print(f"  {disp}  {p*100:6.1f}%  {_bar(p*100)}")
    print(DIVIDER)
    print(f"  Expected runs : {er}")
    print(f"  Strike Rate   : {sr}")
    print(f"  Tendency      : {tend}")
    print(DIVIDER)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="predict_cli",
        description="Phase 1 Cricket Prediction — Empirical + XGBoost"
    )
    parser.add_argument("batter",        nargs="?",  help="Batter name (e.g. 'RG Sharma')")
    parser.add_argument("bowler",        nargs="?",  help="Bowler name (optional)")
    parser.add_argument("--method",      default="both",
                        choices=["empirical", "xgboost", "both"],
                        help="Which model to use (default: both)")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show top-10 bowlers vs batter (or batters vs --bowler)")
    parser.add_argument("--bowler",      dest="bowler_flag",
                        help="Bowler name for --leaderboard mode")
    parser.add_argument("--top",         type=int, default=10,
                        help="Number of entries in leaderboard (default 10)")
    parser.add_argument("--bars",        action="store_true",
                        help="Show ASCII probability bars")

    args = parser.parse_args()

    p = Phase1Predictor()

    # ── Leaderboard mode ──────────────────────────────────────────────────────
    if args.leaderboard:
        if args.batter:
            print(f"\nTop {args.top} bowlers vs {args.batter} (by wicket probability):")
            df = p.leaderboard_bowlers(args.batter, args.top)
            print(df.to_string(index=False))
        elif args.bowler_flag:
            print(f"\nTop {args.top} batters vs {args.bowler_flag} (by expected runs):")
            df = p.leaderboard_batters(args.bowler_flag, args.top)
            print(df.to_string(index=False))
        else:
            print("[Error] Provide a batter name or --bowler for leaderboard mode.")
            sys.exit(1)
        return

    # ── Prediction mode ───────────────────────────────────────────────────────
    if not args.batter:
        parser.print_help()
        sys.exit(1)

    bowler = args.bowler or args.bowler_flag

    if args.method == "both":
        cmp = p.compare(args.batter, bowler)
        if "error" in cmp:
            print(f"[Error] {cmp['error']}")
            sys.exit(1)
        _print_comparison(cmp, show_bars=args.bars)
    else:
        result = p.predict(args.batter, bowler, method=args.method)
        _print_single(result, args.batter, bowler, args.method)


if __name__ == "__main__":
    main()
