"""
fetch_commentary_all_years.py
------------------------------
Fetches ball-by-ball commentary for ICC T20 World Cup matches
(2021, 2022, 2024) from CricBuzz internal API and merges into
master_deliveries_with_commentary.csv.

NOTE: 2016 T20 WC is excluded — CricBuzz does not store ball-by-ball
commentary for 2016 era matches (API returns 0 entries for all matches).

Confirmed series IDs and match ID ranges:
  2021  seriesId=2798   CricBuzz IDs 37987-38250  (40 matches)
  2022  seriesId=3961   CricBuzz IDs 42986-43200  (39 matches)
  2024  seriesId=7476   CricBuzz IDs 87480-87885  (44 matches)

Run from Sports_analytics/source/:
    python fetch_commentary_all_years.py

Re-running is safe: cached JSON files are reused, only missing matches are fetched.
"""

import re, time, json, os
import requests
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(BASE, "..")
DATASET    = os.path.join(ROOT, "master_data", "dataset")
RAW_DIR    = os.path.join(DATASET, "raw_commentary_cb")
os.makedirs(RAW_DIR, exist_ok=True)

DELIVERIES_CSV = os.path.join(DATASET, "master_deliveries.csv")
MATCHES_CSV    = os.path.join(DATASET, "master_matches.csv")
OUTPUT_CSV     = os.path.join(DATASET, "master_deliveries_with_commentary.csv")

HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":  "application/json, text/plain, */*",
    "Referer": "https://www.cricbuzz.com/",
}

# ── team name normalisation ────────────────────────────────────────────────────
_NORM = {
    "india": "india", "pakistan": "pakistan", "australia": "australia",
    "england": "england", "new zealand": "new zealand", "south africa": "south africa",
    "west indies": "west indies", "sri lanka": "sri lanka", "bangladesh": "bangladesh",
    "afghanistan": "afghanistan", "ireland": "ireland", "scotland": "scotland",
    "netherlands": "netherlands", "namibia": "namibia", "zimbabwe": "zimbabwe",
    "oman": "oman", "papua new guinea": "papua new guinea", "canada": "canada",
    "uganda": "uganda", "nepal": "nepal",
    # USA variants
    "united states of america": "usa", "united states": "usa",
    "u.s.a.": "usa", "usa": "usa",
    # UAE
    "united arab emirates": "uae", "u.a.e.": "uae",
}

def norm(name: str) -> str:
    s = str(name).strip().lower()
    return _NORM.get(s, s)


# ── known tournament configs ───────────────────────────────────────────────────
# 2016 excluded: CricBuzz has no ball-by-ball commentary for 2016 era matches.
# 2021/2022: match IDs confirmed by direct probing in both series.
# 2024: already cached from previous fetch_all_commentary.py run.
TOURNAMENTS = [
    {"year": 2021, "series_id": 2798,  "scan_start": 37980, "scan_end": 38260, "scan_step": 1},
    {"year": 2022, "series_id": 3961,  "scan_start": 42980, "scan_end": 43200, "scan_step": 1},
    {"year": 2024, "series_id": 7476,  "scan_start": 87480, "scan_end": 87890, "scan_step": 1},
]


# ── CricBuzz probing ───────────────────────────────────────────────────────────
def probe(cb_id: int) -> dict | None:
    """
    Probe one CricBuzz match ID.
    Returns {cb_id, series_id, t1, t2, date} for T20 international matches, else None.
    """
    url = f"https://www.cricbuzz.com/api/mcenter/{cb_id}/full-commentary/1"
    try:
        r = requests.get(url, headers=HDR, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        mh = data.get("matchDetails", {}).get("matchHeader", {})
        match_type = (mh.get("matchType") or "").lower()
        # We only want T20 international / T20I matches
        if "t20" not in match_type and "international" not in match_type.lower():
            # Also accept based on series name containing "t20 world cup"
            sname = (mh.get("seriesName") or mh.get("seriesDesc") or "").lower()
            if "t20 world cup" not in sname and "world t20" not in sname:
                return None
        t1 = norm(mh.get("team1", {}).get("name", ""))
        t2 = norm(mh.get("team2", {}).get("name", ""))
        if not t1 or not t2:
            return None
        ts = mh.get("matchStartTimestamp", 0)
        date = pd.to_datetime(int(ts), unit="ms").date() if ts else None
        series_id = mh.get("seriesId")
        series_name = mh.get("seriesName") or mh.get("seriesDesc") or ""
        return {
            "cb_id": cb_id,
            "series_id": series_id,
            "series_name": series_name,
            "t1": t1, "t2": t2,
            "date": str(date),
        }
    except Exception:
        return None


def scan_for_series(cfg: dict, known_ids: set) -> list:
    """
    Scan a range of CricBuzz IDs and return all matches belonging to
    this tournament's series_id (or matching 't20 world cup' in name for 2016).
    """
    year       = cfg["year"]
    series_id  = cfg["series_id"]
    start, end = cfg["scan_start"], cfg["scan_end"]
    step       = cfg["scan_step"]
    total      = (end - start) // step

    print(f"\n--- Scanning {year} T20 WC (seriesId={series_id}) IDs {start}-{end} ---")
    found = []

    for i, cb_id in enumerate(range(start, end, step)):
        if cb_id in known_ids:
            continue
        info = probe(cb_id)
        if info:
            sid = info["series_id"]
            is_wc = (sid == series_id)
            if is_wc:
                found.append(info)
                print(f"  Found: {cb_id} | {info['date']} | {info['t1']} vs {info['t2']}")
                known_ids.add(cb_id)

        time.sleep(0.25)
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i+1}/{total} ({start + i*step}), found {len(found)} so far ...")

    print(f"  Total found: {len(found)}")
    return found


# ── commentary parsing ─────────────────────────────────────────────────────────
def parse_commentary(comm_list: list, inning_no: int) -> list:
    rows = []
    for entry in comm_list:
        ball_nbr = entry.get("ballNbr", 0)
        event    = entry.get("event", "NONE")
        if ball_nbr == 0 and event in ("NONE", None):
            continue

        over_number = entry.get("overNumber")
        if over_number is not None:
            try:
                ov_f = float(over_number)
                ov   = int(ov_f)
                bl   = round((ov_f - ov) * 10)
            except (ValueError, TypeError):
                continue
        elif ball_nbr > 0:
            ov = (ball_nbr - 1) // 6
            bl = ((ball_nbr - 1) % 6) + 1
        else:
            continue

        if bl < 1:
            continue

        text = re.sub(r"B\d+\$\s*", "", entry.get("commText", "") or "").strip()
        if not text:
            continue

        rows.append({
            "inning":     inning_no,
            "over":       ov,
            "ball":       bl,
            "short_text": text[:120],
            "full_text":  text,
        })
    return rows


def fetch_match_commentary(cb_id: int, cs_id: str) -> list:
    """Fetch and parse commentary for both innings of a match. Returns list of row dicts."""
    cache = os.path.join(RAW_DIR, f"{cs_id}_rows.json")
    if os.path.exists(cache):
        with open(cache) as f:
            rows = json.load(f)
        if rows:
            return rows

    rows = []
    for inning in [1, 2]:
        url = f"https://www.cricbuzz.com/api/mcenter/{cb_id}/full-commentary/{inning}"
        try:
            r = requests.get(url, headers=HDR, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception as e:
            print(f"    Fetch error inn{inning}: {e}")
            continue

        for inn_obj in data.get("commentary", []):
            inn_id  = inn_obj.get("inningsId", inning)
            inn_no  = 1 if inn_id <= 1 else 2
            parsed  = parse_commentary(inn_obj.get("commentaryList", []), inn_no)
            rows.extend(parsed)
        time.sleep(0.5)

    if rows:
        with open(cache, "w") as f:
            json.dump(rows, f)

    return rows


# ── Cricsheet <-> CricBuzz mapping ────────────────────────────────────────────
def build_mapping(df_year: pd.DataFrame, cb_matches: list) -> tuple[dict, list]:
    """
    Match Cricsheet rows to CricBuzz entries by (teams, date ± 1 day).
    Returns (mapping: cs_id -> cb_id, unmatched: list of cs_ids).
    """
    mapping   = {}
    unmatched = []

    for _, row in df_year.iterrows():
        cs_id   = str(row["match_id"])
        cs_t1   = norm(row["team1"])
        cs_t2   = norm(row["team2"])
        cs_date = row["date"].date() if pd.notna(row["date"]) else None
        cs_pair = frozenset([cs_t1, cs_t2])

        best = None
        for cb in cb_matches:
            if frozenset([cb["t1"], cb["t2"]]) == cs_pair:
                if cs_date and cb["date"] not in ("None", None, ""):
                    diff = abs((cs_date - pd.to_datetime(cb["date"]).date()).days)
                    if diff <= 1:
                        best = cb
                        break
                else:
                    best = cb
                    break

        if best:
            mapping[cs_id] = best["cb_id"]
        else:
            unmatched.append((cs_id, row["team1"], row["team2"],
                              str(row["date"].date()) if pd.notna(row["date"]) else "?"))

    return mapping, unmatched


# ── merge into master CSV ─────────────────────────────────────────────────────
def merge_all_commentary(all_comm: dict) -> None:
    print("\nLoading master_deliveries.csv ...")
    df = pd.read_csv(DELIVERIES_CSV)
    df["match_id"] = df["match_id"].astype(str)
    print(f"  {len(df):,} delivery rows")

    # Build commentary lookup
    comm_rows = []
    for cs_id, rows in all_comm.items():
        for r in rows:
            comm_rows.append({
                "match_id":        cs_id,
                "inning_no":       r["inning"],
                "over":            r["over"],
                "ball_in_over":    r["ball"],
                "commentary_short": r["short_text"],
                "commentary":       r["full_text"],
            })
    df_comm = pd.DataFrame(comm_rows)
    print(f"  Commentary pool: {len(df_comm):,} rows across {df_comm['match_id'].nunique()} matches")

    df["commentary_short"] = None
    df["commentary"]       = None

    matched = 0
    for (mid, inn), g_d in df.groupby(["match_id", "inning_no"]):
        g_d_s = g_d.sort_values(["over", "ball_in_over"])
        g_c   = df_comm[
            (df_comm["match_id"] == mid) & (df_comm["inning_no"] == inn)
        ].sort_values(["over", "ball_in_over"])
        if g_c.empty:
            continue
        idx_d = g_d_s.index.tolist()
        idx_c = g_c.index.tolist()
        for i, didx in enumerate(idx_d):
            if i < len(idx_c):
                df.at[didx, "commentary_short"] = g_c.at[idx_c[i], "commentary_short"]
                df.at[didx, "commentary"]       = g_c.at[idx_c[i], "commentary"]
                matched += 1

    df.to_csv(OUTPUT_CSV, index=False)
    filled  = df["commentary"].notna().sum()
    n_match = df[df["commentary"].notna()]["match_id"].nunique()
    pct     = filled / len(df) * 100
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"  Commentary filled : {filled:,} / {len(df):,} rows ({pct:.1f}%)")
    print(f"  Matches covered   : {n_match} / {df['match_id'].nunique()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading Cricsheet match data ...")
    df_matches = pd.read_csv(MATCHES_CSV)
    df_matches["date"] = pd.to_datetime(df_matches["date"], errors="coerce")
    df_matches["year"] = df_matches["date"].dt.year

    # Load existing scan cache
    scan_cache_path = os.path.join(RAW_DIR, "_all_years_scan.json")
    if os.path.exists(scan_cache_path):
        with open(scan_cache_path) as f:
            all_cb_matches = json.load(f)
        print(f"Loaded scan cache: {len(all_cb_matches)} CricBuzz matches")
    else:
        all_cb_matches = []

    known_ids = {m["cb_id"] for m in all_cb_matches}

    # Scan each tournament year
    for cfg in TOURNAMENTS:
        year = cfg["year"]
        # Check if we already have enough matches for this year
        yr_matches = df_matches[df_matches["year"] == year]
        needed     = len(yr_matches)
        have       = sum(
            1 for m in all_cb_matches
            if m.get("date", "")[:4] == str(year)
        )
        print(f"\n{year}: need {needed} matches, have {have} in cache")
        if have >= needed:
            print(f"  Skipping scan — already have enough")
            continue

        new_found = scan_for_series(cfg, known_ids)
        all_cb_matches.extend(new_found)
        known_ids.update(m["cb_id"] for m in new_found)

    # Save updated scan cache
    with open(scan_cache_path, "w") as f:
        json.dump(all_cb_matches, f, indent=2)
    print(f"\nTotal CricBuzz matches in cache: {len(all_cb_matches)}")

    # Build per-year mappings and fetch commentary
    all_commentary = {}

    for cfg in TOURNAMENTS:
        year = cfg["year"]
        df_year = df_matches[df_matches["year"] == year].copy().sort_values("date")
        print(f"\n=== {year} — {len(df_year)} Cricsheet matches ===")

        cb_year = [m for m in all_cb_matches if str(m.get("date",""))[:4] == str(year)]
        print(f"  CricBuzz matches for {year}: {len(cb_year)}")

        mapping, unmatched = build_mapping(df_year, cb_year)
        print(f"  Mapped: {len(mapping)}/{len(df_year)}")
        if unmatched:
            print(f"  Unmatched ({len(unmatched)}):")
            for u in unmatched[:5]:
                print(f"    {u}")

        for cs_id, cb_id in mapping.items():
            cache = os.path.join(RAW_DIR, f"{cs_id}_rows.json")
            if os.path.exists(cache):
                with open(cache) as f:
                    rows = json.load(f)
                if rows:
                    all_commentary[cs_id] = rows
                    continue

            print(f"  Fetching {cs_id} (cb:{cb_id}) ...")
            rows = fetch_match_commentary(cb_id, cs_id)
            if rows:
                all_commentary[cs_id] = rows
                print(f"    -> {len(rows)} commentary rows saved")
            else:
                print(f"    -> no commentary found")
            time.sleep(0.8)

    # Summary
    total_rows = sum(len(v) for v in all_commentary.values())
    print(f"\n{'='*55}")
    print(f"Matches with commentary : {len(all_commentary)}/149")
    print(f"Total commentary rows   : {total_rows:,}")
    print(f"{'='*55}")

    # Merge into master CSV
    merge_all_commentary(all_commentary)

    # Per-year summary
    print("\nCoverage by year:")
    df_check = pd.read_csv(OUTPUT_CSV, low_memory=False)
    df_check["match_id"] = df_check["match_id"].astype(str)
    df_yr = df_matches[["match_id", "year"]].copy()
    df_yr["match_id"] = df_yr["match_id"].astype(str)
    df_check = df_check.merge(df_yr, on="match_id", how="left")
    has = df_check["commentary"].notna() & (df_check["commentary"].astype(str).str.strip() != "")
    for yr, grp in df_check.groupby("year"):
        n_balls  = len(grp)
        n_comm   = has[grp.index].sum()
        n_match  = grp["match_id"].nunique()
        print(f"  {yr}: {n_match} matches, {n_comm:,}/{n_balls:,} balls ({n_comm/n_balls*100:.1f}%)")


if __name__ == "__main__":
    main()
