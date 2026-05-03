"""
Fetch ball-by-ball commentary for all 44 2024 ICC T20 WC matches from CricBuzz.
Uses the internal API: /api/mcenter/{matchId}/full-commentary/{inningId}

Run from: Sports_analytics/source/
"""
import re, time, json, os, requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "..", "master_data", "dataset")
RAW_COMM_DIR = os.path.join(OUTPUT_DIR, "raw_commentary_cb")
os.makedirs(RAW_COMM_DIR, exist_ok=True)

HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cricbuzz.com/",
}

# Team name normalisation for matching
_TEAM_NORM = {
    "india": "india", "ireland": "ireland", "pakistan": "pakistan",
    "australia": "australia", "england": "england", "bangladesh": "bangladesh",
    "sri lanka": "sri lanka", "south africa": "south africa",
    "new zealand": "new zealand", "west indies": "west indies",
    "afghanistan": "afghanistan", "nepal": "nepal", "netherlands": "netherlands",
    "scotland": "scotland", "canada": "canada", "uganda": "uganda",
    "papua new guinea": "papua new guinea", "namibia": "namibia",
    "oman": "oman", "united states of america": "usa", "usa": "usa",
    "u.s.a.": "usa",
}

def norm(n: str) -> str:
    return _TEAM_NORM.get(str(n).strip().lower(), str(n).strip().lower())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Scan CricBuzz IDs to build Cricsheet → CricBuzz mapping
# ─────────────────────────────────────────────────────────────────────────────
def probe_cb_id(cb_id: int) -> dict | None:
    """
    Fetch match details for a CricBuzz match ID.
    Returns dict with team1, team2, date if it's a T20 WC 2024 match, else None.
    """
    url = f"https://www.cricbuzz.com/api/mcenter/{cb_id}/full-commentary/1"
    try:
        r = requests.get(url, headers=HDR, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        # Real structure: matchDetails -> matchHeader
        mh = data.get("matchDetails", {}).get("matchHeader", {})
        series_id = mh.get("seriesId", 0)
        series_name = (mh.get("seriesName") or mh.get("seriesDesc") or "").lower()

        # Filter: must be T20 WC 2024 (series 7476)
        if series_id != 7476 and "t20 world cup" not in series_name:
            return None

        t1 = norm(mh.get("team1", {}).get("name", ""))
        t2 = norm(mh.get("team2", {}).get("name", ""))
        date_ms = mh.get("matchStartTimestamp", 0)
        date = pd.to_datetime(int(date_ms), unit="ms").date() if date_ms else None
        return {"cb_id": cb_id, "t1": t1, "t2": t2, "date": date}
    except Exception:
        return None


print("Loading master_matches.csv ...")
df_m = pd.read_csv(os.path.join(OUTPUT_DIR, "master_matches.csv"))
df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
df_2024 = df_m[df_m["date"].dt.year == 2024].copy()
df_2024 = df_2024.sort_values("date").reset_index(drop=True)
print(f"  2024 matches: {len(df_2024)}")

# Scan the full known range for T20 WC 2024 matches (June 1 - June 29, 2024)
# IDs are sequential across all CricBuzz matches globally, ~1 T20WC match per 7 IDs
SCAN_RANGES = list(range(87535, 87885))
print(f"Scanning {len(SCAN_RANGES)} candidate CricBuzz IDs ...")

cache_scan_path = os.path.join(RAW_COMM_DIR, "_scan_results.json")
if os.path.exists(cache_scan_path):
    with open(cache_scan_path) as f:
        cb_matches_found = json.load(f)
    print(f"  Loaded {len(cb_matches_found)} from cache")
else:
    cb_matches_found = []
    scanned_ids = set()
    for i, cb_id in enumerate(SCAN_RANGES):
        if cb_id in scanned_ids:
            continue
        scanned_ids.add(cb_id)
        info = probe_cb_id(cb_id)
        if info:
            print(f"  Found T20WC24: {cb_id} | {info['t1']} vs {info['t2']} | {info['date']}")
            cb_matches_found.append({
                "cb_id": cb_id,
                "t1": info["t1"],
                "t2": info["t2"],
                "date": str(info["date"]),
            })
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            print(f"  Scanned {i+1}/{len(SCAN_RANGES)} IDs, found {len(cb_matches_found)} T20WC24 matches so far ...")

    with open(cache_scan_path, "w") as f:
        json.dump(cb_matches_found, f)
    print(f"Scan complete. Found {len(cb_matches_found)} T20 WC 2024 matches.")

# Build Cricsheet -> CricBuzz mapping
MAPPING = {}
unmatched_cs = []

for _, row in df_2024.iterrows():
    cs_id = str(row["match_id"])
    ct1, ct2 = norm(row["team1"]), norm(row["team2"])
    cs_teams = frozenset([ct1, ct2])
    cs_date = row["date"].date() if pd.notna(row["date"]) else None

    matched = False
    for cb in cb_matches_found:
        if frozenset([cb["t1"], cb["t2"]]) == cs_teams:
            # Also check date proximity
            cb_date_str = cb["date"]
            if cb_date_str and cb_date_str != "None":
                cb_date = pd.to_datetime(cb_date_str).date()
                if cs_date and abs((cs_date - cb_date).days) > 2:
                    continue
            MAPPING[cs_id] = cb["cb_id"]
            matched = True
            break

    if not matched:
        unmatched_cs.append((cs_id, row["team1"], row["team2"], str(row["date"].date())))

print(f"\nMapping built: {len(MAPPING)}/44 matched")
if unmatched_cs:
    print(f"Unmatched ({len(unmatched_cs)}): {unmatched_cs[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Parse commentary entries to over+ball
# ─────────────────────────────────────────────────────────────────────────────
def parse_commentary_list(commentary_list: list, inning_no: int) -> list:
    rows = []
    for entry in commentary_list:
        ball_nbr = entry.get("ballNbr", 0)
        event = entry.get("event", "NONE")

        # Skip non-ball events (pre-match, post-match, video snippets)
        if ball_nbr == 0 and event in ("NONE", None):
            continue

        # Get over.ball from overNumber field if available
        over_number = entry.get("overNumber")
        if over_number is not None:
            try:
                ov_f = float(over_number)
                ov = int(ov_f)
                bl = round((ov_f - ov) * 10)
            except (ValueError, TypeError):
                continue
        elif ball_nbr > 0:
            # Fallback: derive from ballNbr (1-indexed, 6 balls per over)
            ov = (ball_nbr - 1) // 6
            bl = ((ball_nbr - 1) % 6) + 1
        else:
            continue

        if bl < 1:
            continue

        comm_text = entry.get("commText", "") or ""
        # Remove format markers like B0$
        comm_text = re.sub(r'B\d+\$\s*', '', comm_text).strip()

        if not comm_text:
            continue

        rows.append({
            "inning": inning_no,
            "over": ov,
            "ball": bl,
            "short_text": comm_text[:120],
            "full_text": comm_text,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Fetch commentary for all mapped matches
# ─────────────────────────────────────────────────────────────────────────────
all_commentary = {}

for cs_id, cb_id in MAPPING.items():
    cache_path = os.path.join(RAW_COMM_DIR, f"{cs_id}_rows.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            rows = json.load(f)
        print(f"  [{cs_id}] cached: {len(rows)} rows")
        all_commentary[cs_id] = rows
        continue

    print(f"Fetching {cs_id} (cb:{cb_id}) ...")
    rows = []
    for inning in [1, 2]:
        url = f"https://www.cricbuzz.com/api/mcenter/{cb_id}/full-commentary/{inning}"
        try:
            r = requests.get(url, headers=HDR, timeout=20)
            if r.status_code != 200:
                print(f"  [{cs_id}] inning {inning}: HTTP {r.status_code}")
                continue
            data = r.json()
        except Exception as e:
            print(f"  [{cs_id}] inning {inning}: Error {e}")
            continue

        for inn_obj in data.get("commentary", []):
            inn_id = inn_obj.get("inningsId", inning)
            # Map inningsId to 1/2
            inn_no = 1 if inn_id in [1, 0] else 2
            comm_list = inn_obj.get("commentaryList", [])
            parsed = parse_commentary_list(comm_list, inn_no)
            print(f"  [{cs_id}] inning {inning} (inningsId={inn_id}): {len(comm_list)} entries -> {len(parsed)} ball rows")
            rows.extend(parsed)

        time.sleep(0.5)

    if rows:
        with open(cache_path, "w") as f:
            json.dump(rows, f)
        all_commentary[cs_id] = rows
        print(f"  [{cs_id}] saved {len(rows)} total rows")
    else:
        print(f"  [{cs_id}] no rows extracted")
    time.sleep(1.0)

print(f"\nMatches with commentary: {len(all_commentary)}")
total_rows_fetched = sum(len(v) for v in all_commentary.values())
print(f"Total commentary rows: {total_rows_fetched}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Merge into master_deliveries_with_commentary.csv
# ─────────────────────────────────────────────────────────────────────────────
if all_commentary:
    print("\nLoading master_deliveries.csv ...")
    df_del = pd.read_csv(os.path.join(OUTPUT_DIR, "master_deliveries.csv"))
    df_del["match_id"] = df_del["match_id"].astype(str)
    print(f"  Total deliveries: {len(df_del)}")

    comm_rows = []
    for cs_id, rows in all_commentary.items():
        for r in rows:
            comm_rows.append({
                "match_id": cs_id,
                "inning_no": r["inning"],
                "over": r["over"],
                "ball_in_over": r["ball"],
                "commentary_short": r["short_text"],
                "commentary": r["full_text"],
            })
    df_comm = pd.DataFrame(comm_rows)
    print(f"  Commentary df: {len(df_comm)} rows, {df_comm['match_id'].nunique()} matches")

    # Initialize columns
    df_del["commentary_short"] = None
    df_del["commentary"] = None

    # Sequential merge: sort both by (over, ball_in_over) within each (match_id, inning_no)
    matched_count = 0
    for (mid, inn), g_d in df_del.groupby(["match_id", "inning_no"]):
        g_d_s = g_d.sort_values(["over", "ball_in_over"])
        g_c = df_comm[
            (df_comm["match_id"] == mid) & (df_comm["inning_no"] == inn)
        ].sort_values(["over", "ball_in_over"])
        if g_c.empty:
            continue
        idx_d = g_d_s.index.tolist()
        idx_c = g_c.index.tolist()
        for i, didx in enumerate(idx_d):
            if i < len(idx_c):
                df_del.at[didx, "commentary_short"] = g_c.at[idx_c[i], "commentary_short"]
                df_del.at[didx, "commentary"] = g_c.at[idx_c[i], "commentary"]
                matched_count += 1

    out = os.path.join(OUTPUT_DIR, "master_deliveries_with_commentary.csv")
    df_del.to_csv(out, index=False)

    filled = df_del["commentary"].notna().sum()
    matches_with = df_del[df_del["commentary"].notna()]["match_id"].nunique()
    print(f"\nSaved: {out}")
    print(f"  Commentary filled: {filled}/{len(df_del)} rows ({filled/len(df_del)*100:.1f}%)")
    print(f"  Matches with commentary: {matches_with}")
    print("\nSample rows with commentary:")
    sample = df_del[df_del["commentary"].notna()].head(3)
    for _, r in sample.iterrows():
        print(f"  match={r['match_id']} inn={r['inning_no']} over={r['over']}.{r['ball_in_over']}")
        print(f"    short: {str(r['commentary_short'])[:80]}")
        print(f"    full:  {str(r['commentary'])[:120]}")
else:
    print("No commentary fetched.")
