"""Build daily news series from GDELT's raw GKG 2.0 files.

GDELT publishes one file every 15 minutes with one row per article: the outlet,
the themes GDELT detected, and the article's tone. This script reads a sample of
those files for each day and stores, per UTC day, region and theme:

  articles  sampled articles from the region's outlets that carry the theme
  total     all sampled articles from the region's outlets (articles / total = share)
  tone      average GDELT tone of the matching articles

Output: data/news/gdelt_<year>.csv (date, region, theme, articles, total, tone).

  python -m pipeline.fetch_news
      fills any missing day in the last `lookback_days` (the daily job)
  python -m pipeline.fetch_news --start 2017-01-01 --end 2017-12-31 --max-minutes 300
      backfills a range; stops cleanly when the time budget runs out and
      resumes where it left off on the next run
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from .common import DATA, HEADERS, http_get, load_config, warn

OUT_DIR = DATA / "news"
COLUMNS = ["date", "region", "theme", "articles", "total", "tone"]

# Positions of the columns we need in a GKG 2.0 row (tab-separated, no header).
COL_SOURCE, COL_THEMES, COL_TONE = 3, 7, 15


def load_domain_regions(cfg: dict) -> dict[str, str]:
    """Map each outlet's domain to one of our regions, using GDELT's lookup table."""
    country_region = {code: region for region, codes in cfg["regions"].items() for code in codes}
    text = http_get(cfg["domain_lookup_url"]).content.decode("utf-8", errors="replace")
    mapping = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() in country_region:
            mapping[parts[0].strip().lower()] = country_region[parts[1].strip()]
    found = set(mapping.values())
    missing = set(cfg["regions"]) - found
    if missing:
        raise RuntimeError(f"outlet lookup has no domains for {sorted(missing)}; "
                           f"its format may have changed ({len(text.splitlines())} lines read)")
    return mapping


def download(url: str) -> bytes | None:
    """Return the file, or None if GDELT has no file for that time slot."""
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def summarize_file(url: str, domain_region: dict, patterns: dict) -> dict | None:
    """Counts and tone sums for one GKG file, keyed by region and theme."""
    blob = download(url)
    if blob is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                df = pd.read_csv(handle, sep="\t", header=None, dtype=str,
                                 usecols=[COL_SOURCE, COL_THEMES, COL_TONE],
                                 quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
    except (zipfile.BadZipFile, pd.errors.ParserError, IndexError, ValueError):
        return None

    df["region"] = df[COL_SOURCE].str.lower().map(domain_region)
    df = df.dropna(subset=["region"])
    # The tone field is a comma-separated list; the first number is the overall tone.
    df["tone"] = pd.to_numeric(df[COL_TONE].str.split(",", n=1).str[0], errors="coerce")
    df = df.dropna(subset=["tone"])
    themes = df[COL_THEMES].fillna("")

    result = {"total": df.groupby("region").size().to_dict(), "themes": {}}
    for theme, pattern in patterns.items():
        matched = df[themes.str.contains(pattern)]
        grouped = matched.groupby("region")["tone"].agg(["size", "sum"])
        result["themes"][theme] = {r: (int(row["size"]), float(row["sum"]))
                                   for r, row in grouped.iterrows()}
    return result


def process_day(day: date, cfg: dict, domain_region: dict, patterns: dict,
                pool: ThreadPoolExecutor) -> tuple[pd.DataFrame | None, str, int]:
    """All rows for one UTC day (None if too few files were available), a status
    line, and the number of files read."""
    step = 24 * 60 // cfg["slots_per_day"]
    suffixes = [".gkg.csv.zip"] + ([".translation.gkg.csv.zip"] if cfg["include_translated"] else [])
    start = datetime(day.year, day.month, day.day)
    # Each slot covers a window of the day (an hour with 24 slots). GDELT publishes
    # a file every 15 minutes and some are missing during its outages, so every
    # 15-minute file in the window is tried in turn until one exists.
    slots = [[f"{cfg['base_url']}{start + timedelta(minutes=m):%Y%m%d%H%M%S}{suffix}"
              for m in range(i * step, (i + 1) * step, 15)]
             for i in range(cfg["slots_per_day"]) for suffix in suffixes]

    def first_available(urls: list[str]) -> dict | None:
        for url in urls:
            result = summarize_file(url, domain_region, patterns)
            if result is not None:
                return result
        return None

    results = [r for r in pool.map(first_available, slots) if r is not None]
    status = f"{len(results)}/{len(slots)} files"
    if len(results) < len(slots) / 2:
        return None, status, len(results)

    rows = []
    for region in cfg["regions"]:
        total = sum(r["total"].get(region, 0) for r in results)
        for theme in patterns:
            pairs = [r["themes"][theme].get(region, (0, 0.0)) for r in results]
            articles = sum(p[0] for p in pairs)
            tone = sum(p[1] for p in pairs) / articles if articles else float("nan")
            rows.append([day.isoformat(), region, theme, articles, total, tone])
        status += f", {region} {total} articles"
    return pd.DataFrame(rows, columns=COLUMNS), status, len(results)


def load_store() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("gdelt_*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f, dtype={"date": str}) for f in files], ignore_index=True)


def save_years(store: pd.DataFrame, years: set[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in sorted(years):
        part = store[store["date"].str.startswith(year)]
        part.sort_values(["date", "region", "theme"]).to_csv(
            OUT_DIR / f"gdelt_{year}.csv", index=False, float_format="%.4f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", help="first day to fetch, YYYY-MM-DD")
    parser.add_argument("--end", help="last day to fetch, YYYY-MM-DD")
    parser.add_argument("--max-minutes", type=float, help="stop cleanly after this long")
    args = parser.parse_args()

    config = load_config()
    cfg = config["news"]
    first_allowed = date.fromisoformat(config["start_date"])
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)  # complete days only

    end = min(date.fromisoformat(args.end), yesterday) if args.end else yesterday
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        start = end - timedelta(days=cfg["lookback_days"] - 1)
    start = max(start, first_allowed)

    store = load_store()
    done = set(store["date"])
    todo = [d for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
            if d.isoformat() not in done]
    if not todo:
        print(f"News is already complete from {start} to {end}.")
        return 0
    print(f"{len(todo)} days to fetch between {todo[0]} and {todo[-1]}", flush=True)

    domain_region = load_domain_regions(cfg)
    print(f"Outlet lookup: {len(domain_region)} domains in our regions", flush=True)
    patterns = {theme: re.compile("(?:^|;)(?:" + "|".join(map(re.escape, codes)) + ")(?:;|$)")
                for theme, codes in cfg["themes"].items()}

    began, fetched, files_read, dirty = time.monotonic(), 0, 0, set()
    with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool:
        for day in todo:
            if args.max_minutes and time.monotonic() - began > args.max_minutes * 60:
                print("Time budget used up; run again to continue.")
                break
            rows, status, n_files = process_day(day, cfg, domain_region, patterns, pool)
            files_read += n_files
            if rows is None:
                warn(f"{day}: skipped, only {status} available")
                continue
            print(f"{day}: {status}", flush=True)
            store = rows if store.empty else pd.concat([store, rows], ignore_index=True)
            fetched += 1
            dirty.add(day.isoformat()[:4])
            if fetched % 10 == 0:  # save regularly so an interrupted run loses little
                save_years(store, dirty)
    save_years(store, dirty)

    if fetched:
        new = store[store["date"].isin({d.isoformat() for d in todo})]
        counts = new.groupby("theme")["articles"].sum()
        for theme in counts[counts == 0].index:
            warn(f"theme '{theme}' matched no articles; check its codes in config.yml")
    print(f"Done: {fetched} of {len(todo)} days fetched.")
    # Fail only if not a single file could be read: that means GDELT is
    # unreachable or the file addresses have changed, not just a gap in its archive.
    return 0 if files_read else 1


if __name__ == "__main__":
    sys.exit(main())
