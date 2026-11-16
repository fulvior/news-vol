"""Download daily news volume and tone from the GDELT DOC 2.0 API.

For every region x theme pair in config.yml this stores, per UTC day:
  articles  number of matching articles
  total     number of all articles GDELT monitored (use articles / total as share)
  tone      average GDELT tone of the matching articles

Output: data/news/gdelt_daily.csv in long format (date, region, theme, ...).

The first run backfills from start_date (many requests: run it locally once).
Later runs only re-fetch the trailing refresh window for each series.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .common import DATA, http_get, load_config, warn

PATH = DATA / "news" / "gdelt_daily.csv"
COLUMNS = ["date", "region", "theme", "articles", "total", "tone"]
KEYS = ["date", "region", "theme"]


def gdelt_timeline(cfg: dict, query: str, mode: str,
                   start: datetime, end: datetime) -> pd.DataFrame:
    """One API call. Returns the timeline rows with a parsed UTC 'date' column."""
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    response = http_get(cfg["endpoint"], params=params)
    time.sleep(cfg["pause_seconds"])
    try:
        payload = response.json()
    except ValueError:
        # GDELT reports query errors as plain text with HTTP 200.
        raise RuntimeError(f"GDELT said: {response.text.strip()[:200]}")

    timeline = payload.get("timeline") or []
    rows = timeline[0].get("data", []) if timeline else []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ").dt.normalize()
    return df


def fetch_series(cfg: dict, query: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Daily articles, total and tone for one query between start and end."""
    frames = []
    chunk = timedelta(days=cfg["chunk_days"])
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + chunk - timedelta(seconds=1), end)
        volume = gdelt_timeline(cfg, query, "timelinevolraw", cursor, chunk_end)
        tone = gdelt_timeline(cfg, query, "timelinetone", cursor, chunk_end)
        cursor = chunk_end + timedelta(seconds=1)
        if volume.empty:
            continue

        # Short windows come back hourly, so always aggregate to days ourselves.
        daily = volume.groupby("date").agg(articles=("value", "sum"), total=("norm", "sum"))
        if not tone.empty:
            daily["tone"] = tone.groupby("date")["value"].mean()
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=["articles", "total", "tone"])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].reindex(columns=["articles", "total", "tone"])


def main() -> int:
    config = load_config()
    cfg = config["gdelt"]
    PATH.parent.mkdir(parents=True, exist_ok=True)

    if PATH.exists():
        store = pd.read_csv(PATH, parse_dates=["date"])
    else:
        store = pd.DataFrame(columns=COLUMNS)
        store["date"] = pd.to_datetime(store["date"])

    # Only complete UTC days: stop at yesterday 23:59:59.
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                               microsecond=0, tzinfo=None)
    end = today - timedelta(seconds=1)
    first = datetime.fromisoformat(config["start_date"])
    failures, series_count = 0, 0

    for region, region_query in cfg["regions"].items():
        for theme, theme_query in cfg["themes"].items():
            series_count += 1
            label = f"{region}/{theme}"
            have = store[(store["region"] == region) & (store["theme"] == theme)]
            if have.empty:
                start = first
            else:
                start = max(first, have["date"].max().to_pydatetime()
                            - timedelta(days=cfg["refresh_days"]))
            try:
                new = fetch_series(cfg, f"{theme_query} {region_query}", start, end)
            except Exception as error:
                failures += 1
                warn(f"{label}: not updated ({error})")
                continue

            new = new[new.index < today]
            if new.empty or new["articles"].sum() == 0:
                warn(f"{label}: GDELT returned no articles; check the query in config.yml")
                continue

            new = new.rename_axis("date").reset_index()
            new["region"], new["theme"] = region, theme
            # New rows replace stored rows for the same day.
            store = pd.concat([store, new[COLUMNS]], ignore_index=True)
            store = store.drop_duplicates(KEYS, keep="last")

            # Save after every series so an interrupted backfill can resume.
            store.sort_values(["region", "theme", "date"]).to_csv(
                PATH, index=False, date_format="%Y-%m-%d", float_format="%.4f")
            print(f"{label}: {len(new)} days fetched, through {new['date'].max():%Y-%m-%d}",
                  flush=True)

    return 1 if failures == series_count else 0


if __name__ == "__main__":
    sys.exit(main())
