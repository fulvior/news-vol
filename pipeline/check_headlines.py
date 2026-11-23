"""Check how usable GDELT's headlines are before building on them.

Reads a sample of GKG files spread over every year (4 days a year, 6 time
slots a day, English and translated feeds) and reports, per year and region:

  - how many articles carry a headline (<PAGE_TITLE> in the Extras field)
  - the same for economically themed articles, the ones we would embed
  - how many headlines are duplicates (syndicated copies) or junk
  - the language headlines are written in
  - how many unique economic headlines a day the pipeline would have to embed

The report goes to the run's summary page on GitHub (and to the terminal);
the sampled headlines are saved as a CSV to download from the run.

  python -m pipeline.check_headlines
"""
from __future__ import annotations

import csv
import html
import io
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .common import DATA, load_config
from .fetch_news import download, load_domain_regions

# Columns of a GKG 2.0 row (tab-separated, no header).
COL_SOURCE, COL_THEMES, COL_QUOTES, COL_TRANSLATION, COL_EXTRAS = 3, 7, 22, 25, 26
SLOTS = ["000000", "040000", "080000", "120000", "160000", "200000"]
FEEDS = {"english": ".gkg.csv.zip", "translated": ".translation.gkg.csv.zip"}

TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
LANGUAGE = re.compile(r"srclc:(\w+)")
OUT_DIR = DATA / "checks"


def sample_days() -> list[date]:
    """Four days a year: mid-January, April, July, October, then the last week."""
    config = load_config()
    first = date.fromisoformat(config["start_date"])
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [date(y, m, 15) for y in range(first.year, yesterday.year + 1)
            for m in (1, 4, 7, 10) if first <= date(y, m, 15) < yesterday - timedelta(days=7)]
    return days + [yesterday - timedelta(days=k) for k in (7, 5, 3, 1)]


def read_file(url: str, feed: str, day: date, domain_region: dict, econ: re.Pattern) -> pd.DataFrame | None:
    blob = download(url)
    if blob is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                df = pd.read_csv(handle, sep="\t", header=None, dtype=str,
                                 usecols=[COL_SOURCE, COL_THEMES, COL_QUOTES, COL_TRANSLATION, COL_EXTRAS],
                                 quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
    except (zipfile.BadZipFile, pd.errors.ParserError, IndexError, ValueError):
        return None

    df = df.fillna("")
    out = pd.DataFrame({
        "year": day.year,
        "day": day.isoformat(),
        "feed": feed,
        "source": df[COL_SOURCE].str.lower(),
    })
    out["region"] = out["source"].map(domain_region).fillna("other")
    out["econ"] = df[COL_THEMES].str.contains(econ)
    raw = df[COL_EXTRAS].str.extract(TITLE, expand=False).fillna("")
    out["title"] = raw.map(lambda t: re.sub(r"\s+", " ", html.unescape(t)).strip())
    out["language"] = df[COL_TRANSLATION].str.extract(LANGUAGE, expand=False).fillna("eng")
    out["has_quotes"] = df[COL_QUOTES].str.len() > 0
    return out


def normalize(title: str) -> str:
    """Key for spotting copies of the same headline: lower case, letters and digits only,
    and without a trailing ' - Outlet name' or ' | Outlet name'."""
    title = re.split(r"\s+[|\-–—]\s+[^|\-–—]{2,40}$", title)[0]
    return re.sub(r"[\W_]+", " ", title.lower()).strip()


def pct(part, whole) -> str:
    return "–" if whole == 0 else f"{100 * part / whole:.0f}%"


def main() -> int:
    config = load_config()
    cfg = config["news"]
    domain_region = load_domain_regions(cfg)
    codes = [c for codes in cfg["themes"].values() for c in codes]
    econ = re.compile(r"(?:^|;)(?:ECON_|EPU_|" + "|".join(map(re.escape, codes)) + r")")

    days = sample_days()
    jobs = [(f"{cfg['base_url']}{d:%Y%m%d}{slot}{suffix}", feed, d)
            for d in days for slot in SLOTS for feed, suffix in FEEDS.items()]
    print(f"Reading {len(jobs)} files from {len(days)} days…", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool:
        frames = list(pool.map(lambda j: read_file(j[0], j[1], j[2], domain_region, econ), jobs))
    read = [f for f in frames if f is not None]
    if not read:
        print("No file could be read.")
        return 1
    df = pd.concat(read, ignore_index=True)
    df = df[df["region"] != "other"]
    df["has_title"] = df["title"].str.len() > 0
    df["key"] = df["title"].map(normalize)
    df["words"] = df["key"].str.split().str.len().fillna(0)
    # Too short to carry meaning once the outlet's name is removed ("Home - BBC News"),
    # or just the outlet's domain.
    df["junk"] = df["has_title"] & ((df["words"] < 4) | (df["title"].str.lower() == df["source"]))
    usable = df["has_title"] & ~df["junk"]

    lines = ["# Headline check", "",
             f"{len(read)} of {len(jobs)} files read, {len(df):,} articles from outlets in our regions.", ""]

    # --- coverage by year and region
    lines += ["## Articles with a usable headline", "",
              "| Year | Region | Articles | With headline | Economic articles | Economic with headline | Unique economic headlines a day* |",
              "|---|---|---|---|---|---|---|"]
    for (year, region), g in df.groupby(["year", "region"]):
        ok = usable.loc[g.index]
        econ_rows = g[g["econ"] & ok]
        unique_per_day = econ_rows.groupby("day")["key"].nunique().mean() * 24 / len(SLOTS) if len(econ_rows) else 0
        lines.append(f"| {year} | {region} | {len(g):,} | {pct(ok.sum(), len(g))} | {g['econ'].sum():,} | "
                     f"{pct(econ_rows.shape[0], g['econ'].sum())} | {unique_per_day:,.0f} |")
    lines += ["", "\\* Estimated at the pipeline's rate of 24 files a day per feed, after removing duplicates within a day.", ""]

    # --- duplicates, junk, length, quotations
    titled = df[df["has_title"]]
    lines += ["## Quality", "",
              "| Region | Junk headlines | Duplicate share (within a day) | Median words | Articles with quotations |",
              "|---|---|---|---|---|"]
    for region, g in titled.groupby("region"):
        good = g[~g["junk"]]
        duplicates = 1 - good.groupby("day")["key"].nunique().sum() / max(len(good), 1)
        all_region = df[df["region"] == region]
        lines.append(f"| {region} | {pct(g['junk'].sum(), len(g))} | {pct(duplicates * len(good), len(good))} | "
                     f"{good['words'].median():.0f} | {pct(all_region['has_quotes'].sum(), len(all_region))} |")
    lines.append("")

    # --- languages
    lines += ["## Headline languages (articles with a usable headline)", "", "| Region | Feed | Languages |", "|---|---|---|"]
    for (region, feed), g in df[usable].groupby(["region", "feed"]):
        top = g["language"].value_counts(normalize=True).head(6)
        lines.append(f"| {region} | {feed} | " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in top.items()) + " |")
    lines.append("")

    # --- examples
    recent = df[usable & df["econ"] & (df["year"] == df["year"].max())].drop_duplicates("key")
    lines += ["## Example economic headlines, latest year", ""]
    for (region, feed), g in recent.groupby(["region", "feed"]):
        lines.append(f"**{region}, {feed}**")
        lines.append("")
        for _, row in g.sample(min(8, len(g)), random_state=1).iterrows():
            lines.append(f"- {row['title'][:160]} ({row['source']}, {row['language']})")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = df[usable & df["econ"]].drop_duplicates(["day", "key"])
    # Up to 300 random headlines per year, region and feed.
    sample = sample.sample(frac=1, random_state=1).groupby(["year", "region", "feed"]).head(300)
    sample = sample.sort_values(["day", "region", "feed"])
    sample[["day", "region", "feed", "language", "source", "title"]].to_csv(
        OUT_DIR / "headline_sample.csv", index=False)
    print(f"Saved {len(sample):,} sampled headlines to {OUT_DIR / 'headline_sample.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
