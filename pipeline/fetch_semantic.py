"""Build daily semantic news series from GDELT headlines.

For every UTC day, reads a sample of GDELT's files (semantic.slots_per_day
time slots, English and translated feeds), keeps headlines in the configured
languages from outlets in our regions, and for each region and topic stores:

  headlines   headlines on the topic (keywords, plus the classifier where the
              calibration found it reliable)
  total       all usable headlines from the region that day
  tone        average semantic tone of the topic's headlines (positive: closer
              to the topic's "stress" sentences, negative: to its "calm" ones)
  tone_sd     how much the topic's headlines disagree in tone

plus, for novelty and for the website:

  the topic's centroid: the average meaning of its headlines that day
  the day's most stressed and calmest headlines per topic

Files, one set per quarter so parallel backfill jobs never touch the same file:
  data/semantic/series_<year>Q<n>.csv
  data/semantic/centroids_<year>Q<n>.npz
  data/semantic/headlines_<year>Q<n>.csv

The topic decisions and centering vectors come from data/semantic/topic_model.npz,
written by the calibration ("Calibrate topics").

  python -m pipeline.fetch_semantic
      fills any missing day in the last `lookback_days` (the daily job)
  python -m pipeline.fetch_semantic --start 2023-01-01 --end 2023-03-31 --max-minutes 320
      backfills a range; stops cleanly when the time budget runs out
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .check_headlines import FEEDS, normalize, read_file
from .common import DATA, load_config, warn
from .fetch_news import load_domain_regions
from .semantic import ScoreModel, ScreenModel, TopicKeywords, centered, tone_scores

OUT_DIR = DATA / "semantic"
MODEL_FILE = OUT_DIR / "topic_model.npz"
SERIES_COLUMNS = ["date", "region", "topic", "headlines", "total", "tone", "tone_sd"]
HEADLINE_COLUMNS = ["date", "region", "topic", "kind", "tone", "source", "title"]
FIRST_DAY = date(2019, 8, 1)          # GDELT headlines are reliable from here


def quarter(day: date | str) -> str:
    d = date.fromisoformat(str(day)[:10])
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


# ---------------------------------------------------------------- storage
class Store:
    """Reads and writes the quarterly files, touching only the quarters in use."""

    def __init__(self):
        self.series: dict[str, pd.DataFrame] = {}
        self.headlines: dict[str, pd.DataFrame] = {}
        self.centroids: dict[str, dict[str, np.ndarray]] = {}
        self.dirty: set[str] = set()

    def load(self, q: str) -> None:
        if q in self.series:
            return
        s = OUT_DIR / f"series_{q}.csv"
        h = OUT_DIR / f"headlines_{q}.csv"
        c = OUT_DIR / f"centroids_{q}.npz"
        self.series[q] = pd.read_csv(s, dtype={"date": str}) if s.exists() else pd.DataFrame(columns=SERIES_COLUMNS)
        self.headlines[q] = pd.read_csv(h, dtype={"date": str}) if h.exists() else pd.DataFrame(columns=HEADLINE_COLUMNS)
        self.centroids[q] = {}
        if c.exists():
            with np.load(c) as z:
                self.centroids[q] = dict(zip(z["keys"].tolist(), z["vectors"]))

    def done_days(self, q: str) -> set[str]:
        self.load(q)
        return set(self.series[q]["date"])

    def add_day(self, day: str, series: pd.DataFrame, headlines: pd.DataFrame, centroids: dict) -> None:
        q = quarter(day)
        self.load(q)
        keep = lambda df: df[df["date"] != day]
        self.series[q] = pd.concat([keep(self.series[q]), series], ignore_index=True)
        self.headlines[q] = pd.concat([keep(self.headlines[q]), headlines], ignore_index=True)
        self.centroids[q] = {k: v for k, v in self.centroids[q].items() if not k.startswith(day)}
        self.centroids[q].update(centroids)
        self.dirty.add(q)

    def save(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for q in sorted(self.dirty):
            for col in ("headlines", "total", "tone", "tone_sd"):
                self.series[q][col] = pd.to_numeric(self.series[q][col])
            self.headlines[q]["tone"] = pd.to_numeric(self.headlines[q]["tone"])
            self.series[q].sort_values(["date", "region", "topic"]).to_csv(
                OUT_DIR / f"series_{q}.csv", index=False, float_format="%.5f")
            self.headlines[q].sort_values(["date", "region", "topic", "kind"]).to_csv(
                OUT_DIR / f"headlines_{q}.csv", index=False, float_format="%.4f")
            keys = sorted(self.centroids[q])
            vectors = np.stack([self.centroids[q][k] for k in keys]).astype(np.float16) if keys else np.zeros((0, 1), np.float16)
            np.savez_compressed(OUT_DIR / f"centroids_{q}.npz", keys=np.array(keys), vectors=vectors)
        self.dirty.clear()


# ---------------------------------------------------------------- one day
class DayProcessor:
    def __init__(self, config: dict):
        self.news = config["news"]
        self.sem = config["semantic"]
        self.topics = list(self.sem["topics"])
        model = np.load(MODEL_FILE)
        saved = [str(t) for t in model["topics"]]
        if saved != self.topics:
            raise RuntimeError(f"topic_model.npz was calibrated for {saved}, but config.yml lists {self.topics}; "
                               "rerun 'Calibrate topics' and replace data/semantic/topic_model.npz")
        self.keywords = {t: TopicKeywords(self.sem["topics"][t]) for t in self.topics}
        self.use = {t: str(model[f"{t}_use"]) for t in self.topics}
        self.classifier = {t: (model[f"{t}_screen_coef"], float(model[f"{t}_screen_intercept"]),
                               float(model[f"{t}_threshold"]))
                           for t in self.topics if self.use[t] == "screen"}
        self.mean_screen = model["mean_screen"] if self.classifier else None
        self.mean_score = model["mean_score"]
        self.screen = ScreenModel(self.sem["screen_model"]) if self.classifier else None
        self.score = ScoreModel(self.sem["score_model"], self.sem["score_prefix"])
        self.anchors = {t: (centered(self.score.encode(self.sem["topics"][t]["calm"]), self.mean_score),
                            centered(self.score.encode(self.sem["topics"][t]["stress"]), self.mean_score))
                        for t in self.topics}
        self.domain_region = load_domain_regions(self.news)
        self.no_theme = re.compile(r"(?!x)x")
        step = 24 * 60 // self.sem["slots_per_day"]
        self.slots = [f"{(i * step) // 60:02d}{(i * step) % 60:02d}00" for i in range(self.sem["slots_per_day"])]
        print("Topic rules: " + ", ".join(f"{t} = {'keywords' if self.use[t] == 'keywords' else 'keywords + classifier'}"
                                          for t in self.topics), flush=True)

    def headlines(self, day: date, pool: ThreadPoolExecutor) -> tuple[pd.DataFrame | None, int, int]:
        jobs = [(f"{self.news['base_url']}{day:%Y%m%d}{slot}{suffix}", feed)
                for slot in self.slots for feed, suffix in FEEDS.items()]
        frames = [f for f in pool.map(lambda j: read_file(j[0], j[1], day, self.domain_region, self.no_theme), jobs)
                  if f is not None]
        if len(frames) < len(jobs) / 2:
            return None, len(frames), len(jobs)
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["region"] != "other") & (df["title"].str.len() > 0) & df["language"].isin(self.sem["languages"])]
        df = df.assign(key=df["title"].map(normalize))
        df = df[(df["key"].str.split().str.len() >= 4) & (df["title"].str.lower() != df["source"])]
        # Within a region, the same headline published by several outlets counts once.
        df = df.drop_duplicates(["region", "key"]).reset_index(drop=True)
        return df, len(frames), len(jobs)

    def process(self, day: date, pool: ThreadPoolExecutor):
        df, read, wanted = self.headlines(day, pool)
        if df is None:
            return None, f"only {read}/{wanted} files"
        titles = df["title"].tolist()
        member, screen_vec = {}, None
        for t in self.topics:
            hit = self.keywords[t].matches(titles)
            if t in self.classifier and len(titles):
                coef, intercept, threshold = self.classifier[t]
                if np.isfinite(threshold):
                    if screen_vec is None:
                        screen_vec = centered(self.screen.encode(titles), self.mean_screen)
                    prob = 1 / (1 + np.exp(-(screen_vec @ coef + intercept)))
                    hit |= prob >= threshold
            member[t] = hit

        any_topic = np.zeros(len(df), dtype=bool)
        for hit in member.values():
            any_topic |= hit
        idx = np.flatnonzero(any_topic)
        vectors = np.zeros((len(df), len(self.mean_score)), dtype=np.float32)
        if len(idx):
            vectors[idx] = centered(self.score.encode([titles[i] for i in idx]), self.mean_score)

        iso = day.isoformat()
        series, top, cents = [], [], {}
        for region in self.news["regions"]:
            in_region = (df["region"] == region).to_numpy()
            total = int(in_region.sum())
            for t in self.topics:
                rows = np.flatnonzero(in_region & member[t])
                tone = mean = sd = np.nan
                if len(rows):
                    calm, stress = self.anchors[t]
                    scores = tone_scores(vectors[rows], calm, stress)
                    mean, sd = float(scores.mean()), float(scores.std()) if len(rows) > 1 else 0.0
                    centroid = vectors[rows].mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        cents[f"{iso}|{region}|{t}"] = (centroid / norm).astype(np.float16)
                    order = np.argsort(scores)
                    for kind, picks in (("stressed", order[::-1][:3]), ("calm", order[:3])):
                        for p in picks:
                            i = rows[p]
                            top.append([iso, region, t, kind, float(scores[p]), df.at[i, "source"], titles[i]])
                series.append([iso, region, t, len(rows), total, mean, sd])
        status = (f"{read}/{wanted} files, " + ", ".join(
            f"{r} {int((df['region'] == r).sum()):,} headlines" for r in self.news["regions"])
                  + f", {len(idx):,} on a topic")
        return (pd.DataFrame(series, columns=SERIES_COLUMNS), pd.DataFrame(top, columns=HEADLINE_COLUMNS), cents), status


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", help="first day, YYYY-MM-DD")
    parser.add_argument("--end", help="last day, YYYY-MM-DD")
    parser.add_argument("--max-minutes", type=float, help="stop cleanly after this long")
    args = parser.parse_args()

    config = load_config()
    sem = config["semantic"]
    if not MODEL_FILE.exists():
        print(f"{MODEL_FILE} is missing: add topic_model.npz from the 'Calibrate topics' run first.")
        return 1
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    end = min(date.fromisoformat(args.end), yesterday) if args.end else yesterday
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=sem["lookback_days"] - 1)
    start = max(start, FIRST_DAY)

    store = Store()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.isoformat() not in store.done_days(quarter(d))]
    if not todo:
        print(f"Semantic series already complete from {start} to {end}.")
        return 0
    print(f"{len(todo)} days to process between {todo[0]} and {todo[-1]}", flush=True)

    processor = DayProcessor(config)
    began, done = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=config["news"]["threads"]) as pool:
        for day in todo:
            if args.max_minutes and time.monotonic() - began > args.max_minutes * 60:
                print("Time budget used up; run again to continue.")
                break
            result, status = processor.process(day, pool)
            if result is None:
                warn(f"{day}: skipped, {status}")
                continue
            store.add_day(day.isoformat(), *result)
            done += 1
            print(f"{day}: {status}", flush=True)
            if done % 5 == 0:
                store.save()
    store.save()
    print(f"Done: {done} of {len(todo)} days processed.")
    return 0 if done or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
