"""Calibrate the semantic topics on a large sample of real GDELT headlines.

1. Downloads headlines from two days a month since August 2019, plus days with
   major market events (about 800 GDELT files), keeping English, German,
   French, Italian and Spanish headlines from outlets in our regions.
2. For each topic, finds clear examples with the keyword patterns in
   config.yml and trains a classifier on them, with the keywords hidden so it
   must learn from the rest of the wording. Its probabilities come from
   cross-validation grouped by day, so no headline is scored by a model that
   saw it, or saw other headlines from the same day.
3. Measures tone for a sample of each topic's headlines with the slower model.
4. Writes a labelling sheet: headlines from every probability band for each
   topic, to be marked 1 (on the topic) or 0 (not).
5. If labels exist in data/labels/topic_labels.csv, estimates the precision of
   each band and recommends each topic's threshold.

Outputs (in data/checks/, kept as a download from the run):
  report.md          the same report shown on the run's summary page
  label_sheet.csv    headlines to label
  topic_scores.csv   every sampled headline with its keyword match and probability per topic
  topic_model.npz    the fitted classifiers, centering vector and thresholds

  python -m pipeline.calibrate_topics
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .check_headlines import FEEDS, normalize, read_file
from .common import DATA, load_config, warn
from .fetch_news import load_domain_regions
from .semantic import ScoreModel, ScreenModel, TopicKeywords, centered, tone_scores

OUT_DIR = DATA / "checks"
LABELS = DATA / "labels" / "topic_labels.csv"
# Headlines without keywords are grouped into bands by rank: the top 0.25% of
# classifier scores, the next 0.25%, and so on. The classifier's probabilities
# are only meaningful as a ranking (it never sees confirmed negatives), so
# thresholds are set by rank and checked with hand labels.
BAND_SHARES = [0.0025, 0.005, 0.01, 0.02, 0.04]   # cumulative share of non-keyword headlines
BAND_NAMES = ["top 0.25%", "0.25–0.5%", "0.5–1%", "1–2%", "2–4%"]
PROVISIONAL_SHARE = 0.01                          # used for tone and timing until labels exist
HEADLINES_PER_DAY = 40_000
BACKFILL_DAYS = 2_330


# ---------------------------------------------------------------- sample
def sample_days(cal: dict) -> list[date]:
    first = date.fromisoformat(str(cal["start"]))
    last = datetime.now(timezone.utc).date() - timedelta(days=2)
    days, month = set(), date(first.year, first.month, 1)
    while month <= last:
        for d in cal["days_per_month"]:
            day = date(month.year, month.month, d)
            if first <= day <= last:
                days.add(day)
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    days |= {date.fromisoformat(str(d)) for d in cal["extra_days"] if first <= date.fromisoformat(str(d)) <= last}
    return sorted(days)


def load_sample(config: dict) -> pd.DataFrame:
    news, sem = config["news"], config["semantic"]
    cal = sem["calibration"]
    domain_region = load_domain_regions(news)
    no_theme = re.compile(r"(?!x)x")              # theme flags are not used here
    days = sample_days(cal)
    jobs = [(f"{news['base_url']}{d:%Y%m%d}{slot}{suffix}", feed, d)
            for d in days for slot in cal["slots"] for feed, suffix in FEEDS.items()]
    print(f"Reading {len(jobs)} files from {len(days)} days…", flush=True)
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=news["threads"]) as pool:
        frames = [f for f in pool.map(lambda j: read_file(j[0], j[1], j[2], domain_region, no_theme), jobs)
                  if f is not None]
    print(f"  {len(frames)} files read in {time.perf_counter() - began:.0f} s", flush=True)
    if not frames:
        raise RuntimeError("no GDELT file could be read")

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["region"] != "other") & (df["title"].str.len() > 0) & df["language"].isin(sem["languages"])]
    df = df.assign(key=df["title"].map(normalize))
    df = df[(df["key"].str.split().str.len() >= 4) & (df["title"].str.lower() != df["source"])]
    df = df.drop_duplicates("key")
    if len(df) > cal["max_headlines"]:
        df = df.sample(cal["max_headlines"], random_state=0)
    return df[["day", "region", "feed", "language", "source", "title"]].reset_index(drop=True)


# ---------------------------------------------------------------- helpers
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def clip(text: str, n: int = 140) -> str:
    text = text.replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"


def examples(rows: pd.DataFrame, value: str, k: int, random: bool = False) -> list[str]:
    if rows.empty:
        return ["- (none)"]
    picks = rows.sample(min(k, len(rows)), random_state=1) if random else rows.head(k)
    return [f"- {r[value]:.2f} · {clip(r['title'])} ({r['language']})" for _, r in picks.iterrows()]


def fit_classifier(X: np.ndarray, y: np.ndarray, groups: np.ndarray, drop_share: float
                   ) -> tuple[np.ndarray, LogisticRegression]:
    """Out-of-fold scores (grouped by day) and a model fitted on everything.

    Headlines without keywords are not all off-topic: some are exactly the ones
    we want to find, and training them as negatives teaches the classifier to
    reject them. So training runs twice: the second round leaves out of the
    negatives the `drop_share` of keyword-free headlines that the first round
    found most similar to the topic, keeping only reliable negatives (a
    standard remedy when only positive examples are known)."""
    make = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups))

    def out_of_fold(keep: np.ndarray) -> np.ndarray:
        prob = np.zeros(len(y))
        for train, test in folds:
            train = train[keep[train]]
            prob[test] = make().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        return prob

    first = out_of_fold(np.ones(len(y), dtype=bool))
    cutoff = np.quantile(first[y == 0], 1 - drop_share)
    reliable = (y == 1) | (first < cutoff)
    return out_of_fold(reliable), make().fit(X[reliable], y[reliable])


def band_edges(prob: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Scores at which the bands start, highest first."""
    return np.array([np.quantile(prob[new], 1 - share) for share in BAND_SHARES])


def band_of(prob: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Band index (0 = top band), or -1 below the lowest band."""
    band = np.full(len(prob), -1)
    for b in range(len(edges) - 1, -1, -1):
        band[prob >= edges[b]] = b
    return band


# ---------------------------------------------------------------- main
def main() -> int:
    config = load_config()
    sem = config["semantic"]
    cal = sem["calibration"]
    topics = sem["topics"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample(config)
    titles = df["title"].tolist()
    print(f"{len(df):,} unique headlines in the sample", flush=True)

    screen = ScreenModel(sem["screen_model"])
    score = ScoreModel(sem["score_model"], sem["score_prefix"])
    screen.encode(titles[:256]); score.encode(titles[:32])        # warm-up

    began = time.perf_counter()
    raw = screen.encode(titles)
    screen_speed = len(titles) / max(time.perf_counter() - began, 1e-9)
    mean = raw.mean(axis=0)
    X = centered(raw, mean)
    del raw
    print(f"Screening model: {screen_speed:,.0f} headlines a second", flush=True)

    # Centering vector for the tone model, from a random subset of all headlines.
    background = df.sample(min(3000, len(df)), random_state=2)["title"].tolist()
    began = time.perf_counter()
    score_mean = score.encode(background).mean(axis=0)
    score_speed = len(background) / max(time.perf_counter() - began, 1e-9)
    print(f"Tone model: {score_speed:,.0f} headlines a second", flush=True)

    groups = df["day"].to_numpy()
    labels = pd.read_csv(LABELS) if LABELS.exists() else None
    model = {"mean": mean, "topics": np.array(list(topics))}
    summary, sections, sheet = [], [], []
    member_any = np.zeros(len(df), dtype=bool)

    for name, spec in topics.items():
        print(f"Topic {name}…", flush=True)
        keywords = TopicKeywords(spec)
        kw = keywords.matches(titles)
        df[f"{name}_kw"] = kw
        prob = np.full(len(df), np.nan)
        title_text = name.replace("_", " ").capitalize()
        section = [f"## {title_text}", ""]

        if kw.sum() < cal["min_examples"]:
            warn(f"{name}: only {kw.sum()} keyword matches, no classifier trained")
            section += [f"Only {kw.sum()} keyword matches: too few to train a classifier. "
                        "The topic uses keywords only until its patterns are widened.", ""]
            df[f"{name}_prob"] = prob
            model[f"{name}_threshold"] = np.float32(np.inf)
            member_any |= kw
            summary.append((name, int(kw.sum()), None, kw.mean(), None))
            sections.append(section)
            continue

        # Hide the keywords from the classifier in the clear examples.
        Xt = X.copy()
        idx = np.flatnonzero(kw)
        Xt[idx] = centered(screen.encode([keywords.mask(titles[i]) for i in idx]), mean)
        prob, fitted = fit_classifier(Xt, kw.astype(int), groups, cal["drop_share"])
        del Xt
        df[f"{name}_prob"] = prob
        new = ~kw
        edges = band_edges(prob, new)
        bands = band_of(prob, edges)
        model[f"{name}_coef"] = fitted.coef_[0].astype(np.float32)
        model[f"{name}_intercept"] = np.float32(fitted.intercept_[0])
        model[f"{name}_edges"] = edges.astype(np.float32)

        # How well the topic is recognised without its keywords: share of keyword
        # matches (with keywords hidden) scoring above the 1% band edge.
        provisional = edges[BAND_SHARES.index(PROVISIONAL_SHARE)]
        recall = float((prob[kw] >= provisional).mean())
        member = kw | (prob >= provisional)
        member_any |= member

        section += [f"Keyword matches: {kw.sum():,} ({kw.mean():.2%} of headlines). "
                    f"With their keywords hidden, the classifier still places {recall:.0%} of them above the score of the "
                    f"top 1% of keyword-free headlines ({provisional:.3f}).", ""]
        langs = sem["languages"]
        share = lambda mask: " · ".join(f"{l} {(df.loc[mask, 'language'] == l).mean():.0%}" for l in langs) if mask.any() else "–"
        top_new = new & (bands >= 0) & (bands <= 2)
        section += [f"Languages: all headlines {share(np.ones(len(df), bool))}; keyword matches {share(kw)}; "
                    f"top 1% of keyword-free headlines {share(top_new)}.", ""]

        view = df.assign(_p=prob)
        for b, band_name in enumerate(BAND_NAMES):
            rows = view[new & (bands == b)]
            k = 10 if b == 0 else 6
            section += [f"**Without keywords, {band_name}** (score from {edges[b]:.3f})", ""]
            section += examples(rows.sort_values("_p", ascending=False) if b == 0 else rows, "_p", k, random=b > 0) + [""]
        section += ["**Keyword matches the classifier doubts most** (possible keyword errors)", ""]
        section += examples(view[kw].sort_values("_p"), "_p", 6) + [""]

        # Tone
        pool = view[member]
        pool = pool.sample(min(cal["tone_sample"], len(pool)), random_state=3)
        if len(pool) >= 20:
            vectors = centered(score.encode(pool["title"].tolist()), score_mean)
            calm = centered(score.encode(spec["calm"]), score_mean)
            stress = centered(score.encode(spec["stress"]), score_mean)
            pool = pool.assign(_tone=tone_scores(vectors, calm, stress))
            ordered = pool.sort_values("_tone")
            section += [f"**Tone** (standard deviation {pool['_tone'].std():.3f}; positive means closer to stress). Most stressed:", ""]
            section += examples(ordered.iloc[::-1], "_tone", 5) + ["", "Calmest:", ""]
            section += examples(ordered, "_tone", 5) + [""]

        # Labelling sheet: keyword matches and each band of keyword-free headlines.
        rng = np.random.default_rng(4)
        to_label = [("keyword", np.flatnonzero(kw), 15)] + [
            (BAND_NAMES[b], np.flatnonzero(new & (bands == b)), 12) for b in range(len(BAND_NAMES))]
        for group, rows, k in to_label:
            for i in rng.choice(rows, size=min(k, len(rows)), replace=False):
                sheet.append({"topic": name, "group": group, "score": round(float(prob[i]), 4),
                              "language": df.at[i, "language"], "title": titles[i], "label": ""})

        # Evaluation against labels, if any.
        threshold = None
        if labels is not None:
            value = pd.to_numeric(labels["label"], errors="coerce")      # blank or other text: not labelled
            mine = labels[(labels["topic"] == name) & value.isin([0, 1])]
            if len(mine):
                y_lab = value[mine.index].astype(int).to_numpy()
                lab_titles = mine["title"].astype(str).tolist()
                lab_kw = keywords.matches(lab_titles)
                # Score labelled headlines the same way as the sample: with their
                # out-of-fold score when they are in it, otherwise with the final model.
                oof = dict(zip(titles, prob))
                missing = [t for t in lab_titles if t not in oof]
                if missing:
                    oof.update(zip(missing, fitted.predict_proba(centered(screen.encode(missing), mean))[:, 1]))
                lab_band = band_of(np.array([oof[t] for t in lab_titles]), edges)
                section += ["**Checked against your labels**", "",
                            "| Group | Labelled | On topic | Precision (95% interval) |", "|---|---|---|---|"]
                k_lab = y_lab[lab_kw]
                lo, hi = wilson(int(k_lab.sum()), len(k_lab))
                section.append(f"| keyword matches | {len(k_lab)} | {int(k_lab.sum())} | "
                               f"{(k_lab.mean() if len(k_lab) else float('nan')):.0%} ({lo:.0%}–{hi:.0%}) |")
                band_precision = {}
                for b, band_name in enumerate(BAND_NAMES):
                    yb = y_lab[(~lab_kw) & (lab_band == b)]
                    lo, hi = wilson(int(yb.sum()), len(yb))
                    band_precision[b] = (yb.mean() if len(yb) else float("nan"), len(yb))
                    section.append(f"| {band_name} | {len(yb)} | {int(yb.sum())} | "
                                   f"{(yb.mean() if len(yb) else float('nan')):.0%} ({lo:.0%}–{hi:.0%}) |")
                # Precision of all keyword-free additions down to a band, each band weighted by its size.
                counts = {b: int((new & (bands == b)).sum()) for b in range(len(BAND_NAMES))}
                for last in range(len(BAND_NAMES) - 1, -1, -1):
                    used = range(last + 1)
                    if any(band_precision[b][1] < 5 for b in used):
                        continue
                    precision = sum(counts[b] * band_precision[b][0] for b in used) / max(sum(counts[b] for b in used), 1)
                    if precision >= cal["target_precision"]:
                        threshold = float(edges[last])
                        section += ["", f"Recommended threshold: **{threshold:.4f}**, adding the keyword-free headlines down to "
                                    f"the {BAND_NAMES[last]} band (estimated precision of those additions {precision:.0%}).", ""]
                        break
                if threshold is None:
                    section += ["", f"No band reaches {cal['target_precision']:.0%} precision: the topic uses keywords only.", ""]
        model[f"{name}_threshold"] = np.float32(threshold if threshold is not None else np.inf)
        summary.append((name, int(kw.sum()), recall, member.mean(), threshold))
        sections.append(section)

    # ---------------------------------------------------------------- report
    lines = ["# Topic calibration", "",
             f"{len(df):,} unique headlines (English, German, French, Italian, Spanish) from {df['day'].nunique()} days, "
             f"{df['day'].min()} to {df['day'].max()}.", "",
             "| Topic | Keyword matches | Recognised without keywords | Share of headlines (keywords + top 1%) | Threshold from labels |",
             "|---|---|---|---|---|"]
    for name, n_kw, recall, share_member, threshold in summary:
        label = (f"{threshold:.4f}" if threshold is not None
                 else ("not labelled yet" if labels is None else "keywords only"))
        recall_text = "no classifier" if recall is None else f"{recall:.0%}"
        lines.append(f"| {name.replace('_', ' ')} | {n_kw:,} | {recall_text} | {share_member:.1%} | {label} |")

    union = member_any.mean()
    per_day = HEADLINES_PER_DAY / screen_speed + HEADLINES_PER_DAY * union / score_speed
    lines += ["", "## Speed", "",
              f"Screening model {screen_speed:,.0f} headlines a second; tone model {score_speed:,.0f}.",
              f"Taking keyword matches plus the top 1% of keyword-free headlines per topic, {union:.1%} of headlines belong to at least one topic and need a tone score. "
              f"For about {HEADLINES_PER_DAY:,} headlines a day that is roughly **{per_day / 60:.1f} minutes a day**; "
              f"a backfill of {BACKFILL_DAYS:,} days about **{BACKFILL_DAYS * per_day / 3600:.1f} hours** of computing "
              f"(downloading the files comes on top).", ""]
    for section in sections:
        lines += section
    report = "\n".join(lines)

    print(report)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    pd.DataFrame(sheet).to_csv(OUT_DIR / "label_sheet.csv", index=False)
    keep = ["day", "region", "language", "title"] + [c for n in topics for c in (f"{n}_kw", f"{n}_prob")]
    df[keep].to_csv(OUT_DIR / "topic_scores.csv", index=False, float_format="%.3f")
    np.savez(OUT_DIR / "topic_model.npz", **model)
    print(f"Saved report, labelling sheet ({len(sheet)} rows), scores and model to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
