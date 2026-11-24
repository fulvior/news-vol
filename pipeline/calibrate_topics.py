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
LABELS_PER_BAND = 12                              # the labelling sheet tops each band up to this many labels
LABELS_KEYWORD = 15                               # and the keyword matches to this many
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


# ---------------------------------------------------------------- evaluation
def evaluate_labels(y: np.ndarray, is_kw: np.ndarray, band: np.ndarray, counts: dict, target: float
                    ) -> tuple[list[str], int | None, float | None]:
    """Precision per group from the labels, and the lowest band down to which the
    keyword-free additions keep the target precision (bands weighted by size)."""
    rows = ["| Group | Labelled | On topic | Precision (95% interval) |", "|---|---|---|---|"]
    k = y[is_kw]
    lo, hi = wilson(int(k.sum()), len(k))
    rows.append(f"| keyword matches | {len(k)} | {int(k.sum())} | "
                f"{(k.mean() if len(k) else float('nan')):.0%} ({lo:.0%}–{hi:.0%}) |")
    precision_by_band = {}
    for b, band_name in enumerate(BAND_NAMES):
        yb = y[(~is_kw) & (band == b)]
        lo, hi = wilson(int(yb.sum()), len(yb))
        precision_by_band[b] = (yb.mean() if len(yb) else float("nan"), len(yb))
        rows.append(f"| {band_name} | {len(yb)} | {int(yb.sum())} | "
                    f"{(yb.mean() if len(yb) else float('nan')):.0%} ({lo:.0%}–{hi:.0%}) |")
    for last in range(len(BAND_NAMES) - 1, -1, -1):
        used = range(last + 1)
        if any(precision_by_band[b][1] < 5 for b in used):
            continue
        precision = sum(counts[b] * precision_by_band[b][0] for b in used) / max(sum(counts[b] for b in used), 1)
        if precision >= target:
            return rows, last, precision
    return rows, None, None


# ---------------------------------------------------------------- main
def main() -> int:
    config = load_config()
    sem = config["semantic"]
    cal = sem["calibration"]
    topics = sem["topics"]
    kinds = cal["classifier_models"]            # which embeddings the classifiers are trained on
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample(config)
    titles = df["title"].tolist()
    print(f"{len(df):,} unique headlines in the sample", flush=True)

    encoders = {"screen": ScreenModel(sem["screen_model"]),
                "score": ScoreModel(sem["score_model"], sem["score_prefix"])}
    model_names = {"screen": sem["screen_model"], "score": sem["score_model"]}
    for enc in encoders.values():
        enc.encode(titles[:64])                   # warm-up

    # Embeddings: every headline with each model used by a classifier, centred.
    X, means, speed = {}, {}, {}
    for kind in ["screen", "score"]:
        texts = titles if kind in kinds else df.sample(min(3000, len(df)), random_state=2)["title"].tolist()
        print(f"Embedding {len(texts):,} headlines with the {kind} model…", flush=True)
        began = time.perf_counter()
        raw = encoders[kind].encode(texts)
        speed[kind] = len(texts) / max(time.perf_counter() - began, 1e-9)
        means[kind] = raw.mean(axis=0)
        if kind in kinds:
            X[kind] = centered(raw, means[kind])
        del raw
        print(f"  {speed[kind]:,.0f} headlines a second", flush=True)

    groups = df["day"].to_numpy()
    labels = pd.read_csv(LABELS) if LABELS.exists() else None
    if labels is not None:
        labels["value"] = pd.to_numeric(labels["label"], errors="coerce")
        labels = labels[labels["value"].isin([0, 1])]
    labelled = set() if labels is None else set(zip(labels["topic"], labels["title"]))

    model = {"topics": np.array(list(topics))}
    for kind in kinds:
        model[f"mean_{kind}"] = means[kind].astype(np.float32)
    model["mean_score"] = means["score"].astype(np.float32)
    summary, sections, sheet = [], [], []
    member_any = np.zeros(len(df), dtype=bool)
    rng = np.random.default_rng(4)

    for name, spec in topics.items():
        print(f"Topic {name}…", flush=True)
        keywords = TopicKeywords(spec)
        kw = keywords.matches(titles)
        new = ~kw
        df[f"{name}_kw"] = kw
        section = [f"## {name.replace('_', ' ').capitalize()}", "",
                   f"Keyword matches: {kw.sum():,} ({kw.mean():.2%} of headlines).", ""]
        mine = None if labels is None else labels[labels["topic"] == name]
        if mine is not None and len(mine):
            y_lab = mine["value"].astype(int).to_numpy()
            lab_titles = mine["title"].astype(str).tolist()
            lab_kw = keywords.matches(lab_titles)
            k = y_lab[lab_kw]
            lo, hi = wilson(int(k.sum()), len(k))
            section += [f"Keyword precision from your labels: {k.mean() if len(k) else float('nan'):.0%} "
                        f"({lo:.0%}–{hi:.0%}, {len(k)} labelled).", ""]

        langs = sem["languages"]
        share = lambda mask: " · ".join(f"{l} {(df.loc[mask, 'language'] == l).mean():.0%}" for l in langs) if mask.any() else "–"
        section += [f"Languages: all headlines {share(np.ones(len(df), bool))}; keyword matches {share(kw)}.", ""]
        section += ["**Keyword matches, random examples**", ""]
        section += examples(df[kw].assign(_p=1.0), "_p", 6, random=True) + [""]

        best = ("keywords", None, 0, None)          # kind, threshold, additions, precision
        provisional_member = kw.copy()
        if kw.sum() < cal["min_examples"]:
            warn(f"{name}: only {kw.sum()} keyword matches, no classifier trained")
            section += [f"Too few keyword matches to train a classifier: keywords only.", ""]
        else:
            for kind in kinds:
                Xt = X[kind].copy()
                idx = np.flatnonzero(kw)
                Xt[idx] = centered(encoders[kind].encode([keywords.mask(titles[i]) for i in idx]), means[kind])
                prob, fitted = fit_classifier(Xt, kw.astype(int), groups, cal["drop_share"])
                del Xt
                df[f"{name}_{kind}_prob"] = prob
                edges = band_edges(prob, new)
                bands = band_of(prob, edges)
                counts = {b: int((new & (bands == b)).sum()) for b in range(len(BAND_NAMES))}
                model[f"{name}_{kind}_coef"] = fitted.coef_[0].astype(np.float32)
                model[f"{name}_{kind}_intercept"] = np.float32(fitted.intercept_[0])
                model[f"{name}_{kind}_edges"] = edges.astype(np.float32)
                provisional = edges[BAND_SHARES.index(PROVISIONAL_SHARE)]
                recall = float((prob[kw] >= provisional).mean())
                if kind == kinds[-1]:
                    provisional_member |= prob >= provisional

                section += [f"### Classifier on the {kind} model ({model_names[kind]})", "",
                            f"With their keywords hidden, {recall:.0%} of keyword matches still score above "
                            f"the top 1% of keyword-free headlines.", ""]
                view = df.assign(_p=prob)
                for b in range(3):
                    section += [f"**Without keywords, {BAND_NAMES[b]}**", ""]
                    section += examples(view[new & (bands == b)], "_p", 6, random=True) + [""]

                have = {b: 0 for b in range(len(BAND_NAMES))}
                if mine is not None and len(mine):
                    oof = dict(zip(titles, prob))
                    missing = [t for t in lab_titles if t not in oof]
                    if missing:
                        oof.update(zip(missing, fitted.predict_proba(
                            centered(encoders[kind].encode(missing), means[kind]))[:, 1]))
                    lab_band = band_of(np.array([oof[t] for t in lab_titles]), edges)
                    have = {b: int(((~lab_kw) & (lab_band == b)).sum()) for b in range(len(BAND_NAMES))}
                    rows, last, precision = evaluate_labels(y_lab, lab_kw, lab_band, counts, cal["target_precision"])
                    section += ["Checked against your labels:", ""] + rows + [""]
                    if last is None:
                        section += [f"No band reaches {cal['target_precision']:.0%} precision (or too few labels "
                                    "in some bands; see the labelling sheet).", ""]
                    else:
                        additions = sum(counts[b] for b in range(last + 1))
                        section += [f"Reaches {cal['target_precision']:.0%} precision down to the {BAND_NAMES[last]} band: "
                                    f"{additions:,} keyword-free headlines added (estimated precision {precision:.0%}).", ""]
                        if additions > best[2]:
                            best = (kind, float(edges[last]), additions, precision)

                # Labelling sheet: tops each band up to LABELS_PER_BAND labelled headlines.
                for b, band_name in enumerate(BAND_NAMES):
                    need = max(0, LABELS_PER_BAND - have[b])
                    rows_b = [i for i in np.flatnonzero(new & (bands == b)) if (name, titles[i]) not in labelled]
                    for i in rng.choice(rows_b, size=min(need, len(rows_b)), replace=False) if rows_b and need else []:
                        labelled.add((name, titles[i]))
                        sheet.append({"topic": name, "group": f"{kind} {band_name}", "score": round(float(prob[i]), 4),
                                      "language": df.at[i, "language"], "title": titles[i], "label": ""})
        # Keyword matches, topped up so keyword precision stays measured as patterns change.
        have_kw = int(keywords.matches(mine["title"].astype(str).tolist()).sum()) if mine is not None and len(mine) else 0
        need_kw = max(0, LABELS_KEYWORD - have_kw)
        rows_k = [i for i in np.flatnonzero(kw) if (name, titles[i]) not in labelled]
        for i in rng.choice(rows_k, size=min(need_kw, len(rows_k)), replace=False) if rows_k and need_kw else []:
            labelled.add((name, titles[i]))
            sheet.append({"topic": name, "group": "keyword", "score": "",
                          "language": df.at[i, "language"], "title": titles[i], "label": ""})

        kind, threshold, additions, precision = best
        model[f"{name}_use"] = np.array(kind)
        model[f"{name}_threshold"] = np.float32(threshold if threshold is not None else np.inf)
        member = kw if kind == "keywords" else kw | (df[f"{name}_{kind}_prob"].to_numpy() >= threshold)
        member_any |= member if labels is not None else provisional_member
        section += [f"**Decision: {'keywords only' if kind == 'keywords' else f'keywords plus the {kind} classifier above {threshold:.4f}'}.**", ""]

        # Tone, on the topic's headlines.
        pool = df[member if labels is not None else provisional_member]
        pool = pool.sample(min(cal["tone_sample"], len(pool)), random_state=3)
        if len(pool) >= 20:
            score = encoders["score"]
            vectors = centered(score.encode(pool["title"].tolist()), means["score"])
            calm = centered(score.encode(spec["calm"]), means["score"])
            stress = centered(score.encode(spec["stress"]), means["score"])
            pool = pool.assign(_tone=tone_scores(vectors, calm, stress))
            ordered = pool.sort_values("_tone")
            section += [f"**Tone** (standard deviation {pool['_tone'].std():.3f}; positive means closer to stress). Most stressed:", ""]
            section += examples(ordered.iloc[::-1], "_tone", 5) + ["", "Calmest:", ""]
            section += examples(ordered, "_tone", 5) + [""]

        summary.append((name, int(kw.sum()), kind, threshold, additions, member.mean()))
        sections.append(section)

    # ---------------------------------------------------------------- report
    lines = ["# Topic calibration", "",
             f"{len(df):,} unique headlines (English, German, French, Italian, Spanish) from {df['day'].nunique()} days, "
             f"{df['day'].min()} to {df['day'].max()}. "
             + ("Labels: none yet." if labels is None else f"Labels: {len(labels):,}."), "",
             "| Topic | Keyword matches | Decision | Keyword-free headlines added | Share of headlines in the topic |",
             "|---|---|---|---|---|"]
    for name, n_kw, kind, threshold, additions, share_member in summary:
        decision = "keywords only" if kind == "keywords" else f"{kind} classifier ≥ {threshold:.4f}"
        if labels is None:
            decision = "not labelled yet"
        lines.append(f"| {name.replace('_', ' ')} | {n_kw:,} | {decision} | {additions:,} | {share_member:.1%} |")

    union = member_any.mean()
    uses_score = any(kind == "score" for _, _, kind, *_ in summary)
    per_day = (HEADLINES_PER_DAY / speed["screen"]
               + (HEADLINES_PER_DAY / speed["score"] if uses_score else HEADLINES_PER_DAY * union / speed["score"]))
    lines += ["", "## Speed", "",
              f"Screen model {speed['screen']:,.0f} headlines a second; score model {speed['score']:,.0f}.",
              f"{union:.1%} of headlines belong to at least one topic. "
              + ("Some topics use the score model's classifier, so every headline goes through it. "
                 if uses_score else "Only topic headlines go through the score model, for tone. ")
              + f"For about {HEADLINES_PER_DAY:,} headlines a day that is roughly **{per_day / 60:.1f} minutes a day**; "
              f"a backfill of {BACKFILL_DAYS:,} days about **{BACKFILL_DAYS * per_day / 3600:.0f} hours** of computing "
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

    pd.DataFrame(sheet, columns=["topic", "group", "score", "language", "title", "label"]).to_csv(
        OUT_DIR / "label_sheet.csv", index=False)
    keep = ["day", "region", "language", "title"] + [c for c in df.columns if c.endswith(("_kw", "_prob"))]
    df[keep].to_csv(OUT_DIR / "topic_scores.csv", index=False, float_format="%.4f")
    np.savez(OUT_DIR / "topic_model.npz", **model)
    print(f"Saved report, labelling sheet ({len(sheet)} new rows), scores and model to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
