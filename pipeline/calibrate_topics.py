"""Calibrate the semantic topics on a sample of real headlines.

Reads the headlines saved by check_headlines (data/checks/headline_sample.csv),
scores every one against each topic in config.yml with both models, and reports:

  - for each topic, example headlines at falling levels of relevance, so we can
    see where "about inflation" stops being true and set the cut-off there
  - whether the fast screening model keeps the headlines the careful model
    rates most relevant (if it doesn't, the two-stage design needs changing)
  - whether non-English headlines are matched as readily as English ones
  - the calmest and most stressed headlines per topic, as a check on tone
  - how fast each model runs, and what that implies for the daily job and
    the backfill

The report goes to the run's summary page on GitHub (and to the terminal); all
scores are saved to data/checks/topic_scores.csv.

  python -m pipeline.calibrate_topics
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from .common import DATA, load_config

SAMPLE = DATA / "checks" / "headline_sample.csv"
OUT = DATA / "checks" / "topic_scores.csv"

# Rough size of the real workload, from the headline check: about 20,000
# headlines per region per day at the pipeline's sampling rate.
HEADLINES_PER_DAY = 40_000
BACKFILL_DAYS = 2_330          # July 2019 to November 2026
PARALLEL_JOBS = 5              # the backfill workflow runs five jobs at a time


# ---------------------------------------------------------------- the models
class FakeEncoder:
    """Stand-in used only for testing this script offline (NEWSVOL_FAKE_MODELS=1):
    hashes character trigrams into a vector, so similar spellings score high."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for i in range(len(t) - 2):
                out[row, hash(t[i:i + 3]) % self.dims] += 1
        return out


class ScreenEncoder:
    def __init__(self, name: str):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(name, force_download=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=2048)


class ScoreEncoder:
    def __init__(self, name: str, prefix: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cpu")
        self.prefix = prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + t for t in texts], batch_size=128)


def normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def embed(encoder, texts: list[str]) -> tuple[np.ndarray, float]:
    """Unit-length embeddings and the speed in texts per second."""
    began = time.perf_counter()
    vectors = normalized(np.asarray(encoder.encode(texts), dtype=np.float32))
    return vectors, len(texts) / max(time.perf_counter() - began, 1e-9)


def max_similarity(vectors: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return (vectors @ anchors.T).max(axis=1)


def mean_similarity(vectors: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return (vectors @ anchors.T).mean(axis=1)


# ---------------------------------------------------------------- report helpers
def clip(text: str, n: int = 140) -> str:
    text = text.replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"


def example_lines(rows: pd.DataFrame, score_col: str) -> list[str]:
    return [f"- {r[score_col]:.3f} · {clip(r['title'])} ({r['language']})" for _, r in rows.iterrows()]


def main() -> int:
    if not SAMPLE.exists():
        print(f"{SAMPLE} not found. Upload headline_sample.csv from the 'Check headlines' run "
              f"to data/checks/ in the repository first.")
        return 1

    config = load_config()
    cfg = config["semantic"]
    topics = cfg["topics"]

    df = pd.read_csv(SAMPLE).dropna(subset=["title"])
    df = df.drop_duplicates("title").reset_index(drop=True)
    titles = df["title"].tolist()
    print(f"{len(df):,} unique headlines loaded", flush=True)

    if os.environ.get("NEWSVOL_FAKE_MODELS"):
        screen, score = FakeEncoder(), FakeEncoder(384)
    else:
        print(f"Loading {cfg['screen_model']} and {cfg['score_model']}…", flush=True)
        screen = ScreenEncoder(cfg["screen_model"])
        score = ScoreEncoder(cfg["score_model"], cfg["score_prefix"])

    # Warm-up so the speed measurement does not include one-off start-up costs.
    screen.encode(titles[:64]); score.encode(titles[:64])

    print("Embedding with the screening model…", flush=True)
    screen_vec, screen_speed = embed(screen, titles)
    print(f"  {screen_speed:,.0f} headlines a second", flush=True)
    print("Embedding with the scoring model…", flush=True)
    score_vec, score_speed = embed(score, titles)
    print(f"  {score_speed:,.0f} headlines a second", flush=True)

    share = cfg["screen_share"]
    n = len(df)
    passed_any = np.zeros(n, dtype=bool)
    lines = ["# Topic calibration", "",
             f"{n:,} unique headlines from the headline check, scored against {len(topics)} topics.",
             "Scores are cosine similarities to the topic's anchor sentences (the closest one counts).",
             "Read each topic's bands from the top down and note where headlines stop being about the topic: "
             "that is where its cut-off belongs.", ""]
    summary_rows = []

    for name, spec in topics.items():
        a_screen, _ = embed(screen, spec["anchors"])
        a_score, _ = embed(score, spec["anchors"])
        calm, _ = embed(score, spec["calm"])
        stress, _ = embed(score, spec["stress"])

        df[f"{name}_screen"] = max_similarity(screen_vec, a_screen)
        df[f"{name}_score"] = max_similarity(score_vec, a_score)
        df[f"{name}_tone"] = mean_similarity(score_vec, stress) - mean_similarity(score_vec, calm)

        screen_rank = df[f"{name}_screen"].rank(ascending=False, pct=True)
        score_rank = df[f"{name}_score"].rank(ascending=False, pct=True)
        passed_any |= (screen_rank <= share).to_numpy()

        # Does the fast model keep what the careful model ranks highest?
        best = score_rank <= 0.01
        recall = {s: (screen_rank[best] <= s).mean() for s in (0.05, 0.10, 0.20)}

        # Language mix: top 2% by the careful model against the whole sample.
        top = df[score_rank <= 0.02]
        english_all = (df["language"] == "eng").mean()
        english_top = (top["language"] == "eng").mean() if len(top) else float("nan")
        summary_rows.append((name, recall, english_top, english_all))

        lines += [f"## {name.replace('_', ' ').capitalize()}", ""]
        bands = [(0.0, 0.005, 10, "Top 0.5%"), (0.005, 0.01, 6, "0.5–1%"), (0.01, 0.02, 6, "1–2%"),
                 (0.02, 0.04, 6, "2–4%"), (0.04, 0.08, 6, "4–8%")]
        ordered = df.assign(_rank=score_rank).sort_values(f"{name}_score", ascending=False)
        for lo, hi, k, label in bands:
            band = ordered[(ordered["_rank"] > lo) & (ordered["_rank"] <= hi)]
            if band.empty:
                continue
            picks = band.head(k) if lo == 0 else band.sample(min(k, len(band)), random_state=1)
            picks = picks.sort_values(f"{name}_score", ascending=False)
            lines += [f"**{label}** (score {band[f'{name}_score'].min():.3f} to {band[f'{name}_score'].max():.3f})", ""]
            lines += example_lines(picks, f"{name}_score") + [""]

        if len(top) >= 8:
            lines += ["**Tone among the top 2%: most stressed**", ""]
            lines += example_lines(top.nlargest(4, f"{name}_tone"), f"{name}_tone") + [""]
            lines += ["**Tone among the top 2%: calmest**", ""]
            lines += example_lines(top.nsmallest(4, f"{name}_tone"), f"{name}_tone") + [""]

    # ---- the two-stage design and the cost
    head = ["## Does the screening model keep the right headlines?", "",
            "Of the 1% most relevant headlines according to the careful model, the share the fast model "
            "ranks in its own top 5%, 10% and 20%. High values in the 5% column mean screening is safe.", "",
            "| Topic | Top 5% | Top 10% | Top 20% | English among top 2% | English overall |",
            "|---|---|---|---|---|---|"]
    for name, recall, english_top, english_all in summary_rows:
        head.append(f"| {name.replace('_', ' ')} | {recall[0.05]:.0%} | {recall[0.10]:.0%} | {recall[0.20]:.0%} | "
                    f"{english_top:.0%} | {english_all:.0%} |")

    union = passed_any.mean()
    per_day = HEADLINES_PER_DAY / screen_speed + HEADLINES_PER_DAY * union / score_speed
    everything = HEADLINES_PER_DAY / score_speed
    head += ["", "## Speed on this machine", "",
             "| | Headlines a second |", "|---|---|",
             f"| Screening model | {screen_speed:,.0f} |", f"| Scoring model | {score_speed:,.0f} |", "",
             f"Passing the top {share:.0%} for each topic sends {union:.0%} of headlines to the scoring model.",
             f"For about {HEADLINES_PER_DAY:,} headlines a day that is roughly **{per_day / 60:.1f} minutes a day**, "
             f"against {everything / 60:.1f} minutes if every headline went through the scoring model.",
             f"A backfill of {BACKFILL_DAYS:,} days would take about **{BACKFILL_DAYS * per_day / 3600:.0f} hours** "
             f"of machine time, or {BACKFILL_DAYS * per_day / 3600 / PARALLEL_JOBS:.0f} hours with "
             f"{PARALLEL_JOBS} jobs in parallel (downloading the files comes on top).", ""]

    report = "\n".join(lines[:6] + head + lines[6:])
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    print(f"Saved all scores to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
