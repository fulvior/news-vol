"""Write the JSON files the website reads.

  site/data/index.json                   instruments, GDELT themes and semantic topics
  site/data/<instrument>.json            dates, volatility, and news share and tone per theme
  site/data/headlines/<region>_<year>.json
      the most stressed and calmest headlines per topic and day, shown on the
      site for the day a visitor selects (loaded only when needed)

The model results (site/data/model_<instrument>.json) are written by fit.py.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from .common import DATA, ROOT, load_config, warn

SITE_DATA = ROOT / "site" / "data"


def as_list(series: pd.Series, digits: int) -> list:
    """Rounded values, with gaps written as null."""
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def label(key: str) -> str:
    return key.replace("_", " ").capitalize()


TOPIC_NAMES = {"inflation": "Inflation", "monetary_policy": "Monetary policy", "equity_markets": "Equity markets",
               "trade_tariffs": "Trade and tariffs", "energy": "Energy", "geopolitics": "Geopolitics",
               "banking_stress": "Banking stress", "growth_labor": "Growth and jobs"}


def export_headlines() -> int:
    """One file per region and year: {date: {topic: {"s": [[tone, source, title], …], "c": […]}}}."""
    files = sorted((DATA / "semantic").glob("headlines_*.csv"))
    if not files:
        return 0
    h = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in files], ignore_index=True)
    out_dir = SITE_DATA / "headlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (region, year), g in h.groupby([h["region"], h["date"].str[:4]]):
        payload: dict = {}
        for row in g.itertuples(index=False):
            entry = payload.setdefault(row.date, {}).setdefault(row.topic, {"s": [], "c": []})
            entry["s" if row.kind == "stressed" else "c"].append([round(float(row.tone), 3), row.source, row.title])
        with open(out_dir / f"{region}_{year}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), ensure_ascii=False)
        written += 1
    return written


def main() -> int:
    config = load_config()
    themes = list(config["news"]["themes"])
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    instruments, latest = [], ""

    for key, spec in config["instruments"].items():
        path = DATA / "features" / f"{key}.csv"
        if not path.exists():
            warn(f"{key}: no feature table yet, skipped")
            continue
        table = pd.read_csv(path)
        payload = {
            "id": key,
            "name": spec["name"],
            "dates": table["date"].tolist(),
            "vol": as_list(table["vol"], 2),
            "share": {t: as_list(table[f"{t}_share"], 3) for t in themes},
            "tone": {t: as_list(table[f"{t}_tone"], 3) for t in themes},
        }
        with open(SITE_DATA / f"{key}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        instruments.append({"id": key, "name": spec["name"], "kind": spec["kind"],
                            "region": spec["news_region"]})
        latest = max(latest, table["date"].iloc[-1])
        print(f"{key}: {len(table)} days exported")

    index = {
        "updated": latest,
        "instruments": instruments,
        "themes": [{"id": t, "name": label(t)} for t in themes],
        "topics": [{"id": t, "name": TOPIC_NAMES.get(t, label(t))} for t in config.get("semantic", {}).get("topics", {})],
        "regions": {r: label(r) if r != "us" else "United States" for r in config["news"]["regions"]},
    }
    with open(SITE_DATA / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print(f"Headline files: {export_headlines()}")
    return 0 if instruments else 1


if __name__ == "__main__":
    sys.exit(main())
