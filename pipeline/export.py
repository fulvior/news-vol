"""Write the JSON files the website reads.

  site/data/index.json          the list of instruments and news themes
  site/data/<instrument>.json   dates, volatility, and news share and tone per theme

Model results (fit, drivers) will be added to these files once fit.py exists.
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
        "regions": {r: label(r) if r != "us" else "United States" for r in config["news"]["regions"]},
    }
    with open(SITE_DATA / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return 0 if instruments else 1


if __name__ == "__main__":
    sys.exit(main())
