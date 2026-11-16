"""Shared paths, config loading, and a polite HTTP helper."""
from __future__ import annotations

import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG_PATH = Path(__file__).with_name("config.yml")

HEADERS = {"User-Agent": "news-vol research pipeline (github.com/fulvior)"}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def warn(message: str) -> None:
    """Print a warning that GitHub Actions shows as an annotation on the run."""
    print(f"::warning::{message}", flush=True)


def http_get(url: str, params: dict | None = None, retries: int = 4,
             backoff: float = 15.0, timeout: float = 90.0) -> requests.Response:
    """GET with retries. Waits longer after each failure or rate-limit reply."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_error}")
