"""Download volatility indices and prices into data/vol/<instrument>.csv.

Every source publishes its full history in one small file, so each run simply
re-downloads everything and overwrites. That keeps the files self-healing if a
provider revises a past value.

Output columns: date, open, high, low, close (open/high/low empty if the source
only publishes a close).
"""
from __future__ import annotations

import sys
from io import StringIO

import pandas as pd

from .common import DATA, http_get, load_config, warn

COLUMNS = ["date", "open", "high", "low", "close"]


def parse_cboe(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    out = pd.DataFrame({"date": pd.to_datetime(df["date"])})
    if "close" in df.columns:
        for col in ("open", "high", "low", "close"):
            out[col] = pd.to_numeric(df.get(col), errors="coerce")
    else:  # some Cboe index files only carry the date and one value column
        out["close"] = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    return out


def parse_stoxx(text: str) -> pd.DataFrame:
    # Expected layout: Date;Symbol;Indexvalue with dates as DD.MM.YYYY
    df = pd.read_csv(StringIO(text), sep=";")
    return pd.DataFrame({
        "date": pd.to_datetime(df.iloc[:, 0], format="%d.%m.%Y"),
        "close": pd.to_numeric(df.iloc[:, -1], errors="coerce"),
    })


def fetch_yahoo(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf  # imported here so the other sources work without it

    hist = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    hist.index = pd.to_datetime(hist.index.date)
    out = hist[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    return out.rename_axis("date").reset_index()


def fetch_instrument(spec: dict, start: str) -> pd.DataFrame:
    source = spec["source"]
    if source == "cboe":
        return parse_cboe(http_get(spec["url"]).text)
    if source == "stoxx":
        return parse_stoxx(http_get(spec["url"]).text)
    if source == "yahoo":
        return fetch_yahoo(spec["ticker"], start)
    raise ValueError(f"unknown source '{source}'")


def clean(df: pd.DataFrame, start: str) -> pd.DataFrame:
    df = df.reindex(columns=COLUMNS)
    df = df.dropna(subset=["date", "close"])
    df = df[df["date"] >= pd.Timestamp(start)]
    # Drop today's row: during market hours it is a partial day, not a close.
    df = df[df["date"] < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def main() -> int:
    config = load_config()
    start = config["start_date"]
    out_dir = DATA / "vol"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for key, spec in config["instruments"].items():
        path = out_dir / f"{key}.csv"
        try:
            try:
                df = fetch_instrument(spec, start)
            except Exception as error:
                if not spec.get("fallback_ticker"):
                    raise
                warn(f"{key}: primary source failed ({error}); using Yahoo fallback")
                df = fetch_yahoo(spec["fallback_ticker"], start)
            df = clean(df, start)

            # Never replace a good file with a truncated download.
            old_rows = len(pd.read_csv(path)) if path.exists() else 0
            if len(df) < old_rows:
                raise RuntimeError(f"got {len(df)} rows, existing file has {old_rows}")

            df.to_csv(path, index=False)
            print(f"{key}: {len(df)} rows, last date {df['date'].iloc[-1]}")
        except Exception as error:
            failures += 1
            warn(f"{key}: not updated ({error})")

    # One broken source should not block the rest; fail only if nothing worked.
    return 1 if failures == len(config["instruments"]) else 0


if __name__ == "__main__":
    sys.exit(main())
