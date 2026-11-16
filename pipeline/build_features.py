"""Line up volatility and news into one table per instrument.

Reads data/vol/<instrument>.csv and data/news/gdelt_<year>.csv and writes
data/features/<instrument>.csv with one row per trading day:

  vol, log_vol          the volatility measure, in annualized percentage points
  har_d, har_w, har_m   yesterday's log volatility and its 5- and 22-day averages
                        (the standard HAR baseline; all three use past values only)
  <theme>_share         share of the region's articles carrying the theme, in %,
  <theme>_tone          and their average tone, on the trading day itself
  <theme>_share_prev    the same two measures over the days from the previous
  <theme>_tone_prev     trading day up to the day before (so weekends are included)

Timing: news is bucketed by UTC day. The "_prev" columns only use days that
ended before the trading day began, so they are safe for forecasting. The
same-day columns include a few hours after the market close (about 3 hours for
New York, 7 for Frankfurt) and are for explaining, not forecasting.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import DATA, load_config, warn

OUT_DIR = DATA / "features"


def load_volatility(key: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / "vol" / f"{key}.csv", parse_dates=["date"]).set_index("date")
    if kind == "implied":
        vol = df["close"]
    else:
        # Garman-Klass estimate of the day's variance from open, high, low, close.
        variance = (0.5 * np.log(df["high"] / df["low"]) ** 2
                    - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2)
        vol = 100 * np.sqrt(252 * variance.where(variance > 0))
    out = pd.DataFrame({"vol": vol.where(vol > 0)})
    out["log_vol"] = np.log(out["vol"])
    lagged = out["log_vol"].shift(1)
    out["har_d"] = lagged
    out["har_w"] = lagged.rolling(5, min_periods=3).mean()
    out["har_m"] = lagged.rolling(22, min_periods=15).mean()
    return out


def load_news() -> pd.DataFrame:
    files = sorted((DATA / "news").glob("gdelt_*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "region", "theme", "articles", "total", "tone"])
    news = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    return news.drop_duplicates(["date", "region", "theme"], keep="last")


def window_sum(daily: pd.DataFrame, first: pd.DatetimeIndex, last: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum the calendar-day table over [first, last] for each pair of dates."""
    cumulative = daily.cumsum()
    upper = cumulative.reindex(last, method="ffill").fillna(0).to_numpy()
    lower = cumulative.reindex(first - pd.Timedelta(days=1), method="ffill").fillna(0).to_numpy()
    return pd.DataFrame(upper - lower, columns=daily.columns)


def news_features(news: pd.DataFrame, region: str, themes: list[str],
                  trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=trading_days)
    regional = news[news["region"] == region]
    if regional.empty:
        for theme in themes:
            for column in ("share", "tone", "share_prev", "tone_prev"):
                out[f"{theme}_{column}"] = np.nan
        return out

    regional = regional.assign(tone_sum=regional["tone"].fillna(0) * regional["articles"])
    calendar = pd.date_range(regional["date"].min(), max(regional["date"].max(), trading_days.max()))
    wide = lambda column: (regional.pivot(index="date", columns="theme", values=column)
                           .reindex(index=calendar, columns=themes).fillna(0))
    articles, tone_sum = wide("articles"), wide("tone_sum")
    total = regional.groupby("date")["total"].first().reindex(calendar).fillna(0).to_frame("total")

    previous = trading_days.to_series().shift(1)
    previous.iloc[0] = trading_days[0] - pd.Timedelta(days=1)
    previous = pd.DatetimeIndex(previous)
    windows = {
        "": (trading_days, trading_days),                               # the day itself
        "_prev": (previous, trading_days - pd.Timedelta(days=1)),       # since the last trading day
    }
    for suffix, (first, last) in windows.items():
        a = window_sum(articles, first, last)
        s = window_sum(tone_sum, first, last)
        t = window_sum(total, first, last)["total"].to_numpy()
        t = np.where(t > 0, t, np.nan)   # no news stored for that window
        for theme in themes:
            count = a[theme].to_numpy()
            out[f"{theme}_share{suffix}"] = 100 * count / t
            out[f"{theme}_tone{suffix}"] = np.where(count > 0, s[theme].to_numpy() / np.where(count > 0, count, 1), np.nan)
            out.loc[np.isnan(t), f"{theme}_tone{suffix}"] = np.nan
    return out


def main() -> int:
    config = load_config()
    themes = list(config["news"]["themes"])
    news = load_news()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for key, spec in config["instruments"].items():
        if not (DATA / "vol" / f"{key}.csv").exists():
            warn(f"{key}: no volatility file yet, skipped")
            continue
        table = load_volatility(key, spec["kind"])
        table = table.join(news_features(news, spec["news_region"], themes, table.index))
        table = table.dropna(subset=["vol"])
        table.to_csv(OUT_DIR / f"{key}.csv", float_format="%.5f", date_format="%Y-%m-%d")
        with_news = int(table[f"{themes[0]}_share"].notna().sum())
        print(f"{key}: {len(table)} trading days, {with_news} with news, "
              f"last {table.index[-1]:%Y-%m-%d}")
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
