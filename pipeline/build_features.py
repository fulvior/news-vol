"""Line up volatility and news into one table per instrument.

Reads data/vol/<instrument>.csv and data/news/gdelt_<year>.csv and writes
data/features/<instrument>.csv with one row per trading day:

  vol, log_vol          the volatility measure, in annualized percentage points
  har_d, har_w, har_m   yesterday's log volatility and its 5- and 22-day averages
                        (the standard HAR baseline; all three use past values only)
  ret                   the day's log return of the region's stock index, in %
  neg_d, neg_w, neg_m   past negative returns (yesterday, 5- and 22-day averages):
                        the leverage effect, added to the baseline in the LHAR model
  <theme>_share         share of the region's articles carrying the theme, in %,
  <theme>_tone          and their average tone, on the trading day itself
  <theme>_share_prev    the same two measures over the days from the previous
  <theme>_tone_prev     trading day up to the day before (so weekends are included)

From the semantic layer (headlines from August 2019, see fetch_semantic.py), per topic:
  sem_<topic>_n         number of headlines on the topic
  sem_<topic>_share     share of the region's headlines on the topic, in %
  sem_<topic>_tone      average semantic tone (positive: worsening news, negative: easing)
  sem_<topic>_disp      dispersion of tone across the topic's headlines
  sem_<topic>_nov       novelty: how far the day's coverage moved from the previous 20 days
  ..._prev              the same over the days since the previous trading day

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


def leverage_terms(price_key: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Past negative returns of the region's stock index (in %): falls in prices raise
    volatility more than rises lower it (the leverage effect). Daily, weekly and
    monthly averages of yesterday's and earlier negative returns, as in the LHAR model."""
    prices = pd.read_csv(DATA / "vol" / f"{price_key}.csv", parse_dates=["date"]).set_index("date")["close"]
    ret = 100 * np.log(prices / prices.shift(1))
    neg = ret.clip(upper=0).shift(1)
    out = pd.DataFrame({"ret": ret,
                        "neg_d": neg,
                        "neg_w": neg.rolling(5, min_periods=3).mean(),
                        "neg_m": neg.rolling(22, min_periods=15).mean()})
    return out.reindex(index)


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


# ---------------------------------------------------------------- semantic layer
SEMANTIC = DATA / "semantic"
NOVELTY_DAYS = 20


def load_semantic() -> pd.DataFrame:
    """Daily semantic series with a novelty column added from the stored centroids."""
    files = sorted(SEMANTIC.glob("series_*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "region", "topic", "headlines", "total", "tone", "tone_sd", "novelty"])
    series = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    series = series.drop_duplicates(["date", "region", "topic"], keep="last")

    keys, vectors = [], []
    for f in sorted(SEMANTIC.glob("centroids_*.npz")):
        with np.load(f) as z:
            if len(z["keys"]):
                keys += z["keys"].tolist()
                vectors.append(z["vectors"].astype(np.float32))
    novelty = {}
    if keys:
        vectors = np.concatenate(vectors)
        cent = pd.DataFrame({"key": keys})
        parts = cent["key"].str.split("|", expand=True)
        cent["date"], cent["region"], cent["topic"] = pd.to_datetime(parts[0]), parts[1], parts[2]
        weights = series.set_index(["date", "region", "topic"])["headlines"]
        for (region, topic), g in cent.groupby(["region", "topic"]):
            g = g.sort_values("date")
            dates = g["date"].to_numpy()
            vec = vectors[g.index.to_numpy()]
            w = np.array([weights.get((d, region, topic), 1) for d in g["date"]], dtype=float)
            for i in range(len(g)):
                past = (dates < dates[i]) & (dates >= dates[i] - np.timedelta64(NOVELTY_DAYS, "D"))
                if past.sum() < 5:
                    continue
                ref = (vec[past] * w[past, None]).sum(axis=0)
                norm = np.linalg.norm(ref)
                if norm > 0:
                    novelty[(g["date"].iloc[i], region, topic)] = 1 - float(vec[i] @ ref / norm)
    series["novelty"] = [novelty.get((d, r, t), np.nan) for d, r, t in
                         zip(series["date"], series["region"], series["topic"])]
    return series


def semantic_features(sem: pd.DataFrame, region: str, topics: list[str],
                      trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=trading_days)
    names = [f"sem_{t}_{m}{suffix}" for t in topics for suffix in ("", "_prev")
             for m in ("n", "share", "tone", "disp", "nov")]
    regional = sem[sem["region"] == region]
    if regional.empty:
        for name in names:
            out[name] = np.nan
        return out

    n = regional["headlines"].fillna(0)
    tone = regional["tone"].fillna(0)
    regional = regional.assign(
        tone_sum=n * tone,
        sq_sum=n * (regional["tone_sd"].fillna(0) ** 2 + tone ** 2),
        nov_n=np.where(regional["novelty"].notna(), n, 0),
        nov_sum=n * regional["novelty"].fillna(0))
    calendar = pd.date_range(regional["date"].min(), max(regional["date"].max(), trading_days.max()))
    wide = lambda column: (regional.pivot(index="date", columns="topic", values=column)
                           .reindex(index=calendar, columns=topics).fillna(0))
    tables = {c: wide(c) for c in ("headlines", "tone_sum", "sq_sum", "nov_n", "nov_sum")}
    total = regional.groupby("date")["total"].first().reindex(calendar).fillna(0).to_frame("total")

    previous = trading_days.to_series().shift(1)
    previous.iloc[0] = trading_days[0] - pd.Timedelta(days=1)
    windows = {"": (trading_days, trading_days),
               "_prev": (pd.DatetimeIndex(previous), trading_days - pd.Timedelta(days=1))}
    with np.errstate(invalid="ignore", divide="ignore"):
        for suffix, (first, last) in windows.items():
            t_sum = window_sum(total, first, last)["total"].to_numpy()
            t_sum = np.where(t_sum > 0, t_sum, np.nan)
            sums = {c: window_sum(tab, first, last) for c, tab in tables.items()}
            for t in topics:
                count = sums["headlines"][t].to_numpy()
                has = np.where(count > 0, count, np.nan)
                mean = sums["tone_sum"][t].to_numpy() / has
                out[f"sem_{t}_n{suffix}"] = np.where(np.isnan(t_sum), np.nan, count)   # headlines behind the measures
                out[f"sem_{t}_share{suffix}"] = 100 * count / t_sum
                out[f"sem_{t}_tone{suffix}"] = mean
                out[f"sem_{t}_disp{suffix}"] = np.sqrt(np.clip(sums["sq_sum"][t].to_numpy() / has - mean ** 2, 0, None))
                nov_n = sums["nov_n"][t].to_numpy()
                out[f"sem_{t}_nov{suffix}"] = sums["nov_sum"][t].to_numpy() / np.where(nov_n > 0, nov_n, np.nan)
    return out


def main() -> int:
    config = load_config()
    themes = list(config["news"]["themes"])
    news = load_news()
    topics = list(config["semantic"]["topics"])
    semantic = load_semantic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for key, spec in config["instruments"].items():
        if not (DATA / "vol" / f"{key}.csv").exists():
            warn(f"{key}: no volatility file yet, skipped")
            continue
        table = load_volatility(key, spec["kind"])
        price_key = spec.get("returns_from", key)
        if (DATA / "vol" / f"{price_key}.csv").exists():
            table = table.join(leverage_terms(price_key, table.index))
        table = table.join(news_features(news, spec["news_region"], themes, table.index))
        table = table.join(semantic_features(semantic, spec["news_region"], topics, table.index))
        table = table.dropna(subset=["vol"])
        table.to_csv(OUT_DIR / f"{key}.csv", float_format="%.5f", date_format="%Y-%m-%d")
        with_news = int(table[f"{themes[0]}_share"].notna().sum())
        print(f"{key}: {len(table)} trading days, {with_news} with news, "
              f"last {table.index[-1]:%Y-%m-%d}")
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
