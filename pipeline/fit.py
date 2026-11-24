"""Explain and forecast volatility with news.

For each instrument, models log volatility with the LHAR baseline (yesterday's
level and its weekly and monthly averages, plus past negative returns of the
region's stock index for the leverage effect) and adds news regressors:

  har              baseline only
  attention        + log share of headlines per semantic topic
  attention_tone   + semantic tone per topic                       (the main model)
  semantic_full    + dispersion and novelty per topic
  non_market       the main model without the equity-markets topic: news about
                   the market itself partly just reports the market's move, so
                   this isolates what the rest of the news adds
  gdelt_themes     + GDELT's own theme shares and tone (for comparison)

All models are fitted on the same days (from the first day with headlines), so
their fit can be compared directly. Two modes:

  explain    news of the trading day itself. Partly reports the market's own
             move, so it describes association, not cause.
  forecast   news up to the day before. Usable for forecasting; also evaluated
             out of sample.

Inference:
  - Newey-West (HAC) standard errors and Holm-adjusted p-values over the news
    coefficients of each model.
  - Stationary block bootstrap (days resampled in blocks, keeping the time
    dependence) for confidence intervals of the coefficients, of the gain in
    fit over the baseline, and of each topic's daily contribution.
  - Out of sample (forecast mode): expanding window refitted monthly, compared
    with the baseline by out-of-sample R², QLIKE loss and a Diebold-Mariano test.

Regressors: attention is log(share + floor). Tone, dispersion and novelty are
shrunk toward their usual level when a day has few headlines, with the weight
of `tone_prior` headlines. Every regressor is standardized, so coefficients are
per standard deviation. All transformation parameters are estimated on the
fitting sample only (in the out-of-sample exercise, on the training window).

Writes site/data/model_<instrument>.json and data/models/report.md.

  python -m pipeline.fit
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .common import DATA, ROOT, load_config, warn

FEATURES = DATA / "features"
OUT_SITE = ROOT / "site" / "data"
OUT_REPORT = DATA / "models" / "report.md"
# Baseline: HAR terms plus the leverage terms (LHAR). A stronger baseline means
# news only gets credit for what past volatility and past returns cannot explain.
HAR = ["har_d", "har_w", "har_m", "neg_d", "neg_w", "neg_m"]
K = 1 + len(HAR)                       # constant + baseline terms; news columns start here
MODES = {"explain": "", "forecast": "_prev"}
SPECS = {
    "har": ("LHAR baseline", []),
    "attention": ("LHAR + attention", ["att"]),
    "attention_tone": ("LHAR + attention + tone", ["att", "tone"]),
    "semantic_full": ("LHAR + attention, tone, dispersion, novelty", ["att", "tone", "disp", "nov"]),
    "non_market": ("LHAR + attention + tone, without equity-market news", ["att", "tone", "-equity_markets"]),
    "gdelt_themes": ("LHAR + GDELT theme shares and tone", ["gdelt"]),
}
MEASURE_NAMES = {"att": "attention", "tone": "tone", "disp": "dispersion", "nov": "novelty",
                 "gshare": "GDELT share", "gtone": "GDELT tone"}


# ---------------------------------------------------------------- regressors
class Design:
    """Builds the regressor matrix. Parameters (usual levels, means, standard
    deviations) are learned from the rows passed to fit() and then applied to
    any rows, so out-of-sample forecasts never use future information."""

    def __init__(self, measures: list[str], topics: list[str], themes: list[str], suffix: str, cfg: dict):
        self.suffix, self.cfg = suffix, cfg
        self.columns: list[tuple[str, str, str]] = []          # (name, group, measure)
        topics = [t for t in topics if f"-{t}" not in measures]
        measures = [m for m in measures if not m.startswith("-")]
        for m in measures:
            if m == "gdelt":
                for th in themes:
                    self.columns += [(f"gdelt_{th}_share", th, "gshare"), (f"gdelt_{th}_tone", th, "gtone")]
            else:
                for t in topics:
                    self.columns.append((f"{t}_{m}", t, m))

    def _raw(self, df: pd.DataFrame, prior: dict) -> pd.DataFrame:
        s, k, floor = self.suffix, self.cfg["tone_prior"], self.cfg["share_floor"]
        out = {}
        for name, group, measure in self.columns:
            if measure == "att":
                out[name] = np.log(df[f"sem_{group}_share{s}"] + floor)
            elif measure in ("tone", "disp", "nov"):
                n = df[f"sem_{group}_n{s}"].fillna(0)
                value = df[f"sem_{group}_{measure}{s}"]
                usual = prior.get(name, 0.0)
                out[name] = (n * value.fillna(usual) + k * usual) / (n + k)
            elif measure == "gshare":
                out[name] = np.log(df[f"{group}_share{s}"] + floor)
            elif measure == "gtone":
                out[name] = df[f"{group}_tone{s}"]
        return pd.DataFrame(out, index=df.index)

    def fit(self, df: pd.DataFrame) -> "Design":
        s = self.suffix
        self.prior = {}
        for name, group, measure in self.columns:
            if measure in ("tone", "disp", "nov"):
                n, v = df[f"sem_{group}_n{s}"].fillna(0), df[f"sem_{group}_{measure}{s}"]
                ok = v.notna() & (n > 0)
                self.prior[name] = float(np.average(v[ok], weights=n[ok])) if ok.any() else 0.0
        raw = self._raw(df, self.prior)
        self.mean = raw.mean()
        self.sd = raw.std().replace(0, 1).fillna(1)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(df), 0))
        raw = self._raw(df, self.prior)
        raw = raw.fillna(self.mean)                 # a theme's tone on a day without articles
        return ((raw - self.mean) / self.sd).to_numpy()

    @property
    def names(self) -> list[str]:
        return [c[0] for c in self.columns]


# ---------------------------------------------------------------- estimation
def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def r2(y: np.ndarray, resid: np.ndarray) -> float:
    return 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())


def hac_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West standard errors (Bartlett weights)."""
    return np.sqrt(np.clip(np.diag(hac_cov(X, resid, lags)), 0, None))


def hac_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    xu = X * resid[:, None]
    omega = xu.T @ xu
    for lag in range(1, lags + 1):
        g = xu[lag:].T @ xu[:-lag]
        omega += (1 - lag / (lags + 1)) * (g + g.T)
    bread = np.linalg.pinv(X.T @ X)
    return bread @ omega @ bread


def wald_news(beta: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Joint test that all news coefficients (after the constant and baseline terms) are zero."""
    b, V = beta[K:], cov[K:, K:]
    if len(b) == 0:
        return float("nan"), float("nan")
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, float(stats.chi2.sf(stat, len(b)))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def stationary_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length, wrapping around."""
    idx = np.empty(n, dtype=int)
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < 1 / block else (i + 1) % n
    return idx


def matrix(df: pd.DataFrame, design: Design) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df[HAR].to_numpy(), design.transform(df)])


# ---------------------------------------------------------------- one instrument
def fit_instrument(key: str, spec_cfg: dict, config: dict) -> dict | None:
    cfg = config["model"]
    topics = list(config["semantic"]["topics"])
    themes = list(config["news"]["themes"])
    path = FEATURES / f"{key}.csv"
    if not path.exists():
        warn(f"{key}: no feature table")
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df = df[df.index >= pd.Timestamp(cfg["semantic_start"])]

    result = {"id": key, "name": spec_cfg["name"], "kind": spec_cfg["kind"], "region": spec_cfg["news_region"],
              "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
              "topics": topics, "modes": {}}
    rng_seed = cfg["bootstrap"]["seed"]

    for mode, suffix in MODES.items():
        # Rows where every model can be estimated: the same sample for all.
        needed = ["log_vol"] + HAR + [f"sem_{t}_share{suffix}" for t in topics] + \
                 [f"{th}_share{suffix}" for th in themes]
        data = df.dropna(subset=needed)
        if len(data) < 300:
            warn(f"{key} {mode}: only {len(data)} usable days, skipped")
            continue
        y = data["log_vol"].to_numpy()
        n = len(y)
        designs = {sid: Design(measures, topics, themes, suffix, cfg).fit(data)
                   for sid, (_, measures) in SPECS.items()}
        Xs = {sid: matrix(data, d) for sid, d in designs.items()}

        # Point estimates and HAC inference.
        specs_out, fits = [], {}
        for sid, (label, _) in SPECS.items():
            X = Xs[sid]
            beta, resid = ols(X, y)
            cov = hac_cov(X, resid, cfg["hac_lags"])
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            wald, wald_p = wald_news(beta, cov)
            fits[sid] = (beta, resid)
            names = ["const"] + HAR + designs[sid].names
            z = beta / np.where(se > 0, se, np.nan)
            p = 2 * stats.norm.sf(np.abs(z))
            news = slice(K, None)
            p_holm = np.full_like(p, np.nan)
            if X.shape[1] > K:
                p_holm[news] = holm(np.nan_to_num(p[news], nan=1.0))
            specs_out.append({"id": sid, "label": label, "regressors": X.shape[1] - K,
                              "r2": r2(y, resid), "wald": wald, "wald_p": wald_p,
                              "names": names, "beta": beta, "se": se,
                              "p": p, "p_holm": p_holm})

        # Bootstrap: refit every model on the same resampled days.
        B, block = cfg["bootstrap"]["replications"], cfg["bootstrap"]["block_length"]
        rng = np.random.default_rng(rng_seed)
        boot_beta = {sid: np.empty((B, Xs[sid].shape[1])) for sid in SPECS}
        boot_gain = {sid: np.empty(B) for sid in SPECS}
        main = cfg["main_spec"]
        recent = min(cfg["recent_days"], n)
        Z_main = Xs[main][-recent:, K:]
        groups = [c[1] for c in designs[main].columns]
        boot_contrib = np.empty((B, recent, len(topics)))
        for b in range(B):
            idx = stationary_indices(n, block, rng)
            yb = y[idx]
            base_beta, base_resid = ols(Xs["har"][idx], yb)
            base_r2 = r2(yb, base_resid)
            for sid in SPECS:
                beta_b, resid_b = (base_beta, base_resid) if sid == "har" else ols(Xs[sid][idx], yb)
                boot_beta[sid][b] = beta_b
                boot_gain[sid][b] = r2(yb, resid_b) - base_r2
            news_beta = boot_beta[main][b, K:]
            for j, t in enumerate(topics):
                cols = [i for i, g in enumerate(groups) if g == t]
                boot_contrib[b, :, j] = Z_main[:, cols] @ news_beta[cols]

        base_r2 = specs_out[0]["r2"]
        for s in specs_out:
            lo, hi = np.percentile(boot_beta[s["id"]], [2.5, 97.5], axis=0)
            design = designs[s["id"]]
            meta = {c[0]: (c[1], c[2]) for c in design.columns}
            s["coefficients"] = [
                {"name": nm, "topic": meta.get(nm, (None, None))[0],
                 "measure": MEASURE_NAMES.get(meta.get(nm, (None, None))[1], nm),
                 "estimate": float(s["beta"][i]), "se_hac": float(s["se"][i]),
                 "p": float(s["p"][i]), "p_holm": None if np.isnan(s["p_holm"][i]) else float(s["p_holm"][i]),
                 "ci": [float(lo[i]), float(hi[i])]}
                for i, nm in enumerate(s["names"])]
            s["r2_gain"] = s["r2"] - base_r2
            # No interval for the gain: it cannot be negative for a larger nested model,
            # so bootstrap intervals for it are biased. Its significance is the joint
            # HAC Wald test, and its real value is judged out of sample.
            for k in ("names", "beta", "se", "p", "p_holm"):
                del s[k]

        # Daily contributions of each topic in the main model, as % change in volatility.
        beta_main = fits[main][0][K:]
        Z_all = Xs[main][:, K:]
        contrib = np.column_stack([Z_all[:, [i for i, g in enumerate(groups) if g == t]]
                                   @ beta_main[[i for i, g in enumerate(groups) if g == t]] for t in topics])
        pct = lambda x: 100 * (np.exp(x) - 1)
        c_lo, c_hi = np.percentile(boot_contrib, [2.5, 97.5], axis=0)
        fitted_har = Xs["har"] @ fits["har"][0]
        fitted_main = Xs[main] @ fits[main][0]
        mode_out = {
            "sample": {"start": data.index[0].strftime("%Y-%m-%d"), "end": data.index[-1].strftime("%Y-%m-%d"), "days": n},
            "specs": specs_out,
            "main_spec": main,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "vol": [round(float(v), 3) for v in data["vol"]],
            "fitted_har": [round(float(v), 3) for v in np.exp(fitted_har)],
            "fitted_main": [round(float(v), 3) for v in np.exp(fitted_main)],
            "contributions": {t: [round(float(v), 3) for v in pct(contrib[:, j])] for j, t in enumerate(topics)},
            "contributions_total": [round(float(v), 3) for v in pct(contrib.sum(axis=1))],
            "recent_ci": {t: {"low": [round(float(v), 3) for v in pct(c_lo[:, j])],
                              "high": [round(float(v), 3) for v in pct(c_hi[:, j])]} for j, t in enumerate(topics)},
            "recent_start": data.index[-recent].strftime("%Y-%m-%d"),
        }
        if mode == "forecast":
            mode_out["out_of_sample"] = out_of_sample(data, y, topics, themes, suffix, cfg)
        result["modes"][mode] = mode_out
        print(f"{key} {mode}: {n} days, R² baseline {base_r2:.3f}, "
              + ", ".join(f"{s['id']} +{s['r2_gain']:.4f}" for s in specs_out[1:]), flush=True)
    return result if result["modes"] else None


def out_of_sample(data: pd.DataFrame, y: np.ndarray, topics, themes, suffix, cfg) -> dict:
    oos = cfg["out_of_sample"]
    n = len(y)
    start = oos["initial_days"]
    if n - start < 100:
        return {"note": "too few days for an out-of-sample evaluation"}
    preds = {sid: np.full(n, np.nan) for sid in SPECS}
    for first in range(start, n, oos["refit_every"]):
        train, test = data.iloc[:first], data.iloc[first:first + oos["refit_every"]]
        for sid, (_, measures) in SPECS.items():
            design = Design(measures, topics, themes, suffix, cfg).fit(train)
            beta, _ = ols(matrix(train, design), y[:first])
            preds[sid][first:first + len(test)] = matrix(test, design) @ beta
    actual = y[start:]
    out = {"start": data.index[start].strftime("%Y-%m-%d"), "days": int(n - start), "specs": {}}
    loss = {}
    for sid in SPECS:
        f = preds[sid][start:]
        ratio = np.exp(2 * (actual - f))                    # realized / forecast variance
        loss[sid] = {"mse": (actual - f) ** 2, "qlike": ratio - np.log(ratio) - 1}
    for sid in SPECS:
        entry = {"mse": float(loss[sid]["mse"].mean()), "qlike": float(loss[sid]["qlike"].mean())}
        if sid != "har":
            entry["r2_oos"] = 1 - entry["mse"] / float(loss["har"]["mse"].mean())
            entry["qlike_ratio"] = entry["qlike"] / float(loss["har"]["qlike"].mean())
            for name in ("mse", "qlike"):
                # Diebold-Mariano: is the mean loss difference positive (news model better)?
                d = loss["har"][name] - loss[sid][name]
                se = float(hac_se(np.ones((len(d), 1)), d - d.mean(), oos["dm_lags"])[0])
                stat = d.mean() / se if se > 0 else float("nan")
                entry[f"dm_{name}"] = stat
                entry[f"dm_{name}_p"] = float(stats.norm.sf(stat))
        out["specs"][sid] = entry
    return out


# ---------------------------------------------------------------- report
def fmt(value, spec: str) -> str:
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def report(results: list[dict]) -> str:
    lines = ["# Model results", ""]
    for r in results:
        lines += [f"## {r['name']}", ""]
        for mode, m in r["modes"].items():
            oos = m.get("out_of_sample", {}).get("specs", {})
            lines += [f"**{mode.capitalize()}** ({m['sample']['start']} to {m['sample']['end']}, {m['sample']['days']} days)", "",
                      "| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |",
                      "|---|---|---|---|---|---|---|"]
            for s in m["specs"]:
                o = oos.get(s["id"], {})
                gain = "–" if s["id"] == "har" else f"{s['r2_gain']:+.4f}"
                lines.append(f"| {s['label']} | {s['r2']:.4f} | {gain} | {fmt(s.get('wald_p'), '.4f')} | {fmt(o.get('r2_oos'), '+.4f')} | "
                             f"{fmt(o.get('dm_mse_p'), '.3f')} | {fmt(o.get('dm_qlike_p'), '.3f')} |")
            main = next(s for s in m["specs"] if s["id"] == m["main_spec"])
            sig = [c for c in main["coefficients"] if c["topic"] and c["p_holm"] is not None and c["p_holm"] < 0.05]
            lines += ["", "Main model, news coefficients significant after Holm adjustment "
                      "(log volatility per standard deviation of the regressor):", ""]
            lines += [f"- {c['topic']} {c['measure']}: {c['estimate']:+.4f} "
                      f"(95% bootstrap {c['ci'][0]:+.4f} to {c['ci'][1]:+.4f})" for c in sig] or ["- none"]
            lines.append("")
    return "\n".join(lines)


def clean(value):
    """JSON cannot hold NaN: write it as null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> int:
    config = load_config()
    OUT_SITE.mkdir(parents=True, exist_ok=True)
    results = []
    for key, spec in config["instruments"].items():
        try:
            r = fit_instrument(key, spec, config)
        except Exception as error:                 # one instrument failing must not stop the others
            warn(f"{key}: model not fitted ({error})")
            continue
        if r is None:
            continue
        with open(OUT_SITE / f"model_{key}.json", "w", encoding="utf-8") as fh:
            json.dump(clean(r), fh, separators=(",", ":"), allow_nan=False)
        results.append(r)
    if not results:
        return 1
    text = report(results)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(text, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
