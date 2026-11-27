# Model results

## VIX

**Explain** (2019-09-23 to 2026-11-25, 1777 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9315 | – | – | – | – | – |
| LHAR + attention | 0.9330 | +0.0014 | 0.0096 | – | – | – |
| LHAR + attention + tone | 0.9473 | +0.0158 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9479 | +0.0164 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9338 | +0.0022 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9363 | +0.0047 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- inflation attention: -0.0073 (95% bootstrap -0.0117 to -0.0022)
- equity_markets tone: +0.0397 (95% bootstrap +0.0337 to +0.0460)

**Forecast** (2019-09-23 to 2026-11-25, 1780 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9319 | – | – | – | – | – |
| LHAR + attention | 0.9322 | +0.0003 | 0.4809 | -0.0094 | 0.886 | 0.962 |
| LHAR + attention + tone | 0.9327 | +0.0008 | 0.0556 | -0.0163 | 0.930 | 0.964 |
| LHAR + attention, tone, dispersion, novelty | 0.9341 | +0.0022 | 0.0001 | -0.0382 | 0.998 | 0.996 |
| LHAR + attention + tone, without equity-market news | 0.9326 | +0.0007 | 0.0456 | -0.0151 | 0.943 | 0.966 |
| LHAR + GDELT theme shares and tone | 0.9326 | +0.0007 | 0.2591 | -0.0335 | 0.996 | 1.000 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## VSTOXX

**Explain** (2019-09-23 to 2026-11-26, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9398 | – | – | – | – | – |
| LHAR + attention | 0.9408 | +0.0010 | 0.0440 | – | – | – |
| LHAR + attention + tone | 0.9626 | +0.0228 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9633 | +0.0235 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9420 | +0.0023 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9501 | +0.0103 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0442 (95% bootstrap +0.0390 to +0.0501)

**Forecast** (2019-09-23 to 2026-11-26, 1781 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9401 | – | – | – | – | – |
| LHAR + attention | 0.9405 | +0.0004 | 0.2687 | -0.0254 | 0.969 | 0.965 |
| LHAR + attention + tone | 0.9409 | +0.0008 | 0.0324 | -0.0445 | 0.992 | 0.990 |
| LHAR + attention, tone, dispersion, novelty | 0.9415 | +0.0015 | 0.0103 | -0.0802 | 1.000 | 1.000 |
| LHAR + attention + tone, without equity-market news | 0.9406 | +0.0005 | 0.2715 | -0.0444 | 0.996 | 0.994 |
| LHAR + GDELT theme shares and tone | 0.9408 | +0.0007 | 0.0800 | -0.0432 | 0.998 | 0.994 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## S&P 500 realized volatility

**Explain** (2019-09-23 to 2026-11-25, 1777 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5212 | – | – | – | – | – |
| LHAR + attention | 0.5352 | +0.0140 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5818 | +0.0605 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5903 | +0.0691 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.5357 | +0.0144 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5597 | +0.0384 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- monetary_policy attention: +0.0393 (95% bootstrap +0.0229 to +0.0593)
- equity_markets attention: +0.0308 (95% bootstrap +0.0090 to +0.0501)
- equity_markets tone: +0.1353 (95% bootstrap +0.1138 to +0.1553)

**Forecast** (2019-09-23 to 2026-11-25, 1780 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5203 | – | – | – | – | – |
| LHAR + attention | 0.5240 | +0.0037 | 0.0268 | -0.0042 | 0.752 | 0.002 |
| LHAR + attention + tone | 0.5340 | +0.0137 | 0.0000 | +0.0104 | 0.160 | 0.000 |
| LHAR + attention, tone, dispersion, novelty | 0.5387 | +0.0184 | 0.0000 | -0.0101 | 0.780 | 0.125 |
| LHAR + attention + tone, without equity-market news | 0.5261 | +0.0058 | 0.0451 | -0.0072 | 0.810 | 0.066 |
| LHAR + GDELT theme shares and tone | 0.5328 | +0.0125 | 0.0000 | -0.0161 | 0.877 | 0.918 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0622 (95% bootstrap +0.0319 to +0.0893)

## EURO STOXX 50 realized volatility

**Explain** (2019-09-23 to 2026-11-26, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4839 | – | – | – | – | – |
| LHAR + attention | 0.4944 | +0.0105 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5375 | +0.0536 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5469 | +0.0630 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.4960 | +0.0121 | 0.0001 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5280 | +0.0441 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.1125 (95% bootstrap +0.0892 to +0.1366)

**Forecast** (2019-09-23 to 2026-11-26, 1781 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4831 | – | – | – | – | – |
| LHAR + attention | 0.4890 | +0.0058 | 0.0031 | -0.0181 | 0.958 | 0.465 |
| LHAR + attention + tone | 0.4975 | +0.0143 | 0.0000 | -0.0133 | 0.836 | 0.620 |
| LHAR + attention, tone, dispersion, novelty | 0.5025 | +0.0193 | 0.0000 | -0.0474 | 0.996 | 0.917 |
| LHAR + attention + tone, without equity-market news | 0.4940 | +0.0108 | 0.0000 | -0.0198 | 0.931 | 0.855 |
| LHAR + GDELT theme shares and tone | 0.4914 | +0.0083 | 0.0034 | -0.0175 | 0.952 | 0.021 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0384 (95% bootstrap +0.0138 to +0.0613)
