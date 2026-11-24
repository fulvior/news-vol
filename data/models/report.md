# Model results

## VIX

**Explain** (2019-09-23 to 2026-11-23, 1775 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9315 | – | – | – | – | – |
| LHAR + attention | 0.9329 | +0.0014 | 0.0099 | – | – | – |
| LHAR + attention + tone | 0.9472 | +0.0158 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9479 | +0.0164 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9337 | +0.0023 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9362 | +0.0048 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- inflation attention: -0.0073 (95% bootstrap -0.0116 to -0.0024)
- equity_markets tone: +0.0396 (95% bootstrap +0.0336 to +0.0461)

**Forecast** (2019-09-23 to 2026-11-23, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9318 | – | – | – | – | – |
| LHAR + attention | 0.9321 | +0.0003 | 0.4747 | -0.0093 | 0.883 | 0.962 |
| LHAR + attention + tone | 0.9326 | +0.0008 | 0.0562 | -0.0162 | 0.930 | 0.964 |
| LHAR + attention, tone, dispersion, novelty | 0.9340 | +0.0022 | 0.0001 | -0.0383 | 0.998 | 0.996 |
| LHAR + attention + tone, without equity-market news | 0.9325 | +0.0007 | 0.0466 | -0.0151 | 0.944 | 0.966 |
| LHAR + GDELT theme shares and tone | 0.9325 | +0.0007 | 0.2627 | -0.0336 | 0.996 | 1.000 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## VSTOXX

**Explain** (2019-09-23 to 2026-11-23, 1775 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9400 | – | – | – | – | – |
| LHAR + attention | 0.9410 | +0.0010 | 0.0581 | – | – | – |
| LHAR + attention + tone | 0.9628 | +0.0227 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9634 | +0.0234 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9423 | +0.0022 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9502 | +0.0102 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0442 (95% bootstrap +0.0386 to +0.0499)

**Forecast** (2019-09-23 to 2026-11-23, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9404 | – | – | – | – | – |
| LHAR + attention | 0.9408 | +0.0004 | 0.2557 | -0.0253 | 0.967 | 0.963 |
| LHAR + attention + tone | 0.9412 | +0.0008 | 0.0230 | -0.0445 | 0.991 | 0.990 |
| LHAR + attention, tone, dispersion, novelty | 0.9418 | +0.0015 | 0.0093 | -0.0803 | 1.000 | 1.000 |
| LHAR + attention + tone, without equity-market news | 0.9409 | +0.0005 | 0.3102 | -0.0453 | 0.996 | 0.994 |
| LHAR + GDELT theme shares and tone | 0.9411 | +0.0007 | 0.0899 | -0.0435 | 0.998 | 0.994 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## S&P 500 realized volatility

**Explain** (2019-09-23 to 2026-11-23, 1775 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5210 | – | – | – | – | – |
| LHAR + attention | 0.5349 | +0.0139 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5813 | +0.0603 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5898 | +0.0688 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.5355 | +0.0145 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5595 | +0.0385 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- monetary_policy attention: +0.0393 (95% bootstrap +0.0226 to +0.0599)
- equity_markets attention: +0.0308 (95% bootstrap +0.0091 to +0.0494)
- equity_markets tone: +0.1347 (95% bootstrap +0.1120 to +0.1556)

**Forecast** (2019-09-23 to 2026-11-23, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5201 | – | – | – | – | – |
| LHAR + attention | 0.5237 | +0.0037 | 0.0263 | -0.0042 | 0.752 | 0.002 |
| LHAR + attention + tone | 0.5338 | +0.0137 | 0.0000 | +0.0103 | 0.163 | 0.000 |
| LHAR + attention, tone, dispersion, novelty | 0.5384 | +0.0183 | 0.0000 | -0.0104 | 0.786 | 0.127 |
| LHAR + attention + tone, without equity-market news | 0.5258 | +0.0058 | 0.0462 | -0.0073 | 0.812 | 0.063 |
| LHAR + GDELT theme shares and tone | 0.5325 | +0.0124 | 0.0000 | -0.0162 | 0.878 | 0.918 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0621 (95% bootstrap +0.0324 to +0.0889)

## EURO STOXX 50 realized volatility

**Explain** (2019-09-23 to 2026-11-23, 1775 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4838 | – | – | – | – | – |
| LHAR + attention | 0.4942 | +0.0104 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5372 | +0.0534 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5467 | +0.0629 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.4957 | +0.0119 | 0.0001 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5278 | +0.0440 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.1125 (95% bootstrap +0.0878 to +0.1368)

**Forecast** (2019-09-23 to 2026-11-23, 1778 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4831 | – | – | – | – | – |
| LHAR + attention | 0.4890 | +0.0059 | 0.0028 | -0.0179 | 0.956 | 0.458 |
| LHAR + attention + tone | 0.4976 | +0.0146 | 0.0000 | -0.0126 | 0.824 | 0.600 |
| LHAR + attention, tone, dispersion, novelty | 0.5025 | +0.0195 | 0.0000 | -0.0471 | 0.995 | 0.911 |
| LHAR + attention + tone, without equity-market news | 0.4940 | +0.0109 | 0.0000 | -0.0195 | 0.928 | 0.850 |
| LHAR + GDELT theme shares and tone | 0.4913 | +0.0082 | 0.0036 | -0.0176 | 0.953 | 0.021 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- growth_labor attention: +0.0293 (95% bootstrap +0.0109 to +0.0484)
- equity_markets tone: +0.0389 (95% bootstrap +0.0145 to +0.0614)
