# Model results

## VIX

**Explain** (2019-09-23 to 2026-11-24, 1776 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9315 | – | – | – | – | – |
| LHAR + attention | 0.9329 | +0.0014 | 0.0098 | – | – | – |
| LHAR + attention + tone | 0.9473 | +0.0157 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9479 | +0.0164 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9338 | +0.0023 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9363 | +0.0048 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- inflation attention: -0.0073 (95% bootstrap -0.0119 to -0.0025)
- equity_markets tone: +0.0396 (95% bootstrap +0.0340 to +0.0464)

**Forecast** (2019-09-23 to 2026-11-24, 1779 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9319 | – | – | – | – | – |
| LHAR + attention | 0.9322 | +0.0003 | 0.4736 | -0.0092 | 0.883 | 0.961 |
| LHAR + attention + tone | 0.9327 | +0.0008 | 0.0554 | -0.0162 | 0.930 | 0.964 |
| LHAR + attention, tone, dispersion, novelty | 0.9341 | +0.0022 | 0.0001 | -0.0383 | 0.998 | 0.996 |
| LHAR + attention + tone, without equity-market news | 0.9326 | +0.0007 | 0.0459 | -0.0151 | 0.943 | 0.966 |
| LHAR + GDELT theme shares and tone | 0.9326 | +0.0007 | 0.2615 | -0.0336 | 0.996 | 1.000 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## VSTOXX

**Explain** (2019-09-23 to 2026-11-24, 1776 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9400 | – | – | – | – | – |
| LHAR + attention | 0.9410 | +0.0010 | 0.0532 | – | – | – |
| LHAR + attention + tone | 0.9628 | +0.0227 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.9634 | +0.0233 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.9423 | +0.0023 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.9502 | +0.0102 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0441 (95% bootstrap +0.0389 to +0.0497)

**Forecast** (2019-09-23 to 2026-11-24, 1779 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.9403 | – | – | – | – | – |
| LHAR + attention | 0.9407 | +0.0004 | 0.2608 | -0.0254 | 0.968 | 0.963 |
| LHAR + attention + tone | 0.9411 | +0.0008 | 0.0229 | -0.0446 | 0.992 | 0.990 |
| LHAR + attention, tone, dispersion, novelty | 0.9418 | +0.0015 | 0.0093 | -0.0805 | 1.000 | 1.000 |
| LHAR + attention + tone, without equity-market news | 0.9409 | +0.0005 | 0.2898 | -0.0451 | 0.996 | 0.994 |
| LHAR + GDELT theme shares and tone | 0.9411 | +0.0007 | 0.0866 | -0.0435 | 0.998 | 0.994 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- none

## S&P 500 realized volatility

**Explain** (2019-09-23 to 2026-11-24, 1776 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5208 | – | – | – | – | – |
| LHAR + attention | 0.5347 | +0.0139 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5811 | +0.0603 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5896 | +0.0688 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.5354 | +0.0146 | 0.0000 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5595 | +0.0387 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- monetary_policy attention: +0.0394 (95% bootstrap +0.0223 to +0.0596)
- equity_markets attention: +0.0306 (95% bootstrap +0.0106 to +0.0504)
- equity_markets tone: +0.1346 (95% bootstrap +0.1132 to +0.1551)

**Forecast** (2019-09-23 to 2026-11-24, 1779 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.5199 | – | – | – | – | – |
| LHAR + attention | 0.5236 | +0.0037 | 0.0267 | -0.0041 | 0.751 | 0.002 |
| LHAR + attention + tone | 0.5337 | +0.0138 | 0.0000 | +0.0106 | 0.156 | 0.000 |
| LHAR + attention, tone, dispersion, novelty | 0.5383 | +0.0184 | 0.0000 | -0.0101 | 0.780 | 0.125 |
| LHAR + attention + tone, without equity-market news | 0.5258 | +0.0059 | 0.0406 | -0.0069 | 0.801 | 0.061 |
| LHAR + GDELT theme shares and tone | 0.5324 | +0.0125 | 0.0000 | -0.0160 | 0.876 | 0.917 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.0622 (95% bootstrap +0.0326 to +0.0899)

## EURO STOXX 50 realized volatility

**Explain** (2019-09-23 to 2026-11-24, 1776 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4836 | – | – | – | – | – |
| LHAR + attention | 0.4940 | +0.0104 | 0.0000 | – | – | – |
| LHAR + attention + tone | 0.5371 | +0.0535 | 0.0000 | – | – | – |
| LHAR + attention, tone, dispersion, novelty | 0.5466 | +0.0630 | 0.0000 | – | – | – |
| LHAR + attention + tone, without equity-market news | 0.4956 | +0.0120 | 0.0001 | – | – | – |
| LHAR + GDELT theme shares and tone | 0.5276 | +0.0440 | 0.0000 | – | – | – |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- equity_markets tone: +0.1126 (95% bootstrap +0.0891 to +0.1348)

**Forecast** (2019-09-23 to 2026-11-24, 1779 days)

| Model | R² | Gain over baseline | News jointly zero, p (HAC Wald) | Out-of-sample R² vs baseline | DM p (MSE) | DM p (QLIKE) |
|---|---|---|---|---|---|---|
| LHAR baseline | 0.4829 | – | – | – | – | – |
| LHAR + attention | 0.4889 | +0.0060 | 0.0025 | -0.0177 | 0.954 | 0.454 |
| LHAR + attention + tone | 0.4975 | +0.0146 | 0.0000 | -0.0124 | 0.820 | 0.598 |
| LHAR + attention, tone, dispersion, novelty | 0.5025 | +0.0196 | 0.0000 | -0.0468 | 0.995 | 0.910 |
| LHAR + attention + tone, without equity-market news | 0.4939 | +0.0110 | 0.0000 | -0.0194 | 0.926 | 0.849 |
| LHAR + GDELT theme shares and tone | 0.4912 | +0.0083 | 0.0034 | -0.0175 | 0.951 | 0.021 |

Main model, news coefficients significant after Holm adjustment (log volatility per standard deviation of the regressor):

- growth_labor attention: +0.0294 (95% bootstrap +0.0114 to +0.0502)
- equity_markets tone: +0.0388 (95% bootstrap +0.0127 to +0.0612)
