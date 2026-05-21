# A/B benchmark — canonical SOGA suite, classic vs --sparse-truncate

- Programs evaluated: 24 (21 OK, 3 failed/skipped)
- N runs per mode: 3, warmup: 1, per-program timeout (classic probe): 60.0s
- Wall time of the whole benchmark: 230.7s

## Aggregate speedup (SOGA-only, excludes preprocessing and CFG construction)

- Geometric mean: **1.73x**
- Median: **1.41x**
- Range: 0.87x to 4.22x
- Programs where sparse is >5% faster: 16
- Programs essentially even (within ±5%): 3
- Programs where sparse is >5% slower: 2

## Per-program results (sorted by SOGA-only speedup, descending)

| Program | d | n_comp | setup ms | classic ms | sparse ms | speedup | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOGA/RandomWalkGauss10.soga | 1 | 1024 | 2.2 | 651.87 | 154.34 | **4.22x** | 0.00e+00 |
| SOGA/ClinicalTrial.soga | 5 | 5795 | 4.8 | 5734.01 | 1568.78 | **3.66x** | 4.44e-16 |
| SOGA/Bernoulli.soga | 2 | 1954 | 4491.6 | 1959.22 | 537.95 | **3.64x** | 6.66e-16 |
| Example/Bernoulli.soga | 2 | 1954 | 3.0 | 1937.92 | 545.44 | **3.55x** | 6.66e-16 |
| SOGA/SurveyUnbias.soga | 4 | 128 | 3.3 | 122.30 | 36.63 | **3.34x** | 2.22e-16 |
| SOGA/CoinBias.soga | 2 | 64 | 6602.3 | 63.42 | 20.49 | **3.09x** | 3.33e-16 |
| SOGA/IndianGPA.soga | 4 | 11 | 8567.6 | 24.69 | 10.62 | **2.32x** | 0.00e+00 |
| SOGA/BernoulliPrune.soga | 2 | 27 | 2.9 | 257.23 | 117.84 | **2.18x** | 2.22e-16 |
| SOGA/ClinicalTrialPrune.soga | 5 | 23 | 4.8 | 264.09 | 123.47 | **2.14x** | 4.44e-16 |
| SOGA/Scale.soga | 3 | 1 | 2.9 | 2.67 | 1.58 | **1.70x** | 0.00e+00 |
| SOGA/ClickGraphPrune.soga | 6 | 35 | 4.8 | 1264.31 | 894.03 | **1.41x** | 2.22e-16 |
| SOGA/BayesPointMachine.soga | 9 | 1 | 10.0 | 7.22 | 5.60 | **1.29x** | 8.70e-14 |
| SOGA/RandomWalkDisc10.soga | 1 | 1024 | 2.7 | 100.18 | 84.11 | **1.19x** | 0.00e+00 |
| SOGA/TrueSkills.soga | 6 | 1 | 2.3 | 2.06 | 1.76 | **1.17x** | 4.26e-14 |
| SOGA/NoisyOr.soga | 10 | 256 | 5.9 | 25.80 | 22.15 | **1.16x** | 0.00e+00 |
| SOGA/MurderMistery.soga | 2 | 2 | 2.0 | 1.20 | 1.07 | **1.13x** | 0.00e+00 |
| SOGA/Grass.soga | 10 | 28 | 7.4 | 11.00 | 10.86 | **1.01x** | 0.00e+00 |
| SOGA/ThreeCoins.soga | 3 | 4 | 1.6 | 1.11 | 1.10 | **1.00x** | 0.00e+00 |
| SOGA/Burglar.soga | 6 | 4 | 5.6 | 4.42 | 4.58 | **0.97x** | 0.00e+00 |
| SOGA/Toy.soga | 1 | 2 | 9330.3 | 0.27 | 0.30 | **0.90x** | 0.00e+00 |
| SOGA/TwoCoins.soga | 3 | 3 | 2.1 | 1.26 | 1.45 | **0.87x** | 0.00e+00 |

**Failures / skipped:**
- `SOGA/ClickGraph.soga`: timeout in classic start_SOGA (>60.0s) (d=?)
- `SOGA/DigitRecognition.soga`: error: TypeError: list indices must be integers or slices, not NoneType (d=?)
- `SOGA/RandomWalkUnif10.soga`: timeout in classic start_SOGA (>60.0s) (d=?)
