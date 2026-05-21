# A/B/C benchmark — canonical SOGA suite, classic vs --sparse-truncate vs --vectorize-truncate

- Programs: 24, runs per mode: 3, warmup: 1, hard timeout: 60.0s
- Wall: 590.9s

## Aggregate (geometric mean over OK programs)

- sparse over classic: **1.51x**
- vectorize over classic: **3.17x**
- vectorize over sparse: **2.10x**
- vectorize over sparse, median: 1.40x
- vectorize over sparse, max: 12.66x
- max relative diff E[var] across modes: 2.78e-07 (equivalence check)

## Per-program results (sorted by vec/sparse speedup desc)

| Program | d | n_comp | classic ms | sparse ms | vectorize ms | sp/cl | vec/cl | vec/sp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SOGA/ClinicalTrial.soga | 5 | 5795 | 5636.13 | 1546.70 | 122.17 | 3.64x | 46.13x | **12.66x** |
| Example/Bernoulli.soga | 2 | 1954 | 1972.04 | 565.52 | 49.40 | 3.49x | 39.92x | **11.45x** |
| SOGA/Bernoulli.soga | 2 | 1954 | 1899.92 | 524.38 | 47.26 | 3.62x | 40.20x | **11.10x** |
| SOGA/IndianGPA.soga | 4 | 11 | 30.24 | 31.65 | 6.11 | 0.96x | 4.95x | **5.18x** |
| SOGA/RandomWalkGauss10.soga | 1 | 1024 | 690.21 | 192.15 | 45.87 | 3.59x | 15.05x | **4.19x** |
| SOGA/SurveyUnbias.soga | 4 | 128 | 138.40 | 40.97 | 10.52 | 3.38x | 13.15x | **3.89x** |
| SOGA/Scale.soga | 3 | 1 | 3.33 | 6.45 | 1.74 | 0.52x | 1.91x | **3.70x** |
| SOGA/CoinBias.soga | 2 | 64 | 63.16 | 20.52 | 7.59 | 3.08x | 8.32x | **2.70x** |
| SOGA/RandomWalkDisc10.soga | 1 | 1024 | 114.83 | 90.74 | 46.82 | 1.27x | 2.45x | **1.94x** |
| SOGA/ClinicalTrialPrune.soga | 5 | 23 | 275.56 | 144.09 | 90.52 | 1.91x | 3.04x | **1.59x** |
| SOGA/BernoulliPrune.soga | 2 | 27 | 257.68 | 127.68 | 85.43 | 2.02x | 3.02x | **1.49x** |
| SOGA/NoisyOr.soga | 10 | 256 | 24.07 | 25.31 | 19.45 | 0.95x | 1.24x | **1.30x** |
| SOGA/Grass.soga | 10 | 28 | 10.93 | 10.64 | 8.24 | 1.03x | 1.33x | **1.29x** |
| SOGA/ThreeCoins.soga | 3 | 4 | 1.20 | 1.46 | 1.31 | 0.83x | 0.92x | **1.11x** |
| SOGA/ClickGraphPrune.soga | 6 | 35 | 1271.32 | 922.41 | 847.21 | 1.38x | 1.50x | **1.09x** |
| SOGA/Burglar.soga | 6 | 4 | 4.43 | 4.43 | 4.13 | 1.00x | 1.07x | **1.07x** |
| SOGA/TrueSkills.soga | 6 | 1 | 2.37 | 1.86 | 1.76 | 1.28x | 1.35x | **1.06x** |
| SOGA/Toy.soga | 1 | 2 | 0.30 | 0.29 | 0.28 | 1.04x | 1.07x | **1.03x** |
| SOGA/TwoCoins.soga | 3 | 3 | 1.28 | 1.35 | 1.32 | 0.95x | 0.97x | **1.02x** |
| SOGA/BayesPointMachine.soga | 9 | 1 | 6.66 | 5.31 | 5.59 | 1.26x | 1.19x | **0.95x** |
| SOGA/MurderMistery.soga | 2 | 2 | 1.38 | 1.09 | 1.35 | 1.27x | 1.02x | **0.81x** |

**Failed / partial:**
- `SOGA/ClickGraph.soga`: classic=timeout warmup, sparse=timeout warmup, vectorize=timeout warmup
- `SOGA/DigitRecognition.soga`: classic=error: TypeError: list indices must be integers or slices, not NoneType, sparse=error: TypeError: list indices must be integers or slices, not NoneType, vectorize=error: TypeError: list indices must be integers or slices, not NoneType
- `SOGA/RandomWalkUnif10.soga`: classic=timeout warmup, sparse=ok, vectorize=ok
