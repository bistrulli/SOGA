# Sigma structure analysis — per-program summary

Run on the canonical SOGA suite with --sparse-truncate --vectorize-truncate active.

Snapshots of (mu, Sigma, pi) captured after every state/observe/merge/prune node.

Metrics computed per component (top 20 by weight):

- eff_rank_ratio: effective rank / d  (fraction of singular values above 1% of the largest)
- sparsity: fraction of off-diagonal entries effectively zero
- blocks_max: max # connected components in the sparsity graph (1 = single dense block)
- lr_err_k: Frobenius relative error of best rank-k approximation

| Program | d range | n_snap | eff_rank/d | sparsity | blocks_max | lr_err k=1 | k=2 | k=5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SOGA/BayesPointMachine.soga | 9–9 | 23 | 0.22 | 0.58 | 9 | 0.061 | 0.000 | 0.000 |
| SOGA/Bernoulli.soga | 2–2 | 50 | 0.50 | 1.00 | 2 | 0.000 | 0.000 | 0.000 |
| SOGA/BernoulliPrune.soga | 2–2 | 50 | 0.50 | 1.00 | 2 | 0.000 | 0.000 | 0.000 |
| SOGA/Burglar.soga | 6–6 | 50 | 0.00 | 1.00 | 6 | 0.000 | 0.000 | 0.000 |
| SOGA/ClickGraph.soga | (status: timeout) | | | | | | | |
| SOGA/ClickGraphPrune.soga | 6–6 | 50 | 0.17 | 1.00 | 6 | 0.000 | 0.000 | 0.000 |
| SOGA/ClinicalTrial.soga | 5–5 | 50 | 0.40 | 0.97 | 5 | 0.704 | 0.000 | 0.000 |
| SOGA/ClinicalTrialPrune.soga | 5–5 | 50 | 0.40 | 0.97 | 5 | 0.704 | 0.000 | 0.000 |
| SOGA/CoinBias.soga | 2–2 | 50 | 0.50 | 1.00 | 2 | 0.000 | 0.000 | 0.000 |
| SOGA/DigitRecognition.soga | (status: error: TypeError) | | | | | | | |
| SOGA/Grass.soga | 10–10 | 50 | 0.00 | 1.00 | 10 | 0.000 | 0.000 | 0.000 |
| SOGA/IndianGPA.soga | 4–4 | 50 | 0.25 | 1.00 | 4 | 0.000 | 0.000 | 0.000 |
| SOGA/MurderMistery.soga | 2–2 | 50 | 0.00 | 1.00 | 2 | 0.000 | 0.000 | 0.000 |
| SOGA/NoisyOr.soga | 10–10 | 50 | 0.00 | 1.00 | 10 | 0.000 | 0.000 | 0.000 |
| SOGA/RandomWalkDisc10.soga | 1–1 | 50 | 0.00 | 0.00 | 1 | 0.000 | 0.000 | 0.000 |
| SOGA/RandomWalkGauss10.soga | 1–1 | 50 | 0.00 | 0.00 | 1 | 0.000 | 0.000 | 0.000 |
| SOGA/RandomWalkUnif10.soga | 1–1 | 50 | 0.00 | 0.00 | 1 | 0.000 | 0.000 | 0.000 |
| SOGA/Scale.soga | 3–3 | 50 | 0.33 | 1.00 | 3 | 0.000 | 0.000 | 0.000 |
| SOGA/SurveyUnbias.soga | 4–4 | 50 | 0.50 | 1.00 | 4 | 0.704 | 0.000 | 0.000 |
| SOGA/ThreeCoins.soga | 3–3 | 50 | 0.00 | 1.00 | 3 | 0.000 | 0.000 | 0.000 |
| SOGA/Toy.soga | 1–1 | 21 | 1.00 | 0.00 | 1 | 0.000 | 0.000 | 0.000 |
| SOGA/TrueSkills.soga | 6–6 | 50 | 0.33 | 1.00 | 6 | 0.707 | 0.000 | 0.000 |
| SOGA/TwoCoins.soga | 3–3 | 50 | 0.00 | 1.00 | 3 | 0.000 | 0.000 | 0.000 |
| Example/Bernoulli.soga | 2–2 | 50 | 0.50 | 1.00 | 2 | 0.000 | 0.000 | 0.000 |
