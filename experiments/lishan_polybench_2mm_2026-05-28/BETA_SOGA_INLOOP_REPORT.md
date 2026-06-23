# Beta validation — SOGA engine in the loop

**Date:** 2026-06-23
**Script:** `beta_soga_in_loop.py` (run with `../../.venv/bin/python`, needs scikit-learn)
**Goal:** repeat the beta validation but with the **real SOGA engine** propagating the
fault (not the hand-rolled analytical quadrature of `beta_input_prior_validation.py`),
so that claims (1) Typhoon-falsified and (2) analytical resilience are validated with
SOGA genuinely in the loop — and so that the **float32 fault-encoding** built here is
the reusable artifact for gamma (the 2mm).

## How SOGA is in the loop

A `.soga` program per (prior, hit-term identity):
```
x0 = gauss(0, s0); ... ; xn1 = gauss(0, sn1);   # Gaussian input prior
out = x0; out = out + x1; ...                    # affine accumulator
r  = gm([w...],[ratios...],[0...]);              # relative-perturbation categorical
err = xhit * r;                                  # RV x RV product -> mul_func (exact)
```
- The fault channel `r` is characterized once by enumerating every single-bit-flip on a
  term `v ~ N(0, sigma_hit²)`: finite flips give relative perturbations `(v'-v)/v`
  (magnitude-independent for sign/mantissa), non-finite flips give the OTR mass. Binned
  into a zero-variance categorical.
- `libSOGAupdate.mul_func` computes `err = xhit * r` EXACTLY for a zero-variance `r`
  (Isserlis): component i = `N(0, sigma_hit²·r_i²)`. The engine also tracks
  `cov(out,err) = sigma_hit²·r_i` — i.e. it captures the masking coupling
  `rho = sigma_hit/norm` (≈0.25 balanced, ≈0.97 for the concentrated big term).
- Post-hoc, per GM component, the correlated bivariate `P(|err| <= eps|out|)` is
  integrated → MSK; SDC = remaining finite mass; OTR = the Bernoulli mass (two-tier).
- Averaged over which term is hit (`k ~ U`).

## Results (n=16, norm=4, big_frac=0.95, samples=1500, n_v=4000, seed=20260623)

| prior | eps | SDC mc | SDC SOGA | MSK mc | MSK SOGA | OTR mc | OTR SOGA |
|---|---|---|---|---|---|---|---|
| balanced | 1e-3 | 48.81 | 50.51 | 50.33 | 48.61 | 0.86 | 0.88 |
| balanced | 1e-2 | 38.41 | 40.32 | 60.75 | 58.80 | 0.84 | 0.88 |
| balanced | 1e-1 | 24.74 | 26.59 | 74.42 | 72.53 | 0.85 | 0.88 |
| concentrated | 1e-3 | 43.93 | 44.96 | 56.03 | 55.00 | 0.04 | 0.04 |
| concentrated | 1e-2 | 32.35 | 33.58 | 67.61 | 66.38 | 0.04 | 0.04 |
| concentrated | 1e-1 | 15.63 | 16.45 | 84.34 | 83.52 | 0.04 | 0.04 |

- **SOGA-engine ≈ MC** to **≤2.04%** worst-case across the grid.
- **Typhoon falsified by the engine:** SDC separation +5.5 … +10.1 pp (MC: +4.9 … +9.1),
  same direction and magnitude, at identical Frobenius norm and identical accumulator law.
- **Two-tier model:** OTR (Bernoulli) reproduced to ≤0.05 pp.

## Honest interpretation

SOGA-in-loop carries a **systematic ~1.5–2 pp** over-estimate of SDC vs the more faithful
analytical model (≤0.24%). This is the **quantified cost of the GM fault abstraction** —
specifically the only approximation: `r ⊥ x_hit` (the engine propagates the *marginal* of
the relative perturbation; MC keeps the exact per-sample coupling). It can be tightened
with finer ratio binning / within-bin spread, but the falsification and curve shape survive
comfortably. Scope: relative `eps>0` only (eps=0 order-dependent, excluded).

## Significance

The float32 single-bit-flip fault is now expressible as a GM the **real engine** ingests
and propagates (Gaussian inputs × multiplicative fault, exact product, joint covariance),
WITHOUT modifying `src/`. This is the encoding gamma reuses on the 2mm product structure.

## Reproduce

```
../../.venv/bin/python beta_soga_in_loop.py            # default n=16 grid
../../.venv/bin/python beta_soga_in_loop.py --n 8 --samples 300 --n-v 1500 --eps-grid 1e-2 1e-1
```
