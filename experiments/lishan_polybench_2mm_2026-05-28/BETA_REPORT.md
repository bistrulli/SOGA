# Beta validation — analytical input-prior resilience vs Monte-Carlo

**Date:** 2026-06-23
**Script:** `beta_input_prior_validation.py`
**Goal:** de-risk the SIGMETRICS direction *without* needing Lishan Yang's GPU/NVBit-FI
campaign — show that an **analytical** prediction of float32 register-FI resilience,
computed from the **input prior alone** (zero fault injection, zero input sampling),
(1) matches Monte-Carlo ground truth and (2) falsifies Typhoon's norm-as-sufficient-statistic.

## Setup

- **Kernel:** single inner product `out = sum_k x_k`, float32 (one GEMM output cell).
- **Fault model:** term `k ~ U{0..n-1}`, bit `b ~ U{0..31}`, single-bit-flip; classes
  MSK (`|err| <= eps*|out|`) / SDC / OTR (flip → Inf/NaN). MC oracle re-accumulates in
  true float32 (`dot_f32` from `float_example.py`).
- **Two priors, matched by construction** (`n=16`, `norm=4`, `big_frac=0.95`):
  - `balanced`: every `x_k ~ N(0, norm²/n)`
  - `concentrated`: `x_0 ~ N(0, 0.95·norm²)`, rest `~ N(0, 0.05·norm²/(n-1))`
  - **Both** have `E[||x||²] = norm²` **and** `Var(out) = norm²` → identical Frobenius
    norm *and* identical accumulator law. The only difference is the per-term magnitude
    distribution, so any SDC/MSK gap isolates the distribution effect.
- **Analytical prediction (no injection):** `P(OTR)` is the closed-form indicator that
  `flip(v,b)` is non-finite, averaged over the prior's exponent law; `P(MSK|v,b) =
  P_r(|v+r| ≥ |δ_b(v)|/eps)` is a Gaussian tail integral (`r ⊥ v ~ N(0, norm²−Var(x_k))`).
  The flipped-term marginal is a 1-component (balanced) / 2-component (concentrated)
  Gaussian mixture → the prediction is a **GM tail integral**, evaluated by 1-D quadrature.

## Results (two independent runs)

| run | samples | n_quad | seed | MC 95% CI | worst \|analytic−MC\| | SDC separation (equal norm) |
|---|---|---|---|---|---|---|
| 1 | 1500 | 4001 | 20260623 | ≤0.11% | **0.36%** | +4.81 … +9.36 pp |
| 2 | 3000 | 16001 | 7 | ≤0.08% | **0.24%** | +4.95 … +9.76 pp |

- **Method validated:** analytical (zero FI, zero input sampling) tracks MC to ≤0.24%
  worst-case across the `eps ∈ {1e-3 … 1e-1}` grid; the residual shrinks with finer
  quadrature (0.36% → 0.24%), confirming it is integration discretization on the
  power-of-two-jumpy `δ_b(v)`, not a model error.
- **Typhoon falsified:** two inputs with **identical Frobenius norm and identical
  accumulator law** have SDC rates differing by **5–10 percentage points**, and the gap
  is **predicted analytically** (MC vs analytic separation agree to ≤0.4 pp). The gap
  grows with `eps` (more tolerance → concentrated's many small terms get rounding-masked).
- **Two-tier model validated:** the OTR (Bernoulli/atomic) channel is reproduced exactly
  (balanced 0.85% vs 0.85%; concentrated 0.04% vs 0.04%) by the closed-form exponent law,
  separate from the GM (MSK/SDC) channel — confirming `02-unifying-error-fault-models.md`.

## Scope / honesty

- Targets the **relative-tolerance regime `eps>0`**. `eps=0` bit-exact masking is
  accumulator-history/order-dependent and is deliberately **not** modeled (see memory
  `int32-eps0-resilience-degenerate`).
- Linear kernel only; propagation through the 2mm product structure is the next step (γ).
- Validates the *method* (analytical-from-prior ≈ ground truth) on CPU. Yang's NVBit-FI
  campaign is still needed for the **hardware-realism** increment, not for the math.

## Reproduce

```
python3 beta_input_prior_validation.py --samples 3000 --n-quad 16001 --seed 7
python3 beta_input_prior_validation.py --csv beta_results.csv          # default config
```
