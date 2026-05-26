# SOGA Fault Resilience POC — Discussion Package for Lishan Yang

**HONESTY DISCLAIMER**: Input-side fault model (faults injected into B before kernel execution). NOT register-level injection (SASSIFI/NVBitFI). Adjacent methodology. Strada Q discipline.

---

## What this is

This package demonstrates SOGA's capability to analytically predict fault resilience
categories (MSK/SDC/OTR) for 2MM (matrix multiply) 32×32 float32 operations, using
an input-side bit-flip fault model. It complements register-level fault injection
studies (SASSIFI, NVBitFI) by providing a fast analytical predictor.

**Primary results (bit-exact model, v2)**:
- SOGA analytical prediction: < 1.6s for full 10-point v-sweep (bit-exact primary)
- SDC relative error vs MC: < 4.2% at all v-points (was ~38x with 5-class model)
- OTR absolute error vs MC: < 0.000013 (< 0.002% absolute)
- SDC monotone in bimodal mixing weight p (Kendall tau=1.000, perfect)

---

## Files

| File | Description |
|------|-------------|
| `lishan_pitch.ipynb` | Interactive notebook with ipywidgets sliders |
| `setup.md` | Step-by-step reproduction instructions |
| `figures/resilience_vs_input_value.png` | Step 1: resilience vs abs(v) (bit-exact) |
| `figures/resilience_vs_input_value_3panel.png` | Step 1: 3-model comparison (MC / 5-class / bit-exact) |
| `figures/resilience_vs_bimodal_p.png` | Step 3: resilience vs bimodal p |

Full experiment source:
`experiments/lishan_resilience_2026-05-25/` in the SOGA repository.

---

## Quick summary of results

### Step 1: Resilience vs input value scale (A=I_32)

Both SOGA and MC show high resilience (MSK > 99.9%) for p_fault=0.01.
The SDC probability is ~5.2e-6 — bit-exact SOGA matches MC within 4.2% relative error.
OTR occurs at v=±1.0 (bit 30 flip → +Inf, correctly detected by bit-exact model).

### Step 3: Resilience vs bimodal mixing weight p (V_low=0, V_high=1)

SDC increases perfectly linearly with p (Kendall tau=1.000, analytically verified).
No counterexample (non-monotone behavior) for identity kernel.
For a non-trivial kernel with mixed-sign rows, non-monotone behavior is plausible.

### Model accuracy (bit-exact primary, v2)

| Metric | 5-class (legacy) | Bit-exact (primary) | Target |
|--------|-----------------|----------------------|--------|
| SDC rel err vs MC | ~38x off | < 4.2% | < 20% |
| OTR abs err vs MC | anti-correlated | < 0.000013 | < 1% |
| MSK Pearson | N/A (both flat) | > 0.90 | > 0.90 |
| Runtime / v-point | 40ms | 150ms | < 200ms |

SOGA bit-exact correctly models:
- SDC (< 4.2% relative error vs MC at all v-points)
- OTR (IEEE 754 bit patterns; per-execution semantics aligned with MC)
- MSK (exact complement: 1 - SDC - OTR)
- SDC monotonicity in bimodal p (tau=1.000, analytical)

---

## How to run

See `setup.md`. Requires: Python 3.9+, numpy, scipy, matplotlib, ipywidgets.

```bash
# 2-minute quickstart
git checkout feat/lishan-resilience-poc
pip install numpy scipy matplotlib ipywidgets jupyter
python3 experiments/lishan_resilience_2026-05-25/setup_matrices.py
jupyter notebook lishan_discussion_package/lishan_pitch.ipynb
```

---

## Relation to Lishan's work (U Alberta Fall 2026)

This POC uses the same resilience taxonomy (MSK/SDC/OTR) as Lishan's fault
injection studies. The SOGA bit-exact analytical model provides:

1. A fast predictor (< 1.6s for 10 v-points vs hours of SASSIFI runs)
2. IEEE 754 bit-exact SDC/OTR computation, matching MC within statistical noise
3. A parametric sweep over input distributions (bimodal Bernoulli p-sweep)
4. Analytical monotonicity guarantees (Kendall tau=1.000 for SDC vs p)

Open questions for collaboration:
- How does the bit-exact model extend to non-identity kernels (convolution, 2MM with A != I)?
- Can the input-side bimodal model capture real GPU activation distributions (ReLU, Gaussian)?
- What is the SOGA analytical model vs SASSIFI register-level gap for the same 2MM kernel?

**Methodological note**: Two SOGA fault models are available in this POC:
- `--mode bit_exact` (default, primary): deterministic enumeration of all 32 IEEE 754 bit-flip
  outcomes per cell; matches MC reference at < 4.2% relative error
- `--mode 5_class` (legacy, historical reference): moment-matched Gaussian approximation of the
  bit-flip distribution; overestimates SDC by ~38x but runs faster and is useful as a ceiling

Contact: emilio.incerto@valuematic.eu
