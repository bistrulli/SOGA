# SOGA Fault Resilience POC — Discussion Package for Lishan Yang

**HONESTY DISCLAIMER**: Input-side fault model (faults injected into B before kernel execution). NOT register-level injection (SASSIFI/NVBitFI). Adjacent methodology. Strada Q discipline.

---

## What this is

This package demonstrates SOGA's capability to analytically predict fault resilience
categories (MSK/SDC/OTR) for 2MM (matrix multiply) 32×32 float32 operations, using
an input-side bit-flip fault model. It complements register-level fault injection
studies (SASSIFI, NVBitFI) by providing a fast analytical upper bound.

**Primary results**:
- SOGA analytical prediction: < 1.4s for full 21-point p-sweep
- MSK probability: < 0.02% absolute error vs MC reference
- SDC monotone in bimodal mixing weight p (tau=1.000 for A=I_32)
- Known limitation: SOGA overestimates SDC by ~38x (mantissa moment-match; documented)

---

## Files

| File | Description |
|------|-------------|
| `lishan_pitch.ipynb` | Interactive notebook with ipywidgets sliders |
| `setup.md` | Step-by-step reproduction instructions |
| `figures/resilience_vs_input_value.png` | Step 1: resilience vs |v| |
| `figures/resilience_vs_bimodal_p.png` | Step 3: resilience vs bimodal p |

Full experiment source:
`experiments/lishan_resilience_2026-05-25/` in the SOGA repository.

---

## Quick summary of results

### Step 1: Resilience vs input value scale (A=I_32)

Both SOGA and MC show high resilience (MSK > 99.9%) for p_fault=0.01.
The SDC probability is ~0.0002 (SOGA) / ~0.000005 (MC) — SOGA is conservative.
OTR occurs at v=1.0 (MC: bit-pattern-specific IEEE 754 overflow) and v>8.5
(SOGA: delta-magnitude model).

### Step 3: Resilience vs bimodal mixing weight p (V_low=0, V_high=1)

SDC increases linearly with p (perfectly monotone, Kendall tau=1.000).
No counterexample (non-monotone behavior) for identity kernel.
For a non-trivial kernel with mixed-sign rows, non-monotone behavior is plausible.

### Model accuracy

SOGA correctly models:
- MSK (dominant probability, < 0.02% error)  
- SDC trend direction and monotonicity
- Symmetry at ±v (exact, analytical)
- SDC=0 at p=0 when V_low=0 (exact)

SOGA over-approximates:
- SDC probability by ~38x (LOW_MANTISSA moment-match; sigma/threshold ~ 1.13)
- OTR location (delta-magnitude model vs IEEE 754 bit patterns)

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
injection studies. The SOGA analytical model provides:

1. A fast predictor (microseconds vs hours of SASSIFI runs) 
2. A conservative upper bound on SDC (overestimates by known factor)
3. A parametric sweep over input distributions (bimodal Bernoulli)

Open questions for collaboration:
- Can the ~38x gap be reduced by improving the mantissa fault model?
- What is the SOGA vs SASSIFI gap for non-identity kernels (convolution, etc.)?
- How does the bimodal model extend to real neural network activation distributions?

Contact: emilio.incerto@valuematic.eu
