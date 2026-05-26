# Design: Bit-Exact Fault Model

**Date**: 2026-05-26
**Branch**: feat/lishan-resilience-poc
**Plan ref**: plan/2026-05-26-bit-exact-fault-model.md

---

## 1. Rationale

The parent plan (2026-05-25-lishan-resilience-poc) used a 5-class moment-matched
fault model that partitioned the 32 bits of IEEE 754 float32 into:
SIGN (1 bit), HIGH_EXP (4), LOW_EXP (4), HIGH_MANTISSA (7), LOW_MANTISSA (16).

Two empirical limitations were identified (L1, L2):

| L# | Limitation | Root Cause |
|----|-----------|-----------|
| L1 | SDC overestimated ~38x (A=I_32) | LOW_MANTISSA Gaussian moment-match smears 2/16 SDC-region bits across all 16; tail integral wrongly inflated |
| L2 | OTR anti-correlated vs MC | (a) Additive shift model `delta = v*(2^k-1)` misses IEEE 754 special-encoding transitions; (b) Per-cell-averaged vs per-execution-OTR aggregation mismatch |

The bit-exact model eliminates both limitations by using `struct.pack/unpack` to
compute the exact shift for each bit independently, exactly as the MC reference does.

---

## 2. Architecture: "MC analytical"

**MC reference** (`simulate_fi_mc.flip_bit`): uses `struct.pack('>f', val32)` to
simulate physical bit-flips on float32. Stochastic aggregation via random sampling.

**Bit-exact SOGA** (`lib/bit_fault_table.py`): uses the **same struct.pack/unpack
routine** deterministically for all (bit, v) tuples, then aggregates analytically
via weighted sum rather than sampling. No approximation; bit-perfect agreement with
MC on any individual fault scenario.

### Key formulas

For each input value v and bit index b (0..31):
- `v_f32 = float32(v)` (cast to float32, same as MC)
- `bits = struct.unpack('>I', struct.pack('>f', v_f32))[0]`
- `flipped = bits XOR (1 << b)`
- `v_post = struct.unpack('>f', struct.pack('>I', flipped))[0]`
- `is_special = not isfinite(v_post)`
- `delta = float32(v_post) - v_f32` if not special, else 0.0

For per-output-cell aggregation (A = I_32, flat input v):
- Non-fault baseline: weight `(1 - p_fault)`, Gaussian `N(mu_base, sigma_base^2)`
- Per-bit fault: weight `p_fault / (m*n*32)`, deterministic shift `delta_b * A[r,i]`
- `is_special_b=True` -> OTR contribution
- `|delta_b * A[r,i]| > eps * |mu_base|` -> SDC contribution
- else -> MSK

---

## 3. OTR Aggregation Semantics

### R0.3 Finding (CRITICAL — closes L2)

Read `simulate_fi_mc.py:simulate_one_fi` carefully:

```python
def simulate_one_fi(A, B, fault_cell, bit, eps=1e-3):
    ...
    is_otr = np.any(~np.isfinite(D_perturbed))  # PER-EXECUTION
    return D_baseline, D_perturbed, is_otr
```

And in `classify_outcome`:
```python
if np.any(~np.isfinite(D_perturbed)):
    return 0.0, 0.0, 1.0  # ENTIRE execution = OTR=1
```

**MC uses PER-EXECUTION OTR**: if ANY output cell is non-finite, the entire execution
is classified as OTR (probability 1.0 for that execution).

**Prior analytical model error**: the `compute_per_cell_SDC` / `compute_sdc_vectorized`
function computed per-cell OTR (fraction of cells that are non-finite), then averaged
over cells. This is mathematically different:

- MC: `P(OTR | fault) = P(any cell non-finite | fault)` = probability that at least one
  output cell is non-finite given a fault occurred
- Old SOGA: `P(OTR | fault) = (1/mn) * sum_cells P(cell_r,s non-finite | fault)`

For A = I_32 (identity): a fault at B[i,j] only affects D[i,j] = B[i,j]. So:
  - MC: OTR = 1 iff D[i,j] = v_post is non-finite (equivalent to per-cell for identity)
  - BUT the old SOGA model then average over all 1024 cells AND multiplied by n_in
    creating a double-counting vs the MC single-execution measure

For general dense A: a fault at any input cell propagates to ALL output cells (since
A is full-rank), so P(any cell non-finite) = P(at least one of mn cells non-finite).
In the dense case the per-cell and per-execution measures can differ significantly.

### Correct analytical formula

**Per-execution OTR** (aligned with MC):

For a single-cell fault at B[i,j] with bit b:
```
P(OTR | fault at (i,j), bit b) = P(v_post_b is_special AND it propagates to any D output)
```

For A = I_32: only D[i,j] is affected, so `P_exec_OTR = is_special[b]` (0 or 1 per bit).

For the sweep over all (fault_cell, bit) uniformly:
```
P_OTR = sum_{i,j,b} P(fault at (i,j)) * P(bit=b) * is_special[i,j,b]
       = (1/(mn)) * (1/32) * sum_{b=0}^{31} is_special[b]   (for A=I, symmetric v)
       = special_count / 32
```

where `special_count = sum_b is_special_b` for value v.

This is the correct formula for the new `predict_resilience_soga.py:compute_per_cell_SDC`.

---

## 4. Specialist-found bug fixes (4 total)

| Bug | Description | Fix |
|-----|-------------|-----|
| A | NaN/Inf guard: `not np.isfinite(v)` would short-circuit Inf input | Guard NaN ONLY: `np.isnan(v)`. Inf flows through for per-bit evaluation |
| B | Delta uses float64 v for subtraction: `float(v_post) - float(v)` | Use float32 cast: `float(np.float32(v_post)) - float(v_f32)` |
| C | Explicit `'<f4'` endianness fails on big-endian | Use `dtype=np.float32` (native byte order) |
| D | OTR aggregation: per-cell-averaged vs per-execution | Use per-execution: P_OTR = special_count/32 |

---

## 5. IEEE 754 reference cases

These are verified against MC reference `flip_bit`:

| v | bit | v_post | delta | is_special |
|---|-----|--------|-------|-----------|
| 1.0 | 31 | -1.0 | -2.0 | False |
| 1.0 | 23 | 0.5 | -0.5 | False |
| 1.0 | 22 | 1.5 | +0.5 | False |
| 1.0 | 30 | +Inf | 0.0 | True |
| 0.0 | 31 | -0.0 | -0.0 | False |
| +Inf | 30 | 1.0 | n/a | False (Inf input, finite output!) |
| +Inf | 22 | NaN | 0.0 | True |

Note: `0.0 -> -0.0` sign flip: delta = -0.0 - 0.0 = -0.0 in float64. Python `==` treats
-0.0 == 0.0; use `math.copysign` to distinguish.

---

## 6. CHANGELOG (behavioral breaking change)

**config_version**: 1 -> 2

The new `predict_resilience_soga.py` (bit-exact primary) replaces the 5-class
moment-matched model as default. The 5-class model is preserved as:
`predict_resilience_soga_5class.py` (historical reference).

Run scripts (`run_soga_step1.py`, `run_step3.py`, `run_sweep.py`) now accept
`--mode {bit_exact, 5_class}` flag (default: `bit_exact`).

Runtime banner is emitted at start of each run:
```
[MODE] Using bit_exact fault model (refinement primary; --mode 5_class for legacy)
```

Old configs missing `fault_model_mode` field default to mode `bit_exact` with
deprecation warning.

---

## 7. Scope and constraints

- **Scope**: 2MM 32x32 float32 with A=I_32 (identity kernel). No SYRK/GEMM/3MM.
- **No libSOGA*.py changes**: all work at the experiments/ script level.
- **No .g4 changes**: no grammar involved.
- **No SASSIFI/NVBitFI calibration**: Strada Q discipline preserved.
- **Honesty disclaimer preserved**: input-side != register-level.

---

## 8. Comparison: 5-class vs bit-exact

| Property | 5-class moment-matched | Bit-exact |
|----------|----------------------|-----------|
| SDC accuracy | ~38x overestimate for A=I_32 | Bit-perfect vs MC |
| OTR accuracy | Anti-correlated (wrong aggregation) | Aligned (per-execution) |
| Runtime | ~50ms per v-point | ~80ms per v-point (32 bits vs 5 classes) |
| Mantissa | Gaussian approximation | Exact per-bit |
| Exponent | Exact (additive shift) | Exact (struct.pack/unpack) |
| Sign | Exact | Exact |

---

_Last updated: 2026-05-26. See plan/2026-05-26-bit-exact-fault-model.md for full context._
