# Codex Cross-Review Prompt — iter 1 — 2026-05-26

You are reviewing the following PLAN for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution).

SOGA strategic context: SOGA computes posterior moments of probabilistic programs via Gaussian Mixture propagation through CFGs. The current plan concerns a fault-resilience POC for matrix kernels (2MM 32×32 float32), where the fault model simulates physical bit-flips using IEEE 754 struct.pack/unpack semantics.

The plan being reviewed is a REFINEMENT of a parent plan that was already executed in 4 iterations (72 tests pass). The refinement replaces a 5-class moment-matched fault model with a bit-by-bit XOR-exact model (same IEEE 754 struct.pack/unpack routine as the MC reference, but analytically aggregated).

Act as an independent reviewer. Do NOT consider yourself the author. Be skeptical.

Evaluate the plan against these 7 criteria:

(a) Correctness of bit-XOR semantics (NaN/Inf/denormal handling) vs IEEE 754 spec. The plan includes 3 specialist-found fixes: NaN/Inf input guard, float32-cast delta, native endianness. Verify all 3 are correctly specified.

(b) Soundness of "bit-exact = MC analytical" claim — is there any hidden gap between bit-exact analytical aggregation and MC sampling? (e.g., sampling without replacement vs aggregation, weight normalization).

(c) Statistical validity of Pearson > 0.95 promise — achievable, or aspirational? Should there be a documented fallback?

(d) Step 3 monotonicity test rigor — Kendall τ + sign-test + binomial p-value. Are the thresholds (τ>0.9, n_wrong≤2 / τ<0.7, n_wrong≥5) calibrated correctly? Multiple-testing correction needed?

(e) Backward-compatibility preservation — git mv rename, --mode flag in runners, config.json schema evolution, HASHES.txt append-only. Any breaking changes that slipped through?

(f) Scope discipline — any tasks that should be split/merged? Any tasks creeping outside scope?

(g) Risk register completeness — 10 risks listed (R-BE1 to R-BE10). Any missing?

IMPORTANT CONTEXT about the existing MC reference implementation:
- flip_bit() in simulate_fi_mc.py uses struct.pack('>f', val32) — big-endian packing ('>' not '!')
- bit_fault_table_array() in the PLAN uses struct.pack('!f', v_f32) in scalar, and v32.view(np.uint32) in vectorized
- '!' and '>' are IDENTICAL in Python struct module (both mean network byte order = big-endian)
- The PLAN's vectorized version uses np.asarray(v_array, dtype=np.float32) + .view(np.uint32) WITHOUT explicit byte order — this uses NATIVE byte order, which on little-endian systems (x86/ARM) produces a DIFFERENT bit layout than the big-endian struct.pack reference
- This is a potential bit-pattern mismatch between the scalar and vectorized implementations

Additional context about the statistical tests:
- 21 p-points for Kendall τ gives df=19, quite adequate for τ threshold 0.9
- The sign-test uses consecutive differences; for 21 points there are 20 differences
- n_wrong≤2 out of 20 is p≈0.0002 under H0 (50/50 fair coin), so the MONOTONE threshold is appropriately strict
- n_wrong≥5 out of 20 has p≈0.98 under H0, meaning this condition alone is NOT sufficient to declare non-monotonicity; the AND with binom_p<0.01 tightens it, but the combined condition requires care
- Multiple testing: 3 curves (MSK, SDC, OTR) are tested for monotonicity; Bonferroni correction would require p<0.003 not p<0.01 for the binomial threshold

Artifact:
---
# Plan: bit-exact-fault-model

**Date**: 2026-05-26
**Slug**: `bit-exact-fault-model`
**Branch (target)**: `feat/lishan-resilience-poc` (stay; per user decision D1=a)
**Parent plan**: `plan/2026-05-25-lishan-resilience-poc.md` (executed 4 iter; 72 tests pass + 2 xfail empirical limitations)
**Effort**: 5 days (R0-R6, ~400 new LOC, 31 new tests)
**Triggered by**: empirical L1 (SDC 38× overshoot) + L2 (OTR anti-correlated) — both confirmed root-caused to bit-class aggregation + additive shift approximation, NOT to matrix-GM core
**Strategic stance**: SOGA becomes "MC analytical" — same per-bit IEEE 754 routine as MC reference, but closed-form aggregation instead of sampling

---

## Goal

Eliminate the 2 empirical limitations of the 5-class moment-matched fault model by computing per-bit XOR-exact shifts using `struct.pack/unpack` (identical IEEE 754 semantics as MC reference). Target results: **<2% rel err on SDC vs MC** (was ~38× overshoot), **<1% abs err on OTR** (was anti-correlated with MC), **Pearson > 0.95** (was 0.90 fallback).

**Two valid Step 3 outcomes**:
- **Outcome A "monotone confirmed"**: bit-exact verifies the moment-match result, much higher precision → robust support for Lishan's Assumption-1
- **Outcome B "counterexample revealed"**: bit-exact uncovers subtle non-monotonicity smoothed by moment-match → first identified failure of Assumption-1 (strongest pitch outcome)

---

## Context

### Why the refinement was needed (root cause)

The parent plan executed M0-M6 successfully but surfaced 2 empirical limitations:

| L# | Limitation | Cause (verified analytically) |
|---|---|---|
| L1 | SDC overestimated ~38× (A=I_32) | LOW_MANTISSA Gaussian moment-match smears 2/16 SDC-region bits across all 16 → tail integral wrongly inflated |
| L2 | OTR anti-correlated vs MC | Additive shift model `δ = v·(2^k-1)` misses IEEE 754 special-encoding transitions (e.g., bit 30 on v=1.0 → +Inf directly via biased-exp overflow, not magnitude check) |

**Root cause analysis** (confirmed by all 4 specialists in Phase D consultation):
- L1: bit-class aggregation + discrete→Gaussian moment-match collapse
- L2: bit-class aggregation + additive shift model approximating bit-XOR semantics
- **Neither limitation is due to matrix-GM core** — the matrix-GM propagator is validated and correct (see MATRIX_GM_SEMANTICS.md)

### Specialist consultations (Phase D, 4 parallel)

- `numerical-stability-expert`: found 3 design bugs in initial `compute_xor_shift` (NaN/Inf input guard missing; delta must use float32-cast minuend; '<f4' explicit endianness portability bug)
- `gaussian-mixture-expert`: confirmed Q1-Q3 formulas; Q4 confirms existing `predict_bimodal_sweep(use_two_component=True)` already implements exact 2-component Bernoulli for A=I_32 (just need to plumb bit-exact in)
- `soga-internal-expert`: confirmed no new SOGA core ops needed; recommended `git mv` for rename, two-script + `--mode` flag in runners, append-only HASHES.txt
- `test-engineer`: 31 new tests structured into 7 batteries; statistical monotonicity test (Kendall τ + sign-test + binomial p-value) with explicit threshold rules

Full expert memos: `/tmp/numerical_bit_exact_memo.md`, `/tmp/gm_bit_exact_memo.md`, `/tmp/soga_internal_memo.md`, `/tmp/test_engineer_memo.md`.

---

## Constraints

1. **NO modifications to `libSOGA*.py`** — only existing matrix-GM API (`libMatrixGaussian.MatrixGaussian.affine_left`)
2. **NO modifications to `.g4` grammars**
3. **NO calibration against SASSIFI/NVBitFI numbers** — Strada Q discipline (input-side ≠ register-level)
4. **Scope: 2MM 32×32 float32 only**. No SYRK/GEMM/3MM/2DCONV. No GPU code.
5. **ε = 10⁻³ relative error default**; parametric in notebook.
6. **Reproducibility**: seed propagation, `HASHES.txt` append-only, `config.json` schema-evolved (backward-compatible).
7. **Branch isolation**: stay on `feat/lishan-resilience-poc` (D1=a); no cross-branch sync.
8. **5-class preservation**: rename existing `predict_resilience_soga.py` → `predict_resilience_soga_5class.py` (D2=a); preserve as historical reference + regression comparison.
9. **Tightened acceptance**: Pearson > 0.95 primary, fallback > 0.90 only if MC sample noise documented (D3=a).
10. **Honesty discipline**: input-side ≠ register-level disclaimer preserved in REPORT and notebook.
11. **31 new tests mandatory**: per test-engineer Q1-Q7 structure; xfail promotion to strict=True if bit-exact achieves the gates.

---

## Approach

### Architecture: bit-by-bit XOR-exact = "MC analytical"

The MC reference (`simulate_fi_mc.py:flip_bit`) uses `struct.pack/unpack` to simulate physical bit-flips on float32. The new SOGA predictor uses the **same routine deterministically** for all (bit, input row, v) tuples, then aggregates analytically via weighted sum rather than sampling.

**Key code contract** (with all 3 specialist-found bugs FIXED):

```python
def compute_xor_shift(v: float, bit_idx: int) -> tuple[float, bool]:
    """IEEE 754 bit-XOR exact. Bit-perfect match against MC reference flip_bit."""
    import struct
    import numpy as np
    # Bug A fix (numerical Q2): NaN/Inf input guard
    if not np.isfinite(v):
        return 0.0, True
    v_f32 = np.float32(v)  # cast for MC consistency
    bits = int.from_bytes(struct.pack('!f', v_f32), 'big')
    flipped = bits ^ (1 << bit_idx)
    v_post = struct.unpack('!f', flipped.to_bytes(4, 'big'))[0]
    is_special = not np.isfinite(v_post)
    # Bug B fix (numerical Q2): minuend uses float32-cast value
    delta = (float(np.float32(v_post)) - float(v_f32)) if not is_special else 0.0
    return delta, is_special
```

**Vectorized version** (numpy view tricks; numerical Q3 portability fix applied):

```python
def bit_fault_table_array(v_array: np.ndarray) -> np.ndarray:
    """Returns shape (N, 32, 2) with (delta_float64, is_special_bool) per (v, bit)."""
    v32 = np.asarray(v_array, dtype=np.float32)         # native float32, no '<f4'
    finite_in = np.isfinite(v32)
    bits = v32.view(np.uint32)                          # native byte order
    bit_idx = np.arange(32, dtype=np.uint32)
    flipped = bits[:, None] ^ (np.uint32(1) << bit_idx[None, :])
    v_post = flipped.view(np.float32)
    is_special = ~np.isfinite(v_post) | ~finite_in[:, None]
    delta = np.where(is_special, 0.0,
                     v_post.astype(np.float64) - v32.astype(np.float64)[:, None])
    return np.stack([delta, is_special.astype(np.float64)], axis=-1)
```

### Per-output-cell aggregation (analytical Option 2 unchanged)

For each output cell `(r,s)`:
- **Baseline** (no fault): weight `(1 - p_fault)`, Gaussian `N(μ_base, σ²_base)`
- **32 fault scenarios** per input row `i` with `j=s`: weight `p_fault / (m·n·32)`, deterministic shift `A[r,i] · δ_bit`
- For `A = I_32`: only `i = r` contributes nonzero shift → **K_per_cell = 1 + 32 = 33**
- For general A: K_per_cell = 1 + 32·m_distinct (≤ 1+32·m = 1025 worst case)

**SDC/MSK/OTR classification** (gm-expert Q2 for σ_base→0):
- `is_special[bit]` → contributes to OTR
- `|delta_bit · A[r,i]| > ε · |μ_base|` → SDC
- else → MSK

**Non-degenerate σ_base > 0** (gm-expert Q3): each non-special scenario is tail-Gaussian, sum weighted.

### Step 3 simplification (gm-expert Q4: EXACT for A=I_32)

For `B_ij ~ p·δ(V_high) + (1-p)·δ(V_low)` IID per cell and A=I_32: `D[r,s] = B[r,s]`, cells independent. Per output cell:

```
Pr(SDC | r,s) = p · Pr(SDC | D = V_high) + (1-p) · Pr(SDC | D = V_low)
```

**Each conditional uses bit-exact (Q3 formula).** No CLT needed. No moment-match. The existing `predict_bimodal_sweep(use_two_component=True)` already implements this dispatch correctly — just plumb in bit-exact for the inner SDC kernel.

For dense A (m_dense ≥ 16): keep existing CLT Gaussian path (already validated). For sparse A or A=I: 2-component exact (recommended primary).

### Non-monotonicity statistical test (gm-expert Q6 + test-engineer Q4)

Given 21-point sweep Pr(SDC | p_k):
1. **Kendall τ**: `scipy.stats.kendalltau(p_list, sdc_vals)` → (τ, p_value)
2. **Sign-test**: count wrong-sign consecutive differences; binomial p-value via `scipy.stats.binomtest`
3. **Robust threshold**:
   - MONOTONE: `τ > 0.9` AND `wrong_sign ≤ 2`
   - NON-MONOTONE: `τ < 0.7` OR (`wrong_sign ≥ 5` AND `binom_p < 0.01`)
   - AMBIGUOUS: between thresholds (warn, do not fail)
   - PLATEAU-DEGENERATE: Pr(SDC) constant within MC noise → exempt from test

### File structure changes

| Action | File |
|---|---|
| ADD | `experiments/lishan_resilience_2026-05-25/lib/bit_fault_table.py` |
| ADD | `experiments/lishan_resilience_2026-05-25/lib/DESIGN_BIT_EXACT.md` |
| RENAME (`git mv`) | `predict_resilience_soga.py` → `predict_resilience_soga_5class.py` |
| ADD | NEW `predict_resilience_soga.py` (bit-exact primary) |
| MODIFY | `run_soga_step1.py`, `run_step3.py`, `run_sweep.py` (add `--mode` flag) |
| MODIFY | `config.json` (add `fault_model_mode`, `mc_samples`; backward-compatible defaults) |
| ADD | 7 test files (31 tests) under `tests/` |
| MODIFY | `REPORT.md` (5-class → historical, bit-exact → primary) |
| MODIFY | `lishan_pitch.ipynb` (Approximation hierarchy section, mode selector) |
| MODIFY | `docs/research-notes/06-input-side-fault-modeling.md` (positioning update) |
| MODIFY | `lishan_discussion_package/README.md`, `setup.md`, `figures/` |
| MODIFY | `HASHES.txt` (append new entries, preserve old) |

---

## Sub-tasks (atomic, milestone-organized)

### R0 — Setup + design lock (0.5 days)

- [ ] **[R0.1]** Write `lib/DESIGN_BIT_EXACT.md` design doc: rationale, IEEE 754 references, the 3 specialist-found bug fixes, comparison vs 5-class, scope.
- [ ] **[R0.2]** Write 5 hand-computed sanity tests (T1-T5): sign(1.0)→-1.0, bit 23→2.0, bit 22→1.5, bit 30→+Inf, sign(0)→-0. All tests RED initially.

### R1 — Bit-fault table module (1 day)

- [ ] **[R1.1]** Implement scalar `compute_xor_shift(v, bit_idx)` WITH all 3 fixes
- [ ] **[R1.2]** Vectorized `bit_fault_table_array(v_array)` returning shape `(N, 32, 2)`
- [ ] **[R1.3]** 100-random-v validation against MC flip_bit via numpy.equal on bit patterns
- [ ] **[R1.4]** 10 IEEE 754 edge cases (T6-T10)

### R2 — Analytical aggregation rewrite (1.5 days)

- [ ] **[R2.1]** `git mv predict_resilience_soga.py predict_resilience_soga_5class.py`
- [ ] **[R2.2]** NEW bit-exact `predict_resilience_soga.py`
- [ ] **[R2.3]** Vectorize aggregation, target < 200ms per v-point
- [ ] **[R2.4]** 5 cross-validation tests
- [ ] **[R2.5]** Add `--mode {bit_exact, 5_class}` flag to 3 runners

### R3 — Step 1 re-validate (0.5 days)

- [ ] **[R3.1]** Re-run v_sweep with `--mode bit_exact`
- [ ] **[R3.2]** 3-panel figure (MC | 5-class | bit-exact)
- [ ] **[R3.3]** 4 acceptance tests: Pearson > 0.95, rel/abs err bounds

### R4 — Step 3 bimodal Bernoulli with bit-exact (1 day)

- [ ] **[R4.1]** Plumb bit-exact into `predict_bimodal_sweep(use_two_component=True)`
- [ ] **[R4.2]** Re-run p_sweep with bit-exact
- [ ] **[R4.3]** 4 statistical tests (Kendall, sign, binomial, threshold rule)
- [ ] **[R4.4]** MC validation at 4 points × 5000 samples, seed=42
- [ ] **[R4.5]** Write counterexample_analysis.md or monotonicity confirmation section

### R5 — Documentation overhaul (0.5 days)

- [ ] **[R5.1]** Rewrite REPORT.md with methodological hierarchy
- [ ] **[R5.2]** Add "Approximation hierarchy" to notebook with mode selector
- [ ] **[R5.3]** Update research note Section 4 positioning
- [ ] **[R5.4]** Update lishan_discussion_package/

### R6 — Final QA + cross-review (0.5 days)

- [ ] **[R6.1]** Code review on full refinement diff
- [ ] **[R6.2]** Codex cross-review (this is it)
- [ ] **[R6.3]** Full pytest ≥103 tests pass; xfail promotion
- [ ] **[R6.4]** Update cross-review history in both plan files

---

## Acceptance criteria

- [ ] All 31 new tests pass; all 72 existing tests pass (≥103 total)
- [ ] Bit-fault table matches MC reference bit-perfect on 100 random v + 10 edge cases
- [ ] Step 1 Pearson > 0.95
- [ ] Step 1 max rel err < 2% (SDC); max abs err < 1% (OTR, MSK)
- [ ] Step 3 monotonicity verdict: MONOTONE, NON-MONOTONE, or AMBIGUOUS
- [ ] `predict_resilience_soga_5class.py` preserved; `git log --follow` works
- [ ] `--mode {bit_exact, 5_class}` flag works on all 3 runners
- [ ] config.json backward-compatible
- [ ] HASHES.txt append-only
- [ ] No `libSOGA*.py` or `.g4` modifications

---

Output format (strict):
VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- <finding 1>
- <finding 2>
RECOMMENDATIONS:
- <action 1>
