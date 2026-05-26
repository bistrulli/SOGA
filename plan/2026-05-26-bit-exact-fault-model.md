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
    # Bug A fix (numerical Q2) — REFINED per Codex iter 1 F2:
    # Guard NaN ONLY (not Inf). For Inf input, individual bit-flips have
    # well-defined results (e.g., bit 30 on +Inf → 1.0 finite, bit 22 → NaN).
    # The test cases R1.4 T8/T9 EXPECT these specific finite/NaN outcomes.
    if np.isnan(v):
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
    is_nan_in = np.isnan(v32)                            # F2: guard NaN only, not Inf
    bits = v32.view(np.uint32)                           # native byte order
    bit_idx = np.arange(32, dtype=np.uint32)
    flipped = bits[:, None] ^ (np.uint32(1) << bit_idx[None, :])
    v_post = flipped.view(np.float32)
    is_special = ~np.isfinite(v_post) | is_nan_in[:, None]  # NaN input always special; Inf input determined per-bit
    delta = np.where(is_special, 0.0,
                     v_post.astype(np.float64) - v32.astype(np.float64)[:, None])
    return np.stack([delta, is_special.astype(np.float64)], axis=-1)
```

### ⚠️ CRITICAL — OTR aggregation semantics (Codex iter 1 F1 + R-BE11)

**Discovered by Codex iter 1**: the MC reference (`simulate_one_fi`) classifies the **EXECUTION** as OTR=1 if ANY output cell is non-finite, while the analytical model originally reported the **per-cell-averaged** P(cell OTR). These are mathematically different quantities. For A=I_32 + single-cell fault: when corrupted_B[i,j] becomes Inf, only D[i,j] is Inf (others unchanged), so per-execution OTR = (1/mn²) · sum P_cell_OTR for sparse A; for dense A the same fault propagates to ALL output cells.

**Required action (NEW R0.3 + R2.2b)**:
1. Read `simulate_fi_mc.py` carefully to ASCERTAIN the actual aggregation semantics (per-execution-OTR vs per-cell-averaged)
2. Align the analytical formula to MATCH the MC semantics. Specifically:
   - If MC = per-execution: `P_OTR_exec = Pr(any output cell non-finite | fault)` = (for A=I_32) `Pr(fault cell becomes Inf|special) · 1` since only one cell affected per fault
   - If MC = per-cell-averaged: `P_OTR = (1/mn) · sum_cells P_cell_OTR`
3. This single fix likely closes most of L2 (OTR anti-correlation) on top of the bit-by-bit improvement
4. Both formulas must produce identical MC ↔ SOGA numbers if aggregation is aligned

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

### Non-monotonicity statistical test (gm-expert Q6 + test-engineer Q4) — REVISED per Codex iter 1 F3-F5

Given 21-point sweep Pr(SDC | p_k):
1. **Kendall τ**: `tau, p_kendall = scipy.stats.kendalltau(p_list, sdc_vals)` — use p-value as primary statistic (not bare τ threshold per F3)
2. **Sign-test (heuristic only)**: `n_wrong = (np.diff(sdc_vals) < 0).sum()` for SDC ascending hypothesis; used only for plateau detection, not as independent test (per F4)
3. **Bonferroni correction (per F5)**: 3 curves tested (MSK, SDC, OTR) → α_per_test = 0.05/3 ≈ 0.0167. Use `p_kendall < 0.0167` as significance threshold per curve.
4. **Decision rule**:
   - **MONOTONE** (Outcome A): `p_kendall < 0.0167` AND `|τ| > 0.7` (significant + sizeable)
   - **NON-MONOTONE** (Outcome B): `p_kendall > 0.05` AND `n_wrong ≥ 10` (no significant monotonic trend + visible irregularity; n=10 of 20 = majority wrong-sign)
   - **AMBIGUOUS**: between thresholds (report, do not fail)
   - **PLATEAU-DEGENERATE**: SDC range `max-min < 5 · max(σ_MC, 0.01)` → exempt from monotonicity test (insufficient signal vs noise)

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

## Alternatives considered

- **A1 — Keep 5-class, accept limitations**: REJECTED. Lishan will see 38× immediately and the pitch credibility collapses; OTR anti-correlation is even harder to defend at a research-collaboration interview.
- **A2 — Explicit per-bit discrete mixture as SOGA matrix-GM components (33 components × matrix-GM ops)**: REJECTED. No benefit over analytical aggregation (Option 2 was unanimous expert choice in parent plan); adds runtime cost without improving accuracy.
- **A3 — Hybrid 5-class + bit-exact only for OTR-boundary cases**: REJECTED. Complexity not justified; full bit-exact is barely more code.
- **A4 — Full SOGA core extension to handle bit-XOR semantically**: REJECTED. Violates "no libSOGA*.py changes" constraint; ten-fold larger scope; not needed (the fix is at script level).
- **A5 — Strada P calibration against SASSIFI numbers now that we have a precise model**: REJECTED. Re-violates Strada Q discipline; outside scope; reserved for follow-up after Lishan engagement.

---

## Sub-tasks (atomic, milestone-organized)

### R0 — Setup + design lock (0.5 days) [#14](https://github.com/bistrulli/SOGA/issues/14)

- [ ] **[R0.1]** [iter:5] [area:experiments/lib] Write `lib/DESIGN_BIT_EXACT.md` design doc: rationale, IEEE 754 references, the 4 specialist+Codex-found bug fixes (NaN-only guard NOT Inf, float32-cast delta, native endianness, OTR aggregation semantics), comparison vs 5-class, scope.
- [ ] **[R0.2]** [iter:5] [agent:test-engineer] [area:tests/test_bit_fault_table.py] Write 5 hand-computed sanity tests (T1-T5 per test-engineer Q1): sign(1.0)→-1.0, bit 23→2.0, bit 22→1.5, bit 30→+Inf, sign(0)→-0 (assert via `math.copysign`, not `==`). All tests RED initially.
- [ ] **[R0.3]** [iter:5] [agent:soga-internal-expert] [area:experiments] **CODEX F1 CRITICAL**: read `/Users/emilio-imt/git/SOGA/experiments/lishan_resilience_2026-05-25/simulate_fi_mc.py` carefully. Document in `lib/DESIGN_BIT_EXACT.md § OTR_SEMANTICS` whether MC reports per-execution-OTR (any cell non-finite → 1.0) or per-cell-averaged-OTR (mean fraction of non-finite cells). This determines the analytical formula in R2.2b.

### R1 — Bit-fault table module (1 day, agent:numerical-stability-expert) [#15](https://github.com/bistrulli/SOGA/issues/15)

- [ ] **[R1.1]** [iter:5] [agent:numerical-stability-expert] [area:lib/bit_fault_table.py] Implement scalar `compute_xor_shift(v, bit_idx)` WITH all 3 fixes:
  - NaN/Inf input guard (early return)
  - `delta = float(np.float32(v_post)) - float(v_f32)` (float32-cast minuend, Bug B)
  - `struct.pack('!f', np.float32(v))` (no native float64 leak)
- [ ] **[R1.2]** [iter:5] [agent:numerical-stability-expert] [area:lib/bit_fault_table.py] Vectorized `bit_fault_table_array(v_array)` returning shape `(N, 32, 2)`:
  - Use `np.asarray(v_array, dtype=np.float32)` (no '<f4' explicit endianness, per Q3)
  - Propagate non-finite input mask through `is_special`
  - Delta computed in float64
- [ ] **[R1.3]** [iter:5] [agent:test-engineer] [area:tests/test_bit_fault_table.py] 100-random-v validation: `np.random.default_rng(42).uniform(-30, 30, 100)` exponents, all 32 bits each. Assert bit-perfect match against MC `flip_bit` via `numpy.equal` on bit patterns (NOT float `==` due to NaN payload subtleties).
- [ ] **[R1.4]** [iter:5] [agent:numerical-stability-expert] [area:tests/test_bit_fault_table.py] 10 IEEE 754 edge cases (T6-T10 per test-engineer Q1 + extras): smallest subnormal, +Inf input + 31 (→-Inf), +Inf + 30 (→1.0), +Inf + 22 (→NaN), fmax + 0 (unchanged at float32 precision), NaN input guard, -0 sign flip → +0, denormal sign flip, denormal exp flip, scalar-vs-vectorized cross-check.

### R2 — Analytical aggregation rewrite (1.5 days, agent:soga-internal-expert) [#16](https://github.com/bistrulli/SOGA/issues/16)

- [ ] **[R2.1]** [iter:5] [area:scripts] `git mv predict_resilience_soga.py predict_resilience_soga_5class.py` (preserves `git log --follow` history per soga-internal Q2).
- [ ] **[R2.2]** [iter:5] [agent:soga-internal-expert] [area:predict_resilience_soga.py] Implement NEW bit-exact `predict_resilience_soga.py` exposing same names (predict_v_sweep, predict_bimodal_sweep, compute_baseline, compute_per_cell_SDC) but with 32-bit loop instead of 5-class. Reuse compute_baseline and affine_left from matrix-GM (no SOGA core changes per Q3).
- [ ] **[R2.2b]** [iter:5] [agent:soga-internal-expert] [area:predict_resilience_soga.py] **CODEX F1**: align OTR aggregation formula with MC semantics from R0.3. Two cases: (i) per-execution-OTR: P_OTR_exec = Pr(any output cell non-finite given fault); for A=I_32 only the corrupted cell is affected, so P_OTR_exec = special_bit_count / 32. (ii) per-cell-averaged: P_OTR_cell = mean over cells of P(this cell non-finite given fault). Document choice in docstring + design doc. Pure aggregation correctness — independent of bit-by-bit vs 5-class; likely closes most of L2 (OTR anti-correlation).
- [ ] **[R2.3]** [iter:5] [agent:soga-internal-expert] [area:predict_resilience_soga.py] Vectorize aggregation: shape `(n_output_cells, 32_bits, m_input_rows)` broadcast. Target runtime < 200 ms per v-point on 32×32. Profile and document.
- [ ] **[R2.4]** [iter:5] [agent:test-engineer] [area:tests/test_predict_soga_bit_exact.py] 5 cross-validation tests per test-engineer Q2:
  - bit-exact must be MORE accurate than 5-class vs MC at v=1.0
  - bit-exact vs MC at v∈{0.1, 0.5, 1.0, 5.0} with n=5000: rel err < 2% SDC, abs err < 1% OTR/MSK
  - K=33 reduction vs K_full validation at m=n=4
- [ ] **[R2.5]** [iter:5] [area:scripts] Add `--mode {bit_exact, 5_class}` flag to `run_soga_step1.py`, `run_step3.py`, `run_sweep.py`. Default `bit_exact`. **BACKWARD-COMPATIBILITY NOTICE (per Codex iter 1 F6)**: this is a DELIBERATE behavioral change — existing scripts/CI invoking these runners without `--mode` will get bit-exact results (intentional refinement primary). MITIGATION: (a) emit a clear runtime banner `"[MODE] Using bit_exact fault model (refinement primary; --mode 5_class for legacy)"` at script start; (b) document the breaking change in `lib/DESIGN_BIT_EXACT.md` and `CHANGELOG` section of REPORT.md; (c) bump `config_version: 2` in config.json (was 1); old configs missing the field default to `1` with deprecation warning.

### R3 — Step 1 re-validate (0.5 days) [#17](https://github.com/bistrulli/SOGA/issues/17)

- [ ] **[R3.1]** [iter:5] [area:scripts] Re-run v_sweep with `--mode bit_exact`, save `results/soga_step1_bitexact.csv`.
- [ ] **[R3.2]** [iter:5] [area:figures] 3-panel figure `figures/resilience_vs_input_value_3panel.png` (MC | 5-class | bit-exact). Shows bit-exact convergence to MC.
- [ ] **[R3.3]** [iter:5] [agent:test-engineer] [area:tests/test_step1_acceptance_bit_exact.py] 4 acceptance tests per test-engineer Q3:
  - `scipy.stats.pearsonr(soga_sdc, mc_sdc)[0] > 0.95` (D3=a tightened)
  - `max(|soga_SDC − mc_SDC| / mc_SDC) < 2%` where `mc_SDC > 1e-5`
  - `max(|soga_OTR − mc_OTR|) < 1%` absolute
  - `max(|soga_MSK − mc_MSK|) < 1%` absolute
  - Re-run at n=3000 if first n=1000 fails Pearson (escalation path)

### R4 — Step 3 bimodal Bernoulli with bit-exact (1 day, agent:gaussian-mixture-expert) [#18](https://github.com/bistrulli/SOGA/issues/18)

- [ ] **[R4.1]** [iter:5] [agent:gaussian-mixture-expert] [area:predict_resilience_soga.py] Plumb bit-exact into existing `predict_bimodal_sweep(use_two_component=True)` path (NO changes to `lib/bimodal_prior.py` per soga-internal Q4 — that module is pure input-distribution math). The 2-component dispatch already works; just bit-exact the inner kernel.
- [ ] **[R4.2]** [iter:5] [area:scripts] Re-run p_sweep with `--mode bit_exact`, save `results/step3_soga_bitexact.csv`. 21 p-points {0.0, 0.05, …, 1.0}.
- [ ] **[R4.3]** [iter:5] [agent:test-engineer] [area:tests/test_step3_monotonicity_bit_exact.py] 4 statistical tests per gm-expert Q6 + test-engineer Q4:
  - Kendall τ via `scipy.stats.kendalltau` (τ, p_value)
  - Sign-test: `n_wrong = (np.diff(sdc_vals) < 0).sum()`
  - Binomial p-value via `scipy.stats.binomtest`
  - Robust threshold rule (REVISED per Codex iter 1 F3-F5; aligned with Approach section): MONOTONE if `p_kendall < 0.0167` (Bonferroni-corrected per F5) AND `|τ| > 0.7`; NON-MONOTONE if `p_kendall > 0.05` AND `n_wrong ≥ 10`; AMBIGUOUS else; plateau-degenerate exempt (SDC range < 5·σ_MC)
- [ ] **[R4.4]** [iter:5] [area:scripts] MC validation at 4 points `p ∈ {0.0, 0.5, p_critical, 1.0}` with 5000 samples each, seed=42. Binomial CI half-width ≤ 0.007.
- [ ] **[R4.5]** [iter:5] [area:experiments] If non-monotonicity detected per gm-expert Q6 rules: write `counterexample_analysis.md` with physical intuition (which p, which bit pattern, why) + MC support + figure highlight. If monotone confirmed: append "monotonicity confirmation" section to REPORT.md.

### R5 — Documentation overhaul (0.5 days, agent:documentation-writer) [#19](https://github.com/bistrulli/SOGA/issues/19)

- [ ] **[R5.1]** [iter:6] [agent:documentation-writer] [area:experiments/REPORT.md] Rewrite: introduce "Methodological hierarchy" section (5-class historical → bit-exact primary); update all numerical tables; remove L1/L2 "known limitations" section (or move to "Historical iterations"); maintain Strada Q disclaimer.
- [ ] **[R5.2]** [iter:6] [agent:documentation-writer] [area:lishan_pitch.ipynb] Add "Approximation hierarchy" section (MC → bit-exact analytical → 5-class fast approximation); add mode selector widget (dropdown or radio); narrative cell explaining input-side fault interpretability for Lishan.
- [ ] **[R5.3]** [iter:6] [agent:documentation-writer] [area:docs/research-notes/06-input-side-fault-modeling.md] Update Section 4 positioning paragraph: "SOGA matches MC reference at IEEE 754 bit precision (analytical) at ~1000× speedup; gap is purely MC sampling noise."
- [ ] **[R5.4]** [iter:6] [agent:documentation-writer] [area:lishan_discussion_package/] Update README.md (1-page) with bit-exact primary result; update setup.md (CLI now needs `--mode`); regenerate figures/ from new CSVs.

### R6 — Final QA + cross-review (0.5 days) [#20](https://github.com/bistrulli/SOGA/issues/20)

- [ ] **[R6.1]** [iter:6] [agent:code-reviewer] [area:final] Code review on full refinement diff (`git diff aa212f2d..HEAD` or against R0 baseline).
- [ ] **[R6.2]** [iter:6] [agent:codex-cross-reviewer] [area:final] Codex cross-review on cumulative artifact (Phase F of /iterate). Focus: (a) bit-XOR correctness vs IEEE 754 spec; (b) "MC analytical" claim soundness; (c) Pearson > 0.95 achievability; (d) Step 3 monotonicity test rigor; (e) backward-compat preservation.
- [ ] **[R6.3]** [iter:6] [area:tests] Full `pytest tests/ -v` on `experiments/lishan_resilience_2026-05-25/`; assert ≥72+31=103 tests pass, 0 xfail remain (the 2 L1/L2 xfails should xpass now via bit-exact). Update HASHES.txt append-only.
- [ ] **[R6.4]** [iter:6] [area:plan] Update `plan/2026-05-25-lishan-resilience-poc.md` cross-review history with refinement-complete note. Update this plan's cross-review history with iter outcomes.

---

## Test plan

### Unit tests (31 new, per test-engineer Q1-Q7)

| File | Tests | Coverage |
|---|---|---|
| `tests/test_bit_fault_table.py` | 10 (5 hand + 5 edge + 100 random + scalar-vs-vec) | R1 |
| `tests/test_predict_soga_bit_exact.py` | 5 (vs 5-class, vs MC, K=33 reduction) | R2 |
| `tests/test_step1_acceptance_bit_exact.py` | 4 (Pearson, rel/abs err) | R3 |
| `tests/test_step3_monotonicity_bit_exact.py` | 4 (Kendall, sign, binomial, threshold rule) | R4 |
| `tests/test_mc_validation_4points.py` | 3 (binomial CI containment) | R4 |
| `tests/test_regression_72.py` | 3 (count assertion, xfail→xpass promotion) | regression |
| `tests/test_smoke_imports_notebook.py` | 2 (import smoke, nbconvert --execute) | R5 |

### Sanity-vs-analytical
- K_reduced=33 (A=I_32) ↔ K_full=1+5·m·n at m=n=4 (cross-validation)
- bit-exact must outperform 5-class vs MC at v=1.0

### Sanity-vs-MC
- Step 1 full sweep, 1000 samples primary, escalate to 3000 if needed
- Step 3 4 points × 5000 samples each (increased from 1000 for subtle non-monotonicity detection)

### Benchmark regression
- Not applicable: no `libSOGA*.py` changes → no `/soga-bench` impact
- xfail promotion: the 2 existing xfail tests for L1/L2 should xpass with bit-exact → mark `xfail(strict=True)` so future regression is caught

### Audit gates
- Not applicable: no `.g4` or `libSOGA*.py` changes. Skip `/audit-grammar` and `/audit-numerical`.

---

## Acceptance criteria

- [ ] All 31 new tests pass; all 72 existing tests pass (≥103 total, 0 unexpected xfail/xpass)
- [ ] Bit-fault table matches MC reference bit-perfect on 100 random v + 10 edge cases
- [ ] Step 1 Pearson > 0.95 between SOGA bit-exact and MC for **MSK and SDC curves only** (per Codex iter 1 F8; OTR uses absolute error metric because OTR values are sparse/zero in most v-points making Pearson degenerate)
- [ ] Step 1 max rel err < 2% (SDC); max abs err < 1% (MSK)
- [ ] Step 1 OTR: max abs err < 1% (Pearson not applicable due to sparseness)
- [ ] OTR aggregation semantics aligned with MC reference (R0.3 + R2.2b; closes Codex iter 1 F1)
- [ ] Step 3 monotonicity verdict: MONOTONE, NON-MONOTONE, AMBIGUOUS (any of three valid); MC-supported at 4 points
- [ ] If counterexample: `counterexample_analysis.md` written with physical intuition + MC support
- [ ] No regression on existing matrix-GM tests (test_matrix_gaussian, test_update_matrix, etc.)
- [ ] `predict_resilience_soga_5class.py` preserved as historical reference; `git log --follow` still works
- [ ] `--mode {bit_exact, 5_class}` flag works on all 3 runners
- [ ] config.json backward-compatible (old configs still load with defaults)
- [ ] HASHES.txt updated append-only
- [ ] REPORT.md, lishan_pitch.ipynb, lishan_discussion_package/ all reflect bit-exact as primary
- [ ] Codex cross-reviewer APPROVE on final
- [ ] No `libSOGA*.py` modifications; no `.g4` modifications

---

## Rollback

All work on `feat/lishan-resilience-poc`. If refinement abandoned:

1. **At any milestone**: `git reset --hard d8a05904` (iter 4 commit, refinement base). Main and parent branches untouched.
2. **Partial recovery**: each R-N milestone is independently revertible; bit-exact files can be removed, 5-class restored as primary.
3. **5-class always available**: the rename `predict_resilience_soga_5class.py` preserves the original implementation; the `--mode 5_class` flag stays functional.

---

## Estimated complexity

**M+ — 5 days (3-4 effective with parallelism), ~400 new LOC, 31 new tests.**

Per-milestone effort:
- R0 (setup + design): 0.5 day — low risk, atomic
- R1 (bit-fault module): 1 day — medium risk, IEEE 754 edge cases + vectorization
- R2 (aggregation rewrite): 1.5 days — highest risk, integration with existing predictor
- R3 (Step 1 verify): 0.5 day — low risk, mechanical
- R4 (Step 3 hunt): 1 day — medium risk, statistical analysis + potential counterexample
- R5 (docs): 0.5 day — low risk
- R6 (QA + cross-review): 0.5 day — codex review may need 2 iter

**Uncertainty flags**:
- **U1**: Pearson > 0.95 might be slightly under at n=1000 due to MC sampling noise; escalation path to n=3000 documented
- **U2**: Step 3 may or may not reveal counterexample — both outcomes valid per D4=a
- **U3**: Codex cross-review for the refinement may surface details about backward-compat that need iteration (budget 1-2 iter)

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-BE1 | `struct.pack/unpack` float32 round-trip precision loss | LOW | R1.3 catches via MC bit-perfect comparison; numerical-stability Bug B fix ensures float32 cast on both sides |
| R-BE2 | NaN input not guarded → propagation of NaN deltas | MEDIUM | Numerical-stability Bug A fix (refined per Codex iter 1 F2): `np.isnan(v)` ONLY guard at entry (Inf input flows through for per-bit IEEE 754 evaluation, e.g., +Inf bit 30 → 1.0 finite); R1.4 NaN + Inf edge case tests |
| R-BE3 | Endianness `'<f4'` explicit fails on big-endian systems | LOW | Numerical-stability Q3 fix: use `np.asarray(v_array, dtype=np.float32)` native byte order |
| R-BE4 | Step 3 "fake" non-monotonicity from MC sampling noise | MEDIUM | R4.4 5000 samples + sign-test binomial p-value < 0.01 + Kendall τ confidence |
| R-BE5 | Vectorization with `struct.pack` slow at scale | LOW | numpy view tricks fully vectorized (R1.2); profile against scalar at R1.3 |
| R-BE6 | Codex cross-review may need 2-3 iterations | LOW | Budgeted for it; if 3 REJECT escalate to user |
| R-BE7 | Dual-maintenance burden (5-class + bit-exact) | LOW | D2=a: 5-class only for regression; new development on bit-exact only; no public API duplication |
| R-BE8 | xfail promotion to strict=True breaks if mantissa approximation worse than expected | MEDIUM | R6.3 verify xpass before promoting; fallback to strict=False with documented gap |
| R-BE9 | Existing `predict_bimodal_sweep` 2-component path subtly different from spec | LOW | gm-expert Q4 confirmed it implements correct formula; R4.1 just plumbs bit-exact kernel in |
| R-BE10 | `git mv` rename breaks downstream imports unexpectedly | LOW | R2.5 explicit edit of all import lines; CI smoke test catches |
| R-BE11 | **OTR aggregation semantics mismatch (per-execution vs per-cell-averaged)** between MC and analytical model | HIGH | **NEW R0.3 + R2.2b** (added per Codex iter 1 F1). Read MC code to ascertain semantics; align analytical formula. Likely closes most of L2 (OTR anti-correlation) on top of bit-exact fix. |
| R-BE12 | Default `--mode bit_exact` is a behavioral breaking change for existing scripts | MEDIUM | R2.5 mitigation: runtime banner + config_version bump + CHANGELOG entry (per Codex iter 1 F6) |
| R-BE13 | Multiple-testing on 3 curves without Bonferroni inflates false-positive rate | MEDIUM | R4.3 Bonferroni α=0.0167 per curve (per Codex iter 1 F5) |

---

## Cross-review history

- **iter 1: APPROVE_WITH_CHANGES** (codex 2026-05-26; audit at `results/codex_review/iter1-2026-05-26/`) — 1 HIGH + 5 MEDIUM + 3 LOW findings:
  - **F1 (HIGH)**: OTR aggregation semantics mismatch — MC reports execution-level (any cell non-finite), analytical was reporting per-cell-averaged. Fix: NEW R0.3 (read MC code) + R2.2b (align formula). Likely the missing piece that closes L2 even with bit-exact.
  - **F2 (MEDIUM)**: Inf input guard contradiction — `compute_xor_shift(+Inf, 30) = 1.0` (finite, per IEEE 754) but original guard `not np.isfinite(v)` returns (0.0, True). Fix: guard NaN ONLY, let Inf flow through naturally.
  - **F3 (MEDIUM)**: NON-MONOTONE threshold τ < 0.7 — use Kendall τ p-value as primary statistic. Fix: use `p_kendall > 0.05` as evidence-against-monotonicity test.
  - **F4 (MEDIUM)**: n_wrong ≥ 5 redundant under H0=random. Fix: raise to n_wrong ≥ 10 (majority), use as heuristic only.
  - **F5 (MEDIUM)**: Multiple-testing correction missing for 3 curves. Fix: Bonferroni α = 0.05/3 ≈ 0.0167 per curve.
  - **F6 (MEDIUM)**: Default `--mode bit_exact` breaks back-compat silently. Fix: runtime banner + config_version bump + CHANGELOG.
  - **F7 (LOW)**: R-BE11 missing in risk register. Fix: added R-BE11, R-BE12, R-BE13.
  - **F8 (LOW)**: Pearson criterion ambiguous about OTR. Fix: explicit MSK/SDC for Pearson; OTR uses abs err only (sparseness makes Pearson degenerate).
  - **F9 (LOW)**: endianness description cosmetic. No code change.
  - Plan v2 applies all 9 fixes; resubmitted for iter 2.
- **iter 2: APPROVE_WITH_CHANGES** (codex 2026-05-26; audit at `results/codex_review/iter2-2026-05-26/`) — F1-F9 verified applied. 2 residual sync issues:
  - **Issue 1 (MED)**: R4.3 task text contained pre-F3/F4 threshold rule (τ<0.7, n_wrong≥5); Approach section was correctly updated but checklist item was missed. Fixed in iter 3 prep.
  - **Issue 2 (LOW)**: R-BE2 mitigation text still described old "not np.isfinite" guard. Fixed in iter 3 prep.
  - Plan v3 applies both residual sync fixes; resubmitted for iter 3.
- **iter 3: APPROVE** (codex 2026-05-26; orchestrator iter 2 residual sweeps applied) — R4.3 threshold rule re-synced with Approach section (p_kendall < 0.0167, n_wrong ≥ 10); R-BE2 mitigation text updated to `np.isnan(v)` guard with Inf flow-through; stale-string grep verified clean outside cross-review history. Plan locked — proceed to user approval gate.
