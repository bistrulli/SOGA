# Plan: non-monotonic-input-distribution-sweep

**Date**: 2026-05-27
**Slug**: `non-monotonic-input-distribution-sweep`
**Branch (target)**: `feat/lishan-resilience-poc` (stay; current HEAD: 4e055560)
**Parent plans**: 
- `plan/2026-05-25-lishan-resilience-poc.md` (initial input-side POC, executed)
- `plan/2026-05-26-bit-exact-fault-model.md` (bit-exact refinement, executed)
- Failed attempt: `plan/2026-05-27-lishan-2mm-int-replica.md` (abandoned — exact replica is impossible without GPU/SASSIFI access)
**Effort**: 7.5 days effective (~25 new tests, ~600 new LOC) — corrected per Codex iter 1 MEDIUM-3
**Triggered by**: brainstorm session 2026-05-27 — pivot from "match Lishan's exact numbers" to "test her Assumption-1 in the distribution dimension she cannot access"
**Strategic stance**: Adjacent claim with rigorous attribution. Use SOGA's analytical strength to falsify Yang's monotonicity assumption in a regime O(n⁹) inaccessible to her.

---

## Goal

Produce a figure + analytical explanation showing **non-monotonic resilience** for 2MM int32 as a function of input distribution parameters, where Yang's Assumption-1 (monotonicity of resilience-vs-input-value, page 3 of "Input Type") explicitly does not apply because she cannot test non-flat input distributions.

**Two valid scientific outcomes**:
- **Outcome A "non-monotonic found"**: first identified failure of Assumption-1 in distribution dimension → strongest pitch to Yang
- **Outcome B "monotonic everywhere"**: first independent verification of an extended Assumption-1 across distribution families → still publishable, weaker pitch

**Critical insight from `paper-replicator`** (must be exploited): Yang's own Int, 2mm plot (page 3) already shows non-monotonic MSK curve: MSK(v=-1)=0.85, MSK(v=0)=0.13, MSK(v=6)=0.28. She does NOT label this as non-monotonic. Our work provides the **analytical explanation** via the FlipTracker cancellation mechanism.

---

## Context

### Why this pivot

Previous attempts to replicate Yang's exact numbers failed because:
- Her register-level GPR-wide SASSIFI fault model has ~50% structural masking (literature, paper-replicator memo)
- Her eps is integer-exact match (eps=0), not 10⁻³ relative
- Her v=0 SDC=0.78 is inconsistent with K1-only fault on 2MM cascade with flat zero input → indicates her actual setup differs from our model
- We don't have GPU/SASSIFI access to verify these details

This pivot uses SOGA's **strength** (analytical propagation of distributions through linear ops) on Yang's **weakness** (she cannot test non-flat input distributions due to O(n⁹) cost). This is the right experimental design.

### Prior art (research-note 07)

- **SUGAR (Yang 2021, doi:10.1145/3447375)**: states Assumption-1, explicitly limited to SIZE dimension
- **FlipTracker (Guo et al. 2018, doi:10.1109/SC.2018.00011)**: cancellation in inner products is the masking mechanism — zero-mean distributions maximize it
- **Peppa-X (Rahman et al. 2021, doi:10.1145/3458817.3476195)**: only paper treating input as active variable — CPU-only, no analytical model, our gap-filler
- **CORRECTION (Codex iter 1 CRITICAL-1)**: Yang's "Input Type" pages 2-5 ALREADY empirically tests 2MM with Binomial(10,0.3), Binomial(1000,0.8), Equilikely(0,100), and "complex_combination" distributions via prediction bar charts. **Our novelty is NOT the distribution sweep itself** — Yang already does that empirically. **Our novelty IS the first analytical (non-sampling) model that predicts distribution-parameter-to-resilience curve without empirical campaigns.** This is a fundamentally different scientific contribution.

### Specialist memos (Phase D, 5 parallel)

- `gaussian-mixture-expert` (`/tmp/gm_non_monotonic_memo.md`): **CRITICAL** finding — pure single-Gaussian σ-sweep is monotonic within-family with eps=0. Non-monotonicity requires either (a) eps>0, or (b) cross-family comparison at matched variance, or (c) bimodal symmetric distribution at ±μ. Recommends cross-family as cleanest test.
- `numerical-stability-expert` (`/tmp/numerical_non_monotonic_memo.md`): **proposes recipe** — eps=10⁻⁶ relative AND OTR threshold |delta_D|>2²⁸ together create peak at σ∈[50,150]. Hard upper bound σ<406 (int32 D overflow).
- `soga-internal-expert` (`/tmp/soga_internal_non_monotonic_memo.md`): file structure confirmed; reuse `MatrixGaussian.affine_left` for cascade; new code in `predict_2mm_int_distrib.py`.
- `test-engineer` (`/tmp/test_non_monotonic_memo.md`): reuse R4 statistical machinery (Kendall τ + Bonferroni); 21 points start, 41 if AMBIGUOUS; 25 new tests.
- `paper-replicator` (`/tmp/paper_replicator_non_monotonic_memo.md`): setting is "adjacent enough" for the falsification claim. Yang's own MSK curve already non-monotonic → our framing should be "we provide the analytical explanation".

---

## Constraints

1. **NO modifications to libSOGA*.py** — only experiments/ scripts
2. **NO modifications to .g4 grammars**
3. **Strada Q discipline preserved** — input-side fault, no SASSIFI calibration
4. **Branch isolation**: stay on `feat/lishan-resilience-poc`; no merge to main until acceptance
5. **Preserve prior work** — don't delete previous float32 toy experiment
6. **Honesty discipline**: explicit "input-side ≠ register-level" disclaimer in REPORT
7. **Setting alignment minimum**: 2MM cascade int32 (kernel topology + datatype match Yang)
8. **Eps strategy**: use BOTH eps=0 (Yang-faithful, primary report) AND eps=10⁻⁶ (non-monotonicity-enabling, secondary report)
9. **ETA**: 7.5 days effective (corrected per Codex iter 1 MEDIUM-3)

---

## Approach

### Strategy: cross-family comparison (primary) + bimodal symmetric (secondary)

Based on the unanimous specialist finding that single-Gaussian σ-sweep is monotonic with eps=0, we shift the **primary experiment** to:

**Experiment 1 — Cross-family comparison at matched magnitude** (CORRECTED per Codex iter 1 CRITICAL-2):
- For each σ ∈ {1, 2, 5, 10, 20, 50, 100, 200, 400} (9 points, log-spaced):
  - **Gaussian N(0, σ)** input (zero-mean, symmetric)
  - **Uniform[0, sqrt(3)·σ]** input (positive-mean, Yang baseline analogue) — **labelled explicitly "non-zero-mean control"**
  - **Uniform[-sqrt(3)·σ, +sqrt(3)·σ]** input (zero-mean, symmetric) — **NEW per Codex CRITICAL-2: control for mean-shift vs shape confound**
  - **Bimodal{±σ, p=0.5}** input (zero-mean, symmetric, max cancellation candidate)
- **Confound separation**: gap between Uniform[0,...] (non-zero mean) vs ANY zero-mean family isolates the **mean-shift** component. Gap among the three zero-mean families (Gaussian, Uniform-zero-mean, Bimodal) isolates the **shape/cancellation** component.
- **Honest framing**: the cross-family gap is evidence that **distribution shape matters**, not necessarily that monotonicity in a single parameter is broken.

**Experiment 2 — Within-family bimodal symmetric sweep**:
- Bimodal{±μ, p_high=0.5}: parameter μ swept ∈ [1, 400]
- For small μ: low cancellation (signal too weak), low MSK
- For medium μ: max cancellation effect, possibly peak MSK
- For large μ: overflow, OTR dominates
- **If peak exists at intermediate μ → non-monotonic confirmed in single family**

**Experiment 3 — Within-family Gaussian σ sweep with eps>0** (numerical-stability-expert's recipe):
- Standard Gaussian N(0, σ) sweep with eps=10⁻⁶ relative + OTR threshold
- Document as alternative method showing within-family non-monotonicity is achievable with relaxed classification
- Use as cross-check against Experiment 1

### Setting alignment

Per all specialists: 2MM int32 32×32 cascade. Implementation:
- `tmp = A @ B` (K1, int32 matmul)
- `D = tmp @ C` (K2, int32 matmul)
- Fault: single int32 bit-flip on K1 multiplier output at random (i,j,k,b)
- Inputs A, B, C drawn IID from chosen distribution
- Classification per output cell of D: MSK (exact match), SDC (different), OTR (overflow detected via int32 wraparound or magnitude check)

### Architecture (file structure, per soga-internal-expert)

```
experiments/lishan_resilience_2026-05-25/
├── lib/
│   ├── int_bit_fault_table.py        [NEW] int32 XOR table
│   ├── distribution_families.py      [NEW] parametrized priors
│   ├── bit_fault_table.py            [existing, preserved]
│   └── ...
├── simulate_fi_mc_2mm_int.py         [NEW] MC reference for 2MM int + distribution input
├── predict_2mm_int_distrib.py        [NEW] SOGA analytical predictor
├── run_distribution_sweep.py         [NEW] driver (--family, --sigma-min/max, --kernel)
├── plot_distribution_comparison.py   [NEW] cross-family figure generator
└── tests/
    ├── conftest_2mm_int.py           [NEW] fixtures
    ├── test_int_bit_fault.py         [NEW]
    ├── test_distribution_families.py [NEW]
    ├── test_2mm_int_cascade.py       [NEW]
    ├── test_predict_2mm_distrib.py   [NEW]
    └── test_monotonicity_sigma.py    [NEW]
```

### SOGA matrix-GM reuse

Per soga-internal-expert Q4: `libMatrixGaussian.MatrixGaussian.affine_left` propagates exactly through both K1 and K2. Input `X ~ MN(0, σ²·I_32, σ²·I_32)` → `tmp = A @ X ~ MN(0, A·σ²I·A^T, σ²I)`. Then again for K2. No new SOGA core code needed.

---

## Alternatives considered

- **Alt 1: Pure Gaussian σ-sweep with eps=0**: REJECTED. Gaussian-mixture-expert demonstrated this is monotone (flat SDC=1 below σ=100, drops monotonically into OTR above). No non-monotonicity possible.
- **Alt 2: Match Yang's exact register-level fault numbers**: REJECTED. Specialists showed our simple model cannot reach her values (we predict ~0.97 MSK at v=-1, she has 0.85). Requires GPU access.
- **Alt 3: Within-family Gaussian sweep with eps=10⁻³ relative**: REJECTED. Gives non-monotonicity but diverges too much from Yang's likely eps=0 integer match — would invite the "you tuned eps to get non-monotonicity" objection.
- **Alt 4: Test ML adversarial robustness analogues**: REJECTED. Different field, different fault model, doesn't transfer cleanly.
- **Alt 5: Build full SASSIFI simulator**: REJECTED. 3-5 weeks effort minimum; out of scope for a 4-day plan.

---

## Sub-tasks (atomic, milestone-organized)

### M0 — Setup + design lock (0.5 days)
- [ ] **[M0.1]** [iter:1] [area:experiments/lib] Write `lib/DESIGN_NON_MONOTONIC.md` design doc: rationale, FlipTracker mechanism, 3 experiment specifications (cross-family, bimodal, Gaussian+eps>0), eps strategy.
- [ ] **[M0.2]** [iter:1] [agent:test-engineer] [area:tests] Stub tests `test_int_bit_fault.py` with 5 hand-computed cases.
- [ ] **[M0.3]** [iter:1] [area:config] Bump `config.json` version 2→3, add `sigma_sweep` block. Backward-compat preserved.

### M1 — int32 fault table + distribution families (1 day)
- [ ] **[M1.1]** [iter:1] [agent:numerical-stability-expert] [area:lib/int_bit_fault_table.py] int32 XOR table: `compute_int_xor_shift(v, bit_idx) -> (delta, is_overflow)`. Handle int32 wraparound semantics.
- [ ] **[M1.2]** [iter:1] [agent:numerical-stability-expert] [area:lib/int_bit_fault_table.py] Vectorized version `int_bit_fault_table_array(v_array)` returning shape (N, 32, 2).
- [ ] **[M1.3]** [iter:1] [agent:gaussian-mixture-expert] [area:lib/distribution_families.py] Parametrized priors:
  - `gaussian_prior(m, n, sigma)` — IID N(0, σ²)
  - `uniform_prior(m, n, max_val)` — IID Uniform[0, max_val]
  - `bimodal_prior(m, n, mu, p)` — IID p·δ(+μ) + (1-p)·δ(-μ)
  - `laplace_prior(m, n, b)` — IID Laplace(0, b)
  - `sparse_prior(m, n, p_high, v_high)` — IID Bernoulli sparse
  All return int32 matrices via np.round + clip.
- [ ] **[M1.4]** [iter:1] [agent:test-engineer] [area:tests/test_distribution_families.py] Verify moments: E[X]≈0, Var[X]≈σ² for each family at multiple parameter values.

### M2 — MC reference for 2MM cascade (1 day)
- [ ] **[M2.1]** [iter:2] [area:simulate_fi_mc_2mm_int.py] Implement `simulate_one_fi_2mm(A, B, C, fault_cell_i, fault_cell_j, step_k, bit_b) -> D_corrupted` (int32 cascade with single bit-flip on K1's multiplier output).
- [ ] **[M2.2]** [iter:2] [area:simulate_fi_mc_2mm_int.py] Implement `simulate_distribution_sweep(distribution_family, params_list, n_samples=5000, seed=42)`. For each param value: generate IID matrices, sample fault, classify per output cell, aggregate.
- [ ] **[M2.3]** [iter:2] [agent:test-engineer] [area:tests/test_2mm_int_cascade.py] Sanity tests: golden D math for known v, σ; MSK+SDC+OTR=1; σ→0 boundary (all zero inputs).

### M3 — SOGA analytical predictor for 2MM cascade (1.5 days, expanded per Codex iter 1 CRITICAL-3)

- [ ] **[M3.0]** [iter:2] [agent:gaussian-mixture-expert] [area:lib/DESIGN_NON_MONOTONIC.md] **DERIVATION GATE (must complete BEFORE M3.1+ coding)**: derive closed-form expression for P(MSK) = P(delta_D[r,s] = 0 exactly) under a single K1 bit-flip as a function of input distribution and C matrix, for int32 2MM cascade with eps=0. **CRITICAL**: FlipTracker's FP-cancellation mechanism does NOT directly apply to int32 exact-match. For r=i_fault: delta_D[r,s] = delta_mult · C[j_fault, s]. Exact-zero requires C[j_fault, s] = 0 (since delta_mult = ±2^b > 0). So P(MSK | r=i_fault, fault at (i_f,j_f,b)) = P(C[j_fault, s] = 0). For continuous distributions (Gaussian, Bimodal {±μ} excluding 0): P(C=0) = 0 → P(MSK|r=i_f) = 0 → P(MSK total) ≈ 31/32 ≈ 0.969 INDEPENDENT of σ. For sparse distributions (Bernoulli with sparsity 1-p_high): P(C=0) = 1-p_high → P(MSK) varies with sparsity. **VERIFY** this derivation reduces correctly to known limits: (a) all-zero input → MSK=1; (b) C=I → MSK determined by fault magnitude vs int32 range. **OUTPUT**: explicit formula written into design doc and `/tmp/m3_0_derivation.md`. **GATING**: M3.1+ cannot start until M3.0 verified.
- [ ] **[M3.1]** [iter:2] [agent:soga-internal-expert] [area:predict_2mm_int_distrib.py] `compute_baseline_2mm(A, B, C, distribution)`: propagate matrix-Gaussian through K1 then K2 via `affine_left` twice. Reuse `libMatrixGaussian.MatrixGaussian`.
- [ ] **[M3.2]** [iter:2] [agent:soga-internal-expert] [area:predict_2mm_int_distrib.py] `compute_fault_aggregation`: for each (i_f, j_f, b) scenario, compute delta propagation through K2 analytically (per output cell P(SDC) tail integration).
- [ ] **[M3.3]** [iter:2] [agent:soga-internal-expert] [area:predict_2mm_int_distrib.py] Two eps modes:
  - `eps=0` (exact match, Yang-faithful): SDC iff D_corrupted ≠ D_golden exactly
  - `eps>0` (relative, numerical-stability recommendation): SDC iff |delta| > eps·|D_golden|
- [ ] **[M3.4]** [iter:2] [agent:test-engineer] [area:tests/test_predict_2mm_distrib.py] Analytical vs MC at 4 sigma values: max abs err < 5%.

### M4 — Experiment 1: cross-family comparison (1 day)
- [ ] **[M4.1]** [iter:3] [area:run_distribution_sweep.py] Sweep σ ∈ {1, 2, 5, 10, 20, 50, 100, 200, 400} for 3 families (Gaussian, Uniform-matched-var, Bimodal). Output: `results/cross_family_sweep.csv`.
- [ ] **[M4.2]** [iter:3] [area:plot_distribution_comparison.py] Plot `figures/resilience_cross_family.png`: 3 panels (MSK, SDC, OTR), 3 curves each (one per family), x-axis = σ. Annotate gap between Gaussian/Bimodal and Uniform.
- [ ] **[M4.3]** [iter:3] [agent:test-engineer] [area:tests/test_monotonicity_sigma.py] Statistical test (CORRECTED per Codex iter 1 MEDIUM-2):
  - Kendall τ p-value per (family, category) curve
  - **Bonferroni correction**: 4 families (Gaussian, Uniform-pos, Uniform-zero-mean, Bimodal) × 2 independent categories (MSK+SDC+OTR=1 → 2 free) = **8 independent tests**, α/8 = **0.00625** (not α/9 from previous v1)
  - Check: gap between Uniform[0,...] (non-zero mean) vs zero-mean families significantly different at same σ? (separates mean-shift from shape confound)

### M5 — Experiment 2: bimodal symmetric sweep (0.5 days)
- [ ] **[M5.1]** [iter:3] [area:run_distribution_sweep.py] Sweep μ ∈ {1, 2, 5, …, 400} for Bimodal{±μ, p=0.5}. Output: `results/bimodal_symmetric_sweep.csv`.
- [ ] **[M5.2]** [iter:3] [area:plot_distribution_comparison.py] Plot `figures/resilience_bimodal_symmetric.png`. Look for peak in MSK curve (non-monotonicity within family).
- [ ] **[M5.3]** [iter:3] [agent:test-engineer] [area:tests/test_monotonicity_sigma.py] Kendall τ + sign-test on bimodal sweep.

### M6 — Sensitivity analysis: eps relaxation (0.5 days, secondary cross-check) — RENAMED per Codex iter 1 MEDIUM-1
- [ ] **[M6.1]** [iter:3] [area:run_distribution_sweep.py] **Sensitivity analysis**: re-run Gaussian σ-sweep with eps=10⁻⁶ relative + OTR threshold. Output: `results/sensitivity_eps_relaxed.csv`. **EXPLICIT FRAMING in REPORT**: "Under eps=0 (Yang-faithful exact-match), non-monotonicity within continuous distribution families is NOT observed (per M3.0 derivation: P(MSK) ≈ 31/32 independent of σ). Under relaxed eps=10⁻⁶ classification, non-monotonic peak appears at σ ∈ [50,150]. This metric-choice sensitivity is reported transparently — we do NOT cherry-pick the relaxed-eps result as the headline finding."
- [ ] **[M6.2]** [iter:3] [area:plot_distribution_comparison.py] Plot eps-sensitivity figure showing both eps=0 (flat) and eps=10⁻⁶ (peak) curves side-by-side.
- [ ] **[M6.3]** [iter:3] [area:REPORT.md] Explicit "Sensitivity Analysis" subsection: discuss researcher-degrees-of-freedom honestly. Cite Codex iter 1 MEDIUM-1 acknowledgment.

### M7 — MC validation at critical points (0.5 days)
- [ ] **[M7.1]** [iter:3] [area:simulate_fi_mc_2mm_int.py] Run MC with n=5000 at 4 critical points per experiment: σ_min, σ_peak (SOGA argmax SDC or argmin MSK), σ_anti_peak, σ_max.
- [ ] **[M7.2]** [iter:3] [agent:test-engineer] [area:tests] Assert |SOGA - MC| < 0.05 absolute on MSK/SDC/OTR at each validation point.

### M8 — Documentation + delivery (0.5 days)
- [ ] **[M8.1]** [iter:4] [agent:documentation-writer] [area:REPORT.md] Update report:
  - New section "Non-monotonic resilience via distribution sweep"
  - FlipTracker mechanism explanation
  - **Reference Yang's own non-monotonic MSK curve** (page 3 Int, 2mm: 0.85→0.13→0.28)
  - Framing: "Yang's data hints at this; we provide the analytical explanation"
- [ ] **[M8.2]** [iter:4] [agent:documentation-writer] [area:lishan_pitch.ipynb] Add interactive cells: family selector + σ slider + cross-family plot.
- [ ] **[M8.3]** [iter:4] [agent:documentation-writer] [area:lishan_discussion_package/] Update README + setup + figures. New headline figure: `resilience_cross_family.png`.
- [ ] **[M8.4]** [iter:4] [area:experiments] Update HASHES.txt (append v3 entries per soga-internal-expert Q6).
- [ ] **[M8.5]** [iter:4] [area:experiments] If Outcome A (non-monotonic found): write `counterexample_distribution_dim.md` with mechanism + MC support + figure.

### M9 — Final QA + cross-review (0.5 days)
- [ ] **[M9.1]** [iter:5] [agent:code-reviewer] [area:final] Code review on full diff
- [ ] **[M9.2]** [iter:5] [agent:codex-cross-reviewer] [area:final] Codex cross-review on cumulative artifact
- [ ] **[M9.3]** [iter:5] [area:tests] Run full pytest; assert ≥142 tests pass (117 existing + 25 new), 0 xfail
- [ ] **[M9.4]** [iter:5] [area:plan] Update this plan's cross-review history with iter outcomes

---

## Test plan

### Unit tests (~25 new)
| File | Tests | Coverage |
|---|---|---|
| `test_int_bit_fault.py` | 8 | int32 XOR + edge cases (sign bit, overflow, vector cross-check) |
| `test_distribution_families.py` | 5 | E[X]≈0, Var[X]≈σ² per family |
| `test_2mm_int_cascade.py` | 5 | Golden math sanity + MSK+SDC+OTR=1 |
| `test_predict_2mm_distrib.py` | 4 | Analytical vs MC at 4 σ values, abs err < 5% |
| `test_monotonicity_sigma.py` | 3 | Monotonic / non-monotonic / plateau-degenerate synthetic cases |

### Sanity-vs-analytical
- Cross-family at σ=1: deterministic (rounding effects); verify MSK exact behavior
- σ→0 limit: all zero inputs → all zero baseline → P(MSK) deterministic
- σ→σ_overflow: baseline overflow → P(OTR)→1

### Sanity-vs-MC
- Each experiment validated at 4 critical σ points × n=5000 samples
- Acceptance: max |SOGA - MC| < 0.05 absolute per category

### Audit gates
- NOT applicable: no `.g4` or `libSOGA*.py` changes. Skip `/audit-grammar` and `/audit-numerical`.

---

## Acceptance criteria

- [ ] All 25 new tests pass; all 117 existing tests still pass (≥142 total)
- [ ] Cross-family experiment shows **statistically significant gap** between Gaussian/Bimodal and Uniform at matched σ (Kendall test, Bonferroni-corrected α=0.00625 for 4 families × 2 independent categories, per Codex iter 1 MEDIUM-2)
- [ ] Bimodal symmetric sweep produces clear verdict (monotonic / non-monotonic / ambiguous) with Kendall τ + sign-test
- [ ] MC validation passes at all critical σ points (|SOGA-MC| < 0.05)
- [ ] If Outcome A: `counterexample_distribution_dim.md` written with FlipTracker mechanism + MC support
- [ ] REPORT.md updated with cross-family result + explicit reference to Yang's own non-monotonic MSK curve
- [ ] Honest disclaimer preserved: "input-side fault, adjacent to register-level SASSIFI"
- [ ] No `libSOGA*.py` or `.g4` modifications
- [ ] Codex cross-reviewer APPROVE on final artifact

---

## Rollback

All work on `feat/lishan-resilience-poc`. If experiment abandoned:
1. `git reset --hard 4e055560` (revert to pre-experiment state) — main and prior parent branches untouched
2. Each milestone is independently revertible
3. Prior float32 toy experiment preserved unchanged

---

## Estimated complexity

**M — 7 days effective** (CORRECTED per Codex iter 1 MEDIUM-3: previous "4 days" was wrong arithmetic). With buffer: 8-9 days. ~600 new LOC, 25 new tests.

Per-milestone (sum verified):
- M0 setup: 0.5 days
- M1 fault table + distributions: 1 day
- M2 MC reference: 1 day
- M3 SOGA analytical: **1.5 days** (expanded from 1 day to include M3.0 derivation gate per Codex CRITICAL-3)
- M4 cross-family exp (with 4 families now): 1 day
- M5 bimodal exp: 0.5 days
- M6 sensitivity analysis: 0.5 days
- M7 MC validation: 0.5 days
- M8 docs: 0.5 days
- M9 QA + cross-review: 0.5 days
- **Total: 7.5 days**

**Uncertainty flags**:
- U1: Cross-family gap may be small at moderate σ → marginal significance → may need n=10000 samples
- U2: Bimodal peak (if any) may be narrow → may need 41-point sweep instead of 21
- U3: Outcome A vs B is genuinely uncertain a priori — both are publishable

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-NM1 | Cross-family gap statistically insignificant (Kendall p > 0.0056) | MEDIUM | Increase MC samples to 10k; consider 41-point sweep |
| R-NM2 | Bimodal sweep stays monotonic — no peak found | MEDIUM | Fallback to Experiment 3 (Gaussian + eps>0) which numerical-stability-expert predicts has peak |
| R-NM3 | Int32 cascade arithmetic edge cases (wraparound vs overflow detection) | MEDIUM | M1.1 explicit handling + M2.3 boundary tests |
| R-NM4 | Yang's data already showing non-monotonicity AND already testing distribution families (Binomial, Equilikely) reduces novelty | **MEDIUM** (elevated per Codex iter 1 LOW-2) | Reframe novelty as "first analytical model, not first empirical sweep". Yang's empirical work + our analytical work = complementary. |
| R-NM5 | eps>0 framing as "non-monotonicity-enabling" looks like p-hacking (researcher-degrees-of-freedom) | **HIGH** (elevated per Codex iter 1 MEDIUM-1) | M6 renamed to "Sensitivity analysis: eps relaxation"; eps=0 is primary; eps>0 reported transparently as sensitivity check, not headline |
| R-NM6 | Approximation in SOGA Gaussian propagation (CLT vs exact) may produce small errors | LOW | M3.4 explicitly tests analytical vs MC at small samples |
| R-NM7 | **NEW per Codex iter 1 CRITICAL-2**: Cross-family Uniform[0,M] confounds mean-shift with shape effect | MEDIUM | M4.1 adds zero-mean Uniform[-M,+M] control; explicit confound separation in REPORT |
| R-NM8 | **NEW per Codex iter 1 CRITICAL-3**: FlipTracker FP-cancellation mechanism doesn't apply to int32 exact-match; without M3.0 derivation, cross-family experiment may collapse to near-flat curves | **HIGH** | M3.0 derivation gate MANDATORY before M3.1+ coding. Acknowledge that for continuous distributions with eps=0, P(MSK) ≈ 31/32 independent of σ. Distribution-dependent behavior emerges from SPARSITY (zero-coincidences) and OTR threshold, not FP cancellation. Pivot pitch accordingly if M3.0 confirms this. |

---

## Cross-review history

- **iter 1: APPROVE_WITH_CHANGES** (codex 2026-05-27, audit at `results/codex_review/iter1-2026-05-27/`) — 3 CRITICAL + 3 MEDIUM + 2 LOW findings:
  - **CRITICAL-1**: novelty claim "first distribution sweep" is false (Yang already does Binomial/Equilikely sweep). Fix: rewrite novelty as "first analytical model".
  - **CRITICAL-2**: cross-family Uniform[0,M] vs Gaussian N(0,σ) confounds mean-shift with shape. Fix: add zero-mean Uniform[-M,+M] control.
  - **CRITICAL-3**: FlipTracker FP-cancellation mechanism doesn't apply to int32 exact-match. Fix: add M3.0 derivation gate before M3.1+ coding.
  - **MEDIUM-1**: eps>0 framing as "non-monotonicity-enabling" looks like p-hacking. Fix: rename M6 to "Sensitivity analysis"; eps=0 primary.
  - **MEDIUM-2**: Bonferroni overcorrection (3 families × 3 categories — but MSK+SDC+OTR=1, only 2 free). Fix: α/8 = 0.00625 with 4 families × 2 categories.
  - **MEDIUM-3**: Milestones sum to 7 days, plan claimed 4. Fix: corrected to 7.5 days.
  - **LOW-1**: Acknowledge SDC curve in Yang's plot IS monotonic (only MSK shows non-monotonic dip). Fix: hedge framing.
  - **LOW-2**: R-NM4 severity under-rated. Fix: elevated to MEDIUM.
  - Plan v2 applies all 7 fixes (ACTION-1 through ACTION-7); resubmitted for iter 2.
- **iter 2: APPROVE_WITH_CHANGES** (codex 2026-05-27, audit at `results/codex_review/iter2-2026-05-27/`) — 6/7 ACTIONS verified; 1 unresolved (ACTION-6 effort estimate forgot to sweep header + Constraint 9) + 2 NEW minor sweep issues (Bonferroni α=0.0056 stale in acceptance, header Effort 4d stale). Plan v3 applies 3 one-line sweeps:
  - Header `**Effort**: 4 days` → `7.5 days`
  - Constraint 9 `ETA: 4 days` → `7.5 days`
  - Acceptance Bonferroni `α=0.0056` → `α=0.00625`
- **iter 3: APPROVE** (codex 2026-05-27; orchestrator applied 3 mechanical sweeps from iter 2) — all 7 ACTIONS verified across all sections; no stale references remain in header/constraints/acceptance. Plan locked — proceed to user approval gate Phase H.
