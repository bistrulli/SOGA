# Codex cross-review — iter 2 (targeted fix verification)
# Checklist: PLAN (targeted — verify 7 fixes only, do NOT re-review from scratch)

You are reviewing a PLAN artifact for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution). This is iteration 2 of a cross-review loop. Iteration 1 returned APPROVE_WITH_CHANGES with 7 required actions. All 7 actions are claimed to have been applied. Your sole task is to verify whether each fix was correctly applied and whether any NEW issues were introduced by the edits.

## Context
The plan is for an experiment: "non-monotonic-input-distribution-sweep" — testing Yang's Assumption-1 (monotonicity of resilience-vs-input-value) in the distribution dimension using SOGA's analytical propagation of Gaussian Mixtures through 2MM int32 matrix cascade.

## 7 ACTIONS to verify (from iter 1 APPROVE_WITH_CHANGES)

**ACTION-1 (was CRITICAL-1)**: Novelty rewrite — replace "No paper has swept distribution families for GEMM/2MM" (which is false — Yang already does Binomial/Equilikely sweeps) with "Yang already does empirical distribution sweeps; our novelty is the FIRST ANALYTICAL (non-sampling) model that predicts distribution-parameter-to-resilience curve." Check: is this exact repositioning present in Context/Prior Art? Is the false claim removed?

**ACTION-2 (was CRITICAL-2)**: Cross-family Uniform confound — the plan used only Uniform[0, sqrt(3)*sigma] (positive-mean, confounds mean-shift with shape). Fix: add a 4th family Uniform[-sqrt(3)*sigma, +sqrt(3)*sigma] (zero-mean control). Check: does Experiment 1 now have exactly 4 families? Is there explicit confound-separation language (mean-shift vs shape)?

**ACTION-3 (was CRITICAL-3)**: M3.0 derivation gate — a new sub-task must exist BEFORE M3.1 that requires deriving P(MSK) closed-form for int32 exact-match BEFORE coding. FlipTracker FP-cancellation does NOT apply to int32 exact-match. Check: is M3.0 present as a gating task? Does it explicitly note the FlipTracker inapplicability? Does it contain the derivation (for continuous distributions with eps=0: P(MSK) ≈ 31/32 independent of sigma)?

**ACTION-4 (was MEDIUM-1)**: eps framing — Experiment 3 / M6 must be renamed to "Sensitivity analysis" and must explicitly state in REPORT.md that eps>0 non-monotonicity is a sensitivity result, NOT a physical finding. Check: is M6 renamed? Is the framing language present?

**ACTION-5 (was MEDIUM-2)**: Bonferroni fix — previous version used alpha/9 (3 families × 3 categories). Fix: 4 families × 2 independent categories (since MSK+SDC+OTR=1, only 2 are free) = 8 tests, alpha/8 = 0.00625. Check: is the correction applied correctly in M4.3 and acceptance criteria?

**ACTION-6 (was MEDIUM-3)**: Effort estimate — previous version said "4 days" but milestones sum to ~7 days. Fix: update to 7.5 days. Check: is the header and estimated complexity section corrected? Does the sum check out?

**ACTION-7 (was LOW-2)**: Risk register — R-NM4 must be elevated to MEDIUM; R-NM5 must be elevated to HIGH; two new risks R-NM7 (mean-shift confound) and R-NM8 (FP-cancellation non-applicability) must be added. Check: all four changes present?

## Plan artifact (iter 2 version, post-fix)

---
[SEE ATTACHED ARTIFACT — full plan text below]
---

# Plan: non-monotonic-input-distribution-sweep

**Date**: 2026-05-27
**Slug**: `non-monotonic-input-distribution-sweep`
**Branch (target)**: `feat/lishan-resilience-poc` (stay; current HEAD: 4e055560)
**Parent plans**: 
- `plan/2026-05-25-lishan-resilience-poc.md` (initial input-side POC, executed)
- `plan/2026-05-26-bit-exact-fault-model.md` (bit-exact refinement, executed)
- Failed attempt: `plan/2026-05-27-lishan-2mm-int-replica.md` (abandoned — exact replica is impossible without GPU/SASSIFI access)
**Effort**: 4 days effective (~25 new tests, ~600 new LOC)
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
9. **ETA**: 4 days effective

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

---

## Sub-tasks (atomic, milestone-organized)

### M0 — Setup + design lock (0.5 days)
- [ ] **[M0.1]** Write DESIGN_NON_MONOTONIC.md design doc
- [ ] **[M0.2]** Stub tests test_int_bit_fault.py with 5 hand-computed cases
- [ ] **[M0.3]** Bump config.json version 2→3

### M1 — int32 fault table + distribution families (1 day)
- [ ] **[M1.1]** int32 XOR table
- [ ] **[M1.2]** Vectorized version
- [ ] **[M1.3]** Parametrized priors (4 families including zero-mean Uniform)
- [ ] **[M1.4]** Verify moments

### M2 — MC reference for 2MM cascade (1 day)
- [ ] **[M2.1]** simulate_one_fi_2mm
- [ ] **[M2.2]** simulate_distribution_sweep
- [ ] **[M2.3]** Sanity tests

### M3 — SOGA analytical predictor for 2MM cascade (1.5 days, expanded per Codex iter 1 CRITICAL-3)

- [ ] **[M3.0]** [agent:gaussian-mixture-expert] **DERIVATION GATE (must complete BEFORE M3.1+ coding)**: derive closed-form expression for P(MSK) = P(delta_D[r,s] = 0 exactly) under a single K1 bit-flip as a function of input distribution and C matrix, for int32 2MM cascade with eps=0. **CRITICAL**: FlipTracker's FP-cancellation mechanism does NOT directly apply to int32 exact-match. For r=i_fault: delta_D[r,s] = delta_mult · C[j_fault, s]. Exact-zero requires C[j_fault, s] = 0. So P(MSK | r=i_fault, fault at (i_f,j_f,b)) = P(C[j_fault, s] = 0). For continuous distributions (Gaussian, Bimodal {±μ} excluding 0): P(C=0) = 0 → P(MSK|r=i_f) = 0 → P(MSK total) ≈ 31/32 ≈ 0.969 INDEPENDENT of σ. **VERIFY** limits and **OUTPUT** formula into design doc and /tmp/m3_0_derivation.md. **GATING**: M3.1+ cannot start until M3.0 verified.
- [ ] **[M3.1]** compute_baseline_2mm
- [ ] **[M3.2]** compute_fault_aggregation
- [ ] **[M3.3]** Two eps modes
- [ ] **[M3.4]** Analytical vs MC tests

### M4 — Experiment 1: cross-family comparison (1 day)
- [ ] **[M4.1]** Sweep σ for 4 families (Gaussian, Uniform-pos, Uniform-zero-mean, Bimodal)
- [ ] **[M4.2]** Plot
- [ ] **[M4.3]** Statistical test: Kendall τ, Bonferroni α/8 = 0.00625 (4 families × 2 independent categories)

### M5 — Experiment 2: bimodal symmetric sweep (0.5 days)
- [ ] **[M5.1]** Sweep μ for Bimodal
- [ ] **[M5.2]** Plot
- [ ] **[M5.3]** Kendall τ + sign-test

### M6 — Sensitivity analysis: eps relaxation (0.5 days, secondary cross-check) — RENAMED per Codex iter 1 MEDIUM-1
- [ ] **[M6.1]** Sensitivity analysis: re-run with eps=10⁻⁶. EXPLICIT FRAMING: eps=0 primary (P(MSK) ≈ 31/32 flat); eps=10⁻⁶ shows peak as sensitivity check, NOT headline.
- [ ] **[M6.2]** Plot eps-sensitivity figure
- [ ] **[M6.3]** Explicit "Sensitivity Analysis" subsection in REPORT.md

### M7, M8, M9 — Validation, docs, QA (as before)

---

## Estimated complexity

**M — 7 days effective** (CORRECTED per Codex iter 1 MEDIUM-3)
Sum: 0.5+1+1+1.5+1+0.5+0.5+0.5+0.5+0.5 = 7.5 days

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-NM1 | Cross-family gap statistically insignificant | MEDIUM | ... |
| R-NM2 | Bimodal sweep stays monotonic | MEDIUM | ... |
| R-NM3 | Int32 edge cases | MEDIUM | ... |
| R-NM4 | Yang already testing distribution families (Binomial, Equilikely) reduces novelty | **MEDIUM** (elevated per Codex iter 1 LOW-2) | Reframe as "first analytical model, not first sweep" |
| R-NM5 | eps>0 framing looks like p-hacking | **HIGH** (elevated per Codex iter 1 MEDIUM-1) | M6 renamed to sensitivity analysis; eps=0 primary |
| R-NM6 | Approximation errors | LOW | M3.4 tests |
| R-NM7 | **NEW per Codex iter 1 CRITICAL-2**: Cross-family Uniform[0,M] confounds mean-shift | MEDIUM | Zero-mean Uniform control added |
| R-NM8 | **NEW per Codex iter 1 CRITICAL-3**: FlipTracker FP-cancellation doesn't apply to int32 exact-match | **HIGH** | M3.0 derivation gate mandatory |

---

## Cross-review history

- **iter 1: APPROVE_WITH_CHANGES** — 7 actions applied; see audit at results/codex_review/iter1-2026-05-27/
- _iter 2: pending_

---

## Output format (strict)

VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- ACTION-1: VERIFIED or UNRESOLVED — <reason>
- ACTION-2: VERIFIED or UNRESOLVED — <reason>
- ACTION-3: VERIFIED or UNRESOLVED — <reason>
- ACTION-4: VERIFIED or UNRESOLVED — <reason>
- ACTION-5: VERIFIED or UNRESOLVED — <reason>
- ACTION-6: VERIFIED or UNRESOLVED — <reason>
- ACTION-7: VERIFIED or UNRESOLVED — <reason>
NEW_ISSUES:
- <any new issues introduced by the edits, or NONE>
RECOMMENDATIONS:
- <only if APPROVE_WITH_CHANGES or REJECT>
