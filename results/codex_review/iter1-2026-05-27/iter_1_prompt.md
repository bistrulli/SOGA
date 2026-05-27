You are reviewing the following PLAN artifact for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution applied to GPU resilience analysis).

ROLE: Act as an independent reviewer. Do NOT consider yourself the author. Be skeptical. Surface real problems, not cosmetic ones.

Evaluate it against these criteria (PLAN checklist):
1. COMPLETENESS: are all sub-tasks atomic and verifiable?
2. NOVELTY: does the plan explain what is NEW vs prior art / baselines?
3. METHODOLOGY: is the approach mathematically sound and reproducible?
4. FEASIBILITY: are estimates realistic (time, complexity)?
5. BASELINE COVERAGE: are PSI / Stan / AQUA / BLOG comparisons included where applicable?

ADDITIONAL FOCUS QUESTIONS (answer each explicitly):
(a) Is the cross-family experimental design (Gaussian vs Uniform vs Bimodal at matched variance) a defensible test of monotonicity in the distribution dimension?
(b) Is the FlipTracker cancellation mechanism (zero-mean inputs maximize cancellation in inner products) the correct theoretical justification?
(c) Is the eps=0 (Yang-faithful) + eps>0 (non-monotonicity enabling) dual mode confusing or rigorous?
(d) Are the statistical thresholds (Kendall tau Bonferroni-corrected 0.0056) appropriate for 3 families x 3 categories?
(e) Is the 4-day budget realistic given the architecture has ~600 LOC + 25 tests?
(f) Honesty discipline: is the "adjacent" framing maintained throughout, or does the plan overclaim?
(g) Risk register completeness: what risks are missing?

CRITICAL CONTEXT you must keep in mind:
- Yang's Int, 2mm plot (page 3 of "Input Type") shows MSK curve: MSK(v=-1)=0.85, MSK(v=0)=0.13, MSK(v=6)=~0.28. She labels this section "Monotonic Trend" even though MSK is NOT monotone. SDC IS roughly monotone increasing (0 -> 0.8 -> 0.6). The plan proposes to "analytically explain" this. 
- Yang's Assumption-1 is explicitly stated as: "Resilience-input_value relationship is monotonic." She adds "This monotonic relationship currently holds for all the cases we have seen." The plan interprets this as falsifiable via distribution sweep.
- Yang's working note already tests Binomial distributions (Figs 4, 5 page 2), and her Prediction bars (page 4) show Binomial_1000_0.8, Binomial_100_0.8, Equilikely_0_100, Binomial_10_0.8 alongside "complex_combination" distributions. Her document is already treating distribution families, not just uniform.
- The plan's "primary novelty" claim is "No paper has conducted a systematic sweep over input distribution families for GEMM or 2MM" — but Yang's own document already does exactly this (Binomial vs Equilikely vs complex_combination for 2MM and 3MM).
- Gaussian-mixture-expert memo: single-Gaussian sigma-sweep is monotone with eps=0. Non-monotonicity requires cross-family OR bimodal OR eps>0.
- The plan claims "Bimodal{+/-mu} maximizes cancellation" but for INTEGER 2MM (int32) with exact-match eps=0, the masking condition is D_corrupted == D_golden, not small |delta|. Cancellation in the FP sense may not translate directly to integer exact-match masking.

Artifact:
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
- **No paper** has demonstrated non-monotonic resilience vs input distribution parameters → genuine novelty

### Specialist memos (Phase D, 5 parallel)

- `gaussian-mixture-expert`: **CRITICAL** finding — pure single-Gaussian σ-sweep is monotonic within-family with eps=0.
- `numerical-stability-expert`: **proposes recipe** — eps=10⁻⁶ relative AND OTR threshold |delta_D|>2²⁸ together create peak at σ∈[50,150].
- `soga-internal-expert`: file structure confirmed; reuse `MatrixGaussian.affine_left` for cascade.
- `test-engineer`: reuse R4 statistical machinery (Kendall τ + Bonferroni); 21 points start, 41 if AMBIGUOUS; 25 new tests.
- `paper-replicator`: setting is "adjacent enough" for the falsification claim.

---

## Constraints

1. **NO modifications to libSOGA*.py** — only experiments/ scripts
2. **NO modifications to .g4 grammars**
3. **Strada Q discipline preserved** — input-side fault, no SASSIFI calibration
4. **Branch isolation**: stay on `feat/lishan-resilience-poc`
5. **Preserve prior work**
6. **Honesty discipline**: explicit "input-side ≠ register-level" disclaimer in REPORT
7. **Setting alignment minimum**: 2MM cascade int32
8. **Eps strategy**: use BOTH eps=0 (Yang-faithful, primary) AND eps=10⁻⁶ (non-monotonicity-enabling, secondary)
9. **ETA**: 4 days effective

---

## Approach

### Strategy: cross-family comparison (primary) + bimodal symmetric (secondary)

**Experiment 1 — Cross-family comparison at matched magnitude**:
- σ ∈ {1, 2, 5, 10, 20, 50, 100, 200, 400}
- Gaussian N(0, σ), Uniform[0, sqrt(3)·σ] (matched variance, positive-only), Bimodal{±σ, equal weight}
- Cross-family gap IS the non-monotonicity evidence

**Experiment 2 — Within-family bimodal symmetric sweep**:
- Bimodal{±μ, p=0.5}: parameter μ swept ∈ [1, 400]
- Expected peak in MSK at intermediate μ

**Experiment 3 — Within-family Gaussian σ sweep with eps>0**:
- Standard Gaussian N(0, σ) with eps=10⁻⁶ + OTR threshold

### SOGA matrix-GM reuse

`libMatrixGaussian.MatrixGaussian.affine_left` propagates through K1 and K2.

---

## Sub-tasks (atomic, milestone-organized)

### M0–M9 [29 sub-tasks total, 5 milestones across 4 iterations]

[Full sub-task list as in original plan — M0 setup, M1 int32 fault+distributions, M2 MC reference, M3 SOGA analytical, M4 cross-family exp, M5 bimodal exp, M6 Gaussian+eps>0, M7 MC validation, M8 docs, M9 QA]

---

## Acceptance criteria

- All 25 new tests pass; all 117 existing tests still pass (≥142 total)
- Cross-family experiment shows statistically significant gap (Kendall, Bonferroni α=0.0056)
- Bimodal sweep produces clear verdict
- MC validation |SOGA-MC| < 0.05
- REPORT.md updated with cross-family result + explicit reference to Yang's own non-monotonic MSK curve
- Honest disclaimer preserved

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-NM1 | Cross-family gap statistically insignificant | MEDIUM | Increase MC samples |
| R-NM2 | Bimodal sweep stays monotonic | MEDIUM | Fallback to Experiment 3 |
| R-NM3 | Int32 cascade arithmetic edge cases | MEDIUM | Explicit handling + boundary tests |
| R-NM4 | Yang's data already showing non-monotonicity reduces novelty | LOW | Pitch FRAMING is "we explain the mechanism" |
| R-NM5 | Codex cross-review may flag eps dichotomy | LOW | Document both clearly |
| R-NM6 | Approximation in SOGA Gaussian propagation | LOW | M3.4 tests analytical vs MC |
---

Output format (strict):
VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- <finding 1>
- <finding 2>
RECOMMENDATIONS:
- <action 1>
