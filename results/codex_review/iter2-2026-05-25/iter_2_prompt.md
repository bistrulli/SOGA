You are reviewing an UPDATED implementation plan for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution). This is ITER 2 — the plan was revised after ITER 1 findings. Your job is to verify whether the 10 iter-1 fixes were correctly applied, check for new issues introduced by the edits, and issue a final verdict.

Evaluate against these criteria:

1. C1 VERIFICATION: K_max formula correctness. The plan now states K_max = 1 + 5·m = 161 worst case, K=6 for A=I_32. Verify: (a) is 1 + 5·32 = 161 correct? (b) is K=6 for A=I_32 correct (only 1 distinct nonzero A[r,i] value per row, 5 classes + baseline)? (c) is M3.5 cross-validation target (K_full = 1 + 5·m·n on m=n=4 = 1+5·4·4 = 81, not "5121") correct? (d) Is the text "K=5121" in M3.5 and elsewhere consistent with "5·m·n where m=32, n=32" = 5·1024 = 5120 plus baseline = 5121?

2. C2 VERIFICATION: Bimodal warning interval. The plan now warns at p∈[0.2,0.8]. Iter-1 review said warn at "central p range e.g. [0.3,0.7]". Check: (a) Is [0.2,0.8] correct — is p=0.2 genuinely in the "high bimodality" regime where single-Gaussian underestimates tails? (b) The CLT mitigation: D[r,s] = sum_i A[r,i]*B[i,s] where B[i,s] ~ Bernoulli. With m=32 IID Bernoulli terms, Lyapunov CLT applies if Lindeberg condition is met. Is this valid when A=I_32 (only 1 term nonzero per row — NOT a CLT situation for A=I)?

3. C3 VERIFICATION: EXACT vs MOMENT-MATCHED split. The plan now shows EXACT for SIGN/EXP, MOMENT-MATCHED for MANTISSA in the fault class table. Check: (a) Is the label split applied consistently throughout the document (table, paragraph, M1.1, M1.4)? (b) Is there an explicit bound on mantissa approximation error? The plan says "empirically expected <2% relative error on Pr(SDC) (validated in M3.5)" — is this bound justified before empirical validation?

4. M1 VERIFICATION: v=1e-30 reasoning. M1.4 now says "shift ratio (2^k-1) is huge regardless of |v|". Is this correct? For HIGH-EXP k=16, delta = v*(2^16 - 1) ≈ 65535*v and D_base = v (for A=I), so ratio = 65535. This is indeed independent of v. Verified?

5. M2 VERIFICATION: logsf cap ordering. R-LR7 now says "cap is residual safety net AFTER H1, verify empirically in M3.3 unit test." Is the ordering dependency now explicit and the test commitment concrete?

6. M3 VERIFICATION: p_critical definition. M5.4 now defines p_critical = argmax_p |dSDC/dp| from SOGA sweep, fallback p=0.5. Is this unambiguous and complete?

7. M4 VERIFICATION: Degenerate p=0/1. M5.1 now handles sigma^2_p < 1e-30 via deterministic comparison. Is the threshold 1e-30 for sigma^2_p appropriate? Note: sigma^2_p = p*(1-p)*(V_high-V_low)^2; for V_high-V_low ~ 1, sigma^2_p < 1e-30 iff p < 5e-31 or p > 1-5e-31. Is this threshold calibrated to the actual sweep range (p in {0.0, 0.05, ..., 1.0})?

8. M5 VERIFICATION: A matrix specification. M0.3 now explicitly extracts A_kernel from lishan_2mm_32x32.soga and mentions optional A_identity variant. Is the A matrix extraction step complete and unambiguous?

9. M6 VERIFICATION: Pearson threshold. M4.3 now uses 0.90 primary, 0.85 fallback, 3000-sample alternative. BUT the Test Plan section (line ~222) still says "Pearson > 0.95 per curve" and the Acceptance Criteria (line ~249) still says "Pearson corr > 0.95". Check for this inconsistency.

10. M7 VERIFICATION: p_fault semantics. M1.4 now explicitly clarifies p_fault as "per-execution probability of at least one fault event". Is this semantics consistent with M2's simulate_fi_mc.py which samples (fault_cell, bit) pairs?

Also check for any NEW issues introduced by the iter-1 edits. Pay particular attention to internal consistency between sections (e.g., Test Plan vs M4.3 vs Acceptance Criteria).

Artifact:
---
# Plan: lishan-resilience-poc

**Date**: 2026-05-25
**Slug**: `lishan-resilience-poc`
**Branch (target)**: `feat/lishan-resilience-poc` — to be created from `feat/matrix-gm-integration`
**Effort**: 2-3 weeks (10-15 effective days)
**Triggered by**: brainstorm session 2026-05-25 — preparing discussion package for Lishan Yang (GMU; SUGAR/SIGMETRICS 2021; Typhoon/SIGMETRICS SRC 2021)
**Strategic stance**: **Strada Q (adjacent claim)** — input-side fault propagation, no calibration against SASSIFI register-level numbers

---

## Goal

Produce `lishan_discussion_package/`: a self-contained bundle with (a) a qualitative replica of Lishan's resilience-vs-input-value 2MM curve (her "Input Type" working note, Sec. II.3.a), and (b) a parametric test of her Assumption-1 (monotonicity of resilience-vs-input-value) in a regime she explicitly cannot test (non-flat input distributions, what she calls "O(n⁹) not doable" for 2DCONV).

**Two valid outcomes — both warrant continuation of collaboration**:
- **Outcome A "monotone"**: first independent verification of Assumption-1 via symbolic propagation
- **Outcome B "counterexample"**: first identified failure mode of Assumption-1

---

## Context

### Background from `/research` (Phase C)

- Lishan Yang, current affiliation **George Mason University** (moving to **University of Alberta** Fall 2026); SUGAR co-authored with Nie+Jog+Smirni at William & Mary; Typhoon solo SRC abstract.
- **GAP confirmed**: no prior analytical (non-MC) tool exists for fault propagation through matrix kernels. TRIDENT/GPU-Trident/VTRIDENT/V-ABFT all stop at scalar per-instruction probability OR variance bounds, NOT a full output distribution. SOGA's contribution is symbolic propagation of arbitrary input distribution through 2MM with closed-form output mixture.
- **Adjacent precedent**: Ares (DAC 2018) injects faults at the **layer-input** level (weight/activation tensors), validating input-side as a recognized methodology in DNN resilience literature. SOGA extends this to dense linear algebra kernels.
- See `docs/research-notes/06-input-side-fault-modeling.md` for the full 13-reference research note with verified DOIs.

### Why we are choosing Strada Q (adjacent, not proxy)

Honesty discipline: register-level FI (SASSIFI/NVBitFI) injects bit-flips into destination registers **after** instructions execute, propagating intra-kernel. SOGA naturally models **input-side** faults (a cell of input is corrupted before kernel execution). These are *adjacent*, not identical. We deliver an analytical input-side capability Lishan does not have; we do NOT claim to reproduce her register-level numbers. This framing avoids the trap of failing a calibration step mid-execution.

### Why this is achievable in 2-3 weeks

The matrix-GM infrastructure on branch `feat/matrix-gm-integration` already validates:
- `programs/Example/lishan_2mm_32x32.soga` runs in ~1s with <5% MC error
- `libMatrixUpdate` exposes `A @ B`, `X + N`, scalar extract with full API
- `libMatrixGaussian` exposes constructors, sampling, dense materialization
- `docs/MATRIX_GM_SEMANTICS.md` and `docs/MATRIX_GM_QUICKSTART.md` document all formulas

We do NOT need to modify SOGA core. All fault-model logic lives in `experiments/lishan_resilience_2026-05-25/` as research scripts that USE the existing API.

---

## Constraints

1. **NO modifications to `libSOGA*.py`** — only use existing matrix-GM API (`libMatrixGaussian`, `libMatrixUpdate`).
2. **NO modifications to `.g4` grammars** — fault aggregation lives in Python, not in DSL.
3. **NO calibration against SASSIFI/NVBitFI numbers** — Strada Q discipline.
4. **Scope: only 2MM 32×32 float32**. No SYRK/GEMM/3MM/2DCONV. No GPU code.
5. **ε = 10⁻³ relative error default** for MSK/SDC classification; parametric in notebook.
6. **Reproducibility**: seed propagation, `HASHES.txt`, `config.json` per CLAUDE.md §8.
7. **Branch isolation**: all work on `feat/lishan-resilience-poc`, no merge until acceptance.
8. **Honesty discipline**: every figure and REPORT section must carry the input-side ≠ register-level disclaimer.
9. **Validation mandatory**: Step 1 requires MC validation with full sweep; Step 3 requires MC validation at ≥3 suspicious points.

---

## Approach

### Architecture: Option 2 (unanimous expert recommendation)

All four specialists converged on **analytical marginalization at output** rather than explicit mixture propagation through SOGA. Concrete strategy:

1. **SOGA does the matrix-Gaussian baseline only**: compute `D_baseline = A @ B` as a single matrix-Gaussian (1 component for flat input, 2 components for bimodal Bernoulli Step 3). Use `libMatrixUpdate` directly.
2. **Fault aggregation in numpy script** (NOT in SOGA): for each (output cell `(r,s)`, fault class `c`, fault input row `i`), compute the per-component tail-Gaussian `Pr(|D[r,s] - D_baseline[r,s]| > ε·|D_baseline[r,s]|)` analytically. Aggregate weighted by `π_{c,i,j=s}`.
3. **Symmetry reduction (H4, corrected per Codex iter 1)**: for fixed output cell `(r,s)`, only faults in input column `j=s` shift the mean; faults in `j≠s` leave `D[r,s]` distribution unchanged (first reduction, mathematically rigorous). Per-output-cell distinct cases collapse from 5121 to **K_max = 1 + 5·m = 161** in the worst case (5 classes × m row positions + baseline). For specific A structures, K is smaller:
   - **A = I_32** (the matrix used in `programs/Example/lishan_2mm_32x32.soga`): `A[r,i] = 𝟙[i=r]`, so only i=r gives nonzero shift; all other i contribute zero shift (effectively MSK). Per output cell K reduces to **K = 1 + 5 = 6** (one per fault class plus baseline).
   - **General A**: K depends on the number of distinct values among `{A[r,i] : i ∈ [m]}`. Plan uses K=161 as a conservative upper bound; actual K is computed at runtime per (r,s).
   M3.5 cross-validation will verify K against the brute-force K=5121 expansion on m=n=4.

### Fault model: 5-class bit-class mixture (moment-matching exact)

| Class | Bits | P(c) | Shift formula | Output distribution per fault scenario |
|---|---|---|---|---|
| SIGN | 31 | 1/32 | δ = −2·v | **EXACT** Gaussian (deterministic shift, linear in v) |
| HIGH-EXP | 27-30 | 4/32 | δ = v·(2ᵏ−1), k∈{16,32,64,128} | **EXACT** Gaussian per (k, sign of v); 4 sub-components; OTR via overflow check |
| LOW-EXP | 23-26 | 4/32 | δ = v·(2ᵏ−1), k∈{1,2,4,8} | **EXACT** Gaussian per k; 4 sub-components |
| HIGH-MANTISSA | 16-22 | 7/32 | δ = ±v·2ᵖ, p∈[−7,−1] | **MOMENT-MATCHED** single Gaussian (true distribution is discrete mixture over (p, sign), 14 sub-components; collapsed via `E[δ|v]=0, Var[δ|v]=v²·E[2^(2p)]`) |
| LOW-MANTISSA | 0-15 | 16/32 | δ = ±v·2ᵖ, p∈[−23,−8] | **MOMENT-MATCHED** single Gaussian (32 sub-components collapsed); contribution to SDC negligible since δ ≪ ε·|μ_golden| typically |

**Correctness scope (corrected per Codex iter 1)**:
- For **SIGN/EXP** classes, the per-(class, bit, v) shift is deterministic, so the conditional output `D[r,s] | fault_scenario` is **exactly Gaussian** (assuming Gaussian input prior). Aggregating across sub-bits per class keeps the mixture exact.
- For **MANTISSA** classes, the shift δ depends on both `p` (random uniform over bit-positions) and sign (random ±1). The true conditional output is a discrete mixture of Gaussians (14 components for HIGH-MANTISSA, 32 for LOW-MANTISSA). We collapse this to a single Gaussian via moment-matching: `E[δ|v]=0` (by sign symmetry), `Var[δ|v] = v² · E[2^(2p)] = v² · (4^(low) · (4^(high−low+1) − 1)) / (3 · (high−low+1))`. This is an **approximation** with bounded error; M1.4 includes a sanity test comparing the moment-matched approximation against the full discrete-mixture computation on a small case.
- **Bound on mantissa approximation error**: for the SDC tail at threshold ε·|μ_golden|, the moment-matched single-Gaussian gives the correct first two moments. Tail probability differs from the true discrete-mixture only by higher-moment correction terms; empirically expected <2% relative error on Pr(SDC) (validated in M3.5).

### Numerical recipe (from `numerical-stability-expert`)

- **OTR pre-classification (H1)**: detect overflow in log-space `log|v| + k·log(2) + log|A[r,i]| > 127·log(2)` before any tail computation. If overflow, directly classify as OTR.
- **Tail-prob (H2)**: use `scipy.stats.norm.logsf` + `logsumexp` for SDC; cap at 1.0.
- **Vectorization (H3)**: numpy broadcast over (r,s); per-class loop. Expected runtime <1s for full v sweep.
- **Dtype discipline**: float64 internal, float32 only for overflow classification.

### Step 1 flow (replica)

```
for v in v_sweep:
    M_B = v · ones(32,32); U_B = ε·I_32; V_B = ε·I_32
    D_base = SOGA(A @ B)
    for (r,s) in output_cells:
        for c in classes:
            P_OTR_c, mask_OTR = check_overflow(v, c, A[r,:])
            for i in rows[~mask_OTR]:
                δ = shift(c, v); μ_post = μ_base + δ·A[r,i]
                P_SDC += tail_gauss(μ_post, σ_base, ε·|μ_base|)
        Pr_MSK_v, Pr_SDC_v, Pr_OTR_v = aggregate(weights, ...)
    plot(v, MSK/SDC/OTR)
```

### Step 3 flow (counterexample hunt)

```
for p in linspace(0, 1, 21):
    D_base_high = SOGA(A @ (V_high · ones))
    D_base_low  = SOGA(A @ (V_low  · ones))
    Pr_MSK_p, Pr_SDC_p, Pr_OTR_p = aggregate_bimodal(p, ...)
    if p in {0.0, 0.5, 1.0}:
        Pr_*_mc_p = simulate_mc(p, ...)
plot(p, MSK/SDC/OTR)
test_monotonicity(curves)
if not monotonic and MC confirms: COUNTEREXAMPLE_FOUND
```

---

## Sub-tasks (atomic, milestone-organized)

### M0 — Branch setup + scaffold (2 days)

- [ ] **[0.1]** Create branch.
- [ ] **[0.2]** Create experiments directory with config.json schema.
- [ ] **[0.3]** **A matrix specification**: extract A_kernel from lishan_2mm_32x32.soga, save as A_kernel.npz. Document its structure. If A=I_32 preferred, generate A_identity.npz as second config. Verify lishan_2mm_32x32.soga runs end-to-end.
- [ ] **[0.4]** Create test_smoke.py.

### M1 — Fault model module (3 days)

- [ ] **[1.1]** Implement FaultClass enum. SIGN/EXP: return list of (sub_class_k, P_sub, delta_deterministic) — EXACT. MANTISSA: return (P_class, E_delta=0, Var_delta=v²·E[2^(2p)]) — moment-matched, documented as APPROXIMATE.
- [ ] **[1.2]** Implement check_overflow in log-space per H1.
- [ ] **[1.3]** Implement shift_moments.
- [ ] **[1.4]** Unit tests (5 edge cases + mantissa moment-match validation):
  - v=0 → all MSK
  - v=1e-30, p_fault=1 → SDC for HIGH-EXP because "shift ratio (2^k-1) is huge regardless of |v|"
  - p_fault=0 → all MSK
  - p_fault=1: MSK >= P(LOW-MANTISSA) − 0.05 ≈ 0.45; p_fault semantics = "per-execution probability of at least one fault event"
  - v=1e38 → OTR >= P(HIGH-EXP)·p_fault = 0.125 when p_fault=1
  - NEW: mantissa moment-match vs full discrete-mixture on m=n=2; max rel err < 5%
- [ ] **[1.5]** Implement tail_gauss vectorized.

### M2 — MC reference (2 days)

- [ ] **[2.1-2.5]** MC simulation implementation and CLI.

### M3 — SOGA analytical prediction (3 days)

- [ ] **[3.1]** compute_baseline using libMatrixGaussian.
- [ ] **[3.2]** compute_per_cell_SDC with K_per_cell = 1 + 5·m_distinct where m_distinct = number of distinct nonzero values in {A[r,i]: i in [m]}. For A=I_32: K=6. For full-A: K≤161.
- [ ] **[3.3]** Overflow pre-classification + log-space tail. Verify scipy boundary at z≈38.5 empirically.
- [ ] **[3.4]** Vectorize over (r,s).
- [ ] **[3.5]** Validation: K reduction (worst case K=1+5·m, A-dependent) must match K=full expansion (1 + 5·m·n) on m=n=4. Run both, compare P_SDC element-wise (max |Δ| < 1e-10 for SIGN/EXP; max rel err < 5% MANTISSA). Test A=I_4 (K=6 per cell) and A=full (K=21 per cell). This certifies symmetry reduction AND mantissa moment-match bound.
- [ ] **[3.6]** CLI run_soga_step1.py.

### M4 — Step 1 verification (1 day)

- [ ] **[4.1]** Orchestrator combining MC + SOGA.
- [ ] **[4.2]** Plot with MC (solid) + SOGA (dashed).
- [ ] **[4.3]** Acceptance gate: pearson(MC, SOGA) > 0.90 per curve (PRIMARY; down from 0.95). Fallback > 0.85. Alternative: 3000 samples if < 0.90. Also max_rel_err < 10% MSK/SDC, max_abs_err < 5% OTR.

### M5 — Step 3 parametric counterexample (3 days)

- [ ] **[5.1]** bimodal_to_matrix_gaussian:
  - M_B = (p·V_high + (1-p)·V_low) · ones(m,n)
  - sigma²_p = p·(1-p)·(V_high - V_low)²
  - U_B = sigma²_p · I_m, V_B = I_n
  - CORRECTED: emit BimodalApproximationWarning when p ∈ [0.2, 0.8] (max bimodality)
  - CLT justification: D[r,s] = sum_i A[r,i]·B[i,s] is approximately Gaussian by Lyapunov CLT over m=32
  - Degenerate case: sigma²_p < 1e-30 → deterministic comparison
- [ ] **[5.2]** Extend compute_per_cell_SDC to accept non-trivial U_B, V_B.
- [ ] **[5.3]** Sweep p in {0.0, 0.05, ..., 1.0} (21 points).
- [ ] **[5.4]** MC at 4 points: {0.0, 0.5, p_critical, 1.0}. p_critical = argmax_p |dSDC/dp| from SOGA sweep; fallback p=0.5 if monotone. Degenerate p=0,1 handled per M5.1.
- [ ] **[5.5]** Monotonicity test: Kendall tau > 0.9 for monotone, < 0.5 for strongly non-monotone.
- [ ] **[5.6]** Plot resilience_vs_bimodal_p.png.
- [ ] **[5.7]** If counterexample: counterexample_analysis.md.

### M6 — Deliverable bundle (2 days)

- [ ] **[6.1-6.8]** REPORT.md, pitch notebook, discussion package, HASHES.txt, code review, final Codex cross-review.

---

## Test plan

### Unit tests
- tests/test_fault_model.py — 5 edge cases per H5 (M1.4)
- tests/test_predict_soga.py — K=26 vs K=5121 cross-validation on m=n=4 (M3.5)
- tests/test_step1_acceptance.py — Pearson + max rel err thresholds (M4.3)

### Sanity-vs-analytical
- Baseline D = A @ B validated against lishan_2mm_32x32.soga (M0.3, M3.1)
- Symmetry reduction K=26 = K=5121 (M3.5)

### Sanity-vs-MC
- Step 1: full sweep, 1000 samples per v point, ~10 v points (M4.3)
- Step 3: 3 representative p points, 1000 samples each (M5.4)
- Acceptance: Pearson > 0.95 per curve; max rel err < 10% MSK/SDC; max abs err < 5% OTR

### No benchmark regression
- No libSOGA*.py modifications

---

## Acceptance criteria

- All 5 fault-model edge cases pass (H5)
- Symmetry reduction K=26 ↔ K=5121 validated (M3.5)
- Step 1 replica: Pearson corr > 0.95 between MC and SOGA per (MSK/SDC) curve
- Step 1 max rel err < 10% (MSK, SDC); max abs err < 5% (OTR)
- Step 3 sweep complete; monotonicity test produces clear verdict
- At least 3 MC-validated points in Step 3
- REPORT.md with input-side disclaimer top-front
- lishan_pitch.ipynb runs end-to-end
- lishan_discussion_package/ self-contained
- HASHES.txt + config.json for reproducibility
- No scope creep

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-LR1 | K=26 symmetry reduction wrong | HIGH | M3.5 cross-validation |
| R-LR2 | Bimodal Bernoulli Gaussian approx breaks for p∈[0.2,0.8] (max bimodality), NOT at extremes | HIGH | M5.1 warning; M5.4 MC at p=0.5; 2-component GM fallback |
| R-LR3 | High-exp overflow handling misclassifies near-boundary | MEDIUM | M1.2 + M1.4 edge case |
| R-LR4 | Pearson < 0.95 | MEDIUM | Diagnose + sample 10000 |
| R-LR5 | Counterexample only at extreme parameter | LOW | M5.7 MC confirmation |
| R-LR6 | Lishan affiliation change | LOW | Engage now |
| R-LR7 | scipy.stats.norm.logsf returns -inf | MEDIUM | Cap at -750 AFTER H1; scipy boundary z≈38.5; verify empirically in M3.3 |
| R-LR8 | MC runtime too long | LOW | Vectorize; reduce to 500 samples |

---

Output format (strict):
VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- <finding 1>
- <finding 2>
RECOMMENDATIONS:
- <action 1>
