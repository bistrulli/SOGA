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
   M3.5 cross-validation will verify K_reduced against the brute-force K_full=1+5·m·n=81 expansion at m=n=4 (the production 32×32 case would be K_full=5121, but we validate at m=n=4 for tractability).

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
- **Bound on mantissa approximation error**: for the SDC tail at threshold ε·|μ_golden|, the moment-matched single-Gaussian gives the correct first two moments. Tail probability differs from the true discrete-mixture only by higher-moment correction terms. **Design target: <5% relative error on Pr(SDC); enforced by M3.5 cross-validation** (Codex iter 2 N3 fix — single enforceable criterion, not two conflicting bounds).

### Numerical recipe (from `numerical-stability-expert`)

- **OTR pre-classification (H1)**: detect overflow in log-space `log|v| + k·log(2) + log|A[r,i]| > 127·log(2)` before any tail computation. If overflow, directly classify as OTR.
- **Tail-prob (H2)**: use `scipy.stats.norm.logsf` + `logsumexp` for SDC; cap at 1.0.
- **Vectorization (H3)**: numpy broadcast over (r,s); per-class loop. Expected runtime <1s for full v sweep.
- **Dtype discipline**: float64 internal, float32 only for overflow classification.

### Step 1 flow (replica)

```
for v in v_sweep:                                    # ~10 input values
    M_B = v · ones(32,32); U_B = ε·I_32; V_B = ε·I_32  # flat input
    D_base = SOGA(A @ B)                             # 1 matrix-Gaussian
    for (r,s) in output_cells:                       # 1024
        for c in classes:                            # 5
            P_OTR_c, mask_OTR = check_overflow(v, c, A[r,:])  # H1
            for i in rows[~mask_OTR]:                # ≤32
                δ = shift(c, v); μ_post = μ_base + δ·A[r,i]
                P_SDC += tail_gauss(μ_post, σ_base, ε·|μ_base|)
        Pr_MSK_v, Pr_SDC_v, Pr_OTR_v = aggregate(weights, ...)
    plot(v, MSK/SDC/OTR)
```

Total: ~10 v-points × 5M tail-prob calls (vectorized) → ~10s end-to-end.

### Step 3 flow (counterexample hunt)

```
for p in linspace(0, 1, 21):                                # bimodal Bernoulli
    # 2-component baseline: B ~ p·δ(V_high) + (1-p)·δ(V_low)
    D_base_high = SOGA(A @ (V_high · ones))                 # 1st baseline component
    D_base_low  = SOGA(A @ (V_low  · ones))                 # 2nd baseline component
    # Apply 5-class fault to BOTH and aggregate
    Pr_MSK_p, Pr_SDC_p, Pr_OTR_p = aggregate_bimodal(p, ...)
    if p in {0.0, 0.5, p_critical, 1.0}:                    # MC validate (4 points per M5.4)
        Pr_*_mc_p = simulate_mc(p, ...)
plot(p, MSK/SDC/OTR)
test_monotonicity(curves)
if not monotonic and MC confirms: COUNTEREXAMPLE_FOUND
```

---

## Alternatives considered

- **Option 1 — Explicit 5121-component mixture in SOGA DSL**: rejected. K=5121 × per-component `A@B` ≈ 168M ops, ~10-30s; subsequent ops blow up further; pruning to K=100 destroys accuracy (98% of fault info lost).
- **Option 3 — Per-class grouped mixture (K=6 in SOGA)**: rejected for v1. Requires deriving closed-form for output variance under random input-cell selector; tractable but non-trivial for 2-week budget. Optional v2 follow-up if Lishan engages.
- **Strada P — Proxy claim with SASSIFI calibration**: rejected upstream by user decision. Risk of mid-execution failure if bridging input-side ↔ register-level doesn't quadrate.
- **Discretize random input then apply deterministic shifts (Approach C from PPL expert)**: rejected. Blows up to K_input × 5121 components, no benefit over moment-matching.

---

## Sub-tasks (atomic, milestone-organized)

### M0 — Branch setup + scaffold (2 days) [#6]

- [ ] **[0.1]** [iter:1] [area:setup] Create branch `feat/lishan-resilience-poc` from `feat/matrix-gm-integration`. Push with `--set-upstream`.
- [ ] **[0.2]** [iter:1] [area:experiments] Create `experiments/lishan_resilience_2026-05-25/` directory with subdirs `results/`, `figures/`, `lib/`. Add `config.json` schema (seed, ε, v_sweep, p_sweep, p_fault).
- [ ] **[0.3]** [iter:1] [agent:soga-internal-expert] [area:experiments] **A matrix specification (Codex iter 1 R1, R5)**: extract the deterministic kernel matrix `A_kernel` from `programs/Example/lishan_2mm_32x32.soga` and save as `experiments/.../A_kernel.npz` (32×32 float64). Document its structure in `config.json` (e.g., banded with values {0.0, 0.1, 0.2, 0.3, 0.5, 1.0} per the existing `.soga` literal). The plan supports any A; K_max=161 is the upper bound. If A=I_32 is preferred for the headline plot (cleanest K=6 reduction), generate a second config `A_identity.npz` and run BOTH as plot variants. Verify `lishan_2mm_32x32.soga` runs end-to-end; capture baseline `(M_D, U_D, V_D)`. Store in `experiments/.../results/baseline_DUV.npz`.
- [ ] **[0.4]** [iter:1] [area:tests] Create `experiments/lishan_resilience_2026-05-25/tests/test_smoke.py` with import-test + baseline-loadability test.

### M1 — Fault model module (3 days) [#7]

- [ ] **[1.1]** [iter:1] [agent:gaussian-mixture-expert] [area:experiments/lib/fault_model.py] Implement `FaultClass` enum (SIGN, HIGH_EXP, LOW_EXP, HIGH_MANTISSA, LOW_MANTISSA) with class-specific exact formulas vs moment-matched approximations (corrected per Codex iter 1 C3):
  - **SIGN/EXP**: return list of (sub_class_k, P_sub, δ_deterministic) — EXACT representation as discrete sub-components (4 for HIGH-EXP, 4 for LOW-EXP, 1 for SIGN)
  - **MANTISSA**: return (P_class, E_δ=0, Var_δ=v²·E[2^(2p)]) — moment-matched single Gaussian; document approximation in docstring
  - All formulas from `gaussian-mixture-expert` memo Q4. Module docstring must explicitly distinguish EXACT (SIGN/EXP) from APPROXIMATE (MANTISSA) handling.
- [ ] **[1.2]** [iter:1] [agent:numerical-stability-expert] [area:experiments/lib/fault_model.py] Implement `check_overflow(v, class_c, A_ri) -> (is_overflow, log_magnitude)` in **log-space**. Per H1: `log|v| + k·log(2) + log|A[r,i]| > 127·log(2)`.
- [ ] **[1.3]** [iter:1] [agent:gaussian-mixture-expert] [area:experiments/lib/fault_model.py] Implement `shift_moments(v, class_c) -> (E_delta, Var_delta)` returning conditional moments per cell value. SIGN/EXP: `Var=0`. MANTISSA: `Var = v² · E[2^(2p)]` formula from gm-expert memo.
- [ ] **[1.4]** [iter:1] [agent:test-engineer] [area:tests/test_fault_model.py] Unit tests on 5 mandatory edge cases (per H5; corrected per Codex iter 1 M1, M7):
  - `v=0` → all MSK (all shifts are zero)
  - `v=1e-30, p_fault=1` → mostly SDC because HIGH-EXP class produces relative shift `2^k − 1` independent of |v| (correction from Codex M1: the reasoning is "shift ratio is huge", not "|shift| absolute is large")
  - `p_fault=0` → all MSK regardless of v
  - **Disambiguation of `p_fault` semantics (Codex M7)**: `p_fault` is the PROBABILITY OF AT LEAST ONE FAULT EVENT IN THE KERNEL EXECUTION (per-execution probability). Given a fault occurs, the cell is chosen uniformly over mn and the bit-class is chosen by P(c). At `p_fault=1`, the test asserts `Pr_MSK ≥ P(LOW-MANTISSA) − 0.05 ≈ 0.45` because most low-mantissa flips produce δ ≪ ε·|μ_golden|
  - `v=1e38` → OTR ≥ P(HIGH-EXP) · p_fault = 0.125 · p_fault (when p_fault=1: ≥0.125)
  - **NEW edge case (Codex iter 1)**: mantissa moment-match validation — compare moment-matched Var[δ|v]·A² against full discrete-mixture expansion on m=n=2 case; max rel err on Pr(SDC) < 5%
- [ ] **[1.5]** [iter:1] [area:lib] Implement `tail_gauss(mu_post, sigma_post, threshold) -> P_SDC` using `scipy.stats.norm.logsf`/`logcdf` + `logsumexp` per H2. Vectorize over array inputs.

### M2 — MC reference (2 days) [#8]

- [ ] **[2.1]** [iter:2] [area:experiments/simulate_fi_mc.py] Implement `flip_bit(value, bit_index) -> float32` via `struct.pack('!f', value)`, XOR, `struct.unpack`. Verify on 5 known cases (sign flip on 1.0 → -1.0, etc.).
- [ ] **[2.2]** [iter:2] [area:experiments/simulate_fi_mc.py] Implement `simulate_one_fi(A, B, fault_cell, bit) -> D_perturbed`. Apply `flip_bit` on `B[fault_cell]`, recompute `D = A @ B_perturbed`. Pure numpy float32.
- [ ] **[2.3]** [iter:2] [area:experiments/simulate_fi_mc.py] Implement `simulate_v_sweep(A, v_list, n_samples=1000, seed) -> {v: (MSK, SDC, OTR)}`. For each v: generate flat B, sample 1000 (fault_cell, bit) pairs uniformly, classify each D_perturbed[r,s] vs D_baseline[r,s] using ε=1e-3, aggregate per-output-cell.
- [ ] **[2.4]** [iter:2] [area:experiments] CLI script `run_mc_step1.py` that calls `simulate_v_sweep` and saves CSV `results/mc_step1.csv` with columns `(v, MSK, SDC, OTR)`.
- [ ] **[2.5]** [iter:2] [area:tests] Smoke test: run `run_mc_step1.py` with 3 v-points × 50 samples; assert runtime < 30s and CSV well-formed.

### M3 — SOGA analytical prediction (3 days) [#9]

- [ ] **[3.1]** [iter:2] [agent:soga-internal-expert] [area:experiments/predict_resilience_soga.py] Implement `compute_baseline(A, M_B, U_B, V_B) -> (M_D, U_D, V_D)` using `libMatrixGaussian.MatrixGaussian` constructor + `_matrix_affine_left` directly. Cross-validate against `lishan_2mm_32x32.soga` output.
- [ ] **[3.2]** [iter:2] [agent:gaussian-mixture-expert] [area:experiments/predict_resilience_soga.py] Implement `compute_per_cell_SDC(M_D, U_D, V_D, A, v, eps, p_fault) -> {(r,s): (P_MSK, P_SDC, P_OTR)}` per the aggregate formula (Q6 of gm-expert memo). Use H4 symmetry reduction: per output cell only j=s faults shift the mean. Per Codex iter 1 R1: K_per_cell = 1 + 5·m_distinct where m_distinct is the number of distinct nonzero values among `{A[r,i] : i ∈ [m]}`. For A=I_32 worst case is K=6; for full-A worst case is K=161.
- [ ] **[3.3]** [iter:2] [agent:numerical-stability-expert] [area:experiments/predict_resilience_soga.py] Wrap with overflow pre-classification (H1) and log-space tail aggregation (H2). Ensure float64 internal, float32 only for overflow check (H6).
- [ ] **[3.4]** [iter:2] [area:experiments/predict_resilience_soga.py] Vectorize over (r,s) using numpy broadcast (H3). Profile runtime — target <1s per v.
- [ ] **[3.5]** [iter:2] [agent:test-engineer] [area:tests/test_predict_soga.py] Validation: K reduction (worst case K=1+5·m, A-dependent) must match K=full expansion (1 + 5·m·n) on a small case m=n=4. Run both, compare per-cell `P_SDC` element-wise (max |Δ| < 1e-10 for SIGN/EXP-only computation; max rel err < 5% when MANTISSA classes included, since mantissa is moment-matched). Test both A=I_4 (K=6 per cell) and A=full (K=21 per cell). This certifies the symmetry reduction AND the mantissa moment-match bound.
- [ ] **[3.6]** [iter:2] [area:experiments] CLI script `run_soga_step1.py` for the v-sweep. Output CSV `results/soga_step1.csv`.

### M4 — Step 1 verification (1 day) [#10]

- [ ] **[4.1]** [iter:3] [area:experiments/run_sweep.py] Orchestrator script combining MC + SOGA Step 1; compute Pearson correlation per curve (MSK, SDC, OTR).
- [ ] **[4.2]** [iter:3] [area:experiments/figures] Plot `resilience_vs_input_value.png` with 3 curves overlaid: MC (solid), SOGA (dashed), per category. Plot title carries input-side disclaimer.
- [ ] **[4.3]** [iter:3] [area:tests/test_step1_acceptance.py] Acceptance gate (corrected per Codex iter 1 M6): assert `pearson(MC, SOGA) > 0.90` per curve as PRIMARY threshold (down from 0.95; 10-point Pearson is statistically fragile at 1000 MC samples). Fallback `> 0.85` with documented gap explanation. Alternative path: increase MC samples to 3000 per v-point if Pearson < 0.90 on first run. Also: `max_rel_err < 10%` on MSK/SDC, `max_abs_err < 5%` on OTR.

### M5 — Step 3 parametric counterexample (3 days) [#11]

- [ ] **[5.1]** [iter:3] [agent:gaussian-mixture-expert] [area:experiments/lib/bimodal_prior.py] Implement `bimodal_to_matrix_gaussian(p, V_low, V_high, m, n) -> (M_B, U_B, V_B)` using moment-matching:
  - `M_B = (p·V_high + (1-p)·V_low) · ones(m,n)`
  - per-cell variance `σ²_p = p·(1-p)·(V_high - V_low)²`
  - `U_B = σ²_p · I_m`, `V_B = I_n` (Kronecker; cells IID)
  - **CORRECTED VALIDITY (Codex iter 1 C2 — warning was inverted)**: emit `BimodalApproximationWarning` when **`p ∈ [0.2, 0.8]`** (high bimodality regime, maximum at p=0.5, where the true Bernoulli distribution is far from Gaussian — single-Gaussian moment-match systematically underestimates SDC tails because the two true modes themselves may lie in the tail region). At `p ∈ {0, 1}` the distribution is degenerate (point mass), so the Gaussian variance σ²_p=0 reduces correctly to a deterministic baseline (handled by M5.2 degenerate-case path).
  - **A-dependent baseline choice (CORRECTED per Codex iter 2 C2(b))**: the CLT mitigation only applies for **dense A** with many nonzero entries per row. For `A = I_32` (the showcase configuration), `D[r,s] = A[r,r]·B[r,s] = B[r,s]` is a SINGLE scaled Bernoulli — CLT does NOT apply. Decision tree:
    * **A dense (≥ m/2 nonzero per row, e.g., A_kernel from `lishan_2mm_32x32.soga`)**: use single-Gaussian moment-match on B; D[r,s] is approximately Gaussian by Lyapunov CLT over the m terms. Validate via M5.4 MC at p=0.5.
    * **A = I_32 or sparse A (< m/4 nonzero per row)**: PRIMARY PATH is the **2-component GM baseline** (propagate B as p·δ(V_high) + (1-p)·δ(V_low), one matrix-Gaussian per mode with near-zero variance, weight by p and 1-p). This is exact at the Bernoulli level and bypasses the bimodality issue entirely. The fault model then applies to each component independently and the mixture is aggregated at output.
    * **Heuristic switch**: count `m_dense = max_r |{i : |A[r,i]| > 1e-10}|`. If `m_dense ≥ 16` use single-Gaussian; else use 2-component GM. Log the choice in `config.json`.
  - **Degenerate case (Codex iter 1 M4)**: when `σ²_p < 1e-30` (i.e., `p < 1e-15 or p > 1-1e-15`), bypass tail_gauss and use deterministic threshold comparison (`|μ_post − μ_golden| > ε·|μ_golden|` is a boolean, not a probability).
- [ ] **[5.2]** [iter:3] [area:experiments/predict_resilience_soga.py] Extend `compute_per_cell_SDC` to accept non-trivial `U_B, V_B` (no longer near-zero). Update tail variance: `σ²_post = var_base + Var[δ|v]·A[r,i]²` per gm-expert Q6.
- [ ] **[5.3]** [iter:3] [area:experiments/run_step3.py] Sweep `p ∈ {0.0, 0.05, 0.10, ..., 1.0}` (21 points); compute SOGA prediction MSK(p), SDC(p), OTR(p).
- [ ] **[5.4]** [iter:3] [area:experiments/simulate_fi_mc.py] Extend MC: sample `B_ij ~ Bernoulli(p)·V_high + (1-Bernoulli(p))·V_low` per cell, then apply bit-flip FI as in M2. Run on **4 representative points** (corrected per Codex iter 1 M3): `p ∈ {0.0, 0.5, p_critical, 1.0}` with 1000 samples each. Define `p_critical = argmax_p |dSDC/dp|` from the SOGA sweep (point of steepest slope); fallback to `p_critical = 0.5` if SOGA reports monotonic. p=0 and p=1 are degenerate (point mass) and should be handled per M5.1; p=0.5 is the worst case for bimodal-Gaussian approximation; p_critical probes the most informative non-trivial point.
- [ ] **[5.5]** [iter:3] [area:experiments/run_step3.py] Monotonicity test: for each curve (MSK, SDC, OTR), check if monotone in p. Use Kendall's τ; threshold τ > 0.9 for "monotone", τ < 0.5 for "strongly non-monotone". Report intermediate cases as ambiguous.
- [ ] **[5.6]** [iter:3] [area:experiments/figures] Plot `resilience_vs_bimodal_p.png` with curves + MC error bars at validated points + monotonicity annotation.
- [ ] **[5.7]** [iter:3] [area:experiments] If counterexample found: write `experiments/.../counterexample_analysis.md` with physical intuition (why the non-monotonic regime exists) + MC confirmation.

### M6 — Deliverable bundle (2 days) [#12]

- [ ] **[6.1]** [iter:4] [agent:documentation-writer] [area:experiments/REPORT.md] Synthesis: (1) setup, (2) Strada Q honesty disclaimer (top-front), (3) Step 1 results + figure, (4) Step 3 results + figure, (5) discussion (citing `docs/research-notes/06-input-side-fault-modeling.md`), (6) reproduction instructions.
- [ ] **[6.2]** [iter:4] [area:experiments/lishan_pitch.ipynb] Front-door notebook: interactive sliders for `ε`, `p_fault`, `v` (Step 1) and `p` (Step 3); live plot updates; cell with one-paragraph framing for Lishan. Use `ipywidgets` if available, else parameterized cells.
- [ ] **[6.3]** [iter:4] [agent:documentation-writer] [area:lishan_discussion_package/README.md] 1-page overview: who/what/why + result summary + how-to-run + link to experiments/.../REPORT.md. Top-level, self-contained.
- [ ] **[6.4]** [iter:4] [area:lishan_discussion_package/setup.md] Clone+branch+install+run instructions (≤ 50 lines). Verified end-to-end on a fresh checkout.
- [ ] **[6.5]** [iter:4] [area:lishan_discussion_package/figures/] Copy key plots from `experiments/.../figures/` (Step 1 replica + Step 3 monotonicity result).
- [ ] **[6.6]** [iter:4] [area:experiments] `HASHES.txt` with SHA256 of all inputs (A matrix, config.json, source files). Seed in config.json fixed at 42.
- [ ] **[6.7]** [iter:4] [agent:code-reviewer] [area:final] Code review on the entire experiments/ + lishan_discussion_package/ artifact.
- [ ] **[6.8]** [iter:4] [agent:codex-cross-reviewer] [area:final] Final cross-review on the cumulative branch artifact (Phase F of /iterate, not /plan).

---

## Test plan

### Unit tests
- `tests/test_fault_model.py` — 5 edge cases per H5 (M1.4) + mantissa moment-match validation
- `tests/test_predict_soga.py` — K_reduced (6 for A=I_4, 21 for A=full at m=n=4) vs K_full=81 (=1+5·4·4) cross-validation on m=n=4 (M3.5)
- `tests/test_step1_acceptance.py` — Pearson + max rel err thresholds (M4.3)

### Sanity-vs-analytical
- Baseline `D = A @ B` validated against `lishan_2mm_32x32.soga` (M0.3, M3.1)
- Symmetry reduction K_reduced = K_full=81 at m=n=4 (M3.5); for 32×32 production case K_full=5121, K_reduced ≤ 161

### Sanity-vs-MC
- Step 1: full sweep, 1000 samples per v point, ~10 v points (M4.3); upgrade to 3000 samples if Pearson < 0.90 (M6 path)
- Step 3: 4 MC-validated points {0.0, 0.5, p_critical, 1.0}, 1000 samples each (M5.4)
- Acceptance: Pearson > 0.90 primary / > 0.85 fallback per curve; max rel err < 10% MSK/SDC; max abs err < 5% OTR

### No benchmark regression
- No `libSOGA*.py` modifications → no `/soga-bench` regression needed

### Audit gates
- **NOT applicable**: no `.g4` change, no `libSOGA*.py` change. Skip `/audit-grammar` and `/audit-numerical`.

---

## Acceptance criteria

- [ ] Branch `feat/lishan-resilience-poc` exists, all commits authored on it
- [ ] All 5 fault-model edge cases pass (H5)
- [ ] Symmetry reduction K_reduced ↔ K_full=81 at m=n=4 validated (M3.5); for 32×32 production K_full=5121 used as worst-case reference only
- [ ] Step 1 replica: Pearson corr > 0.90 primary / > 0.85 fallback between MC and SOGA per (MSK/SDC) curve
- [ ] Step 1 max rel err < 10% (MSK, SDC); max abs err < 5% (OTR)
- [ ] Step 3 sweep complete; monotonicity test produces clear verdict (monotone / counterexample / ambiguous)
- [ ] At least 4 MC-validated points in Step 3 {0.0, 0.5, p_critical, 1.0}
- [ ] If counterexample found: physical-intuition note `counterexample_analysis.md` with MC confirmation
- [ ] `experiments/lishan_resilience_2026-05-25/REPORT.md` with input-side disclaimer top-front
- [ ] `experiments/lishan_resilience_2026-05-25/lishan_pitch.ipynb` runs end-to-end with sliders
- [ ] `lishan_discussion_package/` self-contained (README, REPORT link, setup.md, figures/)
- [ ] `HASHES.txt` + `config.json` for reproducibility
- [ ] Codex cross-reviewer APPROVE on final artifact
- [ ] No scope creep: no SYRK/GEMM/3MM/2DCONV/GPU code added

---

## Rollback

All work on `feat/lishan-resilience-poc` branch. If POC is abandoned:

1. **At any milestone**: `git branch -D feat/lishan-resilience-poc`. Main and `feat/matrix-gm-integration` untouched.
2. **Partial recovery**: each milestone is independently mergeable; revert M-X commits selectively.
3. **No risk to main**: branch was never merged.

---

## Estimated complexity

**M — 14 days (10-15 effective), ~600 new LOC, ~30 new tests.**

Per-milestone effort:
- M0 (setup): 2 days — low risk, atomic
- M1 (fault model): 3 days — medium risk, math correctness critical
- M2 (MC reference): 2 days — low risk, standard FI simulation
- M3 (SOGA prediction): 3 days — medium risk, K reduction (≤161 for general A; =6 for A=I_32) needs validation against K_full=81 at m=n=4
- M4 (Step 1 verify): 1 day — low risk, mechanical
- M5 (Step 3 hunt): 3 days — medium risk, outcome uncertain (counterexample may or may not exist)
- M6 (deliverable): 2 days — low risk if M0-M5 green

Uncertainty flags:
- **U1**: K reduction formula (≤ 1+5·m per output cell, A-dependent) — must be confirmed against K_full=1+5·m·n=81 on m=n=4 in M3.5; if wrong, fall back to K_full (slower but no semantic difference)
- **U2**: Bimodal Bernoulli Gaussian approximation at p far from 0.5 (M5.1) — flagged in gm-expert Q5; may need 2-component GM per cell as fallback
- **U3**: Counterexample to Assumption-1 may not exist — acceptable outcome (monotonicity confirmation also valuable)

---

## Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-LR1 | K reduction formula wrong → silent error in Step 1 numbers | HIGH | M3.5 cross-validation: K_reduced (≤ 1+5·m, A-dependent) vs K_full=1+5·m·n=81 on m=n=4 mandatory before M4 |
| R-LR2 | Bimodal Bernoulli Gaussian approx breaks for `p ∈ [0.2, 0.8]` (high bimodality regime, maximum at p=0.5), NOT at extremes (Codex iter 1 C2 correction) | HIGH | M5.1 emit warning in correct interval; M5.4 MC validation at p=0.5 (worst case); 2-component GM as primary path for sparse A (A=I_32), CLT-Gaussian path for dense A only (Codex iter 2 C2b correction) |
| R-LR3 | High-exp overflow handling (H1) misclassifies near-boundary cases | MEDIUM | M1.2 log-space check + M1.4 edge case `v=1e38` |
| R-LR4 | Pearson correlation < 0.90 primary / < 0.85 fallback in Step 1 → replica fails | MEDIUM | If < 0.90, diagnose: (a) fault model approximation gap, (b) statistical noise (upgrade to 3000 samples per v); accept down to 0.85 with documented gap |
| R-LR5 | Counterexample exists but only at extreme parameter (artifact) | LOW | M5.7 includes physical intuition + multiple MC points to confirm reality |
| R-LR6 | Lishan affiliation/contact change | LOW | Lishan moves to U Alberta Fall 2026; engage NOW while at GMU |
| R-LR7 | scipy.stats.norm.logsf returns -inf for extreme z → P_SDC = 0 wrong | MEDIUM | M3.3 cap at -750 as a residual safety net AFTER H1 pre-classification (per Codex iter 1 M2). scipy.stats.norm.logsf has accurate asymptotic series down to z≈38.5 (yielding logsf≈−742); cap should never fire in practice because H1 catches overflow first. Verify scipy boundary empirically in M3.3 unit test |
| R-LR8 | MC simulation runtime too long for full Step 1 sweep | LOW | Vectorize bit-flip via numpy.frombuffer; alternatively reduce to 500 samples × 8 v-points |

---

## Cross-review history

- **iter 1: APPROVE_WITH_CHANGES** (codex 2026-05-25) — 3 critical findings (C1: K=26 unsubstantiated for general A → corrected to K_max=1+5·m=161 with A=I_32 collapse to K=6; C2: bimodal Bernoulli warning inverted → corrected to p∈[0.2,0.8] is worst case; C3: "moment-matching exact" mislabeled for MANTISSA → split into EXACT for SIGN/EXP and APPROXIMATE for MANTISSA), 7 medium findings (M1 v=1e-30 reasoning, M2 logsf cap verification, M3 p_critical definition, M4 degenerate p=0/1 handling, M5 A matrix never specified, M6 Pearson threshold too tight, M7 p_fault semantics disambiguation), 10 positives (Strada Q framing rigorous, IEEE 754 partition verified, two-outcome design, rollback clean, research note 12 verified refs, no scope creep, SIGN δ=−2v correct, H1 log-space correct, symmetry rigorous for j≠s, risk register comprehensive). Plan v2 applies all 3 critical + 7 medium edits; resubmitted to codex.
- **iter 2: REJECT** (codex 2026-05-25) — 6 unresolved sweep/contradiction findings: (C1c-d) stale "K=26"/"K=5121" labels in Test Plan, Acceptance Criteria, U1, R-LR1; (C2b) CLT argument invalid for A=I_32 (single-term sum is not CLT-applicable); (M6) Pearson 0.95 stale in Test Plan, Acceptance Criteria, R-LR4; (N1) Step-3 MC point count inconsistency (4 vs 3 in different sections); (N2) "max bimodality" wording imprecise; (N3) mantissa error bound contradiction (<2% vs <5% in different sections). Verified-fixes count: 11 of 17 (M1, M2, M4, M5, C3a, C1a-b arithmetic correct).
- **iter 3: APPROVE** (codex 2026-05-25, audit at `results/codex_review/iter3-2026-05-25/`) — all 6 iter-2 mechanical fixes verified: (1) stale K=26/K=5121 swept from Test Plan + AC + U1 + R-LR1 (5121 retained only in Alternatives and historical context); (2) M5.1 CLT scoped to dense A (≥m/2 nonzero/row), A=I_32 routed to 2-component GM primary path with `m_dense ≥ 16` heuristic switch logged in config.json; (3) Pearson 0.95 swept to 0.90 primary / 0.85 fallback; (4) Step-3 MC point count unified to 4 {0.0, 0.5, p_critical, 1.0}; (5) "max bimodality" → "high bimodality regime, maximum at p=0.5"; (6) Mantissa bound unified to "<5% target, enforced by M3.5". Stale-string negative check: all 6 PASS. Plan locked — proceed to user approval gate.

---

## Deliverable manifest

After M6:

```
experiments/lishan_resilience_2026-05-25/
├── REPORT.md
├── config.json
├── HASHES.txt
├── lib/
│   ├── fault_model.py
│   └── bimodal_prior.py
├── simulate_fi_mc.py
├── predict_resilience_soga.py
├── run_sweep.py
├── run_mc_step1.py
├── run_soga_step1.py
├── run_step3.py
├── lishan_pitch.ipynb
├── tests/
│   ├── test_smoke.py
│   ├── test_fault_model.py
│   ├── test_predict_soga.py
│   ├── test_step1_acceptance.py
│   └── test_mc_smoke.py
├── results/
│   ├── baseline_DUV.npz
│   ├── mc_step1.csv
│   ├── soga_step1.csv
│   └── step3_sweep.csv
└── figures/
    ├── resilience_vs_input_value.png
    └── resilience_vs_bimodal_p.png

lishan_discussion_package/
├── README.md
├── REPORT.md  (link to experiments/.../REPORT.md)
├── setup.md
└── figures/
    ├── resilience_vs_input_value.png
    └── resilience_vs_bimodal_p.png

docs/research-notes/
└── 06-input-side-fault-modeling.md  (already created by web-researcher)
```

Total new files: ~25. Total new LOC estimate: ~600 (excluding tests + notebook).
