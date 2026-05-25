# Iter 1 Review — lishan-resilience-poc plan
**Reviewer mode**: Claude self-review (independent, skeptical)
**Date**: 2026-05-25

---

## VERDICT: APPROVE_WITH_CHANGES

---

## FINDINGS

### CRITICAL (math bugs / fatal flaws)

**C1. K=26 symmetry reduction claim is unsubstantiated for a general A matrix.**
The plan states: "Per-output-cell distinct cases collapse from 5121 to 26 (5 classes × 5 row-equivalence groups + baseline)." The first part is correct: for D = A@B and output cell (r,s), only faults in column j=s of B affect D[r,s] (because D[r,s] = sum_i A[r,i]*B[i,s] and B[i,j] for j≠s does not appear). This reduces from 32×32×5 = 5120 fault scenarios to 32×5 = 160.
The "5 row-equivalence groups" claim requires that A has only 5 distinct row-coefficient values A[r,i] for i in {0..31}. For a general 32×32 A matrix this is FALSE — there are up to 32 distinct values. The plan references "numerical-stability-expert" confirming this, but provides no mathematical justification. Either (a) the A matrix used has special structure (e.g. identity, which gives only {0,1} distinct values), or (b) the 5 groups refer to something else (the 5 fault classes collapsing rows with the same A[r,i] value within each class). Without specification of A's structure, the K=26 claim is unverifiable and could silently produce wrong numbers if A does not have 5-value rows. The cross-validation test at m=n=4 (M3.5) partially mitigates this, but the justification must be explicit in the plan.

**C2. Bimodal Bernoulli Gaussian approximation is worst at p=0.5, not at the tails — the plan's warning direction is inverted.**
The plan warns "emit warning when p < 0.1 or p > 0.9 (Gaussian approximation degrades)." This is partially correct: at p→0 or p→1, the distribution approaches a point mass (actually the most Gaussian-like unimodal). The Gaussian approximation of a bimodal distribution is WORST at p=0.5 where the two modes have equal weight and maximal separation. At p=0.5, moment-matching produces a Gaussian centered at (V_high+V_low)/2 with variance (V_high-V_low)^2/4, but the true distribution has probability mass concentrated at the two extremes V_low and V_high, not the center. Tail probabilities (SDC/OTR) computed from this Gaussian will be systematically WRONG — underestimating probability in the tails (which correspond to the actual modes of the bimodal). The warning should cover the CENTRAL p range (say p ∈ [0.3, 0.7]) as well as the extremes. The plan has this inverted, which is a correctness issue for Step 3.

**C3. The per-class SDC probability formula conflates marginalizing over the fault-scenario mixture with computing a single-Gaussian tail probability.**
The plan says "moment-matching is exact for all 5 classes because the shift is always linear in v." This is true for the MEAN and VARIANCE of the shift delta, but the OUTPUT tail probability Pr(|D[r,s] - D_base| > ε*|D_base|) is not determined by the mean and variance of delta alone unless the output is Gaussian. When D_base is Gaussian (N(μ, σ²)) and delta is DETERMINISTIC, D_post is Gaussian(μ+delta, σ²) and the tail probability is exact. But for MANTISSA classes, delta = ±v·2^p where p is random uniform over a discrete set and sign is random — so delta is a discrete mixture, not a point. D_post is then a DISCRETE MIXTURE of Gaussians. Using moment-matching (replacing this mixture with a single Gaussian of the same mean and variance) introduces approximation error. The "exact" label in the plan is therefore misleading for the mantissa classes. The plan should say "exact for SIGN and EXP classes; moment-matched (approximate) for MANTISSA classes."

### MEDIUM (spec gaps, scope ambiguity)

**M1. The edge case test "v=1e-30 → mostly SDC (high-exp dominates)" is mis-reasoned in the plan.**
For HIGH-EXP with k∈{16,32,64,128} and v=1e-30: δ = v·(2^k−1). For k=16, δ ≈ 6.5e-26. The baseline D[r,s] (with identity A and v·ones input B) = v = 1e-30. The RELATIVE shift |δ|/|D_base| ≈ 65535 >> 1, so yes, SDC is highly probable. But this is because 2^k >> 1, not because v is small. For v=1 (k=16), δ = 65535 and D_base = 1, ratio is the same 65535. The SDC probability for HIGH-EXP at a given k is INDEPENDENT of v (for non-zero v). The expected test result "mostly SDC" is correct but for the wrong reason. The unit test (M1.4) must check for SDC irrespective of v scale, not specifically "v=1e-30" as a special case. The test criterion should be "SDC probability for HIGH-EXP >> p_fault·P(HIGH-EXP) for any v in [1e-38, 1e38]."

**M2. R-LR7 cap at -750 is potentially misspecified.**
scipy.stats.norm.logsf returns log(1 - Φ(z)) = log(Φ(-z)) for large z. Using scipy.special.log_ndtr, the function avoids intermediate underflow up to large z values. The plan proposes capping at logsf < -750 to avoid -inf. The issue: for z ≈ 38.5, logsf ≈ -750 (since log(Φ(-38.5)) ≈ -38.5²/2 ≈ -742). For z > 38.5, the value becomes subnormal (<5e-324, the float64 minimum positive). The cap at -750 would set logsf to exactly -750 when the true value is between -745 and -750, which is an error of up to +5 in log-space, corresponding to a factor of e^5 ≈ 150 in probability. For the OTR logic this may not matter (those cells are already classified OTR), but the plan should document that the cap is only applied AFTER the OTR pre-classification (H1) has already removed the overflow cases. If H1 is applied first, the remaining cases should never trigger z > 38.5, making the cap mostly moot. The plan should make this ordering dependency explicit and reconsider whether the cap is needed at all post-H1.

**M3. The "p_critical" selection in Step 3 MC validation (M5.4) is unspecified.**
The plan says "3 representative points (p=0.0, p_critical, p=1.0)." The value of p_critical is left undefined. Since the goal of Step 3 is to test monotonicity, p_critical should be chosen AFTER the SOGA sweep identifies the potentially non-monotone point (if any), not arbitrarily. The plan should specify: "p_critical is the p value with maximum |dSDC/dp| from the SOGA sweep, or p=0.5 if the sweep is monotone."

**M4. The p=0.0 and p=1.0 endpoint behavior needs explicit handling.**
At p=0.0: all input cells take value V_low deterministically. B is deterministic (no randomness), so the bimodal prior degenerates to a point mass. The plan's `bimodal_to_matrix_gaussian` function sets σ²_p = p·(1-p)·(V_high-V_low)² = 0. This makes U_B = 0·I_m (zero covariance matrix), and the Gaussian degenerates to a Dirac delta. The plan must handle this degenerate case: the output D = A@B is deterministic with zero variance. SDC/MSK/OTR then depend only on whether the deterministic output shifts past the threshold under each fault class. The plan does not specify how `compute_per_cell_SDC` handles U_B=0 (division by zero in tail_gauss if σ_base=0).

**M5. Missing cross-reference: the plan does not specify where A (the kernel matrix) comes from for the Step 1 v-sweep.**
The plan initializes `D_base = SOGA(A @ B)` but A is never explicitly defined. The `.soga` file `lishan_2mm_32x32.soga` uses A = identity. If A=I then A[r,i] = δ_{r,i}, which gives only 2 distinct values {0,1} — not 5 groups. This further challenges the K=26 claim. The plan must explicitly define A (or clarify that for Step 1, A is a random/generic 32×32 matrix, in which case A must be fixed and seeded for reproducibility).

**M6. Acceptance threshold for Pearson > 0.95 may be too tight given MC sampling noise.**
With n=1000 samples and ~10 v-points, the MC MSK/SDC/OTR estimates have standard errors proportional to sqrt(p*(1-p)/1000). For MSK near 0.9, SE ≈ 0.009. The 10-point Pearson correlation against a nearly-deterministic SOGA curve will be sensitive to this sampling noise. Empirically, 10-point correlations with SE=0.009 per point can easily show Pearson < 0.95 even if the method is correct. The plan should either (a) increase to n=3000 samples, or (b) set the acceptance threshold at 0.90 with documented rationale, or (c) use the already-provided fallback in R-LR4 (accept down to 0.85 with documented gap) but make this primary rather than a fallback.

**M7. No test for the case p_fault=1 (all bits flip) — listed in H5 but semantically ambiguous.**
The edge case "p_fault=1, v=1.0 → MSK ≥ P(LOW-MANTISSA)" deserves scrutiny. If p_fault=1 means "fault is injected with probability 1 into every cell of B", then every output cell is affected. The prediction MSK ≥ P(LOW-MANTISSA) is because low-mantissa bit flips produce small shifts (2^p for p ∈ [-23,-8] is very small relative to v=1.0). This is a valid test. However, the plan should be explicit that p_fault here is the per-cell fault probability (not bit-flip probability given fault), to avoid ambiguity.

### MINOR (wording, typos)

**m1. Typhoon citation labeled "unverified DOI"** — the research note (line 18) says "lishanyang.github.io/typhoon.pdf [unverified DOI]". The plan should echo this uncertainty: cite Typhoon as "SRC abstract, no stable DOI" consistently across REPORT.md and pitch notebook.

**m2. HIGH-EXP shift formula table** says k∈{16,32,64,128}. IEEE 754 float32 exponent bits are 8 bits (bits 30-23). Flipping bit 30 multiplies/divides the value by 2^128 (biased exponent bit with value 128 = 2^7). The mapping from bit-flip to k value should be: flipping exponent bit b (b ∈ {27,28,29,30}) changes the biased exponent by 2^(b-23). For bit 27: 2^4=16. For bit 28: 2^5=32. For bit 29: 2^6=64. For bit 30: 2^7=128. This is consistent with the table. Correct.

**m3. LOW-EXP** uses k∈{1,2,4,8} mapping bits 23-26. Bit 23: 2^0=1. Bit 24: 2^1=2. Bit 25: 2^2=4. Bit 26: 2^3=8. Consistent. Correct.

**m4. MANTISSA shift p∈[-7,-1] for HIGH-MANTISSA** (bits 16-22). Float32 mantissa = 1.fraction, so mantissa bit b contributes 2^(b-23) to the fractional part. Bit 16: 2^(16-23) = 2^(-7). Bit 22: 2^(22-23) = 2^(-1). So p∈[-7,-1]. Consistent. Correct.

**m5. LOW-MANTISSA p∈[-23,-8]** (bits 0-15). Bit 0: 2^(0-23) = 2^(-23). Bit 15: 2^(15-23) = 2^(-8). Correct.

**m6. The bit partition is mathematically consistent with IEEE 754 float32.** The reviewer confirms: all bit counts sum to 32 (1+4+4+7+16=32), the bit position assignments match the IEEE 754 spec, and the sign/magnitude effects are correctly described. This is a POSITIVE finding.

---

## POSITIVES

1. **Strada Q framing is excellent.** The input-side vs register-level distinction is clearly articulated and consistently enforced. The three-level taxonomy (gate, register, input) in the research note is academically sound and matches the Ares DAC 2018 precedent accurately.

2. **The IEEE 754 bit partition is correctly computed.** All 32 bits are accounted for with the right exponent and mantissa semantics.

3. **Two-outcome framing (monotone / counterexample)** is scientifically honest and robust to the actual result.

4. **Rollback plan** is clean and low-risk: all work on an isolated branch, no merge until acceptance.

5. **The research note (06-input-side-fault-modeling.md)** is well-grounded: 12 references with DOI verification documented, gap analysis is sound, and the novelty claim ("no prior work propagates a full distribution through a linear algebra kernel") is correctly scoped.

6. **Constraint discipline** is strong: no libSOGA*.py modifications, no grammar changes, scope limited to 2MM 32×32. The plan shows no observable scope creep.

7. **SIGN class shift formula δ = -2v** is correctly derived: flipping the sign bit on a float32 value v gives -v, so the delta is -v - v = -2v. Correct.

8. **Overflow H1 condition** `log|v| + k·log(2) + log|A[r,i]| > 127·log(2)` is correct: this tests whether the shifted value magnitude exceeds the float32 maximum (2^127). The log-space check avoids intermediate overflow. Correct.

9. **Symmetry argument for j≠s** is correctly identified: D[r,s] = sum_i A[r,i]*B[i,s], so faults in column j≠s leave D[r,s] unchanged. This part of the reduction (from 5120 to 160 scenarios) is mathematically rigorous.

10. **Risk register** covers the main hazards. The OTR boundary risk (R-LR3), the bimodal approximation risk (R-LR2), and the Pearson threshold risk (R-LR4) are all identified.

---

## RECOMMENDATIONS

**R1 (addresses C1)**: Add a subsection "A matrix specification" to the Approach section defining exactly what A is used in Step 1 and Step 3. If A = identity, state this explicitly and note that K=26 simplifies further (only non-zero A[r,i] = A[r,r] = 1, so actually K = 5 classes × 1 non-zero row + baseline = 6, not 26). If A is a general random matrix, seed it, store it in baseline_DUV.npz, and document the actual number of distinct A[r,i] values (which determines the true number of equivalence groups). Remove the unsupported "5 row-equivalence groups" claim and replace with the actual count.

**R2 (addresses C2)**: Revise the bimodal Gaussian warning in M5.1: warn for p ∈ [0.3, 0.7] (bimodal regime) AND for p < 0.1 or p > 0.9 (near-degenerate regime), but for different reasons. Add a comment explaining that at p=0.5 the approximation error on tail probabilities is maximal (the Gaussian predicts most mass at the mean, the true distribution has mass at the two modes). Consider using a 2-component GM (one per mode) directly at all p values — this eliminates the approximation entirely and is only 2 components. This is the natural Option 3 approach from the alternatives section and adds minimal complexity for Step 3.

**R3 (addresses C3)**: Add a precision qualifier to the moment-matching claim: "Moment-matching is exact for SIGN and EXP classes (deterministic shift conditional on fault scenario); it is an approximation for MANTISSA classes (the shift is a discrete mixture). The approximation error for mantissa classes is bounded by [derive or cite]." Add a dedicated test comparing the mantissa-class moment-matched prediction vs a direct mixture computation on a small case.

**R4 (addresses M2)**: Clarify R-LR7: specify that the H1 OTR pre-classification is applied BEFORE any logsf evaluation, so the cap is only a safety net for residual edge cases. Document that cap=-750 is conservative relative to the float64 subnormal threshold (~-745) and add a test at z=37 (logsf ≈ -686, valid), z=39 (logsf ≈ -761, subnormal territory) to verify behavior.

**R5 (addresses M3)**: Define p_critical in M5.4 as "the p value at which |d(SDC)/dp| is maximized in the SOGA sweep, or p=0.5 if monotone" and add this as a concrete subtask in M5.4.

**R6 (addresses M4)**: Add handling for the degenerate case σ²_p = 0 (p=0 or p=1) in bimodal_to_matrix_gaussian: return a flag indicating "deterministic input" and route to a deterministic branch in compute_per_cell_SDC that skips the Gaussian tail integration and computes SDC/MSK/OTR by direct threshold comparison.

**R7 (addresses M5)**: Add an explicit statement in M0.3 and M3.1: "A is defined as [identity / random seeded matrix]. For Step 1, A = I_32 per lishan_2mm_32x32.soga. Store A in config.json and baseline_DUV.npz."
