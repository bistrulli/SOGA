# Cross-review — iter 1
# Mode: claude-self-review (codex-cli Azure 404 downgrade)
# Reviewer stance: independent, skeptical — NOT the author

## Checklist evaluation

### 1. COMPLETENESS
Sub-tasks M0–M9 are mostly atomic. Specific weaknesses:
- M3.2 ("compute_fault_aggregation ... per output cell P(SDC) tail integration") is under-specified. The analytical derivation for how a single int32 bit-flip on K1's multiplier output propagates through K2 to affect each output cell of D is non-trivial. The plan says "analytically (per output cell P(SDC) tail integration)" but provides no formula. For an n=32 cascade, the fault on one partial product of K1 propagates as a rank-1 perturbation through K2: delta_D = delta_tmp @ C. The distribution of delta_D given a fixed C and a fault on one inner-product accumulation step is the critical computation. This is not the same as simply calling affine_left twice — it requires computing the conditional distribution of one corrupted element of tmp and propagating through C. This sub-task needs a formula pinned before implementation, not discovered during M3.
- M4.3 acceptance criterion "is Gaussian curve significantly different from Uniform at same sigma?" is not the same as demonstrating non-monotonicity. Cross-family difference is a distribution-shape effect test, NOT a test of monotonicity vs non-monotonicity. These are conflated throughout. A Gaussian curve can be consistently BELOW a Uniform curve (a shape effect) while BOTH curves remain monotone individually. The plan equates "cross-family gap exists" with "non-monotonicity evidence" — this equation is logically wrong.
- M5 bimodal sweep: no formula given for when a peak is expected. The plan says "For small mu: low cancellation, low MSK; for medium mu: max cancellation, peak MSK" — but this reasoning is FP-domain cancellation logic. For int32 exact-match, the masking probability is P(delta_D_ij == 0), which requires the fault delta to be exactly cancelled. For a 32x32 accumulation, exact integer cancellation (sum to zero) is a discrete event with very low probability unless all terms are zero. The expected "peak" mechanism is not established for the integer exact-match case.

### 2. NOVELTY
CRITICAL PROBLEM — The plan's primary novelty claim is directly contradicted by Yang's own document visible in context:
- Claim in plan: "No paper has conducted a systematic sweep over input distribution families for GEMM or 2MM"
- Yang's document (pages 2-5): she tests 2MM and 3MM with Binomial(10, 0.3), Binomial(1000, 0.8), Binomial(100, 0.8), Equilikely(0, 100), and "complex_combination_1/2/3" for both benchmarks. Page 4 shows side-by-side prediction bars across these seven families for 2MM and 3MM simultaneously.
- This is a systematic sweep over distribution families for exactly the kernels in scope.
- The plan has a risk entry R-NM4 that says "Yang's data already showing non-monotonicity reduces novelty" and mitigates with "framing is we explain the mechanism." But R-NM4 treats this as a LOW severity novelty risk, not as a CRITICAL novelty underminer. The actual situation is more severe: Yang is already doing the distribution-family comparison experimentally; our analytical model is a potential complement, not a pioneer sweep.
- The research note 07 (cited by the plan) correctly identifies the prior-art gap as "no analytical model" — but the plan then overclaims by asserting the sweep itself is novel.
- The plan ALSO misreads Yang's Int, 2mm plot. Yang labels that section "Monotonic Trend" — and for the quantity she claims is monotone (SDC), it IS roughly monotone (0.0 → 0.80 → 0.60+). MSK is NOT monotone, but Yang never claims MSK is monotone; her Assumption-1 is about the overall resilience-input relationship used for interpolation, not specifically MSK alone. The plan says "she does NOT label this as non-monotonic" as if this is an oversight — but she may intentionally be referring to SDC monotonicity in that section heading.

### 3. METHODOLOGY

ISSUE (a) — Cross-family at matched variance:
The plan specifies Uniform[0, sqrt(3)*sigma] (positive-only, matched variance to Gaussian N(0,sigma)). However:
- Gaussian N(0,sigma) has mean 0; Uniform[0, sqrt(3)*sigma] has mean sqrt(3)*sigma/2 > 0.
- These are NOT matched in mean — they differ in both shape AND mean. The cross-family gap therefore conflates (i) zero-mean vs positive-mean effect and (ii) distribution shape effect. There is no way to attribute the gap to "cancellation mechanism" vs "mean shift" from this design alone.
- Correct matched comparison: Uniform[-sqrt(3)*sigma, sqrt(3)*sigma] (zero-mean, matched variance) vs N(0,sigma). The plan uses a positive-only Uniform to match Yang's baseline, but then claims the difference is due to cancellation — a confounded design.
- The plan explicitly notes the Uniform is "positive-only = Yang's baseline" which is correct for setting alignment, but the cancellation claim requires zero-mean Uniform as the control.

ISSUE (b) — FlipTracker cancellation in integer domain:
- FlipTracker's cancellation pattern is about FP accumulation: a fault delta in one term is masked if subsequent terms in the accumulation sum produce a net correction. For FP, this is a continuous cancellation effect (|delta| shrinks through subsequent operations).
- For int32 EXACT-MATCH (eps=0): masking requires delta_D_ij = 0 exactly. In a 32x32 integer inner product, a fault of size delta_k on one partial product propagates to delta_D_ij = delta_k * C[k,j] (approximately, for the K2 stage). For this to be exactly zero, we need C[k,j]=0 (exactly) or the fault to be exactly cancelled by wrap-around arithmetic. This is not "cancellation" in the FlipTracker sense — it is a discrete event requiring exact zeros in integer arithmetic.
- The plan never addresses this distinction. Using FlipTracker as the theoretical justification for integer exact-match masking is formally incorrect.

ISSUE (c) — eps=0 vs eps>0 dual mode:
- The dual mode is NOT rigorous as presented. The plan uses eps=0 as "Yang-faithful (primary)" and eps>0 as "non-monotonicity enabling (secondary)" — but this framing admits the non-monotonicity only appears when eps is chosen to enable it. A reviewer reading this would immediately note: "you needed to change the error metric to find your non-monotonic result."
- The plan should either: (i) justify which eps value is physically meaningful for int32 arithmetic independent of the desired result, or (ii) be honest that the eps>0 mode is a sensitivity analysis, not a "Yang-faithful" extension.
- As written, the eps dichotomy looks like the investigators chose eps to get the desired outcome (non-monotonicity), which is the exact "you tuned eps to get non-monotonicity" objection listed as LOW severity in R-NM5. It is actually a HIGH severity methodological problem.

ISSUE (d) — Statistical thresholds:
- Kendall tau Bonferroni at 9 tests (3 families x 3 categories): alpha/9 = 0.0056. This is correctly computed.
- However, the 9 tests are NOT independent: MSK + SDC + OTR = 1 by construction, so (family, MSK) and (family, SDC) and (family, OTR) are perfectly correlated (sum constraint). Bonferroni correction assumes independent or positively correlated tests; the sum-to-1 constraint means the effective number of independent tests is at most 2 per family, not 3. The Bonferroni correction should be over 3 families x 2 independent categories = 6, giving alpha/6 = 0.0083.
- This is a minor but real statistical error: the plan overcorrects (0.0056 is more stringent than necessary), which means some true effects might be missed.

### 4. FEASIBILITY

ISSUE (e) — Budget:
- 4 days for ~600 LOC + 25 tests is plausible for an experienced practitioner who understands the domain. The main risk is M3.2 (analytical fault propagation derivation) which could expand to multiple days if the math is not settled before implementation. The plan does not include a formula-derivation task before M3.2 coding starts.
- The plan totals milestone estimates: 0.5 + 1 + 1 + 1 + 1 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 = 7 days of milestone-level work. The plan states "4 days effective" — the discrepancy (7 days of tasks in 4 days) is not explained. Either significant parallelism is assumed or some milestones are underestimated.
- U1, U2, U3 uncertainty flags are acknowledged, which is good.

### 5. BASELINE COVERAGE
- The plan explicitly excludes PSI / Stan / AQUA / BLOG ("Audit gates: NOT applicable"). This is correct for the experimental scope (no new probabilistic program semantics).
- No baseline comparison issue for this plan type.

### 6. HONESTY DISCIPLINE (f)
- The "adjacent" framing (Constraint 6: "input-side ≠ register-level" disclaimer) is present in constraints and acceptance criteria. Good.
- HOWEVER: The Goal section says "Yang's Assumption-1 explicitly does not apply because she cannot test non-flat input distributions" — this is an overstatement. Yang's document shows she IS testing non-flat distributions (Binomial, complex_combination). The plan has not updated its goal statement to reflect this discovery.
- The plan says "No paper has demonstrated non-monotonic resilience vs input distribution parameters → genuine novelty" in the Prior Art section. This claim is made without caveat; the fact that Yang's own unpublished note already tests distribution families is not mentioned here, only in R-NM4 buried in the risk register with LOW severity. This is a honesty-discipline failure in the novelty claim.
- The paper-replicator framing ("Yang's own MSK curve is non-monotonic; we provide analytical explanation") is the strongest part of the honesty framing and should be elevated to the Goal section.

### 7. RISK REGISTER COMPLETENESS (g)
Missing risks:
- R-NM-MISSING-1 (HIGH): Yang is already testing distribution families experimentally (Binomial, complex_combination) — our sweep is NOT the first such experiment, only the first analytical one. This reduces novelty from "pioneer" to "analytical complement." Currently treated as LOW in R-NM4.
- R-NM-MISSING-2 (HIGH): The FlipTracker FP-cancellation mechanism does not straightforwardly apply to int32 exact-match masking. If the integer masking probability is consistently near-zero across all distribution families (because exact cancellation requires C[k,j]=0), the cross-family gap may be negligible. No risk entry covers this.
- R-NM-MISSING-3 (MEDIUM): The cross-family design conflates mean-shift with shape-shift (Uniform is positive-only vs Gaussian/Bimodal are zero-mean). Results may be attributable to mean difference rather than cancellation mechanism.
- R-NM-MISSING-4 (MEDIUM): eps>0 as the primary mechanism for non-monotonicity appearance creates a "researcher degrees of freedom" problem. If eps=0 gives monotone results and eps>0 gives non-monotone, the result is eps-dependent, not a physical property of the system.
- R-NM-MISSING-5 (LOW): Yang's "Typhoon" direction (input types) may be an active unpublished paper. Scooping risk. Currently not in risk register.

---

## Summary verdict

VERDICT: APPROVE_WITH_CHANGES

FINDINGS:
- CRITICAL-1: The plan's primary novelty claim ("no prior paper has swept distribution families for GEMM/2MM") is directly contradicted by Yang's own working note (pages 2-5), which tests Binomial and Equilikely families for 2MM and 3MM with prediction models. The novelty must be repositioned as "first analytical (non-sampling) model for distribution-family effects" — not "first sweep."
- CRITICAL-2: The cross-family experimental design conflates mean-shift (Uniform is positive-only, Gaussian/Bimodal are zero-mean) with distribution-shape/cancellation effects. The claimed "FlipTracker cancellation" attribution is not isolable from the mean difference. A zero-mean Uniform control is needed, or the confound must be explicitly acknowledged.
- CRITICAL-3: The FlipTracker cancellation mechanism (FP-domain, continuous magnitude reduction) is applied to int32 exact-match masking (discrete event: delta_D == 0) without justification. The theoretical basis for why bimodal symmetric inputs would increase P(D_corrupted == D_golden) in integer arithmetic is not established. M3.2 needs a formula before coding.
- MEDIUM-1: The eps=0 (Yang-faithful) vs eps>0 (non-monotonicity enabling) framing is problematic. Non-monotonicity appearing only under eps>0 means the result is metric-choice-dependent, not a physical property. The plan should clearly state this is a sensitivity analysis and not present eps>0 as merely a "secondary mode."
- MEDIUM-2: The Bonferroni correction is applied over 9 tests but MSK+SDC+OTR=1, so the 3 categories per family are not independent. Effective independent tests per family = 2, giving alpha/6 = 0.0083 as the correct correction.
- MEDIUM-3: Milestone time estimates sum to ~7 days of work; the plan claims 4 days effective. The gap is unexplained. M3.2 (analytical derivation) is especially under-budgeted.
- LOW-1: The plan misreads Yang's section heading "Monotonic Trend" as overlooking MSK non-monotonicity. Yang's Assumption-1 is stated for the resilience-input_value relationship used for interpolation, which may refer specifically to the SDC component (which IS roughly monotone). The plan should not assert Yang "does not label this as non-monotonic" without acknowledging that SDC is the monotone curve she relies on.
- LOW-2: Risk R-NM4 (Yang already shows non-monotonicity) is rated LOW. Given that Yang is already doing distribution-family comparisons (Binomial vs Equilikely), this should be MEDIUM or HIGH.

RECOMMENDATIONS:
- ACTION-1 (CRITICAL): Rewrite the novelty claim in the Goal and Prior Art sections. Replace "No paper has swept distribution families for GEMM/2MM" with "Yang's working note (Input Type) empirically tests Binomial/Equilikely families; SOGA provides the first analytical model predicting the distribution-parameter-to-resilience curve without sampling campaigns." This is accurate and still strong.
- ACTION-2 (CRITICAL): Add a zero-mean Uniform[−sqrt(3)*sigma, sqrt(3)*sigma] as a fourth family in Experiment 1, or rename the Uniform[0, sqrt(3)*sigma] family as "Positive-uniform (Yang baseline)" and add explicit language that the Gaussian/Bimodal vs Positive-uniform gap includes a mean effect. Alternatively: run a confound-separation sub-experiment comparing N(mu, sigma) at varying mu to separate mean from shape.
- ACTION-3 (CRITICAL): Add a new sub-task M3.0 before M3.2: "Derive formula for P(MSK) under single K1 bit-flip as a function of input distribution parameters for int32 2MM cascade." Pin the formula with a note on when exact-integer cancellation is non-negligible. This derivation must exist before M3.2 coding begins.
- ACTION-4 (MEDIUM): Rename Experiment 3 from "Gaussian sweep with eps>0 (secondary)" to "Sensitivity analysis: eps>0 relaxation" and explicitly state in REPORT.md that non-monotonicity under eps>0 is a sensitivity result, not a physical finding. Do not use it to claim non-monotonicity was "found."
- ACTION-5 (MEDIUM): Fix the Bonferroni correction: apply over 3 families x 2 independent categories = 6 tests, yielding alpha/6 = 0.0083 as the corrected threshold.
- ACTION-6 (MEDIUM): Add the three missing HIGH/MEDIUM risks to the risk register (mean-shift confound, FP-cancellation inapplicability to int32, eps-sensitivity problem).
- ACTION-7 (LOW): Elevate R-NM4 severity from LOW to MEDIUM. Add explicit statement: "Yang's document tests Binomial(10,0.3), Binomial(1000,0.8), Equilikely(0,100) for 2MM — our novelty is the analytical model, not the sweep itself."
