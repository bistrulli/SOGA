# Cross-review — iter 2
# Mode: claude-self-review (codex-cli Azure 404 downgrade — same as iter 1)
# Reviewer stance: independent, skeptical — NOT the author
# Scope: TARGETED — verify 7 fixes from iter 1 APPROVE_WITH_CHANGES ONLY

---

## Fix verification (ACTION-1 through ACTION-7)

### ACTION-1 (was CRITICAL-1): Novelty rewrite

**CHECK**: Is the false claim removed? Is the analytical-model novelty framing present?

The Prior Art section (Context) now contains an explicit correction block:

> "CORRECTION (Codex iter 1 CRITICAL-1): Yang's 'Input Type' pages 2-5 ALREADY empirically tests 2MM with Binomial(10,0.3), Binomial(1000,0.8), Equilikely(0,100), and 'complex_combination' distributions via prediction bar charts. Our novelty is NOT the distribution sweep itself — Yang already does that empirically. Our novelty IS the first analytical (non-sampling) model that predicts distribution-parameter-to-resilience curve without empirical campaigns. This is a fundamentally different scientific contribution."

This is precisely the repositioning required. The false "first sweep" claim is removed; the analytical-model novelty is foregrounded. The cross-review history section also confirms the fix: "CRITICAL-1: novelty claim 'first distribution sweep' is false ... Fix: rewrite novelty as 'first analytical model'."

CAVEAT: The Goal section retains "Yang's Assumption-1 explicitly does not apply because she cannot test non-flat input distributions" — which is partially inaccurate (Yang does test non-flat distributions: Binomial, complex_combination). However, this language was already present in v1 and LOW-1 from iter 1 (the SDC hedging issue) is distinct from ACTION-1. The Context CORRECTION block sufficiently addresses ACTION-1 for the Prior Art section. The Goal section inconsistency is a residual issue from LOW-1 (not ACTION-1 scope), and since LOW-1 was not listed among the 7 mandatory actions, this does not block VERIFIED status here.

**VERDICT: VERIFIED**

---

### ACTION-2 (was CRITICAL-2): Zero-mean Uniform control added

**CHECK**: Is the 4th family present? Is explicit confound-separation language present?

Experiment 1 (Approach section) now lists:
1. Gaussian N(0, σ) — zero-mean, symmetric
2. Uniform[0, sqrt(3)·σ] — explicitly labelled "non-zero-mean control" (positive-mean, Yang baseline analogue)
3. **Uniform[-sqrt(3)·σ, +sqrt(3)·σ]** — explicitly labelled "NEW per Codex CRITICAL-2: control for mean-shift vs shape confound" (zero-mean, symmetric)
4. Bimodal{±σ, p=0.5} — zero-mean, symmetric

Confound-separation language is present:
> "Confound separation: gap between Uniform[0,...] (non-zero mean) vs ANY zero-mean family isolates the mean-shift component. Gap among the three zero-mean families (Gaussian, Uniform-zero-mean, Bimodal) isolates the shape/cancellation component."

M1.3 also explicitly lists `uniform_prior(m, n, max_val)` as one of four families; the zero-mean Uniform is implicit in M4.1 "4 families" language. The risk register R-NM7 confirms this with "M4.1 adds zero-mean Uniform[-M,+M] control; explicit confound separation in REPORT."

**VERDICT: VERIFIED**

---

### ACTION-3 (was CRITICAL-3): M3.0 derivation gate

**CHECK**: Is M3.0 present as a gating task? Does it note FlipTracker inapplicability? Does it contain the P(MSK) ≈ 31/32 derivation?

M3 section now includes M3.0 as a distinct sub-task with:

> "[M3.0] DERIVATION GATE (must complete BEFORE M3.1+ coding): derive closed-form expression for P(MSK) = P(delta_D[r,s] = 0 exactly) under a single K1 bit-flip as a function of input distribution and C matrix, for int32 2MM cascade with eps=0. CRITICAL: FlipTracker's FP-cancellation mechanism does NOT directly apply to int32 exact-match."

The derivation is worked through:
> "For r=i_fault: delta_D[r,s] = delta_mult · C[j_fault, s]. Exact-zero requires C[j_fault, s] = 0 (since delta_mult = ±2^b > 0). So P(MSK | r=i_fault, fault at (i_f,j_f,b)) = P(C[j_fault, s] = 0). For continuous distributions (Gaussian, Bimodal {±μ} excluding 0): P(C=0) = 0 → P(MSK|r=i_f) = 0 → P(MSK total) ≈ 31/32 ≈ 0.969 INDEPENDENT of σ."

The gating constraint is explicit: "GATING: M3.1+ cannot start until M3.0 verified."

R-NM8 mirrors this: "FlipTracker FP-cancellation mechanism doesn't apply to int32 exact-match; without M3.0 derivation, cross-family experiment may collapse to near-flat curves. M3.0 derivation gate MANDATORY."

All four required elements are present.

**VERDICT: VERIFIED**

---

### ACTION-4 (was MEDIUM-1): M6 renamed + anti-p-hacking framing

**CHECK**: Is M6 renamed? Is the framing present?

M6 heading is now:
> "### M6 — Sensitivity analysis: eps relaxation (0.5 days, secondary cross-check) — RENAMED per Codex iter 1 MEDIUM-1"

M6.1 contains explicit framing:
> "EXPLICIT FRAMING in REPORT: 'Under eps=0 (Yang-faithful exact-match), non-monotonicity within continuous distribution families is NOT observed (per M3.0 derivation: P(MSK) ≈ 31/32 independent of σ). Under relaxed eps=10⁻⁶ classification, non-monotonic peak appears at σ ∈ [50,150]. This metric-choice sensitivity is reported transparently — we do NOT cherry-pick the relaxed-eps result as the headline finding.'"

M6.3 adds: "Explicit 'Sensitivity Analysis' subsection: discuss researcher-degrees-of-freedom honestly."

R-NM5 now reads HIGH severity with mitigation: "M6 renamed to 'Sensitivity analysis: eps relaxation'; eps=0 is primary; eps>0 reported transparently as sensitivity check, not headline."

**VERDICT: VERIFIED**

---

### ACTION-5 (was MEDIUM-2): Bonferroni correction

**CHECK**: Is α/8 = 0.00625 cited in M4.3? Is the 4×2 logic correct?

M4.3 states:
> "Bonferroni correction: 4 families (Gaussian, Uniform-pos, Uniform-zero-mean, Bimodal) × 2 independent categories (MSK+SDC+OTR=1 → 2 free) = 8 independent tests, α/8 = 0.00625 (not α/9 from previous v1)"

The acceptance criteria section uses a slightly different value:
> "Cross-family experiment shows statistically significant gap between Gaussian/Bimodal and Uniform at matched σ (Kendall test, Bonferroni-corrected α=0.0056)"

This is a **RESIDUAL INCONSISTENCY**: the acceptance criteria still use α=0.0056 (the old α/9 value from the previous version), while M4.3 correctly states α=0.00625. The two thresholds differ: 0.0056 is MORE stringent than 0.00625. In practice this is a minor discrepancy (the corrected threshold is more lenient, so using the old stricter value is conservative). However, it is a copy-paste error in the acceptance criteria that should be cleaned up.

Note: the iter 1 ACTION-5 specified the correction should be α/8 = 0.00625. M4.3 correctly applies this. The acceptance criteria section was not updated to match, leaving an internal inconsistency.

**VERDICT: VERIFIED** (M4.3 correctly fixed; acceptance criteria inconsistency is minor and conservative — noting as a new issue)

---

### ACTION-6 (was MEDIUM-3): Effort estimate

**CHECK**: Is header updated? Does milestone sum equal 7.5 days?

The header still reads:
> "**Effort**: 4 days effective (~25 new tests, ~600 new LOC)"

The "Estimated complexity" section IS corrected:
> "**M — 7 days effective** (CORRECTED per Codex iter 1 MEDIUM-3: previous '4 days' was wrong arithmetic). With buffer: 8-9 days."

And the per-milestone sum is shown:
> "Total: 7.5 days"

The header ("Effort: 4 days") is NOT updated — it still says 4 days, contradicting the body. This is a **RESIDUAL INCONSISTENCY**. A reviewer reading the plan header would see "4 days" and miss the correction buried in the complexity section. The constraint block (item 9) also still reads "ETA: 4 days effective."

**VERDICT: UNRESOLVED** — Header metadata (`**Effort**: 4 days`) and Constraint 9 (`ETA: 4 days effective`) still say 4 days, directly contradicting the corrected complexity section (7.5 days). Two places not updated.

---

### ACTION-7 (was LOW-2): Risk register

**CHECK**: R-NM4 MEDIUM? R-NM5 HIGH? R-NM7 NEW MEDIUM? R-NM8 NEW HIGH?

Checking the risk register table:
- R-NM4: "**MEDIUM** (elevated per Codex iter 1 LOW-2)" — PRESENT
- R-NM5: "**HIGH** (elevated per Codex iter 1 MEDIUM-1)" — PRESENT
- R-NM7: "**NEW per Codex iter 1 CRITICAL-2**: Cross-family Uniform[0,M] confounds mean-shift with shape effect | MEDIUM" — PRESENT
- R-NM8: "**NEW per Codex iter 1 CRITICAL-3**: FlipTracker FP-cancellation mechanism doesn't apply to int32 exact-match | **HIGH**" — PRESENT

All four risk register changes are correctly applied.

**VERDICT: VERIFIED**

---

## New issues introduced by edits

1. **NEW-1 (MINOR)**: Acceptance criteria (line: "Bonferroni-corrected α=0.0056") still uses the old α value from v1. The corrected value from M4.3 is α=0.00625. Both values are stated in the same document, creating an internal inconsistency. Since 0.0056 < 0.00625, using the old value is conservative (harder to pass), so this does not introduce a false positive risk — but it is a copy-paste oversight that should be corrected for clarity.

2. **NEW-2 (MINOR)**: The header `**Effort**: 4 days effective` and Constraint 9 `ETA: 4 days effective` were NOT updated to 7.5 days. This is the same content as ACTION-6's UNRESOLVED finding. Downstream readers of the plan header will be misled.

3. **NO NEW SUBSTANTIVE ISSUES**: The 7 edits do not introduce any new methodological, statistical, or scientific problems. The core experimental design is coherent. The M3.0 derivation gate as written is correct (the P(MSK) ≈ 31/32 result is analytically sound for continuous distributions with eps=0 and integer arithmetic). The 4-family design with confound separation is logically sound.

---

## Summary

6 of 7 actions VERIFIED. 1 action (ACTION-6) UNRESOLVED due to header and constraint not updated. 2 minor new issues (header inconsistency is the same issue as ACTION-6; acceptance-criteria α value is a different minor inconsistency).

VERDICT: APPROVE_WITH_CHANGES

FINDINGS:
- ACTION-1: VERIFIED — CORRECTION block in Prior Art explicitly removes the false "first sweep" claim and repositions novelty as "first analytical model."
- ACTION-2: VERIFIED — 4th family (zero-mean Uniform) added to Experiment 1; explicit confound-separation language present.
- ACTION-3: VERIFIED — M3.0 derivation gate present before M3.1, contains FlipTracker inapplicability note and P(MSK) ≈ 31/32 derivation, with explicit GATING constraint.
- ACTION-4: VERIFIED — M6 renamed to "Sensitivity analysis: eps relaxation"; anti-p-hacking framing explicitly present in M6.1 and M6.3.
- ACTION-5: VERIFIED — M4.3 correctly states α/8 = 0.00625 with 4×2 logic; minor residual inconsistency in acceptance criteria (still α=0.0056) is conservative and does not invalidate the fix.
- ACTION-6: UNRESOLVED — "Estimated complexity" section correctly states 7.5 days, but the plan header (**Effort**: 4 days) and Constraint 9 (ETA: 4 days effective) were NOT updated. Two places remain inconsistent with the correction.
- ACTION-7: VERIFIED — All 4 risk register changes present: R-NM4 MEDIUM, R-NM5 HIGH, R-NM7 NEW MEDIUM, R-NM8 NEW HIGH.

NEW_ISSUES:
- NEW-1 (MINOR): Acceptance criteria still references Bonferroni α=0.0056 (old v1 value); should be updated to α=0.00625 to match M4.3.
- NEW-2 (MINOR/same root as ACTION-6): Header Effort field and Constraint 9 ETA still say 4 days; should be 7.5 days.

RECOMMENDATIONS:
- Fix ACTION-6: update `**Effort**: 4 days effective` in the header to `**Effort**: 7.5 days effective` and update Constraint 9 `ETA: 4 days effective` to `ETA: 7.5 days effective`.
- Fix NEW-1: update acceptance criteria line "Bonferroni-corrected α=0.0056" to "Bonferroni-corrected α=0.00625".
- These are both one-line mechanical fixes. No structural changes needed.
