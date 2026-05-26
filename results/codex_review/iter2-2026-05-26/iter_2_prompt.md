You are reviewing a PLAN artifact for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution).

This is ITER 2 — a TARGETED re-review. In iter 1 this plan received APPROVE_WITH_CHANGES with 9 findings (F1-F9). The plan has been revised. Your sole task is to VERIFY that each of the 9 fixes was correctly applied. Do NOT re-review the entire plan.

FIXES TO VERIFY:

**F1 (HIGH) — OTR aggregation semantics**: NEW R0.3 task added (read simulate_fi_mc.py to determine MC's OTR aggregation semantics) and NEW R2.2b task (align analytical formula). New section "⚠️ CRITICAL — OTR aggregation semantics" in Approach. New R-BE11 in risk register.

**F2 (MEDIUM) — Inf input guard contradiction**: scalar `compute_xor_shift` now guards `np.isnan(v)` only (not `not np.isfinite(v)`). Comment block explicitly states Inf input flows through, individual bits handled per IEEE 754.

**F3 (MEDIUM) — NON-MONOTONE τ threshold**: revised Non-monotonicity statistical test section. Now uses p_kendall (not bare τ threshold) for primary test.

**F4 (MEDIUM) — n_wrong ≥ 5 redundancy**: threshold raised to n_wrong ≥ 10.

**F5 (MEDIUM) — Bonferroni correction**: α_per_test = 0.05/3 ≈ 0.0167 documented; applied in monotonicity decision rule.

**F6 (MEDIUM) — Default --mode backward-compat**: R2.5 task now includes runtime banner mitigation, config_version bump, CHANGELOG entry. Marked as DELIBERATE behavioral change. R-BE12 added.

**F7 (LOW) — R-BE11 missing**: added R-BE11 (OTR semantics), R-BE12 (default --mode breaking), R-BE13 (multiple testing).

**F8 (LOW) — Pearson criterion**: Acceptance Criteria now explicitly says "Pearson > 0.95 for MSK and SDC curves only; OTR uses abs err".

**F9 (LOW) — Endianness description**: no code change needed (cosmetic).

SPECIFIC STALE STRINGS TO CHECK (grep these for unexpected occurrences OUTSIDE the cross-review history section):
- "not np.isfinite" (should appear in code only as `is_special = not np.isfinite(v_post)` for the OUTPUT side, and in R-BE2 mitigation text — NOT as the INPUT guard)
- "τ < 0.7" (old NON-MONOTONE threshold — should appear ONLY in cross-review history)
- "n_wrong ≥ 5" (old threshold — should appear ONLY in cross-review history)

KNOWN RESIDUAL ISSUES TO EVALUATE:

1. Line 89: `is_special = not np.isfinite(v_post)` in scalar `compute_xor_shift` — this checks the POST-flip value (v_post), NOT the input v. This is CORRECT behavior (F2 was about the INPUT guard, not the output check). Verify this is not a false alarm.

2. Line 244 in R4.3 sub-task: `NON-MONOTONE if τ<0.7 OR (n_wrong≥5 AND binom_p<0.01)` — this CONFLICTS with the Approach section (lines 151-161) which uses Bonferroni-corrected p_kendall as primary and n_wrong≥10. Assess: is R4.3 a new inconsistency introduced by the edits that F3/F4/F5 DID fix in the Approach section but FAILED to fix in the R4.3 task text?

3. Line 351 (R-BE2): mitigation says "explicit `not np.isfinite(v)` guard at entry" — this is the OLD (wrong) mitigation text that contradicts F2's fix. F2's fix is NaN-only. The risk register mitigation for R-BE2 still says the wrong thing.

Evaluate all three items above and determine their severity.

Output format (strict):
VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- <finding 1: Fx status VERIFIED or UNRESOLVED, with line reference>
- <finding 2: ...>
RECOMMENDATIONS:
- <action 1>
