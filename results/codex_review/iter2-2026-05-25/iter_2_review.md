# Iter 2 Review — lishan-resilience-poc plan
**Reviewer**: Codex CLI (gpt-5.3-codex, azure)
**Mode**: codex-cli
**Date**: 2026-05-25

---

## VERDICT: REJECT

---

## FINDINGS

- C1(a,b) are correct: `1 + 5*32 = 161`, and `K=6` for `A=I_32` is correct under the class-aggregated counting used here.
- C1(c,d) are not fully fixed: for `m=n=4`, full expansion is `K_full = 1 + 5*4*4 = 81`; stale `K=26` / `K=5121` references remain in Test Plan, Sanity-vs-analytical, Acceptance Criteria, and R-LR1.
- C2(a) is only partially valid: `[0.2,0.8]` is a conservative warning band, but calling it "max bimodality" is inaccurate (maximum is at `p=0.5`).
- C2(b) is incorrect for `A=I_32`: CLT justification does not hold when each row has only one nonzero term (`D[r,s]` is not a many-term sum).
- C3(a) is mostly consistent: SIGN/EXP marked EXACT, MANTISSA marked MOMENT-MATCHED across table/paragraph/M1.1.
- C3(b) is not justified as written: "empirically expected <2%" is a pre-validation claim and conflicts with M3.5's `<5%` criterion.
- M1 (v=`1e-30`) reasoning is correct: the shift ratio `(2^k-1)` is independent of `|v|` in the stated identity-case setup.
- M2 logsf-cap ordering fix is correctly applied: cap is positioned after H1 and tied to explicit M3.3 verification.
- M3 (`p_critical`) is directionally improved but still underspecified: finite-difference method, endpoint handling, and tie-breaks are not defined.
- M4 threshold (`sigma_p^2 < 1e-30`) is acceptable for the declared sweep; it effectively reduces to endpoint handling (`p=0/1`) on `{0,0.05,...,1}`.
- M5 A-matrix extraction is now explicit and materially complete (`A_kernel` extraction + optional `A_identity` variant).
- M6 is inconsistent: M4.3 uses Pearson `>0.90` primary / `>0.85` fallback, but Test Plan + Acceptance + R-LR4 still use `0.95`.
- M7 is only partially fixed: `p_fault` semantics are clarified in M1.4, but M2 does not explicitly enforce matching MC sampling semantics.
- New inconsistency introduced: Step-3 MC point count conflicts (`M5.4` says 4 points incl. `p_critical`; Test Plan says 3 points; Step-3 flow pseudocode shows `{0,0.5,1}`).

## RECOMMENDATIONS

- Replace all stale K references with one consistent scheme: `K_full=1+5mn` (81 for 4x4, 5121 for 32x32) and reduced `K=1+5*m_distinct` (6 for identity, 21 for dense 4x4).
- Unify Pearson acceptance thresholds across M4.3, Test Plan, Acceptance Criteria, and R-LR4.
- Make CLT conditional on effective row support (e.g., many nonzero terms); add explicit non-CLT handling for sparse/identity rows.
- Reword bimodality warning rationale (or narrow to a central range) so "max bimodality" language is mathematically accurate.
- Recast mantissa "<2%" as a hypothesis/target until validated; keep one enforceable bound in M3.5.
- Specify `p_critical` computation details (discrete derivative formula, tie-break, endpoint policy).
- Make Step-3 MC-point count consistent everywhere and codify `p_fault` MC sampling semantics explicitly in M2.
