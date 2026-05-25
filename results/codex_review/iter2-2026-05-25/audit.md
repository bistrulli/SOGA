# Audit — lishan-resilience-poc iter 2 cross-review
**Run ID**: iter2-2026-05-25
**Date**: 2026-05-25
**Artifact**: plan/2026-05-25-lishan-resilience-poc.md (v2, post iter-1 edits)
**Checklist**: PLAN

---

## Tool availability

```
codex-cli: AVAILABLE (version 0.124.0, model gpt-5.3-codex, provider azure)
```

No downgrade event. Codex CLI used as primary reviewer.

---

## Chronological log

| Time | Event |
|------|-------|
| 2026-05-25 | iter 2 started; artifact snapshotted to iter_2_artifact.md |
| 2026-05-25 | iter_2_prompt.md written (10 verification criteria + new-issue check) |
| 2026-05-25 | codex exec run; exit 0; verdict parsed |
| 2026-05-25 | iter_2_review.md saved |
| 2026-05-25 | Independent orchestrator analysis confirms all Codex findings |
| 2026-05-25 | REJECT confirmed — 4 unresolved issues with strong overlap between Codex and orchestrator |

---

## Verdicts by iteration

| Iter | Reviewer | Verdict | Primary grounds |
|------|----------|---------|----------------|
| 1 | Claude self-review | APPROVE_WITH_CHANGES | 3 critical + 7 medium findings |
| 2 | Codex CLI (gpt-5.3-codex) | REJECT | 4 unresolved issues after edits (stale K labels, CLT invalid for A=I, Pearson inconsistency x3 locations, MC count inconsistency) |

---

## Iter-1 findings verification summary (iter 2)

| Finding | Status | Notes |
|---------|--------|-------|
| C1 — K formula correctness | PARTIALLY VERIFIED | K_max=161 and K=6 for A=I_32 are correct. Stale "K=26" labels remain in Test Plan (line 223, 228), Acceptance Criteria (line 247), Uncertainty register (U1, line 280), Risk register R-LR1 (line 296). These are UNRESOLVED remnants. |
| C2 — Bimodal warning interval | PARTIALLY VERIFIED | Warning correctly shifted to p∈[0.2,0.8]. "Max bimodality" language is imprecise (maximum bimodality is at p=0.5, not the whole interval). CLT justification is INVALID for A=I_32 (only 1 nonzero term per row — not a sum of many terms). |
| C3 — EXACT vs MOMENT-MATCHED split | VERIFIED (with caveat) | Table, paragraph, M1.1, M1.4 all correctly split SIGN/EXP=EXACT, MANTISSA=MOMENT-MATCHED. Caveat: "<2% relative error" is stated as empirical fact before any validation; M3.5 sets a 5% criterion. Contradiction. |
| M1 — v=1e-30 reasoning | VERIFIED | "Shift ratio (2^k-1) is huge regardless of |v|" is correct. |
| M2 — logsf cap ordering | VERIFIED | R-LR7 now correctly positions cap AFTER H1, with empirical verification committed to M3.3. |
| M3 — p_critical definition | PARTIALLY VERIFIED | p_critical = argmax |dSDC/dp| is specified, fallback p=0.5 given. Underspecified: finite-difference formula, tie-break, endpoint policy not stated. |
| M4 — degenerate p=0/1 | VERIFIED | sigma²_p < 1e-30 threshold is correct for the sweep range {0, 0.05, ..., 1.0}. |
| M5 — A matrix specification | VERIFIED | M0.3 now extracts A_kernel from lishan_2mm_32x32.soga, mentions A_identity variant. Complete. |
| M6 — Pearson threshold | UNRESOLVED | M4.3 correctly updated to 0.90 primary / 0.85 fallback. BUT Test Plan line 233 still says "0.95", Acceptance Criteria line 248 still says "0.95", R-LR4 still says "< 0.95". Three stale copies. |
| M7 — p_fault semantics | PARTIALLY VERIFIED | M1.4 clarified semantics. M2 simulate_fi_mc.py description does not explicitly enforce matching semantics. |

---

## New issues introduced by iter-1 edits

| ID | Issue | Severity |
|----|-------|----------|
| N1 | Step-3 MC point count inconsistency: M5.4 says 4 points {0.0, 0.5, p_critical, 1.0}; Test Plan (line 232) says "3 representative points"; Step-3 flow pseudocode shows only {0.0, 0.5, 1.0}; Constraint 9 says "≥3 suspicious points". Three conflicting counts. | MEDIUM |
| N2 | CLT argument invalid for A=I_32: M5.1 invokes Lyapunov CLT with m=32 terms, but for A=I_32, D[r,s] = B[r,s] — a single Bernoulli cell, not a sum. CLT does not apply. The fallback to 2-component GM is the correct path but the CLT justification as written gives a false sense of safety. | MEDIUM |
| N3 | "<2% relative error" bound on mantissa approximation (Correctness scope paragraph) is pre-validation and contradicts M3.5 enforcing <5%. Must pick one consistent number or rephrase as "target/hypothesis". | MINOR |

---

## Escalation status

Two consecutive REJECT verdicts have NOT occurred (iter 1 was APPROVE_WITH_CHANGES, iter 2 is REJECT). No 3-consecutive-REJECT escalation triggered. Proceeding to report findings to orchestrator.

STRONG_REJECT marker: NOT issued (only one REJECT, iter 2). However, orchestrator should note that both reviewers (iter-1 Claude, iter-2 Codex) independently flagged C2/CLT and M6/Pearson on independent passes.
