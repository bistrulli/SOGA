# Iter 3 Review Prompt — lishan-resilience-poc plan
**Date**: 2026-05-25
**Mode**: claude-self-review (codex-cli unavailable)
**Scope**: TARGETED — verify only 6 mechanical fixes from iter-2 REJECT

## Instruction to reviewer

Act as an independent reviewer. Do NOT consider yourself the author. Be skeptical.
This is a TARGETED review: verify ONLY the 6 fixes required after iter-2 REJECT.
Do NOT re-review the entire plan from scratch.

## 6 fixes to verify

1. C1(c-d): Stale K=26 / K=5121 labels swept from Test Plan, Acceptance Criteria, U1, R-LR1.
   Architecture line ~72 should say K_full=1+5·m·n=81 at m=n=4. K_full=5121 only in Alternatives section.

2. C2(b): M5.1 CLT argument scoped to dense A (≥ m/2 nonzero per row).
   A=I_32 routed to 2-component GM primary path. Heuristic switch m_dense ≥ 16 documented. Logged in config.json.

3. M6: Pearson threshold swept to 0.90 primary / 0.85 fallback in Test Plan, Acceptance Criteria, R-LR4.

4. N1: Step-3 MC point count unified to 4 everywhere (Test Plan, Acceptance Criteria, M5.4).

5. N2: "max bimodality" replaced by "high bimodality regime (maximum at p=0.5)" or equivalent in M5.1 and R-LR2.

6. N3: Mantissa bound unified to "<5% target, enforced by M3.5" in Correctness scope paragraph. No "<2%" mention.

## Stale strings to grep for (any match outside cross-review history = FAIL)

- "K=26"
- "K=5121" outside Alternatives section
- "Pearson > 0.95" / "0.95"
- "max bimodality" as standalone term
- "<2% relative" or "<2%" in mantissa error bound context
- "3 representative p points" or "≥3 MC"
