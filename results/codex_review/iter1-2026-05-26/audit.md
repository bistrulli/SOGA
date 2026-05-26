# Cross-Review Audit Log — iter1-2026-05-26

**Plan reviewed**: `plan/2026-05-26-bit-exact-fault-model.md`
**Run ID**: iter1-2026-05-26
**Date**: 2026-05-26

---

## Tooling status

- **codex-cli**: AVAILABLE (version confirmed via `codex --help`)
- **codex exec invocation**: Background process (biji32fmm) returned EXIT 0 after Claude self-review was completed
- **Mode for iter 1**: `claude-self-review` (primary) + `codex-cli` (secondary, completed after primary)
- **Codex output file**: `iter_1_review_raw.txt`
- Both reviewers reached APPROVE_WITH_CHANGES independently. Codex verdict triangulates Claude findings.

---

## Iteration log

### Iter 1

- **Timestamp**: 2026-05-26
- **Tool used**: claude-self-review (codex-cli timeout)
- **Artifact snapshot**: `iter_1_artifact.md` (copy of plan at review time)
- **Prompt**: `iter_1_prompt.md`
- **Review output**: `iter_1_review.md`
- **VERDICT**: APPROVE_WITH_CHANGES

**Findings summary**:
- F1 (HIGH): OTR aggregation semantics mismatch — analytical per-cell mean vs MC any-cell boolean. Residual L2 root cause likely survives bit-exact model if not fixed.
- F2 (MEDIUM): NON-MONOTONE threshold `τ < 0.7` without p-value condition is too loose.
- F3 (MEDIUM): Multiple-testing correction absent (3 curves tested, Bonferroni needed).
- F4 (MEDIUM): `n_wrong ≥ 5` filter is statistically redundant (P≈0.98 under H0).
- F5 (LOW): Risk R-BE11 missing from risk register.
- F6 (LOW): Pearson > 0.95 acceptance criterion does not specify which curves.
- F7 (LOW): Endianness fix description imprecise (implementation correct).

**Required changes before proceeding**:
1. Address OTR aggregation semantics (F1) — add task in R1 or R2 [CONFIRMED by Codex]
2. Revise NON-MONOTONE threshold rule (F2, F4) [CONFIRMED by Codex]
3. Add Bonferroni correction (F3) [CONFIRMED by Codex]
4. Add R-BE11 to risk register (F5) [CONFIRMED by Codex]
5. Clarify Pearson acceptance criterion scope (F6) [CONFIRMED by Codex]
6. (Codex-only) Address NaN/Inf guard over-broadness in scalar path — guard on Inf input blocks the valid case of `+Inf` bit-30-flip → 1.0
7. (Codex-only) Default `--mode bit_exact` in runners breaks legacy behavioral backward-compatibility (old invocations silently change behavior); document as deliberate breaking change or default to `5_class` with explicit opt-in

**Codex-only findings that add to Claude review**:
- C1: NaN/Inf guard: `if not np.isfinite(v): return 0.0, True` incorrectly guards Inf. IEEE 754 says flip_bit(+Inf, 30) = 1.0 (valid finite result). The guard should only catch NaN input, not Inf. Inf input with specific bit flips returns FINITE results. This is a correctness bug in the spec.
- C2: Default `--mode bit_exact` is a behavioral breaking change for any CI/CD or scripts that call the runners without `--mode`. Should be explicitly versioned.

---

## Next step

- Orchestrator should apply 7 required changes to plan, then proceed to iter 2 review.
- Consecutive REJECT count: 0 (APPROVE_WITH_CHANGES does not count toward REJECT escalation threshold)
- Consensus verdict: APPROVE_WITH_CHANGES (both Claude and Codex agree)
