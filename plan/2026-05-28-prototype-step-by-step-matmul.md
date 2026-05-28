# Plan: prototype-step-by-step-matmul

**Date**: 2026-05-28
**Slug**: `prototype-step-by-step-matmul`
**Branch (target)**: `feat/lishan-resilience-poc` (stay; current HEAD: 38ad282a)
**Parent plans**:
- `plan/2026-05-25-lishan-resilience-poc.md` (input-side POC, executed)
- `plan/2026-05-26-bit-exact-fault-model.md` (bit-exact refinement, executed)
- `plan/2026-05-27-non-monotonic-input-distribution-sweep.md` (current `Strada Q` pitch)
**Effort**: 1-2 days effective (S+)
**Triggered by**: brainstorm 2026-05-28 — user hypothesis that "one-shot" matmul abstraction is the root cause of the discrepancy with Lishan Yang's NVBit-FI numbers
**Strategic stance**: feasibility de-risk for a NEW pitch ("analytical NVBit-FI"); complements the current `Strada Q` pitch (non-flat distributions) without replacing it.

---

## Goal

Build a 2×2 SOGA program that decomposes matrix multiplication into per-scalar mult-add operations, with one Bernoulli fault injection at an intermediate accumulator step. Validate the analytical SOGA output against (a) a hand-derived closed-form formula and (b) a Monte Carlo reference simulator. Produce a **GO/NO-GO decision** for committing to a full 32×32 implementation (Option B).

**Two valid outcomes**:
- **GO**: 3-way match (SOGA = analytical = MC within ε); runtime/memory extrapolation to 32×32 is feasible → open Option B in a subsequent plan.
- **NO-GO**: prototype reveals fundamental blocker (component explosion, numerical instability, semantic mismatch); document blockers, retain current `Strada Q` pitch as the only viable angle.

---

## Context

### Why this prototype matters

D2 (internal SOGA vs MC consistency on input-side fault model) was closed 2026-05-28: SOGA bit_exact matches an analytical ground truth at machine precision, and matches MC within Wilson 95% CI. Residual gap with Lishan's published numbers is attributed to D1 (abstraction gap: input-side vs register-level SASS).

The natural SOGA-native fix to D1 is to decompose the matmul into scalar mult-add ops so that **intermediate accumulator faults** can be injected analytically. This is the closest analytical equivalent of NVBit-FI we can build without leaving the SOGA paradigm.

Before committing 2-3 weeks to a full 32×32 implementation, we validate the approach on a minimal 2×2 case. If the prototype fails (component explosion, moment-matching cascade, σ=0 degeneracies), we save weeks of misdirected work.

### Specialist consensus (Phase D, 3 parallel)

- `soga-internal-expert` (`/tmp/soga_internal_prototype_memo.md`): **GO**. No DSL changes, no core changes for 2×2. Reference syntax anchors: `TwoCoins.soga` (Bernoulli), `RandomWalkDisc10.soga` (affine accumulator in loop), `ClinicalTrialPrune.soga` (prune). Key parser pitfall: use `0 - acc` not `(-1) * acc` to avoid `mul_func` misclassification. Runtime <100ms for 2×2; ~minutes for 32×32 with `prune(K=8)` per step.
- `gaussian-mixture-expert` (`/tmp/gm_prototype_memo.md`): the affine path (A deterministic, B random) is **EXACT** — zero moment-matching error. Closed-form for 2×2 with fault at intermediate accumulator: `(1-p)·N((a00+a01)v, (a00²+a01²)σ²) + p·N((a01-a00)v, same σ²)`. For 32×32 scaling: top-K (`ranking_prune`), NOT Runnalls. K = 1 + N_fault_sites per cell.
- `numerical-stability-expert` (`/tmp/numerical_prototype_memo.md`): for 2×2 with p=0.01, the current `prob_tol=1e-10` is sufficient. For 32×32 scaling (Option B), `prob_tol` should drop to 1e-15 AND `make_psd` should be inserted in `add_func/mul_func` for d>1 — but these are libSOGA*.py changes requiring `/audit-numerical`, **out of scope for the 2×2 prototype**.

---

## Constraints

1. **NO modifications to `src/libSOGA*.py`** — prototype uses the existing SOGA core unchanged.
2. **NO modifications to `grammars/*.g4`** — prototype uses the existing DSL.
3. **Scope strictly 2×2** — no scaling experiments here; document 32×32 needs as FOLLOW-UP risks.
4. **Maximum 2 days effective work** — abort if scope creeps.
5. **Analytical ground truth derivation MUST be included** — without closed-form verification, the 3-way match is incomplete.
6. **MC reference simulator must be independently implemented** — not reused from `simulate_fi_mc.py`, which models input-side faults (different semantics). MUST accept `--seed N` argument; seed value logged in `config.json`.
7. **σ_reg = 10⁻⁶** for input B cells (per `gaussian-mixture-expert` memo: avoids Tallis div-by-zero).
8. **Parser safety**: use `0 - acc` for sign flip, never `(-1) * acc`.
9. **Reproducibility (CLAUDE.md §8)**: `HASHES.txt` (SHA256 of `.soga` input + scripts) and `config.json` (seed, grid params, tolerances) MUST exist in `experiments/lishan_prototype_2026-05-28/`.

---

## Approach

### Architecture: 2×2 scalar decomposition with 1 fault site

The prototype program (`programs/Example/lishan_prototype_2x2_scalar.soga`) computes only `D[0,0] = sum_k A[0,k] * B[k,0]` with A = I_2, B random. Sequence:

```
acc = 0;
acc = acc + a00 * b00;          // partial after first FMA
fault = bern(p);
if fault == 1 {
    acc = 0 - acc;               // sign flip (intermediate fault)
} else {
    acc = acc;
} end if;
acc = acc + a01 * b10;          // partial after second FMA
d00 = acc;
```

Closed-form derivation (per `gaussian-mixture-expert` memo Q2):
- After first add: `acc ~ N(a00·v, a00²·σ_b²)`
- After Bernoulli merge: `acc ~ (1-p)·N(a00·v, a00²σ_b²) + p·N(-a00·v, a00²σ_b²)`
- After second add: `acc ~ (1-p)·N((a00+a01)·v, (a00²+a01²)·σ_b²) + p·N((a01-a00)·v, (a00²+a01²)·σ_b²)`

For A=I_2 (a00=1, a01=0): no-fault component `N(v, σ_b²)`, fault component `N(-v, σ_b²)`. The two components are well-separated (mean gap 2v), so top-K (K=2) preserves both exactly.

### Why this isolates the right risk

We're testing the **GM cascade through affine-plus-Bernoulli sequences with intermediate fault injection**. This is exactly the kernel that would run 64K times for 32×32. The 2×2 case validates:
- Bernoulli sign-flip produces correct 2-component mixture
- Subsequent affine update applied per-component (gives us closed-form parameters to match)
- Final E[·], Var[·], Pr(|·|>ε) extracted correctly via existing CLI
- σ_reg=10⁻⁶ avoids any degenerate Tallis path

If these all hold, scaling to 32×32 is a question of (a) operation count × runtime, (b) pruning efficacy. Both estimable from prototype telemetry.

---

## Alternatives considered

- **Alt B (Direct 32×32 implementation)**: rejected. 2-3 weeks of work; if any of the prototype hazards (parser misclassification, σ=0 degeneracy, prune efficacy) fails, the work is wasted. De-risking with 2×2 first costs 1-2 days.
- **Alt C (Hybrid: matrix-GM baseline + numpy fault aggregation)**: rejected for this slot. Doesn't test the scalar-GM cascade, which is the actual risk for Option B. If Option A succeeds, Option C becomes redundant; if Option A fails, Option C may be revisited as a fallback.
- **Alt D (Theoretical-only: derive closed-form P_SDC for register-level uniform fault distribution)**: rejected. Would require deep symbolic algebra; output is one formula, not a tool. Doesn't demonstrate SOGA's capability.

---

## Sub-tasks (atomic)

### iter 1 — Foundation (target: end of day 1)
- [ ] `[iter:1]` `[agent:soga-internal-expert]` `[area:experiments]` Write `programs/Example/lishan_prototype_2x2_scalar.soga` — D[0,0] only, A=I_2, B random N(v, σ²=1e-12), 1 Bernoulli sign-flip between the two adds
- [ ] `[iter:1]` `[agent:gaussian-mixture-expert]` `[area:docs]` Write `experiments/lishan_prototype_2026-05-28/ANALYTICAL.md` — closed-form derivation of E[D[0,0]], Var[D[0,0]], Pr(|D[0,0]-v|>ε) parametric in (v, p, σ_b, ε)
- [ ] `[iter:1]` `[agent:test-engineer]` `[area:experiments]` Write `experiments/lishan_prototype_2026-05-28/mc_reference.py` — independent Python MC simulator that runs the SAME algorithm (sample b00, b10; sample fault; sign-flip if needed; output d00) with n_samples=100000

### iter 2 — Validation (target: morning day 2)
- [ ] `[iter:2]` `[agent:test-engineer]` `[area:experiments]` Write `experiments/lishan_prototype_2026-05-28/compare_3way.py` — runs SOGA, reads `Dist` output, extracts E[D[0,0]], Var[D[0,0]], Pr via `marg_cdf`; compares against analytical formula and MC with Wilson 95% CI
- [ ] `[iter:2]` `[agent:numerical-stability-expert]` `[area:experiments]` Stress test grid (v, p): v ∈ {0.5, 1.0, 2.0}, p ∈ {0.001, 0.01, 0.05}. Verify σ_reg=1e-6 path stable; flag any NaN/Inf; verify zero-crossing degenerate handling (v=0 not in main grid but tested at edge)
- [ ] `[iter:2]` `[agent:soga-internal-expert]` `[area:experiments]` Runtime profile: wall-clock per (v, p); component count after each merge; extrapolation table to 32×32

### iter 3 — Decision report (target: afternoon day 2)
- [ ] `[iter:3]` `[agent:documentation-writer]` `[area:docs]` Write `experiments/lishan_prototype_2026-05-28/REPORT.md` — verdict GO/NO-GO with quantitative basis; if GO, kick off `plan/2026-06-XX-full-scalar-decomp-32x32.md` skeleton; if NO-GO, document blockers
- [ ] `[iter:3]` `[agent:bs-detector]` `[area:meta]` BS-check final REPORT.md for unwarranted claims (e.g., "fast", "scales", without numbers)

---

## Test plan

### Unit-level
- For (v=1, p=0.01, A=I_2):
  - Analytical: E[D[0,0]] = (1-p)·v + p·(-v) = v(1-2p) = 0.98
  - Analytical: Var[D[0,0]] = (1-p)·(σ_b² + v²·4p) ≈ σ_b² + 4·v²·p(1-p)
  - SOGA output must match to relative error < 1e-6

### MC validation (n=10⁵)
- For each (v, p) in 3×3 grid: run MC, compute mean ± Wilson CI for E[D] and Pr(SDC); SOGA value must be inside CI

### Sanity edge cases
- v=0 (zero-crossing): outputs E[D]=0 (with both components canceling), Pr(SDC) handling per `numerical-stability-expert` Q3
- p=0 (no fault): single Gaussian, matches one-shot matmul exactly
- p=1 (always fault): single Gaussian at -v, matches inverted one-shot

### Performance gate
- 2×2 wall-clock < 1s; if > 1s flag and investigate before iter 3
- Component count after final merge: exactly 2 (or 1 after merge of identical-variance components, if SOGA collapses them)

---

## Acceptance criteria

- [ ] SOGA E[D[0,0]] matches analytical formula to relative error < 1e-6 (3 v-values × 3 p-values = 9 points)
- [ ] SOGA Var[D[0,0]] matches analytical formula to relative error < 1e-4
- [ ] MC at n=10⁵ matches SOGA within Wilson 95% CI on all 9 points
- [ ] Runtime 2×2 < 1s (preferably < 100ms)
- [ ] Component count ≤ 2 **after the final merge node** (intermediate states may differ)
- [ ] REPORT.md explicit GO/NO-GO verdict with **quantitative** basis (numbers, not adjectives)
- [ ] code reviewer approves
- [ ] codex cross-reviewer approves (or single-pass APPROVE_WITH_CHANGES applied)

---

## Rollback

All artifacts isolated to:
- `programs/Example/lishan_prototype_2x2_scalar.soga` (1 file)
- `experiments/lishan_prototype_2026-05-28/` (new directory)

Rollback: `rm -rf experiments/lishan_prototype_2026-05-28/ && rm programs/Example/lishan_prototype_2x2_scalar.soga`. **No SOGA core changes, no grammar changes** — zero blast radius outside prototype scope.

---

## Estimated complexity

**1-2 days effective work** (S+). Confidence: HIGH (3 expert memos converge on GO with no blockers identified for 2×2 scope).

Uncertainty flags:
- If SOGA's `printOutput` doesn't expose `Pr(acc > thr)` cleanly, may need 10-line post-hoc wrapper (per `soga-internal-expert` memo Pitfall 6) → +1h
- If parser misclassifies `0 - acc` (low probability per Pitfall 2) → potential half-day debugging

---

## Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| P1 | σ=0 from `bern(p)` components triggers Tallis div-zero in downstream truncate | High | Use σ_reg=1e-6 for B (already constraint #7) |
| P2 | Parser misclassifies `(-1)*acc` as `mul_func` | Medium | Use `0 - acc` (already constraint #8) |
| P3 | Component count from Bernoulli not actually collapsing to 2 after merge | Medium | Profile in iter 2; if K>2, investigate before claiming "scales to 32×32" |
| P4 | Closed-form derivation errors → false validation match | High | Cross-check via independent MC; both must converge to same number |
| P5 | Scope creep into 4×4 / 8×8 / 32×32 within prototype budget | Medium | Hard stop at 2×2; defer all scaling to Option B plan |

---

## Cross-review history

- iter 1: APPROVE_WITH_CHANGES (codex 2026-05-28) — applied 3 changes:
  1. Added `HASHES.txt` and `config.json` to artifact list (CLAUDE.md §8 reproducibility)
  2. Constraint 6 + MC reference must accept `--seed N`; seed logged in `config.json`
  3. Tightened component count AC: "≤ 2 **after the final merge node**" (removed ambiguity on intermediate states)
- audit trail: `results/codex_review/plan-2026-05-28-matmul-cr/`

---

## What success looks like (concrete artifacts at end of plan)

1. `programs/Example/lishan_prototype_2x2_scalar.soga` — runnable
2. `experiments/lishan_prototype_2026-05-28/ANALYTICAL.md` — derivation
3. `experiments/lishan_prototype_2026-05-28/mc_reference.py` — independent MC (accepts `--seed N`)
4. `experiments/lishan_prototype_2026-05-28/compare_3way.py` — validation script
5. `experiments/lishan_prototype_2026-05-28/results.csv` — 9 data points (v, p) × (analytical, SOGA, MC, MC_lo, MC_hi, pass/fail)
6. `experiments/lishan_prototype_2026-05-28/REPORT.md` — GO/NO-GO verdict
7. `experiments/lishan_prototype_2026-05-28/HASHES.txt` — SHA256 of `.soga` + scripts (reproducibility)
8. `experiments/lishan_prototype_2026-05-28/config.json` — seed, grid params, tolerances

If GO: a 1-page skeleton of `plan/2026-06-XX-full-scalar-decomp-32x32.md` ready for the next iteration.
