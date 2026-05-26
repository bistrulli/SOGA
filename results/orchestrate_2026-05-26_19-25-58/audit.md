# Orchestration Audit — 2026-05-26_19-25-58

**Plan**: plan/2026-05-26-bit-exact-fault-model.md
**Branch**: feat/lishan-resilience-poc
**Start**: 2026-05-26 (autonomous bounded loop, max 5 iterations)

---

## Phase 0 — Context loaded

- Baseline: 72 passed + 2 xfailed + 1 xpassed (74 total run) — clean working tree at 8d70bf58
- MC OTR semantics (R0.3): ASCERTAINED from simulate_fi_mc.py:
  - `classify_outcome` returns `(0.0, 0.0, 1.0)` if `np.any(~np.isfinite(D_perturbed))` — this is PER-EXECUTION OTR
  - The entire execution is classified OTR=1 if ANY output cell is non-finite
  - The existing analytical model uses PER-CELL-AVERAGED OTR — this IS the L2 mismatch
- Existing `predict_resilience_soga.py`: 5-class moment-matched model, ~599 lines
- Existing tests: test_step1_acceptance.py has 2 xfail tests (pearson_msk, pearson_sdc)
  - test_step3_monotonicity.py has 1 xfail (mc_sdc_monotone_tau)

## R0.3 Finding
MC `simulate_one_fi` uses `np.any(~np.isfinite(D_perturbed))` for OTR classification.
`classify_outcome` in the v_sweep loop returns (0,0,1) if ANY cell non-finite.
**Verdict**: MC = per-execution OTR (whole execution = OTR if any cell non-finite).
For A=I_32: B[i,j] fault only affects D[i,j] (identity matrix), so OTR iff the single
faulted output cell is non-finite. Per-execution and per-cell are equivalent for A=I.
HOWEVER: the analytical model was averaging OTR over all cells (the 32x32=1024 cells)
and dividing by n_in again. This creates systematic undercount of OTR.

## Iteration 1 — R0 + R1 (Setup + Bit-fault table module)

### Tasks
- R0.1: DESIGN_BIT_EXACT.md
- R0.2: test_bit_fault_table.py (5 hand-computed sanity tests, RED initially)
- R0.3: OTR semantics documented (done above)
- R1.1: compute_xor_shift scalar implementation
- R1.2: bit_fault_table_array vectorized implementation
- R1.3: 100-random-v validation
- R1.4: 10 IEEE 754 edge cases

