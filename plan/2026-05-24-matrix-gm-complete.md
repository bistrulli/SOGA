# Plan: matrix-gm-complete — close the 4 remaining CRASH cases

**Date**: 2026-05-24
**Slug**: `matrix-gm-complete`
**Branch**: `feat/matrix-gm-integration`
**Effort estimate**: ~20-30 working days (4-6 weeks of focused work)
**Status**: draft, awaiting cross-review + user approval
**Triggered by**: empirical test session 2026-05-24 — 4 matrix-GM patterns crash; user mandate "deve essere completo in tutto", "deve avere riflessione sulla semantica e i conti che SOGA deve considerare"
**Prerequisite reads**:
- `docs/research-notes/04-matrix-element-extract-write.md` — element extract/write theory
- `docs/research-notes/05-matrix-gm-merge-and-random-matmul.md` — merge + random@random theory

---

## 1. Goal

Lift matrix-GM from "linear forward-prop only" to **first-class first-citizen in every SOGA syntactic construct**:
- `if/else` on programs with matrix variables in scope (currently `NotImplementedError [M3.6]`)
- `for i in range(N) { ... X[i,j] ... }` — matrix indexing with loop variables (currently `TypeError NoneType`)
- `Y = X1 @ X2` — random × random matmul (currently `NotImplementedError [M4.1]`)
- `X[i,j] = expr` — matrix element write (currently `NotImplementedError` runtime)

For each construct, define semantics formally (per research notes 04/05), derive the exact propagation formulas, implement, verify with sanity-vs-analytical tests and Monte Carlo, document for the user.

## 2. Context

After 17 commits on `feat/matrix-gm-integration`, the matrix-GM extension reaches **8 of the 8 forward-propagation features** working end-to-end (matrix_gm, +, A@X, X@B, transp, X[i,j] extract with cross-cov, observe element, observe row/col_sum). What remains broken: 4 patterns where matrix variables interact with **non-linear control flow** or **random×random arithmetic** or **destructive element writes**.

These four cases require non-trivial theoretical handling:
- Merge requires accepting MVGM (Matrix-Variate Gaussian Mixture) as the post-merge family. Closed-form, no Kronecker structure loss (research note 05).
- Loop with index requires runtime resolution of the loop variable in matrix-element references (engineering only, no theory).
- Random × random matmul produces a non-Gaussian distribution — must approximate via delta-method linearization + nearest-Kronecker projection (research note 05).
- Element write destroys Kronecker separability — must densify the affected matrix variable's covariance (research note 04 Finding B1).

## 3. Constraints

1. **No regression on scalar programs**: Bernoulli `E[theta]=0.25689`, ClickGraphPrune `E[simAll]=0.61409`, full pytest 293+ tests must stay green.
2. **No regression on the existing 8 matrix-GM features**: t1..t8 smoke battery from session 2026-05-24 must all PASS post-fix.
3. **Sanity-vs-analytical for every new operation**: each of the 4 fixes ships with at least one ground-truth analytical test (closed-form expected value at the unit-test level).
4. **No silent approximation**: when a numerical approximation is used (delta-method, nearest-Kronecker, dense fallback), the user must see a clear log message or an opt-out flag.
5. **Branch hygiene**: all commits on `feat/matrix-gm-integration`. Never push without explicit user gate. Branch is already pushed to origin (17 commits).
6. **Semantic write-ups**: each implemented operation is documented with the math it implements (the closed-form formula) so a reader / student / reviewer can verify the SOGA computation is what they expect.

## 4. Approach (5 phases, dependency-ordered)

### Phase 1 — Loop-variable resolution in matrix indexing (smallest)
Pure engineering. Extend the regex / dispatcher in two places:
- `libSOGAupdate.update_rule` hybrid routing for `scalar = X[i,j]`: accept `i`, `j` as either NUM or IDV, resolve IDV via `data[idv][0]` at runtime (SOGA's loop counter convention).
- `libMatrixTruncate._classify_constraint` (`_RE_ELEMENT`, `_RE_ROW_SUM`, `_RE_COL_SUM`): accept IDV indices and resolve via `data` dict at dispatch time.

No grammar change (the grammar already accepts `IDV [ (NUM|IDV) , (NUM|IDV) ]`). No theory.

### Phase 2 — Merge with matrix variables (medium)
Lift `libSOGAmerge.merge` from `NotImplementedError [M3.6]` to a working implementation per research note 05 §Problem 1:

- Concatenate components from all branches with weight rescaling `p_i · pi_k`.
- Each component retains its matrix-variate `(M_k, U_k, V_k)` per matrix variable.
- Cross-cov scalar↔matrix per component carried forward unchanged.
- Variables present in only one branch: the absent branch carries the prior (snapshot the joint state at the `if` entry).
- After concat: optionally invoke existing `ranking_prune(K_max)` or `classic_prune` (which already exists for scalar; extend to multi-component matrix block).

Component reduction (Salmond/Runnalls): research note 05 §Q1b notes that post-merge `Sigma_m` is generally not Kronecker-separable; option A (keep dense) vs option B (re-project via Van Loan-Pitsianis). For v1 of this plan, accept option A (dense merged components) — this keeps the merge mathematically exact. Document the per-merged-component cost: `O((mn)²)`. Add memory budget guard.

### Phase 3 — Random × random matmul `Y = X1 @ X2` (medium-large)
Implement per research note 05 §Problem 2:

- Mean: `M_Y = M_X1 @ M_X2` (exact).
- Cov: delta-method linearization → sum of two Kronecker products.
- Project to nearest single Kronecker via Van Loan-Pitsianis rank-1 SVD → `(U_Y, V_Y)`.
- PSD enforce on `U_Y`, `V_Y`.

Add to dispatcher in `libMatrixUpdate.update_rule_matrix` for `MATMUL_RAW` when **both** operands are matrix vars. Add `_nearest_kronecker` if not already present (research note 05 has the implementation).

Component pruning: full mixture product has `J · K` components (Cartesian); apply `ranking_prune(K_max)` immediately.

Document the approximation: emit `MatmulApproxWarning(error_estimate)` when the second singular value of the rearrangement is > 5 % of the first (indicates the approximation discards a non-negligible Kronecker term).

### Phase 4 — Matrix element write `X[i,j] = expr` (largest)
Three sub-cases per research note 04:
- **B1**: `X[i,j] = c` (deterministic constant) — Schur-complement update on vec(X).
- **B2**: `X[i,j] = z` (scalar variable z) — same formula with `var_z > 0` denominator.
- **B3**: `X[i,j] = expr` (scalar expression) — reduce to B2 by evaluating expr first via scalar update.

All require **densification** of X's covariance (V⊗U → (mn × mn) dense matrix). Add a `'dense'` tag variant to `GaussianMixBlock.cov_blocks[k][frozenset({X})]`:
- Tag `'kron'` (current): value is `(U, V)` tuple.
- Tag `'dense'` (new): value is `Sigma` ndarray (mn × mn).

All downstream matrix operations on a dense-mode variable use the dense covariance path. For v1, raise `NotImplementedError` if a Kronecker-required op (e.g., transp, affine) is attempted on a dense-mode variable (acceptable simplification — element write is typically near the end of a program).

LHS routing: extend `update_rule` to detect `^matname[i,j]=...` LHS pattern, dispatch to `_matrix_element_write(dist, mat_name, i, j, rhs_expr, data)`.

### Phase 5 — End-to-end validation + documentation (3-5 days)
- Extend the 8-feature smoke battery to 16 features (add 8 more: each broken case + back-prop chains).
- Add MC ground-truth tests for delta-method and dense fallback (20k samples).
- Update `Manual/ReusabilityGuide.md` with a "Matrix DSL" section covering all working patterns.
- Write `docs/MATRIX_GM_SEMANTICS.md` — for each construct, the formal semantics + the SOGA computation (math + code reference). User-facing.
- Run `/audit-numerical` + `/audit-grammar` regression. Run `/soga-bench Table3` to verify no regression on scalar benchmarks.

## 5. Alternatives considered

### Alt-A: Random × random via direct dense product (no linearization)
Compute the exact non-Gaussian distribution of `Z = X @ Y` symbolically (closed-form moments via Isserlis). **Rejected**: the result is not in any Gaussian family, breaks SOGA's invariant (every joint is a Gaussian mixture). Cannot proceed further. Approximation is mandatory.

### Alt-B: Element write via full marginalisation
Marginalize X[i,j] out of the joint, then re-instantiate with the new value. **Rejected**: marginalisation in matrix-variate form is itself Kronecker-breaking (research note 04). No benefit over direct Schur-complement.

### Alt-C: Merge via moment-matching collapse to single matrix-variate
Collapse the J + K components down to 1 by matching the first two moments of the joint mixture (Salmond 1990). **Rejected as default**: loses multimodality, which is exactly what `if/else` introduces. Keep as an OPTIONAL post-merge reduction policy if K_max is exceeded.

### Alt-D: Defer Phase 4 (element write) to v2
Keep `X[i,j] = expr` as `NotImplementedError`. **Rejected by user**: "deve essere completo in tutto... devo poter usare le matrici in tutto gli elementi sintattici". Phase 4 is in scope.

## 6. Sub-tasks (atomic, milestone-organized)

### M.fix1 — Loop-variable resolution in matrix indexing (3-5 days)

- [ ] **[fix1.1]** [iter:1] [area:libSOGAupdate.py] Extend hybrid routing regex in `update_rule` to accept IDV indices: `^([A-Za-z]\w*)\s*\[\s*([A-Za-z]\w*|\d+)\s*,\s*([A-Za-z]\w*|\d+)\s*\]$`. Resolve identifier indices via `data[idv][0]` (SOGA loop-counter convention) at dispatch time. Numeric strings parsed via `int(_m.group(N))`.
- [ ] **[fix1.2]** [iter:1] [area:libMatrixTruncate.py] Extend `_RE_ELEMENT`, `_RE_ROW_SUM`, `_RE_COL_SUM` regexes to accept IDV in index positions. Add `_resolve_index(tok, data)` helper that returns `int(tok)` if tok is numeric else `int(data[tok][0])`. Update `_classify_constraint` to use it.
- [ ] **[fix1.3]** [iter:1] [agent:test-engineer] [area:tests/test_update_matrix.py + new tests/test_loop_matrix.py] Add end-to-end SOGA test: `for i in range(4) { d = X[i,i]; observe(d > 0); }` for X with known diagonal mean. Verify final E[d] and E[X] post-observe across all 4 iterations.
- [ ] **[fix1.4]** [iter:1] [area:smoke battery] Add to 8-feature smoke battery as t9_loop_index.soga; verify PASS.

### M.fix2 — Merge with matrix variables (5-7 days)

- [ ] **[fix2.1]** [iter:2] [agent:soga-internal-expert] [area:libSOGAmerge.py] Lift `NotImplementedError [M3.6]` from `merge()`. Implement concatenation per research note 05 §Problem 1:
  - Per branch with weight `p_i` and dist with `K_i` components, append all `K_i` components to merged with weights `p_i · pi_k`.
  - Concatenate `gm_block.mu_blocks` and `gm_block.cov_blocks` per component.
  - Concatenate `gm.mu` and `gm.sigma` per component.
  - Renormalize merged_pi.
- [ ] **[fix2.2]** [iter:2] [agent:gaussian-mixture-expert] [area:libSOGAmerge.py] Handle "variable present in only one branch" case: detect mismatch in `var_entries` or `var_list` between branches. If a variable exists in branch i but not branch j, snapshot its prior (carried from the joint state at `if` entry) and inject into branch j's components. Document the priorscoping rule explicitly.
- [ ] **[fix2.3]** [iter:2] [agent:soga-internal-expert] [area:libSOGAmerge.py] Extend `classic_prune` / `ranking_prune` to handle matrix block gm_block alongside scalar gm. When pruning component k, drop both `gm.mu[k]/sigma[k]` AND `gm_block.mu_blocks[k]/cov_blocks[k]`.
- [ ] **[fix2.4]** [iter:2] [agent:numerical-stability-expert] [area:libSOGAmerge.py + libSOGAsharedMatrix.py] Memory-budget guard: after merge, sum K_components × per-component-cost. If exceeds `--matrix-merge-budget-mb` (default 1024), emit `MatrixMergeMemoryWarning` and trigger pruning to reduce K. Default budget conservative since merge can multiply component count.
- [ ] **[fix2.5]** [iter:2] [agent:gaussian-mixture-expert] [area:libSOGAmerge.py] OPTIONAL: implement post-merge Runnalls reduction for matrix components — extend `B(i,j)` criterion with `Sigma_m = V_m ⊗ U_m` determinant. Document that post-merge components may have non-Kronecker dense Sigma_m (sum of two Kron); choice of option A (keep dense) vs option B (Van Loan-Pitsianis re-project) is configurable via CLI flag `--merge-projection={dense|kron}`.
- [ ] **[fix2.6]** [iter:2] [agent:test-engineer] [area:tests/test_merge_matrix.py (new)] Unit + e2e tests:
  - Unit: 2-branch merge with simple matrix var, verify per-component (M, U, V) preserved and weights correct.
  - Unit: variable-scope test — matrix declared inside `if`, only present in then-branch, verify post-merge handling.
  - E2E SOGA: `matrix[2][2] X; X = matrix_gm(...); theta = gm(...); if theta > 0 { y = X[0,0]; } else { y = X[1,1]; } end if;` — verify E[y] is correct mixture of (X[0,0] given theta>0) and (X[1,1] given theta<0).
  - Sanity-vs-analytical: closed-form for the e2e case with X ~ MN(0, I, I) and theta deterministic.
- [ ] **[fix2.7]** [iter:2] [area:smoke battery] Add t10_if_else_matrix.soga; verify PASS.

### M.fix3 — Random × random matmul `Y = X1 @ X2` (7-10 days)

- [ ] **[fix3.1]** [iter:3] [agent:numerical-stability-expert] [area:libMatrixGaussian.py] Implement `nearest_kronecker(Sigma, m, n)` per research note 05 §Q2b. Rank-1 SVD on the rearrangement R[Sigma] of size (m² × n²). Return `(U, V)` after PSD enforcement. Add unit test verifying recovery of exact Kronecker products and known failure modes (rank-2 input).
- [ ] **[fix3.2]** [iter:3] [agent:gaussian-mixture-expert] [area:libMatrixUpdate.py] Implement `matmul_random_random_component(M_X, U_X, V_X, M_Y, U_Y, V_Y)` per research note 05 recipe:
  - Mean: `M_Z = M_X @ M_Y`.
  - Cov delta: `Sigma = kron(M_Y.T @ V_X @ M_Y, U_X) + kron(V_Y, M_X @ U_Y @ M_X.T)`.
  - Project: `(U_Z, V_Z) = nearest_kronecker(Sigma, m, n)`.
  - PSD enforce.
  - Approximation warning: compute `s_2 / s_1` of the SVD; if > 0.05, emit `MatmulApproxWarning`.
- [ ] **[fix3.3]** [iter:3] [agent:soga-internal-expert] [area:libMatrixUpdate.py] Wire into `update_rule_matrix` dispatcher: lift the current `NotImplementedError [M4.1]` from the `MATMUL_RAW` branch when both operands are matrix vars. Iterate over all pairs of components (i from X, j from Y), invoke `matmul_random_random_component`, append to result mixture with weight `pi_X_i · pi_Y_j`.
- [ ] **[fix3.4]** [iter:3] [agent:numerical-stability-expert] [area:libMatrixUpdate.py] Component blow-up management: if `J · K > K_max`, invoke `ranking_prune(K_max)` immediately after. Add CLI flag `--matmul-prune-K` (default 50).
- [ ] **[fix3.5]** [iter:3] [agent:test-engineer] [area:tests/test_update_matrix.py + new test_random_matmul.py]
  - Unit: 2×2 case with known M_X, U_X, V_X, M_Y, U_Y, V_Y — verify E[Z], Cov(vec(Z)) against analytical formula (research note 05 §Q2a).
  - MC ground truth: 20k samples of Z = X @ Y, compare empirical mean and per-element variance with our approximation. Tolerance < 5 % relative.
  - Failure-mode test: case where both Kronecker terms equal — verify the warning fires.
- [ ] **[fix3.6]** [iter:3] [area:smoke battery] Add t11_random_matmul.soga; verify PASS.

### M.fix4 — Matrix element write `X[i,j] = expr` (7-10 days)

- [ ] **[fix4.1]** [iter:4] [agent:soga-internal-expert] [area:libSOGAsharedMatrix.py] Add storage tag for matrix self-cov:
  - Current: `cov_blocks[k][frozenset({X})] = (U, V)` tuple (Kronecker).
  - New: `cov_blocks[k][frozenset({X})] = ('kron', U, V)` OR `('dense', Sigma_mn_x_mn)`.
  - Update `GaussianMixBlock.get_cov` to dispatch on tag.
  - Backward-compat: existing code paths that read `(U, V)` get a wrapper that materialises from `('dense', Sigma)` if needed (via `nearest_kronecker`, with warning). Other matrix ops on a `'dense'` variable raise `NotImplementedError` for v1 (acceptable: element writes are typically end-of-program).
- [ ] **[fix4.2]** [iter:4] [agent:numerical-stability-expert] [area:libMatrixUpdate.py] Implement `_densify_matrix_var(block, k, mat_name) -> np.ndarray` that materialises `V⊗U` on demand. Gated by memory budget (existing M5.2 guard suffices).
- [ ] **[fix4.3]** [iter:4] [agent:gaussian-mixture-expert] [area:libMatrixUpdate.py] Implement `_matrix_element_write(block, k, mat_name, i, j, c_or_mu, var_z=0)` per research note 04 Findings B1+B2:
  - `idx = j*m + i`
  - `gain = (V[:,j] ⊗ U[:,i]) / (V[j,j] · U[i,i] + var_z)` if Kronecker; else `Sigma[:, idx] / (Sigma[idx, idx] + var_z)`
  - `vec_M_new = vec_M + gain · (c_or_mu - vec_M[idx])`
  - `Sigma_new = Sigma - outer(gain, sel) ; sel = (V[:,j] ⊗ U[:,i]) or Sigma[:, idx]`
  - Triggers densification if not already dense.
- [ ] **[fix4.4]** [iter:4] [agent:soga-internal-expert] [area:libSOGAupdate.py + libMatrixUpdate.py] LHS routing: detect `^([A-Za-z]\w*)\s*\[(\d+|\w+)\s*,\s*(\d+|\w+)\]` LHS pattern (matrix-element write). Three RHS cases per research note 04:
  - B1: RHS is NUM (deterministic constant) → call `_matrix_element_write` with `c=NUM, var_z=0`.
  - B2: RHS is IDV referencing an existing scalar var → look up its (mu, var) per component, call with `c=mu, var_z=var`.
  - B3: RHS is a general scalar expression → first evaluate via scalar `update_rule` into a temp var, then reduce to B2.
- [ ] **[fix4.5]** [iter:4] [agent:numerical-stability-expert] [area:libMatrixUpdate.py] After element write, the matrix var's cov is dense. Document that subsequent matrix ops (transp, affine, +N) are NOT implemented for dense-mode vars in v1. Add clear error message naming `_densify_matrix_var`.
- [ ] **[fix4.6]** [iter:4] [agent:test-engineer] [area:tests/test_element_write.py (new)]
  - Unit B1: `X = matrix_gm(...)` then write `X[0,0] = 5.0`. Verify `E[X[0,0]] = 5.0` and other elements have Schur-complement-corrected means and variances.
  - Unit B2: `z = gm(...)` then write `X[0,0] = z`. Verify E[X[0,0]] = E[z], and Var[X[0,0]] is correct rank-1 downdate denominator `+ var_z`.
  - Unit B3: scalar expr (e.g., `2 * z + 1`) on RHS.
  - Sanity vs analytical: 2x2 case with hand-computed Schur complement.
  - MC ground truth: 20k samples vs analytical, tolerance < 1 % on first two moments.
- [ ] **[fix4.7]** [iter:4] [area:smoke battery] Add t12_element_write.soga; verify PASS.

### M.fix5 — End-to-end validation + documentation (3-5 days)

- [ ] **[fix5.1]** [iter:5] [agent:soga-benchmark-runner] [area:bench] Run full `/audit-numerical` Phase 5 on all 12 features (t1..t12). Compare with M0 baseline.
- [ ] **[fix5.2]** [iter:5] [agent:soga-benchmark-runner] [area:bench] Run `/soga-bench Table3_sogaprograms.txt`. Acceptance: no benchmark slows by > 1 %, all 18 produce identical E[·] within `1e-10`.
- [ ] **[fix5.3]** [iter:5] [agent:documentation-writer] [area:Manual/ReusabilityGuide.md] Add complete "Matrix DSL" section listing all working patterns + semantics + worked examples.
- [ ] **[fix5.4]** [iter:5] [agent:documentation-writer] [area:docs/MATRIX_GM_SEMANTICS.md (new)] Per-construct semantic write-up. For each of the 12 working patterns, write: (a) DSL syntax, (b) example, (c) formal semantics (the distribution-transfer rule), (d) the closed-form formula SOGA implements, (e) approximation flags if any, (f) reference to the source code function. **User-facing reference document; mandatory.**
- [ ] **[fix5.5]** [iter:5] [agent:codex-cross-reviewer] [area:audit] Final cumulative cross-review of `git diff origin/main..HEAD` (max 5 iterations).
- [ ] **[fix5.6]** [iter:5] [agent:bs-detector] [area:audit] Run on full branch diff. Address any 🔴 findings before final push.

## 7. Test plan

### Unit tests (per-fix)
- `tests/test_loop_matrix.py` — loop-variable resolution (fix1). ~10 tests.
- Extension to `tests/test_merge_matrix.py` (new) — branch merge (fix2). ~15 tests.
- Extension to `tests/test_update_matrix.py` — random×random (fix3). ~10 tests.
- `tests/test_element_write.py` (new) — element write B1/B2/B3 (fix4). ~15 tests.

Per-fix unit-test target: ≥ 90 % branch coverage on the new code path.

### Sanity-vs-analytical
For each fix, at least one test with a hand-computed analytical ground truth:
- fix1: `for i in range(4) { d = X[i,i]; }` with X = diag(1,2,3,4) deterministic → E[d_iter_i] = i+1 exactly.
- fix2: 2-branch if/else with X ~ MN(0, I, I) and known scalar trigger probability → E[X | post-merge] = mixture of conditioned means.
- fix3: 2×2 matmul X1 @ X2 with X1 ~ MN(I, I, I), X2 ~ MN(I, I, I) → E[Z] = I @ I = I, Cov via delta formula.
- fix4: B1 `X[0,0] = 5` with X ~ MN(M, I, I) → post-condition mean of other elements via Schur complement.

### Sanity-vs-MC
For each non-trivial approximation (fix3 NKP, fix4 Schur), 20k MC samples → empirical moments within tolerance:
- fix3: 5 % relative on E[Z], 10 % on Cov(vec(Z)) per-element.
- fix4: 1 % relative on first two moments of resulting joint.

### Benchmark regression
- `/soga-bench Table3_sogaprograms.txt` — 18 scalar benchmarks; no slowdown > 1 %; no E[·] drift > 1e-10.
- 8-feature smoke battery (t1..t8) — all PASS post-fix1..4.
- 12-feature extended smoke battery (t1..t12, adding t9..t12 for the new patterns) — all PASS at end of fix5.

### Audit gates
- `/audit-numerical` after each numerical kernel change (fix2.4, fix3.2, fix4.3): 🟢 GO required.
- `/audit-grammar` not triggered (no .g4 changes planned). Recheck if any sub-task ends up touching grammar.

## 8. Acceptance criteria

- [ ] AC1: t1..t12 smoke battery all PASS (8 existing + 4 new).
- [ ] AC2: pytest tests/ -q: 293 + ~50 new tests = ~340+ tests all green.
- [ ] AC3: Bernoulli scalar regression E[theta] = 0.25689 unchanged.
- [ ] AC4: ClickGraphPrune scalar regression all 6 E[·] unchanged.
- [ ] AC5: Sanity-vs-analytical for each fix passes hand-computed ground truth.
- [ ] AC6: MC validation passes 5 % / 10 % / 1 % tolerances per fix.
- [ ] AC7: `/audit-numerical` 🟢 GO after each fix.
- [ ] AC8: `/audit-grammar` regression unchanged (or 🟢 if any grammar tweak surfaces).
- [ ] AC9: `docs/MATRIX_GM_SEMANTICS.md` complete with all 12 patterns documented.
- [ ] AC10: `Manual/ReusabilityGuide.md` updated.
- [ ] AC11: codex-cross-reviewer APPROVE on cumulative branch diff.
- [ ] AC12: bs-detector no 🔴 findings.
- [ ] AC13: working tree clean, no uncommitted changes at gate.
- [ ] AC14: SOGA CLI prints E[X] for matrix vars (already done in M2.fix — verify).

## 9. Rollback

Each phase is its own commit (or commit group). To roll back:
- `git revert <SHA>` of the problematic commit
- Worst case: revert to `de3dd048` (last verified-good commit, current head as of plan creation)
- Branch `feat/matrix-gm-integration` already pushed; never force-push.

## 10. Estimated complexity

| Phase | Effort | Risk | Theoretical novelty |
|-------|--------|------|---------------------|
| fix1 (loop idx) | S — 3-5 days | Low | None (engineering) |
| fix2 (merge) | M — 5-7 days | Medium (variable-scope edge cases) | Low (research note 05 §1) |
| fix3 (random@random) | L — 7-10 days | High (first PPL to do this; approximation must be validated) | High (research note 05 §2) |
| fix4 (element write) | L — 7-10 days | High (densification, dense-mode invariant) | Medium (research note 04 §B) |
| fix5 (validation+docs) | M — 3-5 days | Low | None |
| **Total** | **~30-40 working days, 6-8 calendar weeks** | | |

Optimistic: 4 weeks with parallel agent execution + few rework cycles.

## 11. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | fix3 delta-method approximation fails MC validation on some marginal-cov regimes | Medium | High | Document failure modes; offer dense fallback flag `--matmul-mode=dense` |
| R2 | fix4 dense-mode invariant breaks downstream matrix ops (transp, affine on dense X) | High | Medium | v1: raise NotImplementedError for these; v2 extend ops to dense path |
| R3 | fix2 component blow-up (J·K growth across nested if/else) | High | High | Mandatory `prune(K_max)` after merge; budget warnings |
| R4 | fix1 loop counter resolution misses edge case (loop var named like data) | Low | Low | Test with adversarial names; document precedence (data wins over loop counter, since they share the data dict) |
| R5 | Documentation drift between code and `MATRIX_GM_SEMANTICS.md` | Medium | Medium | Each new op references the source-code function by line; CI lint check (optional) |
| R6 | bs-detector false positives on approximation warnings | Low | Low | Tune; suppress with explanatory comment |
| R7 | codex-cross-reviewer rejects the delta-method approximation as too lossy | Medium | Medium | Cite research note 05; argue that all PPL alternatives also approximate (sampling) |

## 12. Out-of-scope (explicit non-goals)

- v2 matrix-matrix cross-covariance (different matrix variables correlated). Currently `NotImplementedError`. Defer until benchmark demands it.
- v2 matrix-matrix conditional (observe involving two matrix vars).
- v2 Kronecker recovery via flip-flop ALS for dense-mode variables. Defer until profiling shows the dense path is a bottleneck.
- v2 covariance back-prop on scalar observe (research note 04 Opt-2). Currently mean-only. Defer.
- Matrix-element LHS chained ops (e.g., `X[i,j] = X[k,l] + 1`). Out of scope.

## 13. Verification commands

```bash
# After each fix:
.venv/bin/python -m pytest tests/ -q                    # full suite
.venv/bin/python src/SOGA.py -f programs/Example/Bernoulli.soga       # scalar regression
.venv/bin/python src/SOGA.py -f programs/SOGA/ClickGraphPrune.soga    # scalar regression
for t in /tmp/t1..t12_*.soga; do
  .venv/bin/python src/SOGA.py -f $t > /tmp/out.log && echo PASS $t || echo FAIL $t
done

# Numerical audit:
/audit-numerical

# Grammar audit (if any .g4 touched):
/audit-grammar

# Full benchmark regression:
/soga-bench Table3_sogaprograms.txt --compare-vs origin/main
```

## 14. Cross-review history

_To be populated by Phase F (codex-cross-reviewer)._

---

_End of plan. Awaiting user approval to start `/iterate plan/2026-05-24-matrix-gm-complete.md`._
