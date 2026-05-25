# Plan: matrix-gm-full — general-covariance constructor with auto-Kronecker detection

**Date**: 2026-05-25
**Slug**: `matrix-gm-full`
**Branch**: `feat/matrix-gm-integration` (head `0d0aa859`)
**Effort estimate**: ~5-7 working hours, 1-2 /iterate iterations
**Status**: draft, awaiting user approval
**Triggered by**: user request 2026-05-25 — "modo semplice per specificare GM matrix in forma generale + auto-detect Kronecker per non avere ambiguità per lo studente"
**Relation to prior plans**:
- builds on `plan/2026-05-22-matrix-gm-lishan.md` (M1-M6 done)
- builds on `plan/2026-05-23-m1-grammar-cleanup.md` (grammar pipeline)
- builds on `plan/2026-05-24-matrix-gm-complete.md` (fix1-fix5 + Isserlis + O5/O7)

---

## 1. Goal

Add a second matrix-variate constructor to SOGA DSL — `matrix_gm_full(M, Sigma)` —
that accepts a **full (mn × mn) covariance matrix** Sigma alongside the existing
3-arg Kronecker form. At runtime, SOGA **auto-detects** whether Sigma is
Kronecker-separable via Van Loan-Pitsianis rank-1 SVD; if yes, it stores as
efficient (U, V) Kronecker factors; if no, as a dense sentinel.

The student-facing benefit: a single, unambiguous way to specify any matrix-
Gaussian prior without needing to manually pre-decompose Kronecker structure
or reason about separability — SOGA picks the optimal storage automatically.

## 2. Context

Current state (after `0d0aa859`):
- `matrix_gm(M, U, V)` requires explicit Kronecker factors. User must
  pre-decompose: if they have a desired full Sigma, they have to compute U, V
  by hand. For non-Kronecker Sigma they cannot express the prior at all.
- Internally, `GaussianMixBlock.cov_blocks[k][frozenset({X})]` already supports
  both `(U, V)` (Kronecker) and `(None, Sigma)` (dense sentinel) — the latter
  is created by `fix4` element-write. The infrastructure is ready; what's
  missing is the user-facing constructor.

Theoretical foundation:
- Van Loan-Pitsianis 1993 (research note 05 §Q2b): rank-1 SVD on the
  rearrangement R[Σ] returns the nearest Kronecker product. If R[Σ] has
  rank exactly 1 (i.e. `s_2 / s_1 ≈ 0`), the decomposition is exact.
- This is the same algorithm used in fix3 (random×random matmul NKP),
  with a residual-threshold check added.

## 3. Constraints

1. **No regression** on existing scalar SOGA (Bernoulli `E[theta]=0.25689`,
   ClickGraphPrune `E[simAll]=0.61409`) and existing matrix-GM showcase
   (`matrix_gm_showcase.soga` + `matrix_gm_advanced.soga`).
2. **No grammar ambiguity**: `matrix_gm` and `matrix_gm_full` must lex
   unambiguously (longest-match wins; MATRIX_GM_FULL declared before
   MATRIX_GM in the lexer).
3. **Backward compat**: existing `matrix_gm(M, U, V)` 3-arg form keeps working.
4. **Sanity-vs-analytical** on every code path: at least one test where the
   user-specified full Sigma equals V⊗U exactly (Kronecker-detected), one
   where it equals V⊗U + ε (near-Kronecker), and one truly non-separable.
5. **Auto-detect transparency**: emit an info-level log (or warning) showing
   which mode was chosen and the residual ratio.
6. **Branch hygiene**: all commits on `feat/matrix-gm-integration`, no push
   without user approval.

## 4. Approach

Extend SOGA.g4 and ASGMT.g4 with a `matrix_gm_full(M, Sigma)` rule and a
new `MATRIX_GM_FULL` lexer keyword.  Regenerate parsers; ensure ASGMT
grammar accepts the new constructor in `mat_atom` alongside `matrix_gm`.

In `libMatrixUpdate._parse_matrix_expr`, add detection of
`body.startswith("matrix_gm_full(")` returning a new op `MATRIX_GM_FULL`.
Add a parsing helper `_parse_matrix_gm_full_text` that extracts M (m×n)
and Sigma (mn×mn) as numpy arrays from the nested-list text.

Add a numerical helper `_try_kronecker_decompose(Sigma, m, n, tol=1e-8)`
in `libMatrixGaussian.py` (or reuse existing `_nearest_kronecker`):
returns `(U, V, residual_ratio)`. The residual is the SVD `s_2 / s_1`
ratio. If `residual_ratio < tol`, the decomposition is exact (within
machine precision); otherwise it's lossy.

In `update_rule_matrix`, when op == "MATRIX_GM_FULL":
- Parse M, Sigma.
- Validate shape consistency with declared `matrix[m][n] X`.
- Run `_try_kronecker_decompose(Sigma, m, n)`.
- If residual < `KRON_DETECT_STRICT` (default 1e-8):
  - Store as Kronecker: `block.cov_blocks[k][frozenset({X})] = (U, V)`
  - Log `KroneckerDetectionInfo(residual)` at INFO level
- Else:
  - Store as dense: `block.cov_blocks[k][frozenset({X})] = (None, Sigma)`
  - Log `DenseCovarianceInfo(residual)`
- If `KRON_DETECT_STRICT ≤ residual < KRON_DETECT_LOOSE` (1e-3 by default):
  - Emit `KroneckerNearMissWarning("matrix_gm_full Σ near-Kronecker with residual=...; stored as DENSE for safety")`

Subsequent operations on this variable follow the existing dispatch:
- If Kronecker-stored → fast path
- If dense-stored → dense path (extract_scalar_from_matrix already
  handles this thanks to the 0d0aa859 fix; matmul/transp/affine raise
  NotImplementedError on dense — same v1 limitation as post-write).

Update `matrix_gm_showcase.soga` with a new SECTION 9 demonstrating
the Kronecker-detected case. Update `matrix_gm_advanced.soga` with a
new PART C demonstrating the truly-dense case. Update QUICKSTART.md
with a "Specifying covariance: Kronecker vs Full" section.

## 5. Alternatives considered

### Alt-A — Polymorphic `matrix_gm(M, ...)`: 2-arg or 3-arg
Single keyword, 2-arg = full / 3-arg = Kronecker, disambiguated by argument
count.
**Rejected**: ambiguity for the reader. From the program text it's not
visually clear whether the 2nd argument is a full covariance or a row factor.
The user explicitly asked for "no ambiguity for the student".

### Alt-C — Three convenience constructors (matrix_iid, matrix_gm, matrix_gm_full)
Adds `matrix_iid(M, sigma)` for the common iid-noise case.
**Rejected (for this plan)**: doubles the grammar work and adds a third
keyword for a case that can already be expressed as
`matrix_gm(M, sigma*I, sigma*I)`. Can be added as a follow-up if student
feedback shows the iid case is the dominant pattern.

### Alt-D — No DSL change, document workaround
Keep DSL as-is; teach the student to pre-decompose by hand.
**Rejected**: defeats the user's explicit request for zero-ambiguity UX.

## 6. Sub-tasks (atomic, in iteration order)

### Iteration 1 — Core implementation (4-5 hours)

- [ ] **[B.1]** [iter:1] [agent:antlr-grammar-expert] [area:grammars/SOGA.g4, grammars/ASGMT.g4]
  Add `MATRIX_GM_FULL : 'matrix_gm_full';` lexer keyword (BEFORE `MATRIX_GM`).
  Add `matrix_gm_full: MATRIX_GM_FULL '(' mlist ',' mlist ')';` parser rule.
  Add `matrix_gm_full` to `vars` (SOGA.g4) and `mat_atom` (ASGMT.g4).
  Update header comment with regen instructions.
- [ ] **[B.2]** [iter:1] [agent:antlr-grammar-expert] [area:src/*]
  Regenerate SOGA + ASGMT parsers via `java -jar tools/antlr-4.10-complete.jar ... -o ../src/`.
  Verify `bash scripts/check_grammar_sync.sh` exits 0.
  Re-run `tests/test_grammar_matrix.py` (35 tests): all must stay green.
- [ ] **[B.3]** [iter:1] [agent:soga-internal-expert] [area:src/libMatrixUpdate.py]
  Add `_parse_matrix_gm_full_text(text)` helper.
  Returns `(M: np.ndarray, Sigma: np.ndarray)`.
  Validates: M shape (m, n); Sigma shape (mn, mn); Sigma symmetric.
- [ ] **[B.4]** [iter:1] [agent:numerical-stability-expert] [area:src/libMatrixGaussian.py]
  Add `_try_kronecker_decompose(Sigma, m, n, tol=1e-8)` helper.
  Reuse the existing `_nearest_kronecker` SVD machinery; expose the
  `s_2 / s_1` ratio as residual.
  Returns `(U_approx, V_approx, residual_ratio)`.
  Unit tests: exact Kronecker (residual ≈ 0), near-Kronecker (residual ≈ ε),
  rank-2 in rearrangement (residual ~ 0.5).
- [ ] **[B.5]** [iter:1] [agent:soga-internal-expert] [area:src/libMatrixUpdate.py]
  Extend `_parse_matrix_expr` to detect `matrix_gm_full(` body prefix
  and return `op="MATRIX_GM_FULL"`.
  Add MATRIX_GM_FULL handler in `update_rule_matrix`:
  1. Parse M, Sigma.
  2. Validate shapes (raise ValueError on mismatch).
  3. `_try_kronecker_decompose(Sigma, m, n)`.
  4. Store as Kronecker if residual < 1e-8, else dense (`(None, Sigma)`).
  5. Initialise cross-covs with scalars as zero.
  6. Append `ve` to `block.var_entries` if not present.
- [ ] **[B.6]** [iter:1] [agent:soga-internal-expert] [area:src/libSOGAsharedMatrix.py]
  Add `GaussianMixBlock.from_matrix_gm_full(M, Sigma, ve, ...)` constructor
  for the case `dist.gm_block is None` (first matrix variable in the program).
- [ ] **[B.7]** [iter:1] [agent:numerical-stability-expert] [area:src/libMatrixGaussian.py]
  Add `KroneckerDetectionInfo`, `KroneckerNearMissWarning`, `DenseCovarianceInfo`
  warning classes; emit appropriately in the MATRIX_GM_FULL dispatcher.
  Configurable thresholds via env vars `SOGA_KRON_STRICT`, `SOGA_KRON_LOOSE`.

### Iteration 2 — Tests + docs + showcase (2-3 hours)

- [ ] **[B.8]** [iter:2] [agent:test-engineer] [area:tests/test_matrix_gm_full.py (NEW)]
  Unit tests, ~15:
  - Exact Kronecker input (Σ = V⊗U for known V, U): assert stored as (U, V)
  - Near-Kronecker input (Σ = V⊗U + 1e-12·I): assert stored as Kronecker, warning emitted if residual > 1e-8
  - Non-separable Σ (rank-2 rearrangement): assert stored as dense sentinel
  - Marginal moments: Var[X[i,j]] = Σ[j·m+i, j·m+i] in dense mode
  - Cross-cov: Σ[k, j·m+i] equals stored cross-cov column in dense mode
  - Operations on Kronecker-detected: matmul / transp / + (Kronecker fast path)
  - Operations on dense: extract / observe element (work); transp / affine (NotImplementedError)
- [ ] **[B.9]** [iter:2] [agent:test-engineer] [area:tests/test_matrix_gm_full_e2e.py (NEW)]
  End-to-end SOGA tests via subprocess on .soga programs, ~8:
  - `matrix_gm_full([[0,0],[0,0]], I_4)`: detects Kronecker (I_4 = I_2⊗I_2), runs.
  - `matrix_gm_full(M, full_dense_Sigma)`: stores dense, runs.
  - Mixed: matrix_gm() and matrix_gm_full() in same program (different variables).
  - Observe element on a dense-mode variable.
  - Scalar extract on a dense-mode variable.
- [ ] **[B.10]** [iter:2] [agent:test-engineer] [area:tests/test_grammar_matrix.py]
  Extend grammar tests with ~6 cases for `matrix_gm_full`:
  - Positive: 2-arg parses correctly.
  - Negative: 3-arg `matrix_gm_full(M, U, V)` rejected (wrong arity).
  - Negative: 1-arg rejected.
  - Lex precedence: `matrix_gm_full(` does NOT trigger MATRIX_GM.
- [ ] **[B.11]** [iter:2] [area:programs/Example/matrix_gm_showcase.soga]
  Add SECTION 9 with `matrix_gm_full` Kronecker-detected example
  (e.g. iso 2×2 specified as 4×4 identity Σ).
- [ ] **[B.12]** [iter:2] [area:programs/Example/matrix_gm_advanced.soga]
  Add PART C with a truly-non-Kronecker Σ on a 2×2 matrix variable.
  Document the warning emitted and the dense storage cost.
- [ ] **[B.13]** [iter:2] [agent:documentation-writer] [area:docs/MATRIX_GM_QUICKSTART.md]
  Add "Specifying covariance: Kronecker vs Full" section, ~half page.
  Show 3 example forms (iid, Kronecker explicit, full with auto-detect).
  Document the threshold flags.
- [ ] **[B.14]** [iter:2] [agent:documentation-writer] [area:docs/MATRIX_GM_SEMANTICS.md]
  Add §16 "matrix_gm_full constructor" with formal grammar, dispatcher,
  auto-detection algorithm and threshold semantics.
- [ ] **[B.15]** [iter:2] [agent:soga-benchmark-runner] [area:regression]
  - Bernoulli E[theta]=0.25689 unchanged.
  - ClickGraphPrune E[simAll]=0.61409 unchanged.
  - matrix_gm_showcase E[y00]=2.57684 unchanged.
  - matrix_gm_advanced all values unchanged.
  - Full pytest must pass (≥ 349 + ~21 new = 370+).
- [ ] **[B.16]** [iter:2] [area:commits]
  3-5 incremental commits (grammar / regen / dispatcher / tests / docs).
  All commits on `feat/matrix-gm-integration`. NO push until user approves.

## 7. Test plan

### Unit tests (new)
- `tests/test_matrix_gm_full.py` (~15 tests)
  Tests `_try_kronecker_decompose` algorithm, `_parse_matrix_gm_full_text`
  parsing, and dispatch through `update_rule_matrix` with both Kronecker-
  detected and dense paths.

### End-to-end SOGA tests (new)
- `tests/test_matrix_gm_full_e2e.py` (~8 tests)
  Run programs via `subprocess.run([sys.executable, "SOGA.py", "-f", ...])`,
  parse stdout, assert moments match analytical ground truth.

### Sanity-vs-analytical
- `matrix_gm_full([[0,0],[0,0]], I_4)` → variance per element = 1, mean 0
  (matches `matrix_gm([[0,0],[0,0]], I_2, I_2)` exactly — Kronecker-detected)
- `matrix_gm_full(M, full_Sigma_with_correlation_0.5_between_X00_and_X11)`
  → Cov(X[0,0], X[1,1]) = 0.5, Var[X[i,j]] = 1 (dense mode)

### Grammar regression
- `tests/test_grammar_matrix.py` (35 existing tests must stay green)
- 6 new tests for matrix_gm_full parsing
- Scalar canonical programs (Bernoulli, ClickGraphPrune) must parse identically

### Smoke battery
- `programs/Example/matrix_gm_showcase.soga` (already passing): re-verify
- `programs/Example/matrix_gm_advanced.soga` (already passing): re-verify
- New SECTION 9 in showcase + PART C in advanced: pass empirically

### Benchmark regression
- Run `python3 src/SOGA.py -f programs/SOGA/Bernoulli.soga` — E[theta] unchanged
- Run `python3 src/SOGA.py -f programs/SOGA/ClickGraphPrune.soga` — all 6 moments unchanged

## 8. Acceptance criteria

- [ ] **AC1** — Grammar regen passes `bash scripts/check_grammar_sync.sh`.
- [ ] **AC2** — `matrix_gm_full(M, V⊗U)` for exact Kronecker stored as `(U, V)`;
       verified via `dist.gm_block.cov_blocks[k][frozenset({X})] == (U, V)`.
- [ ] **AC3** — `matrix_gm_full(M, Σ)` for non-separable Σ stored as dense sentinel `(None, Σ)`.
- [ ] **AC4** — Subsequent ops on Kronecker-detected variable: matmul, transp,
       add work end-to-end. On dense-mode variable: extract + observe element work.
- [ ] **AC5** — Auto-detect emits `KroneckerNearMissWarning` only when
       `1e-8 ≤ residual < 1e-3`; `DenseCovarianceInfo` for residual ≥ 1e-3.
- [ ] **AC6** — `tests/test_matrix_gm_full.py` ≥ 15 tests passing.
- [ ] **AC7** — `tests/test_matrix_gm_full_e2e.py` ≥ 8 tests passing.
- [ ] **AC8** — Updated `matrix_gm_showcase.soga` produces same E[·] values
       as before (regression) + the new SECTION 9 example.
- [ ] **AC9** — Updated `matrix_gm_advanced.soga` similarly + new PART C.
- [ ] **AC10** — `docs/MATRIX_GM_QUICKSTART.md` has new section + cheatsheet updated.
- [ ] **AC11** — `docs/MATRIX_GM_SEMANTICS.md` has §16 with formal grammar + algorithm.
- [ ] **AC12** — Full pytest: ≥ 349 + new tests = ~370+ all passing.
- [ ] **AC13** — Bernoulli E[theta] = 0.25689 unchanged; ClickGraphPrune unchanged.
- [ ] **AC14** — Working tree clean; all changes committed on `feat/matrix-gm-integration`.

## 9. Rollback

- If a sub-task fails irrecoverably: `git revert <SHA>` of the offending commit.
- If grammar change introduces parse regression: revert SOGA.g4 + ASGMT.g4 to
  pre-plan state, regen parsers, re-run `tests/test_grammar_matrix.py`.
- Worst case: revert to commit `0d0aa859` (head before this plan).

## 10. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| RB1 | ANTLR4 lex ambiguity: `matrix_gm_full(` matches as `matrix_gm` + `_full(` | Medium | High | Declare MATRIX_GM_FULL **before** MATRIX_GM in lexer; verify with parse test |
| RB2 | Auto-detect threshold (1e-8) wrong for some programs | Low | Low | Configurable via env var; document; user can pin to dense via flag |
| RB3 | Nested-list parser stack overflow on large Σ | Low | Medium | For mn=64, Σ has 4096 entries; numbers around 10KB text — manageable; add max-size sanity check |
| RB4 | Dense-mode propagation through subsequent matmul: existing limitation could surface in new tests | Medium | Low | Document explicitly; raise NotImplementedError as already done |
| RB5 | Showcase regression: new SECTION 9 changes component count | Low | Medium | Run `matrix_gm_showcase.soga` before/after, verify E[·] values byte-identical |
| RB6 | Naming conflict: `matrix_gm_full` collides with an existing identifier in any benchmark | Very Low | Medium | Grep `programs/SOGA/**/*.soga` for the new keyword; verified 0 hits earlier for similar new keywords |

## 11. Estimated complexity

- Grammar change (SOGA.g4, ASGMT.g4): 30 min
- ANTLR regen + verify: 15 min
- Dispatcher (`_parse_matrix_expr` + MATRIX_GM_FULL handler): 1 hour
- Auto-detect helper (`_try_kronecker_decompose`): 30 min
- GaussianMixBlock dense init: 30 min (mostly reusing existing dense path from fix4)
- Unit tests: 1.5 hours
- End-to-end tests: 1 hour
- Showcase + advanced updates: 45 min
- Quickstart + semantics docs: 45 min
- Regression spot-checks + commit + push gate: 30 min

**Total**: 5-7 working hours. Single-iter executable.

## 12. Out-of-scope (explicit non-goals)

- Multi-component `matrix_gm_full` constructor (mixture of matrix-Gaussians).
  Mixtures emerge from if/else branching; a direct multi-component constructor
  is a different feature.
- `matrix_iid(M, sigma)` convenience constructor (Alt-C). Defer to a follow-up.
- Auto-recovery from dense to Kronecker after operations (flip-flop ALS).
  Once dense, the variable stays dense.
- Backwards-incompatible changes to `matrix_gm(M, U, V)`. Stays as-is.
- Variance back-prop on observe (Opt-2 from research note 04). Mean-only stays.

## 13. Verification commands

```bash
# After each sub-task — quick spot-check
.venv/bin/python -m pytest tests/test_matrix_gm_full.py -v
.venv/bin/python -m pytest tests/ -q                              # full suite

# Showcase + advanced end-to-end
(cd src && timeout 30 ../.venv/bin/python SOGA.py -f ../programs/Example/matrix_gm_showcase.soga)
(cd src && timeout 30 ../.venv/bin/python SOGA.py -f ../programs/Example/matrix_gm_advanced.soga)

# Grammar regen check
bash scripts/check_grammar_sync.sh

# Scalar regression
(cd src && timeout 30 ../.venv/bin/python SOGA.py -f ../programs/Example/Bernoulli.soga)
(cd src && timeout 30 ../.venv/bin/python SOGA.py -f ../programs/SOGA/ClickGraphPrune.soga)
```

## 14. Cross-review history

_To be populated by Phase F (codex-cross-reviewer)._
Codex CLI unavailable in this environment → falling back to `bs-detector`
+ inline review at end of /iterate.

---

_End of plan. Awaiting user approval to start `/iterate plan/2026-05-25-matrix-gm-full.md`._
