# Matrix-GM Extension — Known Limitations

This document catalogues the latent bugs discovered during the matrix-GM stress
test campaign (plan: `plan/2026-05-25-matrix-gm-stress-test.md`).

None of these bugs have been fixed in the `feat/matrix-gm-integration` branch.
They are exposed as `pytest.mark.xfail(strict=True)` probes in
`tests/stress_matrix_gm/test_fragile_paths.py`.  A dedicated follow-up plan
should address each before the branch is used in production.

---

## Known Bug Table

| Bug ID | Symptom | Source location (approx.) | Workaround | Exposed by test | Follow-up |
|--------|---------|--------------------------|------------|-----------------|-----------|
| **C1 / Path 5** | `observe(X[i,j] > c)` densifies the matrix covariance to a `(None, Sigma)` sentinel via the Schur-complement path. A subsequent `Y = A @ X` call in `_matrix_affine_left` attempts to unpack `(None, Sigma)` as `(U, V)` and executes `A @ None`. The subprocess worker crashes with `TypeError`; the main process hits the multiprocessing queue timeout and prints `Warning: SOGA Timeout occurred`. | `src/libMatrixUpdate.py:268` (`_matrix_affine_left`), `src/libSOGAsharedMatrix.py:163` (`get_cov`) | Do not apply affine operations to a matrix variable after it has been conditioned via `observe(X[i,j] op c)`. Keep the `observe` statement as the last matrix operation in the program or extract the needed scalar before the observe. | `test_F1_observe_then_left_affine_hangs` | Dedicated plan: "fix dense-sentinel affine ops" |
| **C7** | `X = matrix_gm_full(M, Sigma_dense)` stores a dense sentinel `(None, Sigma)`. A subsequent `Y = X @ B` (`_matrix_affine_right`) executes `U_x.copy()` where `U_x is None` → `AttributeError`. Similarly `Y = A @ X` (`_matrix_affine_left`) executes `A @ None` → `ValueError`. | `src/libMatrixUpdate.py:282` (`_matrix_affine_right:285`), `src/libMatrixUpdate.py:268` (`_matrix_affine_left:274`) | After calling `matrix_gm_full`, only use operations that support the dense sentinel: element extract `X[i,j]`, element write `X[i,j] = c`, and `observe(X[i,j] op c)`. Do not apply affine (`A@X`, `X@B`) or transpose operations. | `test_F2a_dense_sentinel_right_affine_fails`, `test_F2b_dense_sentinel_left_affine_fails` | Same plan as C1 |
| **Gap 3** | After `y0 = X[0,0]`, an `observe(X[1,0] > 0)` updates the matrix distribution of `X` via Tallis truncation and a Kalman-style mean shift. When the row covariance `U` has off-diagonal entries, `E[X[0,0]]` changes (correlated update). However, the scalar `y0` extracted before the observe is NOT Kalman-updated: `E[y0]` stays at the prior mean while `E[X[0,0]]` shifts upward. Root cause: the cross-covariance between `y0` and `X` in `cov_blocks` is never propagated back through the truncation update. | `src/libMatrixTruncate.py` (matrix truncate path), `src/libSOGAsharedMatrix.py` (cross-cov scalar-matrix block) | Extract scalar variables AFTER any observe operations, not before. Alternatively, re-extract `y0 = X[0,0]` after the observe to get the updated value. | `test_F3_stale_cross_cov_after_observe` | Dedicated plan: "Gap3 cross-cov back-propagation" |
| **F4** | After `y = X[0,0]` (extract) and `X[0,0] = 5.0` (Schur element write, which sets `X[0,0]` to a deterministic constant), the cross-covariance entry `Cov(y, X[0,0])` should become 0 (because `X[0,0]` is now a constant). However, `_matrix_element_write_component` only updates `cov_blocks[k][{mat_name}]` and does not update any cross-covariance entries involving other variables. The stale cross-cov `frozenset({"y","X"})` retains the pre-write value `U[0,0]*V[0,0] != 0`. | `src/libMatrixUpdate.py:843` (`_matrix_element_write_component`) — does not iterate over cross-cov keys | Extract scalar variables AFTER element writes, not before. Equivalently, do not rely on scalar variables extracted before an `X[i,j] = c` write for downstream covariance computations involving the written element. | `test_F4_extract_write_stale_cross_cov` | Same plan as Gap3 |

---

## Scope

These bugs are present in `feat/matrix-gm-integration` as of commit `7d84b1c2`.
They affect:
- Programs that combine `observe(X[i,j] op c)` with subsequent affine ops (C1/Path5)
- Programs that use `matrix_gm_full` (dense-sentinel constructor) with subsequent affine ops (C7)
- Programs that extract scalars before conditioning on correlated matrix entries (Gap3)
- Programs that extract scalars before element writes to the same matrix (F4)

Programs that use only `matrix_gm` (Kronecker constructor) with affine, sum, transpose,
and Isserlis operations — without observe followed by affine — are not affected.

---

## Follow-up

The fix plan (not in scope here) should:
1. Add dense-sentinel guards to `_matrix_affine_left` and `_matrix_affine_right`
   (raise `NotImplementedError` with a clear message, or densify and operate).
2. Propagate Kalman update back to extracted scalar cross-cov entries during
   matrix truncation (Gap3 fix).
3. Update cross-cov entries in `_matrix_element_write_component` when scalar
   variables have a cross-cov entry with the written matrix variable (F4 fix).

All fixes require `/audit-numerical` pre-flight per project conventions (see `CLAUDE.md §7`).

---

_Generated: 2026-05-25. Campaign run: `results/qa_stress_test/`. Plan: `plan/2026-05-25-matrix-gm-stress-test.md`._
