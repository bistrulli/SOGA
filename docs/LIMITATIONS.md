# Matrix-GM Extension — Known Limitations

This document catalogues the latent bugs discovered during the matrix-GM stress
test campaign (plan: `plan/2026-05-25-matrix-gm-stress-test.md`).

These bugs are now exposed at runtime via clean errors/warnings (commit `433adc67`).
A dedicated follow-up plan should address the underlying mathematical fix before
the branch is used in production.

---

## Known Bug Table

| Bug ID | Symptom | Source location (approx.) | Runtime signal | Workaround | Exposed by test | Follow-up |
|--------|---------|--------------------------|----------------|------------|-----------------|-----------|
| **C1 / Path 5** | `observe(X[i,j] > c)` densifies the matrix covariance to a `(None, Sigma)` sentinel via the Schur-complement path. A subsequent `Y = A @ X` call in `_matrix_affine_left` previously crashed with `TypeError`; the subprocess worker died silently and the main process hit the queue timeout. | `src/libMatrixUpdate.py` (`_matrix_affine_left`), `src/libSOGAsharedMatrix.py` (`get_cov`) | `NotImplementedError("[C1/C7] left-affine (A@X) on dense-sentinel covariance not implemented. ... See docs/LIMITATIONS.md §C1/C7.")` — raised immediately in the worker process, traceback visible in stderr; rc=0 with `Warning: SOGA Timeout occurred` in stdout. | Do not apply affine operations to a matrix variable after it has been conditioned via `observe(X[i,j] op c)`. Keep the `observe` statement as the last matrix operation in the program or extract the needed scalar before the observe. | `test_F1_observe_then_left_affine_hangs` | Dedicated plan: "fix dense-sentinel affine ops" |
| **C7** | `X = matrix_gm_full(M, Sigma_dense)` stores a dense sentinel `(None, Sigma)`. A subsequent `Y = X @ B` (`_matrix_affine_right`) previously crashed with `AttributeError` (U_x.copy() on None). Similarly `Y = A @ X` (`_matrix_affine_left`) crashed with `ValueError`. | `src/libMatrixUpdate.py` (`_matrix_affine_right`, `_matrix_affine_left`, `_matrix_scale`, `_matrix_transpose`) | `NotImplementedError("[C1/C7] right-affine (X@B) / left-affine (A@X) / scale (c*X) / transpose (X^T) on dense-sentinel covariance not implemented. ... See docs/LIMITATIONS.md §C1/C7.")` — explicit clear error for each of the 4 affected operations. | After calling `matrix_gm_full`, only use operations that support the dense sentinel: element extract `X[i,j]`, element write `X[i,j] = c`, and `observe(X[i,j] op c)`. Do not apply affine (`A@X`, `X@B`), scale, or transpose operations. | `test_F2a_dense_sentinel_right_affine_fails`, `test_F2b_dense_sentinel_left_affine_fails` | Same plan as C1 |
| **Gap 3** | After `y0 = X[0,0]`, an `observe(X[1,0] > 0)` updates the matrix distribution of `X` via Tallis truncation and a Kalman-style mean shift. When the row covariance `U` has off-diagonal entries, `E[X[0,0]]` changes (correlated update). However, the scalar `y0` extracted before the observe is NOT Kalman-updated: `E[y0]` stays at the prior mean (silent-wrong). | `src/libMatrixTruncate.py` (`_truncate_matrix_element_ineq`) — cross-cov never propagated back | `StaleCrossCovWarning("Scalar 'y0' has non-zero cross-cov with 'X' (norm=…); observe on 'X' will not update 'y0'. Value of 'y0' remains STALE. See docs/LIMITATIONS.md §Gap3.")` — emitted before the component loop when any scalar has non-zero cross-cov with the observed matrix variable. | Extract scalar variables AFTER any observe operations, not before. Alternatively, re-extract `y0 = X[0,0]` after the observe to get the updated value. | `test_F3_stale_cross_cov_after_observe` | Dedicated plan: "Gap3 cross-cov back-propagation" |
| **F4** | After `y = X[0,0]` (extract) and `X[0,0] = 5.0` (Schur element write), the cross-covariance entry `Cov(y, X[0,0])` should become 0 but is not updated (silent-wrong). | `src/libMatrixUpdate.py` (`_matrix_element_write_component`) — cross-cov keys not iterated | `StaleCrossCovWarning("Scalar 'y' has non-zero cross-cov with 'X' (norm=…); element write on 'X' will not update 'y'. Value of 'y' remains STALE. See docs/LIMITATIONS.md §F4.")` — emitted at end of `_matrix_element_write_component` when any scalar has non-zero cross-cov with the written matrix variable. | Extract scalar variables AFTER element writes, not before. | `test_F4_extract_write_stale_cross_cov` | Same plan as Gap3 |

---

## Scope

These bugs are present in `feat/matrix-gm-integration` as of commit `7d84b1c2`.
Runtime signals were added in commit `433adc67` (plan `2026-05-25-matrix-gm-safety-patches.md`).
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
