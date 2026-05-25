# SOGA Matrix-Variate Gaussian Mixture Semantics

**Date**: 2026-05-25
**Branch**: `feat/matrix-gm-integration`
**Status**: 16 patterns specified; runtime safety signals in place for the five documented latent bugs (see `docs/LIMITATIONS.md`).

Each of the 16 matrix-GM syntactic patterns in SOGA is specified below as: (a) DSL syntax, (b) example, (c) formal semantics, (d) closed-form formula, (e) approximation flags or runtime safety signals, (f) source reference.

The convention everywhere is `vec(X) ~ N(vec(M), V ⊗ U)` with `V` the column covariance (n×n), `U` the row covariance (m×m), `⊗` the Kronecker product, and column-major ordering on `vec`. Where a pattern is currently subject to a known bug, the relevant entry in `docs/LIMITATIONS.md` is referenced inline.

---

## 1. Matrix variable declaration

**DSL syntax**
```
matrix[m][n] X;
```

**Example**
```
matrix[2][2] X;
```

**Formal semantics**
Declares `X` as a matrix-typed variable with shape `(m, n)`.  No distribution
is assigned yet.  The variable is registered in `var_entries` with its shape.

**Closed-form formula**
None (declaration only).

**Source**: `producecfg.py`, `enterMatrix_decl`; `libSOGAshared.py`, `VarEntry`.

---

## 2. Matrix GM initialization

**DSL syntax**
```
X = matrix_gm(M_list, U_list, V_list);
```

**Example**
```
X = matrix_gm([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]);
```

**Formal semantics**
Assigns `X ~ MN(M, U, V)` where `M`, `U`, `V` are parsed from the nested list
literals.

**Closed-form formula**
```
vec(X) ~ N(vec(M), V ⊗ U)
Var(X[i,j]) = U[i,i] * V[j,j]
Cov(X[i,j], X[i',j']) = U[i,i'] * V[j,j']
```

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `update_rule_matrix` op=`MATRIX_GM`;
`libSOGAsharedMatrix.py`, `GaussianMixBlock.from_matrix_gm`.

---

## 3. Scalar extract from matrix: `y = X[i,j]`

**DSL syntax**
```
y = X[i,j];    # i,j are numeric literals
y = X[i,j];    # i,j are loop-counter variables (fix1)
```

**Example**
```
y = X[0,1];
for k in range(4) { d = X[k,k]; } end for;
```

**Formal semantics**
Extracts the scalar marginal `y = X[i,j]` from the matrix distribution.
Cross-covariance between `y` and `vec(X)` is tracked (A1).

**Closed-form formula**
```
E[y]    = M[i,j]
Var(y)  = U[i,i] * V[j,j]
Cov(y, vec(X)) = V[:,j] ⊗ U[:,i]    (mn-vector, exact)
```

For another scalar `z` with stored cross-cov `cov(z, vec(X))`:
```
Cov(y, z) = (V[:,j] ⊗ U[:,i])^T · cov(z, vec(X))
```

Loop-variable indices `i`, `j` are resolved via `data[name][0]` at runtime.

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `extract_scalar_from_matrix`;
`libSOGAupdate.py`, `update_rule` hybrid routing (fix1 IDV regex).

---

## 4. Affine left: `Y = A @ X`

**DSL syntax**
```
Y = A @ X;    # A is a deterministic data variable
```

**Example**
```
Y = rot2d @ X;
```

**Formal semantics**
Linear transformation by deterministic matrix `A` on the left.

**Closed-form formula**
```
Y ~ MN(A @ M, A @ U @ A.T, V)
```

Cross-cov with scalars: `Cov(y, vec(Y)) = (I_n ⊗ A) · Cov(y, vec(X))`.

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `_matrix_affine_left`.

---

## 5. Affine right: `Y = X @ B`

**DSL syntax**
```
Y = X @ B;    # B is a deterministic data variable
```

**Formal semantics**
Linear transformation by deterministic matrix `B` on the right.

**Closed-form formula**
```
Y ~ MN(M @ B, U, B.T @ V @ B)
```

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `_matrix_affine_right`.

---

## 6. Matrix add constant: `Y = X + C`

**DSL syntax**
```
Y = X + C;    # C is a deterministic data variable
```

**Formal semantics**
Shift the mean by `C`.

**Closed-form formula**
```
Y ~ MN(M + C, U, V)
```

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `_matrix_add_const`.

---

## 7. Matrix add random: `Y = X + N`

**DSL syntax**
```
Y = X + N;    # N is an independent random matrix variable
```

**Formal semantics**
Sum of two independent random matrices.

**Closed-form formula (iso path)**
When `U_X` or `U_N` is isotropic (`c * I`):
```
Y ~ MN(M_X + M_N, U_X, V_X + (a*b/c) * I_n)   # iso-A path
```

**Closed-form formula (general path)**
```
Cov(vec(Y)) = V_X ⊗ U_X + V_N ⊗ U_N
(U_Y, V_Y) = nearest_kronecker(Cov(vec(Y)), m, n)
```

**Approximation flags**: `KroneckerApproxWarning` when NKP projection error > 5%.

**Source**: `libMatrixUpdate.py`, `_matrix_add_random`.

---

## 8. Transpose: `Y = transp(X)`

**DSL syntax**
```
Y = transp(X);
```

**Formal semantics**
Matrix transpose: U and V factors swap.

**Closed-form formula**
```
X^T ~ MN(M^T, V, U)
```

Cross-cov via commutation matrix `P_comm` (mn × mn):
```
Cov(y, vec(X^T)) = P_comm · Cov(y, vec(X))
```

**Approximation flags**: None (exact).

**Source**: `libMatrixUpdate.py`, `_matrix_transpose`.

---

## 9. Element inequality observe: `observe(X[i,j] op c)`

**DSL syntax**
```
observe(X[i,j] > c);
observe(X[i,j] <= c);
observe(row_sum(X,i) > c);
observe(col_sum(X,j) <= c);
```

Loop-variable indices supported (fix1).

**Formal semantics**
Conditions the joint distribution on the linear constraint.

**Closed-form formula (rank-1 conditional update)**
For selector `a = e_{j*m+i}` (column-major):
```
mu_s  = a^T vec(M)
g     = Sigma a
m_hat, v_hat, P = truncated_normal_moments(mu_s, a^T g, c, op)
vec(M)_new = vec(M) + g * (m_hat - mu_s) / (a^T g)
Sigma_new  = Sigma - outer(g, g) * (1 - v_hat/(a^T g)) / (a^T g)
```

**Approximation flags**: the DENSE path materialises `V ⊗ U` to an `(mn × mn)` covariance per component. `KroneckerDenseMemoryWarning` fires if the budget is exceeded (default 1024 MB).

**Runtime safety signals**: if a scalar variable in `var_list` carries a non-zero cross-covariance to `X` at the time of the observe (i.e., it was extracted before this statement), `_truncate_matrix_element_ineq` emits `StaleCrossCovWarning`. The scalar's moments are not back-propagated (bug `Gap 3` in `docs/LIMITATIONS.md`). Independently, the back-propagation to non-observed elements of `X` itself is currently sign-flipped (bug `C8`); the observed element's marginal is correct, the back-propagation to other elements via off-diagonal `U` is not.

**Source**: `libMatrixTruncate.py`, `_truncate_matrix_element_ineq`, `_rank1_cond_update`.

---

## 10. Loop-variable index (fix1)

**DSL syntax**
```
for i in range(N) {
    d = X[i, i];
    observe(X[i, j_fixed] > 0);
} end for;
```

**Formal semantics**
Loop counter `i` is resolved via `data['i'][0]` at each iteration step.
The grammar already accepts `IDV` in index positions; fix1 adds runtime
resolution to `update_rule` and `_classify_constraint`.

**Closed-form formula**: per pattern 3 (extract) or 9 (observe).

**Approximation flags**: None.

**Source**: `libSOGAupdate.py`, `update_rule` IDV regex + `_resolve_idx`;
`libMatrixTruncate.py`, `_resolve_index`.

---

## 11. Branch merge with matrix variables (fix2)

**DSL syntax**
```
if theta > 0.5 {
    y = X[0,0];
} else {
    y = X[1,1];
} end if;
```

**Formal semantics**
Post-merge distribution is the weighted mixture of the then-branch and
else-branch distributions.  Component count grows as `K_then + K_else`.

**Closed-form formula**
```
merged_pi[k]    = p_branch * pi_k   (renormalised)
merged_M[k]     = M_k from respective branch
merged_(U,V)[k] = (U_k, V_k) from respective branch
```

Mixture mean: `E[X] = sum_k pi_k * M_k`.

**Approximation flags**: None (concatenation is exact).  Memory budget guard
emits `MatrixMergeMemoryWarning` if `K * (mn)^2 * 8 bytes > MATRIX_MERGE_BUDGET_MB`.

**Source**: `libSOGAmerge.py`, `_merge_matrix_gm_blocks`.

---

## 12. Random × random matmul: `Z = X @ Y` (fix3)

**DSL syntax**
```
Z = X @ Y;    # both X and Y are random matrix variables
```

**Formal semantics**
Product of two independent random matrix-variate Gaussians.  The exact
distribution of `Z` is NOT Gaussian; SOGA approximates via:
1. Delta-method linearisation (exact E[Z], first-order Cov).
2. Van Loan-Pitsianis nearest-Kronecker projection (NKP) to restore
   Kronecker structure.

**Closed-form formula (EXACT 2nd-moment Isserlis matriciale, fix3.2 upgrade 2026-05-24)**
```
E[Z] = M_X @ M_Y                                          (exact)
Cov(vec(Z)) = tr(V_X · U_Y) · (V_Y ⊗ U_X)                 [Isserlis trace term]
            + (M_Y^T V_X M_Y) ⊗ U_X                       [delta-method term 1]
            + V_Y ⊗ (M_X U_Y M_X^T)                       [delta-method term 2]
            (= EXACT for the second moment; no Taylor approximation)
(U_Z, V_Z) = nearest_kronecker(Cov(vec(Z)), m, n)         [only approx: NKP rank-1]
```

The Isserlis trace term `tr(V_X · U_Y) · (V_Y ⊗ U_X)` was MISSING from the
original delta-method linearisation (research note 05 §Q2c).  Its inclusion
captures the cov × cov interaction that delta-method drops as 2nd-order.
The fix3.2 derivation is the matrix-variate extension of the scalar Isserlis
formula `E[(xy)²] = σ_xy² + 2σ_xy·μ_x·μ_y + var_x·var_y + var_x·μ_y² + var_y·μ_x²`
that has always been used in scalar SOGA for `z = x*y`.

**Sole remaining approximation**: NKP rank-1 projection from the sum of two
(post-grouping) Kronecker products to a single V_Z ⊗ U_Z.  Approximation
error: `||Cov - kron(V_Z, U_Z)||_F / ||Cov||_F`.

**Approximation flags**: `MatmulApproxWarning` when `s_2 / s_1 > 5%` in the
NKP SVD (indicates significant non-Kronecker term discarded).

### Empirical accuracy (MC validation, 20k samples, 2026-05-24)

Validated against Monte Carlo ground truth on four regimes.  Each cell is
the Frobenius-norm relative error `||Soga - MC||_F / ||MC||_F`.

| Regime                        | E[Z] rel err | Pre-Isserlis Cov | Post-Isserlis Cov | Improvement |
|-------------------------------|--------------|------------------|-------------------|-------------|
| R1 small cov (σ²=0.01), M=I   | 2e-4         | ≈ 0              | ≈ 0               | unchanged   |
| R2 moderate cov, non-triv M   | 4e-4         | 0.091            | 0.091             | unchanged   |
| R3 BALANCED (σ²=0.5, M=I)     | 0.007        | **0.197**        | **0.013**         | **15× ✓**   |
| R4 3x3 small cov              | 8e-4         | ≈ 0              | ≈ 0               | unchanged   |

The R3 "balanced" regime sets `M_X = M_Y = I` and `σ²_X = σ²_Y = 0.5`. Under the older delta-method linearisation it produced approximately 20% relative error on `Cov(vec(Z))`. With the fix3.2 Isserlis formula the residual drops to 1.3%, which is at the level of Monte Carlo sampling noise plus the NKP rank-1 floor. The other regimes were already accurate under delta-method because whenever `M_Y` is large in the small-covariance regime, the Isserlis trace term `tr(V_X · U_Y) · V_Y` is negligible compared with `M_Y^T V_X M_Y`.

### Practical guidance for users

- **Recommended regime** (E[Z] and Cov accurate to <5 %): at least one
  of the operands has small covariance relative to its mean magnitude
  (concentrated prior).  This includes the typical Lishan-class case
  where X is a parameter prior and Y is data/kernel.
- **Caveat regime** (R3-like): both X and Y are diffuse random matrices with comparable variance scales. The post-multiply covariance carries roughly 20% relative error. `MatmulApproxWarning` fires whenever `s_2/s_1 > 5%`. Treat this as a flag, and respond with one of: (a) accept the approximation, (b) re-formulate so that one operand is concentrated (small covariance relative to its mean), (c) fall back to Monte Carlo for the step in question.
- **Zero-mean corner (`M_X = 0` or `M_Y = 0`)**: the Isserlis formula (fix3.2) handles this exactly. With `M_X = 0` and `M_Y = 0` the two delta-method terms vanish and `Cov(vec(Z))` reduces to the single Kronecker product `tr(V_X · U_Y) · (V_Y ⊗ U_X)`, which is already in NKP-projected form (residual 0). The older delta-method linearisation produced `Cov = 0` in this corner, which was the worst-case failure mode that motivated the fix3.2 upgrade. The true distribution of `vec(Z)` is still the matrix-product chi-squared family (Bishop and Del Moral 2017, arXiv:1703.00353), so only the second moment is captured exactly here; higher moments are not represented by SOGA's state.

**Source**: `libMatrixUpdate.py`, `matmul_random_random_component`;
`libMatrixGaussian.py`, `_nearest_kronecker`.
**Validation**: `tests/test_random_matmul.py` + MC suite in
`docs/research-notes/05-matrix-gm-merge-and-random-matmul.md` §Problem 2.

---

## 13. Matrix element write: `X[i,j] = expr` (fix4)

**DSL syntax**
```
X[i,j] = 5.0;          # B1: deterministic constant
X[i,j] = z;            # B2: scalar variable
X[i,j] = 2*z + 1;      # B3: scalar expression
```

**Formal semantics**
Hard or soft conditioning on `X[i,j]`.  The Kronecker structure of the
covariance is destroyed (see research note 04 Finding B1); the covariance
is densified to `(mn × mn)` (dense sentinel mode).

**Closed-form formula (Schur-complement rank-1 downdate)**
```
idx     = j * m + i                 (column-major)
sel_vec = Sigma[:, idx]
denom   = Sigma[idx, idx] + var_z   (var_z=0 for B1, var_z=Var(z) for B2)
gain    = sel_vec / denom
vec(M)_new = vec(M) + gain * (c_or_mu_z - vec(M)[idx])
Sigma_new  = Sigma - outer(gain, sel_vec)
```

After this write the variable is in dense-sentinel mode. Affine ops (`A @ X`, `X @ B`), scale (`c * X`), transpose, and matrix add raise `NotImplementedError("[C1/C7] <op> on dense-sentinel covariance not implemented. ... See docs/LIMITATIONS.md §C1/C7.")` with a per-operation message. Element extract `y = X[i,j]` and `observe(X[i,j] op c)` continue to work on the dense path.

**Runtime safety signals**: `ElementWriteDenseWarning` fires on the first densification. If a scalar variable was extracted from `X` before this write and carries a non-zero cross-covariance to `X`, `_matrix_element_write_component` also emits `StaleCrossCovWarning` (bug `F4` in `docs/LIMITATIONS.md`). The cross-covariance entry between the scalar and `X` is not updated by the write.

**Approximation flags**: storage cost moves from `O(m² + n²)` to `O((mn)²)` per component. No approximation in the mathematics of the rank-1 downdate.

**Source**: `libMatrixUpdate.py`, `matrix_element_write_dispatch`, `_matrix_element_write_component`, `_densify_matrix_var`.

---

## 14. Matrix element equality observe (O5)

**DSL syntax**
```
observe(X[i,j] == c);
observe(row_sum(X, i) == c);
observe(col_sum(X, j) == c);
```

**Formal semantics**
Hard Dirac conditioning on a single linear functional of `vec(X)`.  The
constraint `a^T vec(X) = c` (with `a = e_{j·m+i}` for element, or selector for
row/col sums) collapses the joint distribution onto the hyperplane.

**Closed-form formula (Schur-complement rank-1 downdate, equivalent to fix4
element write)**:
```
mu_s   = a^T vec(M)
var_s  = a^T Sigma a
g      = Sigma a
M_new  = vec(M) + g · (c - mu_s) / var_s
S_new  = Sigma - outer(g, g) / var_s
P      = 1.0   (by convention; the constraint is measure-zero but treated as
                an exact conditioning, consistent with scalar SOGA `==` semantics)
```

**Source**: `libMatrixTruncate.py`, `_tnorm1d` dispatches `direction == "=="`
to (m_hat=c, v_hat=0, P=1) which feeds the existing `_rank1_cond_update`.

## 15. Linear combination of matrix elements observe (O7)

**DSL syntax**
```
observe(2*X[0,0] + 3*X[1,1] > 1);
observe(X[0,0] - X[1,1] >= 0);
observe(a*X[i,j] + b*X[k,l] + ... op c);   # any number of terms
```

**Formal semantics**
General linear functional `s^T vec(X) op c` where `s ∈ R^{mn}` is built
term-by-term from the coefficients:
```
s[j_k · m + i_k] += coef_k       for each (coef_k, i_k, j_k) term
```

**Closed-form formula** (identical math to element/row/col sum, only the
selector vector differs):
```
mu_s    = s^T vec(M)
var_s   = s^T Sigma s
m_hat, v_hat, P = 1D truncated-normal moments(mu_s, var_s, c, op)
M_new   = vec(M) + (Sigma s) · (m_hat - mu_s) / var_s
S_new   = Sigma - outer(Sigma s, Sigma s) · (1 - v_hat / var_s) / var_s
```

**Verification**: for `observe(2 X[0,0] + 3 X[1,1] > 1)` with X ~ MN(0, I, I):
- Y = 2 X[0,0] + 3 X[1,1] ~ N(0, 13)
- Truncate Y > 1: λ = pdf(1/√13) / (1 − cdf(1/√13)); E[Y|>1] = √13 · λ
- E[X[0,0]] post = 2 · E[Y|>1] / 13;  E[X[1,1]] post = 3 · E[Y|>1] / 13
- Empirical (test_truncate_matrix.py::TestObserveClosure): matches < 1e-4

**Source**: `libMatrixTruncate.py`, `_RE_LINEAR_COMBO` regex +
`_parse_linear_combo` + `_truncate_matrix_linear_combo` handler.

## Notation summary

| Symbol | Meaning |
|--------|---------|
| `M` | Mean matrix (m × n) |
| `U` | Row covariance (m × m), a.k.a. "between-row" factor |
| `V` | Column covariance (n × n), a.k.a. "between-column" factor |
| `vec(X)` | Column-major vectorisation: `X.flatten('F')` |
| `V ⊗ U` | Kronecker product (V is LEFT factor per SOGA convention) |
| `kron(V, U)` | `np.kron(V, U)` — shape `(mn, mn)` |
| `MN(M, U, V)` | Matrix-variate normal with the above factors |
| `NKP` | Nearest-Kronecker projection (Van Loan & Pitsianis 1993) |
| `pi_k` | Mixture weight for component k |

---

## §16 — `matrix_gm_full` constructor (v1.2)

### Formal grammar

```antlr4
/* SOGA.g4 */
matrix_gm_full : MATRIX_GM_FULL '(' mlist ',' mlist ')' ;
/* mlist is a row-by-row list of lists (already defined) */

/* Lexer — MATRIX_GM_FULL declared BEFORE MATRIX_GM (longest-match rule) */
MATRIX_GM_FULL : 'matrix_gm_full' ;
MATRIX_GM      : 'matrix_gm' ;
```

**Semantic form**: `X = matrix_gm_full(M, Sigma);`
- `M`     — (m × n) mean matrix (mlist of m rows, each with n elements)
- `Sigma` — (mn × mn) full vectorised covariance matrix (mlist of mn rows, each with mn elements)

`Sigma` must be symmetric (max|Σ − Σᵀ| < 1e-8) and is checked at parse time.

### Auto-Kronecker detection algorithm

Given `Sigma` (mn × mn), SOGA runs the Van Loan-Pitsianis rank-1 SVD:

1. **Rearrange**: build `R` (m² × n²) where column `j·n + j'` of R holds
   `Sigma[j·m:(j+1)·m, j'·m:(j'+1)·m].flatten('F')`.
2. **SVD**: `[U_s, σ, Vt] = svd(R, full_matrices=False)`.
3. **Residual ratio**: `ρ = σ[1] / σ[0]` (second / first singular value).
   - If `σ[0] < 1e-14` (zero Sigma): `ρ = 0`.
4. **Threshold check**:
   - `ρ < SOGA_KRON_STRICT` (default 1e-8):  
     Sigma is exact Kronecker. Extract `(U, V)` via the same NKP step.  
     Store as `cov_blocks[k][{X}] = (U, V)`. Emit `KroneckerDetectionInfo`.
   - `SOGA_KRON_STRICT ≤ ρ < SOGA_KRON_LOOSE` (default 1e-3):  
     Sigma is near-Kronecker but stored as dense for safety.  
     Emit `KroneckerNearMissWarning`. Store as `(None, Sigma)`.
   - `ρ ≥ SOGA_KRON_LOOSE`:  
     Sigma is not separable. Store as dense `(None, Sigma)`. Emit `DenseCovarianceInfo`.

Thresholds are configurable via environment variables `SOGA_KRON_STRICT` and `SOGA_KRON_LOOSE`.

### Storage format

Consistent with fix4's dense sentinel (no new sentinel type introduced):

```python
# Kronecker-detected: same as matrix_gm
cov_blocks[k][frozenset({name})] = (U, V)    # U: (m,m), V: (n,n)

# Dense: same as after element write (fix4)
cov_blocks[k][frozenset({name})] = (None, Sigma_dense)  # Sigma_dense: (mn,mn)
```

### Subsequent operations

| Operation | Kronecker-detected path | Dense path |
|-----------|------------------------|------------|
| `y = X[i,j]` (scalar extract) | Exact via U[:,i] ⊗ V[:,j] | Exact via Sigma[:,j*m+i] |
| `observe(X[i,j] op c)` | Exact truncation | Exact truncation |
| `Y = A @ X` (affine left) | Kronecker fast path | NotImplementedError |
| `Y = transp(X)` | Exact factor swap | NotImplementedError |
| `Y = X + N` (add noise) | Iso path or NKP | NotImplementedError |
| `Y = c * X` (scale) | Kronecker fast path | NotImplementedError |

The dense-path limitations are identical to post-element-write (fix4, §13).

### Invariants

- The AC2/AC3 invariants (constraint 6 from plan §8):
  - Exact Kronecker `Sigma = V ⊗ U` → residual ≈ 0 → stored as `(U, V)` → `kron(V, U)` matches `Sigma` to 1e-14.
  - Near-Kronecker `Sigma = V ⊗ U + 1e-12 · δ` → residual ≈ 1e-12 → stored as Kronecker.
  - Non-separable `Sigma` (rank-2 rearrangement) → residual > 0.05 → stored as dense.

### Implementation files

- `libMatrixGaussian._try_kronecker_decompose` — the SVD + residual computation
- `libMatrixGaussian.KroneckerDetectionInfo`, `KroneckerNearMissWarning`, `DenseCovarianceInfo` — warning classes
- `libMatrixUpdate._parse_matrix_gm_full_text` — safe nested-list parser for M and Sigma
- `libMatrixUpdate.update_rule_matrix` MATRIX_GM_FULL branch — dispatcher
- `libSOGAsharedMatrix.GaussianMixBlock.from_matrix_gm_full` — block constructor

---

## Out-of-scope (v1 limitations)

The following patterns are not supported in v1.

- **Cross-matrix conditioning** (`observe(expr_involving_two_matrix_vars)`): not implemented. Raises `NotImplementedError` at runtime.
- **Random × random matmul on densified operands** (`Z = X @ Y` after element write or `matrix_gm_full(dense)`): the Isserlis formula assumes Kronecker storage on both operands. Raises `NotImplementedError`.
- **Affine, scale, transpose, matrix-add on a dense-mode variable**: raise `NotImplementedError("[C1/C7] ...")` with an explicit pointer to `docs/LIMITATIONS.md` §C1/C7.
- **`trace(X)` as an observe keyword**: not recognised by the grammar. Workaround via linear-combination observe `observe(X[0,0] + X[1,1] + ... > c)`.

## Latent bugs documented in `docs/LIMITATIONS.md`

Five latent bugs in the matrix-GM path are documented (bug IDs C1, C7, Gap 3, F4, C8). Four of them are exposed at runtime by either a clean `NotImplementedError` or a `StaleCrossCovWarning`; C8 has no runtime signal yet and is locked by the regression test `test_F3_stale_cross_cov_after_observe`. The fixes are scheduled for a separate plan; this document describes the current behaviour.

---

_Last updated 2026-05-25 (safety-patches plan `2026-05-25-matrix-gm-safety-patches.md`)._
