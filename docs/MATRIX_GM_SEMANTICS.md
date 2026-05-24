# SOGA Matrix-Variate Gaussian Mixture Semantics

**Date**: 2026-05-24
**Branch**: `feat/matrix-gm-integration`
**Status**: Complete (12 patterns verified)

This document specifies, for each of the 12 matrix-GM syntactic patterns
supported by SOGA, the following:
(a) DSL syntax, (b) example, (c) formal semantics, (d) closed-form formula,
(e) approximation flags, (f) source code reference.

All formulas use the convention `vec(X) ~ N(vec(M), V ⊗ U)` where `V` is
the column covariance (n×n), `U` is the row covariance (m×m), and `⊗` is
the Kronecker product.  Column-major ordering throughout.

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

**Approximation flags**: DENSE materialises `V ⊗ U` as `(mn × mn)` matrix per
component.  `KroneckerDenseMemoryWarning` if budget exceeded (default 1024 MB).

**Source**: `libMatrixTruncate.py`, `_truncate_matrix_element_ineq`.

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

The R3 "balanced" regime (M_X = M_Y = I and σ²_X = σ²_Y = 0.5) — where
delta-method previously had ~20% relative error on `Cov(vec(Z))` — is now
exact (1.3% residual is MC sampling noise + NKP rank-1 floor).  Other regimes
were already accurate via delta-method; the Isserlis correction term
`tr(V_X · U_Y) · V_Y` becomes negligible relative to `M_Y^T V_X M_Y` whenever
M_Y is large (small-covariance regime).

### Practical guidance for users

- **Recommended regime** (E[Z] and Cov accurate to <5 %): at least one
  of the operands has small covariance relative to its mean magnitude
  (concentrated prior).  This includes the typical Lishan-class case
  where X is a parameter prior and Y is data/kernel.
- **Caveat regime** (R3-like): both X and Y are diffuse random matrices
  with comparable variance scales.  In this case the post-multiply
  covariance carries ~20 % relative error.  The `MatmulApproxWarning`
  fires (s_2/s_1 > 5 %) — users should treat it as a flag to either
  (a) accept the approximation, (b) re-formulate the computation, or
  (c) use Monte Carlo for the affected step.
- **Hard limit**: the delta-method linearisation is a first-order Taylor
  expansion around the means.  It is *not* well-defined when either
  operand has zero mean (the linearisation becomes degenerate).  For
  the `M_X = 0` or `M_Y = 0` corner, the exact distribution of vec(Z)
  is the matrix-product chi-squared family (Bishop & Del Moral 2017,
  arXiv:1703.00353) — no current PPL handles this symbolically.

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

After this write, the variable is in "dense mode".  Subsequent
Kronecker-dependent ops (affine_left, transpose) raise `NotImplementedError`.
Only element extract (`y = X[i,j]`) and observe (`observe(X[i,j] op c)`)
remain valid on a dense-mode variable (v1 limitation).

**Approximation flags**: `ElementWriteDenseWarning` always emitted on first
densification.  Cost: `O((mn)^2)` storage vs `O(m^2 + n^2)` for Kronecker.

**Source**: `libMatrixUpdate.py`, `matrix_element_write_dispatch`,
`_matrix_element_write_component`, `_densify_matrix_var`.

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

## Out-of-scope (v1 limitations)

The following patterns are NOT supported in v1 and raise `NotImplementedError`:

- `observe(expr_involving_two_matrix_vars)` — cross-matrix conditioning
- `Z = X @ Y` where `X` and `Y` have been densified by element writes
- Affine/transpose on a dense-mode variable (after element write)
- `trace(X)` inequality observe

These are tracked as v2 items in the risk register.

---

_Generated 2026-05-24 as part of feat/matrix-gm-integration fix5._
