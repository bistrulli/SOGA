# Research note — Matrix-GM merge and random × random matmul

**Date**: 2026-05-24
**Triggered by**: /plan to fix 4 broken matrix-GM features (merge, loop-index, random@random, element-write)
**Scope**: Theoretical foundation for (1) branch merge with matrix-variate components, (2) Z = X @ Y for two random matrix-variate Gaussians.

---

## Top references (all URLs verified)

| # | Title | Authors | Year | URL/DOI |
|---|-------|---------|------|---------|
| 1 | Inference of Probabilistic Programs with Moment-Matching Gaussian Mixtures | Randone, Bortolussi, Incerto, Tribastone | 2024 | arXiv:2311.08235 (POPL) |
| 2 | A Kullback-Leibler Approach to Gaussian Mixture Reduction | Runnalls | 2007 | doi:10.1109/TAES.2007.4383588 |
| 3 | Matrix Variate Distributions | Gupta, Nagar | 2000 | ISBN 9781584880462 |
| 4 | Automated Kronecker Product Approximation (KoPA) | Tsilifis, Reineking, Xiao | 2022 | arXiv:1912.02392 (JMLR) |
| 5 | Matrix Product Moments in Normal Variables | Bishop, Del Moral | 2017 | arXiv:1703.00353 |
| 6 | Approximation with Kronecker Products | Van Loan, Pitsianis | 1993 | Semantic Scholar:61ae81bbbf527e3463abe8d1c067d935de129fe5 |

---

## Problem 1 — Branch merge with matrix-variate components

**Verdict**: closed-form, trivially correct. No new theory; same as scalar mixture merge.

The Gaussian-mixture definition is dimension-agnostic. SOGA's merge node combines two branches with weights `p_then`, `p_else=1-p_then`:

```
merged_pi[k]  = p_then · pi_then[k]      for k in then-branch
merged_pi[k]  = p_else · pi_else[k]      for k in else-branch
merged_components = concat(then_components, else_components)
```

For a matrix variable, each component carries `(M_k, U_k, V_k)`. After merge there are `K_then + K_else` components per matrix variable. **No Kronecker structure is lost by the merge itself** — each component retains its own Kronecker factors. The post-merge gm_block has K_then + K_else mu_blocks / cov_blocks entries.

### Reductions (Runnalls/Salmond) post-merge

Standard reduction operates on vectorized form `vec(M_k)`, `Sigma_k = V_k ⊗ U_k`. The Runnalls KL-bound:

```
B(i, j) = (1/2) [ w_m · log|Sigma_m| − w_i · log|Sigma_i| − w_j · log|Sigma_j| ]
```

uses determinants and the moment-preserving merge formula (Bishop §9). **The post-merge Sigma_m is generally NOT Kronecker-separable** (sum of two Kronecker products), so:

- Option A: keep Sigma_m dense (mn × mn) for the merged component
- Option B: re-project Sigma_m via Van Loan-Pitsianis nearest-Kronecker (rank-1 SVD on rearrangement)

Option B keeps the matrix-variate family closed at the cost of O(error). Option A is exact but exits the Kronecker representation.

### Variables present in only one branch

Same handling as scalar: missing variable carries its prior. For matrix vars, this means propagating `MN(M_prior, U_prior, V_prior)` into the branch where it's missing. SOGA would be the first PPL to formalize this for matrix-variate Gaussians (no published treatment found).

### Cross-cov scalar↔matrix through merge

Treated as part of the joint covariance block per component. Standard merge formula (Bishop §9, Salmond 1990) applies unchanged.

### Concrete formula for libSOGAmerge.merge

```python
def merge_matrix_gm(list_dist, list_p):
    """Concatenate components from N branches with weight rescaling.
    Each entry in list_dist has its own gm + gm_block.
    list_p[i] = probability mass of branch i.
    """
    merged_pi = []
    merged_scalar_mu = []
    merged_scalar_sigma = []
    merged_mu_blocks = []   # for gm_block
    merged_cov_blocks = []  # for gm_block

    for dist_i, p_i in zip(list_dist, list_p):
        for k in range(dist_i.gm.n_comp()):
            merged_pi.append(p_i * dist_i.gm.pi[k])
            merged_scalar_mu.append(dist_i.gm.mu[k])
            merged_scalar_sigma.append(dist_i.gm.sigma[k])
            if dist_i.gm_block is not None:
                merged_mu_blocks.append(dist_i.gm_block.mu_blocks[k])
                merged_cov_blocks.append(dist_i.gm_block.cov_blocks[k])
    # Normalize
    total = sum(merged_pi)
    merged_pi = [p / total for p in merged_pi]
    return Dist(var_list, GaussianMix(merged_pi, merged_scalar_mu, merged_scalar_sigma),
                var_entries=var_entries,
                gm_block=GaussianMixBlock(var_list, var_entries, merged_pi,
                                         merged_mu_blocks, merged_cov_blocks))
```

Source: Randone et al. 2024 (arXiv:2311.08235), Bishop PRML §9.

---

## Problem 2 — Random × Random matmul Z = X @ Y

**Verdict**: Z is NOT matrix-variate Gaussian. Approximation MANDATORY. Best approach: delta-method linearization + Van Loan-Pitsianis nearest-Kronecker projection.

### Why exact propagation fails

For X ~ MN(M_X, U_X, V_X), Y ~ MN(M_Y, U_Y, V_Y) independent:

- E[Z] = M_X @ M_Y (exact, by linearity + independence)
- vec(Z) is NOT Gaussian (product of two Gaussians)
- The exact second moment exists (via Isserlis' theorem, see Q2c below) but the resulting covariance does not correspond to any MN distribution

### Delta-method linearization (recommended for SOGA)

Linearize around the means:

```
Z − E[Z] ≈ (X − M_X) @ M_Y + M_X @ (Y − M_Y)
```

Apply `vec(AXB) = (B^T ⊗ A) vec(X)`:

```
vec(Z − E[Z]) ≈ (M_Y^T ⊗ I_m) vec(X − M_X) + (I_n ⊗ M_X) vec(Y − M_Y)
```

By independence + mixed-product property `(A⊗B)(C⊗D) = (AC)⊗(BD)`:

```
Cov(vec(Z)) ≈ (M_Y^T V_X M_Y) ⊗ U_X  +  V_Y ⊗ (M_X U_Y M_X^T)
```

**This is a sum of two Kronecker products, generically NOT Kronecker-separable** (matches Finding B1 of research-note 04).

Failure mode: 3rd-order term O(||X − M_X|| · ||Y − M_Y||). Accurate when at least one marginal covariance is small.

### Van Loan-Pitsianis nearest-Kronecker projection

To return to the matrix-variate family (single V'⊗U'), project Sigma_Z onto the nearest Kronecker product via rank-1 SVD on the rearrangement R[Sigma_Z]:

```python
def nearest_kronecker(Sigma, m, n):
    # Rearrange Sigma (mn × mn) into R (m² × n²)
    R = np.zeros((m*m, n*n))
    for i in range(m):
        for j in range(m):
            block = Sigma[i*n:(i+1)*n, j*n:(j+1)*n]
            R[i*m+j, :] = block.ravel()
    u1, s1, vt1 = np.linalg.svd(R, full_matrices=False)
    U_approx = u1[:, 0].reshape(m, m) * np.sqrt(s1[0])
    V_approx = vt1[0, :].reshape(n, n) * np.sqrt(s1[0])
    return U_approx, V_approx
```

For `Sigma_Z = A⊗B + C⊗D`, the rearrangement `R[Sigma_Z]` has rank 2. Rank-1 NKP captures the dominant term; approximation error bounded by `||second-term||_F`. Acceptable when one of the two contributing terms dominates.

### Concrete recipe for libMatrixUpdate.matmul_random_random

```python
def matmul_random_random_component(M_X, U_X, V_X, M_Y, U_Y, V_Y):
    # Exact mean
    M_Z = M_X @ M_Y
    m, n = M_Z.shape
    # Delta-method covariance (sum of 2 Kronecker products)
    A = M_Y.T @ V_X @ M_Y
    B = U_X
    C = V_Y
    D = M_X @ U_Y @ M_X.T
    Sigma_full = np.kron(A, B) + np.kron(C, D)
    # NKP projection
    U_Z, V_Z = nearest_kronecker(Sigma_full, m, n)
    U_Z = make_psd(make_sym(U_Z))
    V_Z = make_psd(make_sym(V_Z))
    return M_Z, U_Z, V_Z
```

The full X-mixture × Y-mixture product has J·K components (`pi_Z_ij = pi_X_i · pi_Y_j`). Pruning to K_max should follow immediately.

### Existing PPL support

- **PSI**: no MatrixNormal. Scalar-Gaussian only.
- **NumPyro**: `MatrixNormal` exists (continuous.py:1895) but inference is sampling-only. No symbolic matmul.
- **Pyro**: same as NumPyro.
- **Stan**: no MatrixNormal. HMC-only.
- **Hakaru**: no MatrixNormal.

**SOGA would be the first PPL to propagate Z = X @ Y symbolically when both are matrix-variate Gaussian random variables**, even via approximation. Novelty angle.

### Exact second moment (for testing only)

```
E[ZZ^T] = U_X tr((M_Y M_Y^T + U_Y tr(V_Y)) V_X^T) + M_X U_Y M_X^T tr(V_Y)
Cov_row(Z) = E[ZZ^T] − M_X M_Y M_Y^T M_X^T
```

Use as ground truth in `/audit-numerical` regression — verify delta-method approximation error against this exact formula on small cases (m=n=2).

---

## Gaps SOGA could fill (papers waiting to be written)

1. **First PPL with symbolic Z = X @ Y for matrix-variate Gaussian RVs**. The delta + NKP recipe above is implementable, sound, and unpublished as a PPL inference rule.
2. **Matrix-variate Gaussian mixture (MVGM) with branch merge semantics**. Randone et al. 2024 covers the scalar case; the matrix extension is straightforward but undocumented.
3. **Variable-scope rules across branches for matrix-variate state**. The "missing in one branch → carry prior" rule is unmentioned in the matrix-variate literature.

---

## Recommended next actions

- [ ] Save this as research note 05 — DONE (this file).
- [ ] Implement `nearest_kronecker(Sigma, m, n)` in `src/libMatrixGaussian.py`.
- [ ] Implement `matmul_random_random_component` per the delta + NKP recipe.
- [ ] Add unit tests vs. MC with Sigma_Z dense ground truth.
- [ ] Implement `merge_matrix_gm` in `libSOGAmerge.merge` — replaces the current NotImplementedError.
- [ ] Document the "variable missing in one branch" rule in CLAUDE.md or the matrix DSL guide.

---

_End research note 05._
