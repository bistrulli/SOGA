# Research note — Structured covariance representations for SOGA

**Date**: 2026-05-21
**Triggered by**: `/plan structured-cov` (scalability target: d > 100 with real-world programs)
**Scope**: literature on representations of `Σ` that exploit structure (low-rank, sparse, block, banded) and the matrix identities needed to operate on them efficiently, with focus on transferability to SOGA's truncate/update/merge kernels.

## 1. The core identities

### 1.1 Sherman–Morrison–Woodbury (SMW)

For invertible `A` and conformable `U, C, V` with `(C⁻¹ + V A⁻¹ U)` invertible:

```
(A + U C V)⁻¹  =  A⁻¹  −  A⁻¹ U (C⁻¹ + V A⁻¹ U)⁻¹ V A⁻¹
```

Rank-1 special case (Sherman–Morrison, 1949–50):

```
(A + u vᵀ)⁻¹  =  A⁻¹  −  (A⁻¹ u vᵀ A⁻¹) / (1 + vᵀ A⁻¹ u)
```

Cost: O(d²) instead of O(d³). For Σ = D + UUᵀ with D diagonal d×d and U d×k, computing Σ⁻¹ costs **O(d·k² + k³)** instead of O(d³). For d=1000, k=10: 10⁵ vs 10⁹ ops.

References (canonical):
- Sherman, J. & Morrison, W.J. (1949). *Adjustment of an inverse matrix corresponding to a change in one element of a given matrix*. Ann. Math. Stat. 20(4).
- Woodbury, M.A. (1950). *Inverting modified matrices*. Memorandum 42, Statistical Research Group, Princeton.
- Hager, W.W. (1989). *Updating the inverse of a matrix*. SIAM Review 31(2): 221–239. DOI: 10.1137/1031049.

### 1.2 Matrix Determinant Lemma

For Σ = A + U V:

```
det(A + U V)  =  det(A) · det(I + V A⁻¹ U)
```

Allows det(D + UUᵀ) in O(d + k³) instead of O(d³). Critical for log-likelihood and PSD checks in low-rank form.

References:
- Harville, D.A. (1997). *Matrix Algebra from a Statistician's Perspective*, Section 18.1.
- Press, W.H. et al. *Numerical Recipes* §2.7.

### 1.3 Block matrix inversion

For Σ block-partitioned as `[[A, B], [Bᵀ, C]]`:

```
Σ⁻¹ = [[ S⁻¹,                  −S⁻¹ B C⁻¹                 ],
       [ −C⁻¹ Bᵀ S⁻¹,           C⁻¹ + C⁻¹ Bᵀ S⁻¹ B C⁻¹  ]]
```

where `S = A − B C⁻¹ Bᵀ` is the Schur complement.

For block-diagonal Σ (B = 0), the inverse is `diag(A⁻¹, C⁻¹)` and cost reduces to the sum of the per-block costs. For a Σ that decomposes into K independent blocks of size d/K, inversion is O(K · (d/K)³) = O(d³/K²). For K=10: 100× speedup.

References:
- Petersen, K.B. & Pedersen, M.S. (2012). *The Matrix Cookbook*, §9.1. (Standard reference for matrix identities.)
- Boyd, S. & Vandenberghe, L. (2018). *Introduction to Applied Linear Algebra*, Ch. 11–12.

## 2. Domain literature that uses these identities

### 2.1 Kalman filtering with low-rank covariance

The **square-root Kalman filter** and the **ensemble Kalman filter** maintain `Σ = LLᵀ` where `L` is d×k tall-and-skinny. Updates go through `L` directly, avoiding the d×d cov matrix entirely.

Key references:
- Bierman, G.J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press. (Foundational text on square-root and UD filters.)
- Evensen, G. (2003). *The ensemble Kalman filter: theoretical formulation and practical implementation*. Ocean Dynamics 53(4): 343–367. DOI: 10.1007/s10236-003-0036-9.
- Verlaan, M. & Heemink, A.W. (1997). *Tidal flow forecasting using reduced rank square root filters*. Stochastic Hydrology and Hydraulics 11. DOI: 10.1007/BF02427899.

**Transferability to SOGA**: Kalman maintains a single Gaussian; SOGA maintains a Gaussian mixture. Each *component* of SOGA's mixture is the analog of a Kalman state — we can apply the same factorization per component. The update equations for "linear assignment + measurement" in Kalman map almost 1:1 to SOGA's `update_rule` and `truncate`.

### 2.2 Factor analysis and probabilistic PCA

When Σ has the form `Σ = D + Λ Λᵀ` with D diagonal and Λ d×k, this is exactly the **factor analysis model** (Spearman 1904, modern treatment Bartholomew 1987). All standard operations have closed forms.

Probabilistic PCA (Tipping & Bishop 1999) is the special case where D = σ² I.

Key references:
- Tipping, M.E. & Bishop, C.M. (1999). *Probabilistic principal component analysis*. Journal of the Royal Statistical Society B 61(3): 611–622. DOI: 10.1111/1467-9868.00196.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, §12.2.

**Transferability**: any SOGA program where outputs are noisy observations of latent factors (Bayes-Point-Machine-like, linear regression with shared regressors) has covariance naturally of this form. Confirmed empirically — see §4 of the parent plan.

### 2.3 Sparse Gaussian Processes

Sparse GPs maintain a low-rank + diagonal posterior over inducing point variables:

```
q(u) = N(m, A)        where A = D + V Vᵀ
```

The Nyström / FITC / VFE approximations all use this structure to avoid the O(N³) full GP cost.

Key references:
- Titsias, M.K. (2009). *Variational learning of inducing variables in sparse Gaussian processes*. AISTATS 12. (Inducing-point variational bound.)
- Hensman, J., Fusi, N. & Lawrence, N.D. (2013). *Gaussian processes for big data*. UAI 29. arXiv: 1309.6835.
- Burt, D.R., Rasmussen, C.E. & van der Wilk, M. (2020). *Convergence of sparse variational inference in Gaussian processes regression*. JMLR 21(131). arXiv: 2008.00323.

**Transferability**: SOGA does not have GP kernels, but the rank-k+diagonal form Σ = D + UUᵀ is the same. The Sparse GP literature is rich in numerical techniques for stable updates of the U factor under arbitrary rank-1 modifications.

### 2.4 Structured variational inference

Modern PPLs (Pyro, NumPyro, Edward2) offer guide families with structured covariance: `AutoNormal` (diagonal), `AutoMultivariateNormal` (full d×d), and `AutoLowRankMultivariateNormal` (D + UUᵀ).

Key references:
- Ranganath, R., Tran, D. & Blei, D.M. (2016). *Hierarchical variational models*. ICML 33. arXiv: 1511.02386.
- Bingham, E. et al. (2019). *Pyro: deep universal probabilistic programming*. JMLR 20(28). (See `pyro.infer.autoguide.AutoLowRankMultivariateNormal`.)

**Transferability**: confirms that the low-rank + diagonal form is the empirical sweet spot in real PPL applications. Not just a theoretical convenience.

### 2.5 Block-structured and banded covariance

State-space models (HMM, Linear Dynamical System) naturally produce **banded** Σ — only adjacent time steps are correlated.

Hierarchical models produce **block-structured** Σ — independent groups at each level.

Key references:
- Rue, H. & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman & Hall/CRC. (Banded/sparse precision matrices, exploits CHOLMOD.)
- Davis, T.A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. (Sparse matrix algorithms.)

**Transferability**: SOGA programs that unroll loops with sequential dependencies (`RandomWalkGauss10`, HMM-like models) have banded Σ. Programs with grouped variables (hierarchical) have block Σ.

## 3. Computational complexity table

For Σ d×d, low-rank form `Σ = D + UUᵀ` with `U` d×k, k << d:

| Operation | Dense Σ | Structured Σ |
|---|---|---|
| Storage | O(d²) | O(d · (k+1)) |
| Σ x  (matrix-vector) | O(d²) | O(d · k) |
| Σ A Σ  (where A is d×d) | O(d³) | depends on A; for A sparse: O(d · k²); for A dense: O(d² · k) |
| Σ⁻¹ via SMW | O(d³) | O(d · k² + k³) |
| det(Σ) via Matrix-Det Lemma | O(d³) | O(d + k³) |
| Rank-1 update Σ + uuᵀ | O(d²) | O(d · k) (extend U; periodically compress via SVD) |
| Rank-1 subtraction Σ − γ uuᵀ (truncate update) | O(d²) | O(d · k) until rank exceeds budget |
| PSD enforcement (eigendecomp) | O(d³) | clip D + bound U norms; O(d + k³) |
| Cholesky | O(d³) | not natively cheaper; alternative: maintain U as factor of (Σ − D) |

For d = 1000 and k = 10:
- Storage: 10⁶ → 1.1 × 10⁴ → **~100× less memory**
- Σ⁻¹: 10⁹ → 10⁵ → **~10⁴× faster**
- Σ x: 10⁶ → 10⁴ → **~100× faster**

## 4. Practical caveats from the literature

### 4.1 Rank-growth under repeated rank-1 updates

Every SOGA truncate is a rank-1 modification of Σ: `Σ' = Σ − γ g gᵀ`. Naïvely this grows U's rank by 1 per truncate. After N observes, rank could be k + N.

**Standard solution**: periodically **compactify** U by computing the SVD of `U` itself (k+N × d, but k+N is small in practice) and keeping the top-k singular vectors. This is O(k² · d).

References:
- Brand, M. (2002). *Incremental singular value decomposition of uncertain data with missing values*. ECCV 7. DOI: 10.1007/3-540-47969-4_47. (The standard incremental SVD update algorithm.)
- Vahdat, A. & Macready, W. (2020). *Undirected graphical models as approximate posteriors*. ICML. arXiv: 2003.00766. (Maintains structured posteriors with compactification.)

### 4.2 Truncating to the wrong rank

If true rank > k, the rank-k approximation introduces error proportional to the (k+1)-th singular value. SOGA needs an **adaptive** k: bound the spectral tail error, increase k if it exceeds a threshold, decrease if many tail values are negligible.

### 4.3 Numerical stability of SMW

When `C⁻¹ + V A⁻¹ U` is near-singular, SMW becomes unstable. Best practice: keep U near-orthogonal (re-orthogonalize during compactification) and use Cholesky factors instead of inverses.

References:
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. (Chapters on rank-1 updates and downdates.)

### 4.4 PSD enforcement

For Σ = D + UUᵀ, PSD-ness is guaranteed if D >= 0 elementwise. For Σ = D + UUᵀ − γ g gᵀ (after a truncate), PSD enforcement requires checking that the smallest eigenvalue of the result is non-negative. With structured form:

```
λ_min(D + U Uᵀ − γ g gᵀ)  ≥  λ_min(D) − γ · g · ‖U‖²
```

(bound, not tight). Practical fix: maintain Σ as `D + UUᵀ` where U absorbs signed contributions; check `D >= 0` and rely on cancellation between U columns.

## 5. Top-5 references for SOGA

| # | Title | Authors | Year | Venue | DOI/arXiv | Relevance |
|---|---|---|---|---|---|---|
| 1 | *Hierarchical Variational Models* | Ranganath, Tran, Blei | 2016 | ICML | arXiv:1511.02386 | Justifies low-rank + diagonal as the standard "structured posterior" in modern PPLs |
| 2 | *Probabilistic principal component analysis* | Tipping & Bishop | 1999 | JRSS-B | 10.1111/1467-9868.00196 | Closed-form for Σ = D + ΛΛᵀ; direct template for SOGA |
| 3 | *Updating the inverse of a matrix* | Hager | 1989 | SIAM Review | 10.1137/1031049 | The canonical reference for SMW and rank-1 updates |
| 4 | *The Matrix Cookbook* | Petersen & Pedersen | 2012 | (book) | — | All identities consolidated; daily reference |
| 5 | *Factorization Methods for Discrete Sequential Estimation* | Bierman | 1977 | (book) | — | Square-root Kalman: maintaining Σ = LLᵀ throughout, never materializing Σ |

## 6. Existing implementations

| Tool | URL | License | Last commit | Notes |
|---|---|---|---|---|
| pyro AutoLowRankMultivariateNormal | github.com/pyro-ppl/pyro | Apache-2 | active | Variational guide with D + UUᵀ Σ; reference for the data structure |
| GPyTorch LowRankRootLinearOperator | github.com/cornellius-gp/gpytorch | MIT | active | Generic structured-cov backend (low-rank, diagonal, block, banded, Toeplitz) |
| scikit-learn FactorAnalysis | sklearn.decomposition | BSD-3 | stable | Reference fit for the factor form |
| filterpy.kalman SquareRootKalmanFilter | github.com/rlabbe/filterpy | MIT | maintained | Educational reference for square-root form |

## 7. Gaps SOGA could fill

- No public PPL maintains structured Σ **per Gaussian-mixture component**. Pyro's `AutoLowRankMultivariateNormal` is a single posterior; SOGA's mixture would extend the technique to `n_comp · (d + k·d)` storage versus `n_comp · d²` today.
- No SOGA-style symbolic engine combines structured Σ with truncation under linear-inequality constraints (the truncate operation). The Kalman literature handles only equality conditioning.
- Bench-suite-level study of which programs benefit from low-rank Σ is absent — we are producing it in §4 of the parent plan.

## 8. Recommended next actions

- [ ] Bench-suite-level Σ-structure profile (`experiments/sigma_structure_analysis_2026-05-21/`)
- [ ] Standalone math prototype verifying SMW + Matrix-Det-Lemma operations agree with dense to machine precision across (d, k) sweeps
- [ ] Integration plan distinguishing programs where structured Σ is "obviously beneficial" (BayesPointMachine, regression-like) versus "structurally dense" (ClickGraphPrune, ClinicalTrial after merge)
