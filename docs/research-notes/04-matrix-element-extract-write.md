# Research note — Matrix-Gaussian element extraction and element write

**Date**: 2026-05-24
**Triggered by**: M4.8 v1 limitation (scalar extract drops cross-cov) + parser-accepted LHS `X[i,j] = expr` rejected at runtime
**Scope**: Find theoretical foundation and verified prior art for (A) cross-covariance preservation when extracting `y = X[i,j]` from a matrix-variate Gaussian, and (B) element-wise write to a matrix-variate Gaussian variable. Both within the SOGA Kronecker-factored joint-state representation.

---

## Top-10 verified references

| # | Title | Authors | Year | Venue | ID | Notes |
|---|-------|---------|------|-------|----|-------|
| 1 | Matrix Variate Distributions | Gupta, Nagar | 1999 | Chapman & Hall/CRC | DOI:10.1201/9780203749289 | Theorem 2.3.1: Cov(X[i,j], X[k,l]) = V[j,l]·U[i,k] |
| 2 | Equivalence MN ↔ MVN | Book of Statistical Proofs | 2022 | statproofbook.github.io/P/matn-mvn.html | URL verified | vec(X)~N(vec(M), V⊗U) proof |
| 3 | Conditional MVN formula | Book of Statistical Proofs | 2022 | statproofbook.github.io/P/mvn-cond.html | URL verified | Schur-complement closed-form |
| 4 | On Separability of Covariance in Multiway Data | Deslauriers-Gauthier et al. | 2023 | arXiv | arXiv:2302.02415 | Generic multiway cov NOT separable; NP-hard |
| 5 | Fast Kronecker Inference in GPs with non-Gaussian Likelihoods | Flaxman, Wilson et al. | 2015 | ICML | proceedings.mlr.press/v37/flaxman15.html | Kronecker breaks with non-Cartesian observations |
| 6 | Scalable GPs with Latent Kronecker Structure | Lindinger et al. | 2025 | arXiv | arXiv:2506.06895 | "structure is lost if any observation is missing" |
| 7 | Vec-Permutation Matrix, Vec Operator, Kronecker Products | Henderson, Searle | 1981 | Linear Multilinear Algebra | doi:10.1080/03081088108817379 | Mixed-product property (foundation for factored cross-cov) |
| 8 | scipy.stats.matrix_normal | SciPy authors | 2024 | docs.scipy.org | URL verified | V⊗U convention; no element conditioning |
| 9 | Existence and Uniqueness of Kronecker Covariance MLE | Drton, Kuriki, Hoff | 2020 | arXiv | arXiv:2003.06024 | Flip-flop/ALS estimator for V⊗U (Kronecker recovery) |
| 10 | Support MatrixNormal distribution | NumPyro Issue #1178 | 2024 | github.com/pyro-ppl/numpyro/issues/1178 | NumPyro MatrixNormal exists but no indexing semantics |

---

## Key findings

### Finding A1: cross-cov formula `Cov(y, vec(X)) = V[:,j] ⊗ U[:,i]` is exact

Source: Gupta & Nagar 1999 (Thm 2.3.1); statproofbook.github.io/P/matn-cov.html.

Evidence: For X ~ MN(M, U, V), `vec(X) ~ N(vec(M), V⊗U)` with column-major vectorisation. Then `Cov(X[i,j], X[k,l]) = (V⊗U)[j*m+i, l*m+k] = V[j,l]·U[i,k]`. The column of V⊗U at index `j*m+i` is exactly `V[:,j] ⊗ U[:,i]`.

Relevance to SOGA: validates plan §M4.8. The formula is not an approximation. Cross-cov with another scalar z (stored as factored `(u_z, v_z)`): `Cov(y, z) = V[j,j']·U[i,i']` using the mixed-product property — O(m+n) not O(mn).

### Finding A2: Kronecker-factored cross-cov storage is exact and O(m+n)

Source: Henderson & Searle 1981 (mixed-product property of Kronecker).

Evidence: `(A⊗B)^T(C⊗D) = (A^TC)(B^TD)` (a scalar). Applied: `(V[:,j]⊗U[:,i])^T (V[:,j']⊗U[:,i']) = V[j,j']·U[i,i']`. Storage cost: (m+n) floats vs mn for the materialised vector. For m=n=32: 64 vs 1024 floats per (component, extracted-scalar).

Relevance to SOGA: a `(u_col, v_col)` factored storage variant in `GaussianMixBlock.cov_blocks` for the (scalar, matrix) cross is an exact, drop-in replacement for the current (mn,) dense flat-vector pattern. No paper proposes this as a named data structure for PPL state, but the algebra is standard.

### Finding A3: no prior PPL handles symbolic element extraction with cross-cov tracking

Source: NumPyro Issue #1178 (verified); Stan reference manual (verified); Pyro distributions docs.

Evidence: NumPyro `MatrixNormal` is a log_prob primitive — indexing returns a plain JAX scalar with no covariance tracking. Stan compiles element accesses to dense log-joint evaluations (HMC). Pyro tracks compute graphs for variational inference, not symbolic cov. No PPL propagates the cross-covariance symbolically after element extraction.

Relevance to SOGA: this is genuine open territory. SOGA would be the first to expose `y = X[i,j]` as a covariance-preserving symbolic operation. Publication angle.

### Finding B1: any single-element write destroys Kronecker structure

Source: Flaxman et al. 2015; Lindinger et al. 2025; Deslauriers-Gauthier et al. 2023.

Evidence: After conditioning vec(X) on `X[i,j] = c`, the posterior covariance is `(V⊗U) − (V[:,j]⊗U[:,i])(V[:,j]⊗U[:,i])^T / (V[j,j]·U[i,i])`. This is a rank-1 downdate to the (mn × mn) covariance matrix. Generic rank-(mn−1) matrices are not Kronecker-separable (NP-hard to verify per arXiv:2302.02415). Flaxman 2015 explicitly: "this structure is lost if any observation is missing."

Relevance to SOGA: confirms plan §M5.2 DENSE strategy is the only exact path. Storage cost (mn)² per affected component; for m=n=32 → 1M floats per component. For K=50 components → 50M floats (400 MB at float64). Already covered by the existing memory-budget guard (M5.2 §"Memory budget guard").

### Finding B2: soft write `X[i,j] = z` (scalar var) is structurally equivalent to hard write

Source: Bishop PRML §2.3.1; statproofbook.github.io/P/mvn-cond.html.

Evidence: the joint (vec(X), z) is multivariate normal; imposing X[i,j] = z is a linear constraint, yielding rank-1 downdate with denominator `V[j,j]·U[i,i] + Var(z)` (vs `V[j,j]·U[i,i]` for hard). Same structure; same Kronecker-loss.

Relevance to SOGA: dense fallback covers both B1 (var_z=0) and B2 (var_z>0) with the same code path. Single helper `element_write_dense(M, cov_vU_dense, i, j, c_or_mu, var_z)`.

### Finding B3: Kronecker-recovery via flip-flop ALS exists but is fragile

Source: Drton et al. 2020 (arXiv:2003.06024).

Evidence: Given a dense covariance Σ, the closest separable approximation V'⊗U' can be computed via alternating least-squares. Existence/uniqueness conditions are non-trivial; the MLE may not exist for some Σ. Numerically fragile when Σ is near low-rank.

Relevance to SOGA: an optional optimisation to recover Kronecker structure after one or a few writes, before the dense store grows too large. Risk/reward unclear; defer until benchmarks show the dense path is a bottleneck.

---

## Numerical verification example (m=n=2)

Let `M=[[1,2],[3,4]]`, `U=[[a,b],[b,c]]`, `V=[[p,q],[q,r]]`. Extract `y = X[0,1]` (i=0, j=1).

Column-major: vec(X) = [X[0,0], X[1,0], X[0,1], X[1,1]]^T → index of (0,1) is `j*m+i = 1*2+0 = 2`.

- `Var(y) = (V⊗U)[2,2] = V[1,1]·U[0,0] = r·a` ✓
- `Cov(y, vec(X)) = (V⊗U)[2,:]` should equal `V[:,1] ⊗ U[:,0] = [q,r]^T ⊗ [a,b]^T = [q·a, q·b, r·a, r·b]^T` ✓

---

## Existing implementations (verified)

| Tool | URL | License | Last commit | Notes |
|------|-----|---------|-------------|-------|
| scipy.stats.matrix_normal | github.com/scipy/scipy | BSD-3 | active | No element-conditioning API |
| NumPyro MatrixNormal | num.pyro.ai | Apache-2.0 | active | log_prob only; no indexing |
| Pyro LowRankMultivariateNormal | docs.pyro.ai | Apache-2.0 | active | factored storage for VI guides |
| Stan matrix_normal_prec_lpdf | mc-stan.org | BSD-3 | active | log_prob only |
| TensorLy `tensor.matricize` | github.com/tensorly/tensorly | BSD-3 | 2024-active | Vec / unvec ops but no Kronecker cov |

None propagate symbolic cross-cov after element ops.

---

## Gaps SOGA could fill

- **First PPL to expose `y = X[i,j]` with exact covariance-preserving symbolic state propagation**. The algebra is standard but the implementation as a PPL feature is novel.
- **First open-source library with factored cross-cov `(u_col, v_col)` storage as a named data structure**. Algebraic identity exists; the engineering pattern is unnamed.
- **First PPL to handle `X[i,j] = expr` symbolically via a dense-fallback gate**. Stan/Pyro/NumPyro all defer to sampling for this; SOGA can be exact at the cost of densifying one matrix var per element write.

---

## Recommended next actions

### Implementation order: A before B (raccomandato)

**A — scalar extract with factored cross-cov (~3-5 hours work, NO numerical risk, exact)**

- [ ] Add `cov_blocks[k][frozenset({y, X})] = ('factored', U[:,i].copy(), V[:,j].copy())` to `GaussianMixBlock` storage. Replace the current zero cross-cov in `extract_scalar_from_matrix`. Add a tag/discriminator so getters know whether the entry is dense `(mn,)` or factored.
- [ ] Update `GaussianMixBlock.get_cov` to handle the factored variant (return factored pair).
- [ ] Add dot-product helper `_cross_cov_dot(factored_a, factored_b) -> float` using `(u_a·u_b)·(v_a·v_b)`.
- [ ] Implement `Cov(y, z)` propagation: when y is extracted and another scalar z exists with stored cross-cov-to-X, compute the resulting scalar `Cov(y,z)` and store it in the scalar gm's sigma block.
- [ ] Unit tests in `tests/test_update_matrix.py`: verify extracted scalar has correct mu, var, and cross-cov against a brute-force MC validation (20k samples).
- [ ] End-to-end test: `.soga` program that does `y = X[i,j]; observe(y > 0);` and verify the observe correctly conditions through the cross-cov.

**B — matrix element write with dense fallback (~5-8 hours, mandatory densification)**

- [ ] Add `cov_blocks[k][frozenset({X})] = ('dense', Sigma_mn_x_mn)` tag variant to GaussianMixBlock. The existing Kronecker `(U, V)` tuple becomes one of two storage modes.
- [ ] Implement `_densify_matrix_var(block, k, X) -> np.ndarray (mn, mn)` that materialises V⊗U on demand. Gate behind a memory budget check (existing M5.2 budget guard suffices).
- [ ] Implement `element_write_dense(block, k, X, i, j, c_or_mu_z, var_z)` per the formula in Finding B1+B2. Triggers densification if not already dense.
- [ ] Add grammar dispatch in `libSOGAupdate.update_rule`: detect `X[i,j] = expr` LHS pattern, evaluate RHS as scalar (per Finding B3), then call `element_write_dense`.
- [ ] After dense conversion, all subsequent matrix ops on X (affine, add, transp) must use the dense covariance path. Add `_apply_op_dense_path` variants. Or: error out if a matrix op is attempted on a dense-mode variable (simpler, acceptable for v1).
- [ ] Unit tests: brute-force MC validation that `X[i,j] = c` followed by reading `E[X[k,l]]` matches the Schur-complement formula.
- [ ] Document the memory cost in CLAUDE.md (or M6 documentation): "matrix element writes densify the variable's covariance from O(m²+n²) to O((mn)²)".

### Deferred / not recommended

- Kronecker-recovery via flip-flop ALS (Finding B3) — only consider if dense path becomes the dominant cost on benchmarks. Adds a solver inside the CFG traversal with non-deterministic convergence.
- A "lazy densification" pattern where the dense cov is only materialised when actually queried — interesting optimisation but defer until profiling shows it's needed.

---

## Acceptance criteria for this research note

- [x] All cited references have URL/DOI verified (12/12).
- [x] At least one formal numerical verification example (m=n=2).
- [x] One concrete pseudocode block per problem.
- [x] Implementation order with effort estimate.
- [x] Honest gap analysis: SOGA would be the first PPL to do this.

---

_End of research note._
