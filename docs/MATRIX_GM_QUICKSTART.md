# Matrix-GM Quickstart

User guide to the matrix-variate Gaussian extension of SOGA. Formal semantics live in `docs/MATRIX_GM_SEMANTICS.md`; theoretical derivations in `docs/research-notes/04-matrix-element-extract-write.md` and `05-matrix-gm-merge-and-random-matmul.md`; the catalogue of currently-known bugs in `docs/LIMITATIONS.md`.

## What this extension adds

Matrix variables `X ∼ MN(M, U, V)` are propagated symbolically through the same control-flow graph used by scalar SOGA. The Kronecker covariance `vec(X) ∼ N(vec(M), V ⊗ U)` is stored as the pair `(U, V)` whenever the operation preserves separability, which keeps memory at `O(m² + n²)` instead of `O((mn)²)`.

Linear operations against a deterministic kernel (`A @ X`, `X @ B`, `X + N`, `transp(X)`, `c * X`) are exact and preserve the Kronecker factorisation. Element extract `y = X[i,j]` is exact and tracks the cross-covariance `Cov(y, vec(X)) = V[:,j] ⊗ U[:,i]`. Observe statements on matrix elements, row/column sums, linear combinations, and `==` equality all reduce to a rank-1 Schur update on `vec(X)`.

Two operations carry an unavoidable approximation. Random × random matmul `Z = X1 @ X2` computes `Cov(vec(Z))` exactly via the Isserlis 3-term formula but the result is generally a sum of two Kronecker products, which we project onto the nearest single Kronecker via Van Loan–Pitsianis (rank-1 SVD on the rearrangement). Element write `X[i,j] = c` is exact via Schur but destroys Kronecker separability, so the covariance is densified to a full `(mn × mn)` matrix from that point on.

Branching, loops with matrix-indexed accesses, and component pruning behave as in scalar SOGA. Five latent bugs in the matrix-GM path are documented and exposed at runtime by `NotImplementedError` or `StaleCrossCovWarning`; see `docs/LIMITATIONS.md`.

## Running the showcase programs

```bash
# Baseline showcase: 9 sections, all-exact operations (no approximation)
python3 src/SOGA.py -f programs/Example/matrix_gm_showcase.soga

# Advanced showcase: random × random matmul, element write, dense matrix_gm_full
python3 src/SOGA.py -f programs/Example/matrix_gm_advanced.soga
```

Each runs in roughly one second. Output is printed to stdout: scalar moments as `E[var]: <value>` lines, matrix moments as `E[X]:` followed by a numpy-formatted 2D array.

## What was added (relative to scalar-only SOGA)

| Construct | DSL syntax | Notes |
|-----------|-----------|--------|
| Matrix declaration | `matrix[m][n] X;` | |
| Kronecker constructor | `X = matrix_gm(M, U, V);` | exact, `O(m² + n²)` storage |
| Full-covariance constructor | `X = matrix_gm_full(M, Sigma);` | auto-detects Kronecker structure |
| 2D data | `data A = [[1,2],[3,4]];` | |
| Matmul against deterministic kernel | `Y = A @ X;`, `Y = X @ B;` | exact, Kronecker preserved |
| Matrix add | `Y = X + N;` | exact on the isotropic path |
| Transpose | `Y = transp(X);` | exact |
| Scalar extract | `y = X[i,j];` | exact, cross-cov tracked |
| Loop with matrix index | `for i in range(N) { ... X[i,j] ... }` | runtime index resolution |
| Branch with matrix in scope | `if cond { ... X ... } else { ... }` | merge concatenates components |
| Random × random matmul | `Z = X1 @ X2;` | Isserlis (exact 2nd moment) + NKP projection |
| Element write | `X[i,j] = expr;` | exact via Schur; covariance densifies |
| Element observe (inequality) | `observe(X[i,j] > c)` | exact rank-1 Schur update |
| Element observe (equality) | `observe(X[i,j] == c)` | hard conditioning |
| Aggregate observe | `observe(row_sum(X,i) op c)`, `observe(col_sum(X,j) op c)` | exact |
| Linear-combination observe | `observe(2*X[0,0] + 3*X[1,1] > 1)` | exact |
| Observe on extracted scalar | `y = X[i,j]; observe(y > 0);` | mean-only back-prop to `E[X]` |
| Scalar SOGA semantics | unchanged | regression-tested across 385 unit tests |

## Walkthrough of the showcase program

The file `programs/Example/matrix_gm_showcase.soga` is split into 9 commented sections, each self-contained. Read it top to bottom.

| Section | Pattern | Storage cost | Approximation |
|---------|---------|--------------|---------------|
| 0 | `data A = [[2,0],[0,1]];` | static | none |
| 1 | `matrix[2][2] X;` + `matrix_gm(M, U, V)` | O(m²+n²) Kronecker | none |
| 2 | `Y = A @ X` (deterministic kernel) | O(m²+n²) | none |
| 3 | `Y = Y + N` (iso noise) | O(m²+n²) | iso path: none |
| 4 | `Z = transp(Y)` | O(m²+n²) | none |
| 5 | `y00 = Y[0,0]` (scalar extract with cross-cov) | O(mn) per scalar | none |
| 6 | `observe(y00 > 0)` (back-prop to Y mean) | scalar truncate | A2-mean only (Var[Y] not updated) |
| 7 | `if theta > 0 {...} else {...}` (merge) | components multiply | none on per-component |
| 8 | `for i { diag = X[i,i] }` (loop index) | scalar extract per iter | none |

The advanced showcase (`matrix_gm_advanced.soga`) covers the two patterns that carry an approximation:

| Pattern | Approximation | Empirical envelope |
|---------|---------------|---------------------|
| `Z = X1 @ X2` (random × random) | rank-1 NKP projection of the exact 3-term Isserlis covariance | Frobenius relative error on Cov < 2% in all tested regimes; mean exact |
| `X[i,j] = c` (element write) | none in the math, but `V ⊗ U` densifies to `(mn × mn)` | Schur update exact; subsequent Kronecker-only ops raise `NotImplementedError` |

## DSL cheatsheet (one screen)

```soga
/* Data (must come FIRST in the program) */
data scalars1d = [1.0, 2.0, 3.0];        /* 1D list */
data kernel2d  = [[1,0],[0,1]];          /* 2D matrix */

/* Matrix declarations (after data, before assignments) */
matrix[m][n] X;
matrix[m][n] Y;

/* Initialise matrix variable */
X = matrix_gm(
    [[m11,m12],[m21,m22]],               /* M (m × n) */
    [[u11,0],[0,u22]],                   /* U (m × m, symmetric PSD) */
    [[v11,0],[0,v22]]                    /* V (n × n, symmetric PSD) */
);

/* Linear operations (Kronecker-exact) */
Y = kernel2d @ X;                         /* deterministic-left matmul */
Y = X @ kernel2d;                         /* deterministic-right matmul */
Y = X + Y;                                /* matrix add */
Y = transp(X);                            /* transpose */
Y = 2 * X;                                /* scalar scale */

/* Scalar from matrix (cross-cov preserved) */
y = X[0,1];

/* Observe (all patterns) */
observe(X[i,j] > 0);                      /* element inequality */
observe(X[i,j] == 1);                     /* element equality (hard) */
observe(row_sum(X, 0) > 0);               /* row aggregation */
observe(col_sum(X, 0) <= 5);              /* col aggregation */
observe(2 * X[0,0] + 3 * X[1,1] > 1);     /* linear combination */
observe(y > 0);                            /* scalar derived from matrix
                                              → back-prop to E[X] */

/* Element write (densifies the matrix's covariance) */
X[i,j] = 5;                                /* deterministic value */
X[i,j] = z;                                /* scalar variable */

/* Control flow with matrices in scope */
theta = gm([0.5,0.5], [-1,1], [0.1,0.1]);
if theta > 0 {
    pick = X[0,0];
} else {
    pick = X[1,1];
} end if;
prune(10);                                /* bound K ≤ 10 after merge */

for i in range(2) {
    d = X[i,i];
} end for;
```

## Specifying covariance: Kronecker vs Full

SOGA offers two constructors for matrix-variate Gaussian priors.  Choose based
on how you know (or want to express) the covariance structure.

### Form 1 — Explicit Kronecker factors (original constructor)

```soga
X = matrix_gm(M, U, V);     /* vec(X) ~ N(vec(M), V ⊗ U) */
```

You supply the row factor `U` (m×m) and column factor `V` (n×n) directly. Storage cost is `m² + n²` entries. Every linear operation (matmul, add, transpose, scale) takes the Kronecker fast path: exact in closed form, memory-efficient.

Use this form when:
- You know the row/column factorisation from the model structure
  (e.g. spatial row covariance × temporal column covariance).
- You want the smallest memory footprint.

### Form 2 — Full covariance with auto-detection (new in v1.2)

```soga
X = matrix_gm_full(M, Sigma);   /* vec(X) ~ N(vec(M), Sigma) */
```

You supply the full `(mn × mn)` covariance matrix `Sigma`.
SOGA auto-detects whether `Sigma` is Kronecker-separable via Van Loan-Pitsianis
rank-1 SVD on the rearrangement `R[Sigma]`.

| Sigma structure | Residual `s₂/s₁` | Storage | Info emitted |
|-----------------|------------------|---------|--------------|
| Exact `V ⊗ U` | < `SOGA_KRON_STRICT` (1e-8) | `(U, V)` factors | `KroneckerDetectionInfo` |
| Near-Kronecker | in `[1e-8, 1e-3)` | dense `(None, Sigma)` | `KroneckerNearMissWarning` |
| Truly dense | ≥ `SOGA_KRON_LOOSE` (1e-3) | dense `(None, Sigma)` | `DenseCovarianceInfo` |

The thresholds are configurable via environment variables:
```bash
export SOGA_KRON_STRICT=1e-10   # stricter Kronecker detection
export SOGA_KRON_LOOSE=0.01     # wider near-miss band
```

**Example — Kronecker-detected (I₄ = I₂ ⊗ I₂):**
```soga
matrix[2][2] X;
X = matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
/* SOGA detects Sigma = I_4 = I_2 ⊗ I_2; stores (I_2, I_2) — Kronecker fast path */
```

**Example — Dense (non-Kronecker off-diagonal coupling):**
```soga
matrix[2][2] W;
W = matrix_gm_full(
    [[1,0],[0,2]],
    [[2,0,0,0.5],[0,1,0,0],[0,0,1,0],[0.5,0,0,2]]  /* Cov(W[0,0],W[1,1])=0.5 */
);
/* SOGA detects non-Kronecker; stores dense sentinel (None, Sigma) */
/* DenseCovarianceInfo or KroneckerNearMissWarning emitted */
```

**Dense-mode limitations**: identical to post-element-write. Affine ops (`A @ X`, `X @ B`), scale (`c * X`), transpose, and matrix add on a dense-mode variable now raise `NotImplementedError("[C1/C7] ...")` with an explicit message pointing back to `docs/LIMITATIONS.md`. Element extract `y = X[i,j]` and `observe(X[i,j] op c)` still work on the dense path.

### Quick comparison

| | `matrix_gm(M, U, V)` | `matrix_gm_full(M, Sigma)` |
|--|--|--|
| What you provide | Row/column factors | Full (mn×mn) covariance |
| Student burden | Must pre-decompose Sigma into (U,V) | None (SOGA auto-detects) |
| Storage (if Kronecker) | O(m²+n²) | O(m²+n²) after detection |
| Storage (if dense) | N/A (must be Kronecker) | O((mn)²) |
| Linear ops | All fast-path | Kronecker-detected: fast; dense: NotImplementedError for Kronecker-ops |
| Recommended when | You know U, V explicitly | You have Sigma; unsure of separability |


## What v1 does not support

- **`trace(X)` as an observe keyword**: not exposed. Workaround: expand the diagonal sum by hand, `observe(X[0,0] + X[1,1] + ... > c)`. The linear-combination path handles this exactly.
- **Variance back-prop on scalar observe**: when `observe(y)` fires on `y = X[i,j]`, only the mean of `X` is updated (exact Kalman gain). `Cov(vec(X))` is intentionally not downdated; a full downdate would densify `X` and lose the Kronecker storage.
- **Kronecker-only ops on a dense-mode variable**: after `X[i,j] = c` or `X = matrix_gm_full(M, Sigma_dense)`, calls to `A @ X`, `X @ B`, `transp(X)`, `c * X`, and `X + N` raise `NotImplementedError("[C1/C7] ...")`. Extract and observe still work on the dense path.
- **Cross-covariance between two distinct matrix variables**: not tracked. After `Y = A @ X`, the cross `Cov(vec(X), vec(Y))` is computed when an extracted scalar needs it but is not stored as a matrix-to-matrix block.
- **Identifier names**: the SOGA `IDV` lexer rule does not accept underscores or hyphens. Use `wcorner` or `wCorner`, never `w_corner`.
- **Declaration ordering**: every `data` declaration must precede the first matrix declaration, assignment, observe, or loop, per the `progr` rule in `grammars/SOGA.g4`.

## Where the approximations are

Two places introduce approximation. The first is the nearest-Kronecker rank-1 projection (Van Loan and Pitsianis 1993) applied after random × random matmul `Z = X1 @ X2` and after sums of components whose covariance is itself a sum of distinct Kronecker products. The exact `Cov(vec(Z))` is a rank-2 element in the rearrangement-SVD sense, so projecting to rank 1 incurs `||proj - exact||_F / ||exact||_F`. Across the four regimes we have validated against Monte Carlo (20k samples), the Frobenius relative error stays below 2% in the small-cov and moderate-cov regimes and reaches 1.3% in the balanced regime that previously produced 19.7% under the older delta-method formula.

The second is the mean-only back-propagation when `observe(y > c)` fires on a scalar `y` extracted from a matrix `X`. SOGA updates `E[X]` via the exact Kalman gain but does not downdate `Cov(vec(X))`; updating the covariance would require densifying `X` to `(mn × mn)`, which is the same cost as element write. This omission is deliberate (it preserves the Kronecker factorisation across subsequent operations), and it matches the precision profile of scalar SOGA where second-moment downdate is also approximated.

Everything else (linear operations, element observe of any form, element write, merge, loop with matrix index, scalar extract) is exact for the second moments that SOGA propagates.

## Where to learn more

| Topic | File |
|-------|------|
| Formal semantics (15 §, one per construct, closed-form formulas) | `docs/MATRIX_GM_SEMANTICS.md` |
| Element extract + write theory + Schur formula derivation | `docs/research-notes/04-matrix-element-extract-write.md` |
| Merge + random×random theory + Isserlis derivation | `docs/research-notes/05-matrix-gm-merge-and-random-matmul.md` |
| Implementation plan with 5 milestones (M1–M6) | `plan/2026-05-22-matrix-gm-lishan.md` |
| Implementation plan for 4 CRASH fixes | `plan/2026-05-24-matrix-gm-complete.md` |
| Grammar regeneration | `Manual/ReplicabilityGuide.md § Grammar Regeneration` |

## Reproduction (regression-validated 2026-05-25)

After any rebase, merge, or grammar regeneration, run:

```bash
# Full test suite: 491 tests = 385 (scalar PPL + matrix-GM internals) + 106 (stress campaign)
.venv/bin/python -m pytest tests/ -q

# Scalar regression spot checks
python3 src/SOGA.py -f programs/Example/Bernoulli.soga          # E[theta] = 0.25689
python3 src/SOGA.py -f programs/SOGA/ClickGraphPrune.soga       # E[simAll] = 0.61409

# Matrix-GM smoke battery
python3 src/SOGA.py -f programs/Example/matrix_gm_showcase.soga  # E[y00]=2.57684, E[xfull00]=5.0
python3 src/SOGA.py -f programs/Example/matrix_gm_advanced.soga  # E[wcorner]=5.0, E[wd00]=1.0

# Grammar regeneration sanity
bash scripts/check_grammar_sync.sh
```

## Known issues

The stress test campaign (May 2026) and the safety-patches campaign that followed it documented five latent bugs in the matrix-GM path. Four of them have a runtime signal (a `NotImplementedError` with an explicit message, or a `StaleCrossCovWarning`); the fifth (C8) is locked by a regression test but has no runtime signal yet. Full details, including source locations, test probes, and the planned fix, are in [`docs/LIMITATIONS.md`](LIMITATIONS.md).

| Bug | Trigger | Workaround |
|-----|---------|------------|
| C1 (= Path 5) | `observe(X[i,j] op c)` followed by `Y = A @ X` or `Y = X @ B` | Apply affine before observe, or split into two matrix variables |
| C7 | `matrix_gm_full(M, Sigma_dense)` followed by affine, scale, or transpose | Use `matrix_gm(M, U, V)` when factors are known; otherwise restrict to extract and observe |
| Gap 3 | `y0 = X[0,0]; observe(X[1,0] > c)` with off-diagonal U | Extract scalars after, not before, observe statements |
| F4 | `y = X[0,0]; X[0,0] = c` (write after extract) | Extract scalars after element writes when covariance is needed downstream |
| C8 | `observe(X[i,j] op c)` and then reading `E[X[i',j']]` (i' ≠ i or j' ≠ j) with correlated U | Read only the observed element, or pre-derive correlated elements analytically. Sign error in dense Kalman back-prop |

Programs that stay within the safe envelope (only `matrix_gm`, only affine plus extract plus observe on the same element) hit none of the five.
