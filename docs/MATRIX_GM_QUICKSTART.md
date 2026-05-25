# Matrix-GM Quickstart

A 2-page user-facing guide to SOGA's matrix-variate Gaussian DSL extension.
For formal semantics see `docs/MATRIX_GM_SEMANTICS.md`.  For theory see
`docs/research-notes/04-matrix-element-extract-write.md` and `05-matrix-gm-merge-and-random-matmul.md`.

## TL;DR

SOGA now propagates **matrix-variate Gaussian mixtures** through a control-flow
graph.  Matrix variables `X ∼ MN(M, U, V)` are first-class citizens alongside
scalar variables.  Linear operations (matmul with deterministic kernel, add,
transpose) are **exact** in O(m²+n²) Kronecker storage.  Non-linear ops
(matrix × matrix random, element write, observe) use exact closed-form
2nd-moment computations followed by a single approximation step (nearest-
Kronecker projection or covariance densification).  Branching `if/else` and
loops with matrix indexing work natively.  All observe patterns
(inequality, equality, row/col sum, linear combination of elements,
back-prop on scalar extracted from matrix) are supported end-to-end.

## Quick run

```bash
# Basic showcase (8 sections, all-exact operations)
python3 src/SOGA.py -f programs/Example/matrix_gm_showcase.soga

# Advanced showcase (random × random matmul + element write)
python3 src/SOGA.py -f programs/Example/matrix_gm_advanced.soga
```

Both run in ≈ 1 second.  Output is printed to stdout: scalar moments
(`E[var]`) plus matrix moments (`E[X]` as a numpy-formatted 2D array).

## What was added (relative to scalar-only SOGA)

| Construct | DSL syntax | Status |
|-----------|-----------|--------|
| Matrix declaration | `matrix[m][n] X;` | new |
| Matrix-variate Gaussian init | `X = matrix_gm(M, U, V);` | new |
| 2D data | `data A = [[1,2],[3,4]];` | new |
| Matmul (data × random) | `Y = A @ X;`  or  `Y = X @ B;` | new (Kronecker exact) |
| Matrix add | `Y = X + N;` | new (iso path exact) |
| Transpose | `Y = transp(X);` | new (exact) |
| Scalar extraction | `y = X[i,j];` | new (cross-cov tracked) |
| Loop with matrix index | `for i in range(N) { ... X[i,j] ... }` | new |
| Branch with matrix in scope | `if cond { ... X ... } else { ... }` | new (merge concatenates components) |
| Random × random matmul | `Z = X1 @ X2;` | new (Isserlis + NKP rank-1) |
| Element write | `X[i,j] = expr;` | new (Schur, densifies) |
| Element observe (inequality) | `observe(X[i,j] > c)` | new |
| Element observe (equality) | `observe(X[i,j] == c)` | new |
| Aggregate observe | `observe(row_sum(X,i) op c)` , `observe(col_sum(X,j) op c)` | new |
| Linear-combo observe | `observe(2*X[0,0] + 3*X[1,1] > 1)` | new |
| Observe on extracted scalar | `y = X[i,j]; observe(y > 0);` | new (back-prop to E[X]) |
| Scalar SOGA semantics | unchanged | regression-tested |

## Walk-through of the showcase program

The file `programs/Example/matrix_gm_showcase.soga` is split into 8 commented
sections.  Read it top-to-bottom; each section is self-contained.

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

The advanced showcase (`matrix_gm_advanced.soga`) covers the two
approximation-bearing patterns:

| Pattern | Approximation | Empirical envelope |
|---------|---------------|---------------------|
| `Z = X1 @ X2` (random × random) | NKP rank-1 projection of an exact 3-term Isserlis covariance | ≤ 2 % Frobenius rel err on Cov in all tested regimes; mean exact |
| `X[i,j] = c` (element write) | none in math, but variable densifies (V⊗U → mn × mn) | exact Schur update; subsequent Kronecker-only ops fail |

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

You supply the row factor `U` (m×m) and column factor `V` (n×n) directly.
Storage cost: `m² + n²` entries.  All linear operations (matmul, add, transp,
scale) use the Kronecker fast path — exact and memory-efficient.

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

**Dense-mode limitations**: identical to post-element-write.
Subsequent `affine_left`, `transp`, `Y = X + N` on a dense-mode variable
raise `NotImplementedError`.  Element extract `y = X[i,j]` and
`observe(X[i,j] op c)` still work.

### Quick comparison

| | `matrix_gm(M, U, V)` | `matrix_gm_full(M, Sigma)` |
|--|--|--|
| What you provide | Row/column factors | Full (mn×mn) covariance |
| Student burden | Must pre-decompose Sigma → (U,V) | None — SOGA auto-detects |
| Storage (if Kronecker) | O(m²+n²) | O(m²+n²) after detection |
| Storage (if dense) | N/A — must be Kronecker | O((mn)²) |
| Linear ops | All fast-path | Kronecker-detected: fast; dense: NotImplementedError for Kronecker-ops |
| Recommended when | You know U, V explicitly | You have Sigma; unsure of separability |


## When NOT to use it (current v1 limits)

- **`trace(X)` observe**: not implemented as keyword; workaround: write
  out the diagonal sum manually `observe(X[0,0] + X[1,1] + ... > c)` — fully
  exact via the linear-combo path.
- **Variance back-prop on scalar observe**: only mean propagates back to
  `E[X]`.  `Var[X]` is NOT updated by `observe(y)` where `y = X[i,j]`
  (Opt-2 deferred; would require densification of `X`).
- **Kronecker-only ops on a dense-mode variable**: after `X[i,j] = c`,
  the variable is in dense covariance mode.  Subsequent `Y = A @ X`,
  `Y = transp(X)`, `Y = X + N` raise `NotImplementedError`.
  Element extract `y = X[i,j]` and observe `X[i,j] op c` still work.
- **Cross-covariance between two different matrix variables**: not tracked
  in v1.  After `Y = A @ X`, `Cov(vec(X), vec(Y))` is computed for cross-cov
  with extracted scalars but not stored as a matrix-matrix block.
- **Identifier names**: SOGA's IDV grammar does **not** allow underscores or
  hyphens.  Use `wcorner` or `wCorner`, not `w_corner`.
- **Declaration ordering**: all `data` declarations must precede matrix
  declarations and any assignments / observes / loops (SOGA's `progr` rule).

## Approximations in 1 paragraph

For the v1 matrix-GM extension, the only approximations introduced are
(1) **Van Loan-Pitsianis nearest-Kronecker rank-1 projection** applied
after a random × random matmul or after a merge of components whose
post-merge covariance is a sum of multiple Kronecker products
(empirical envelope ≤ 2 % Frobenius rel error on Cov in all MC-validated
regimes); (2) **mean-only back-propagation** when an observe fires on a
scalar derived from a matrix (the matrix mean is updated via the exact
Kalman gain; the covariance is intentionally not downdated to preserve
the Kronecker factorisation — full covariance update would require
densification of the matrix variable, which is the same machinery used
internally by the element-write path).  All other operations (linear,
observe on matrix elements / row sum / col sum / linear combinations /
equality, element write, merge, loop, extract) are **exact** for the
2nd-moment moments tracked by SOGA, matching the precision of scalar
SOGA on its operations.

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

After any rebase / merge / regenerate:

```bash
# Full test suite (must pass 349)
.venv/bin/python -m pytest tests/ -q

# Scalar regression spot checks
python3 src/SOGA.py -f programs/Example/Bernoulli.soga      # E[theta] = 0.25689
python3 src/SOGA.py -f programs/SOGA/ClickGraphPrune.soga    # E[simAll] = 0.61409

# Matrix-GM smoke (showcase + advanced)
python3 src/SOGA.py -f programs/Example/matrix_gm_showcase.soga
python3 src/SOGA.py -f programs/Example/matrix_gm_advanced.soga

# Grammar sync check
bash scripts/check_grammar_sync.sh
```

## Known issues

The stress test campaign (2026-05-25) uncovered four latent bugs in the
matrix-GM extension.  They are documented in full in
[`docs/LIMITATIONS.md`](LIMITATIONS.md).

**Quick reference (do not use these combinations until fixed):**

| Bug | Broken combination | Workaround |
|-----|--------------------|------------|
| C1 / Path 5 | `observe(X[i,j] op c)` followed by `Y = A @ X` or `Y = X @ B` | Apply affine ops before observe, or use a separate matrix variable |
| C7 | `matrix_gm_full(M, Sigma_dense)` followed by `Y = A @ X` or `Y = X @ B` | Use `matrix_gm(M, U, V)` if Kronecker factors are known; or restrict to extract/observe ops |
| Gap 3 | `y0 = X[0,0]` before `observe(X[1,0] > c)` with correlated rows | Extract scalars AFTER observe statements |
| F4 | `y = X[0,0]` before `X[0,0] = 5.0` (element write) | Extract scalars AFTER element writes if covariance is needed |

See [`docs/LIMITATIONS.md`](LIMITATIONS.md) for source locations, test probes, and
the follow-up plan reference.
