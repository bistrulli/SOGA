# SOGA — Enhancement Notes & Roadmap

This document collects the findings, improvement opportunities, and design
discussions accumulated during the optimization work on branch
`feat/sparse-truncate-impl` (May 2026). It is intended as a living roadmap
for evolving SOGA from a research prototype into a more general-purpose
inference engine.

The document is organized in four sections:

- **§1** — what has already been implemented and shipped on this branch
- **§2** — open improvements, classified by layer (architecture, performance,
  robustness, packaging)
- **§3** — directions explored and intentionally rejected, with the reasoning
  preserved so future work does not re-derive the same conclusions
- **§4** — proposed research / product roadmap

---

## 1. Implemented in this iteration

Two CLI flags and a comprehensive A/B/C benchmark were added.

### 1.1 `--sparse-truncate`

Replaces the per-component SVD rotation in `libSOGAtruncate.ineq_func` with a
rank-1 conditional Gaussian update. Mathematically equivalent (verified to
machine epsilon on 7200 random samples in
`experiments/sparse_truncate_prototype_2026-05-20/`). Reduces per-component
cost from O(d^3) to O(d^2). Confirmed externally: matches Algorithm 1
("Recursive PDF Truncation") of Perälä & Ali-Löytty, MoMM 2008.

Files: `src/libSOGAtruncate.py` (`_ineq_func_sparse`,
`_truncated_normal_moments_1d`), `src/SOGA.py`, `src/libSOGA.py`.

### 1.2 `--vectorize-truncate`

Eliminates the per-component Python loop in `truncate()`. Stacks all GM
components into `(n_comp, d, d)` tensors and applies the rank-1 update with a
single batched numpy call. Implies `--sparse-truncate`. Numpy BLAS handles
SIMD and multi-core automatically.

Files: `src/libSOGAtruncate.py` (`_truncate_vectorized`,
`_ineq_truncate_vectorized_impl`, `_eq_truncate_vectorized_impl`,
`_truncated_normal_moments_1d_batched`).

### 1.3 Test suite

51 pytest tests across `tests/test_sparse_truncate.py` and
`tests/test_vectorize_truncate.py`. Cover equivalence on random inputs (4
shapes × dimensions × n_comp × directions), aux `gm()` variables,
per-component delta substitution, n_comp=1 and n_comp=2000 stress, equality
conditioning, end-to-end CLI on Bernoulli/BayesPointMachine/TrueSkills.

### 1.4 A/B/C benchmark

Three benchmark scripts in
`experiments/sparse_truncate_prototype_2026-05-20/`:

- `bench_all_canonical.py` — 2-way (classic vs sparse) on 24 canonical
  programs
- `bench_all_3way.py` — 3-way (classic vs sparse vs vectorize)
- `bench_failed_isolated.py` — isolated retest of programs that timed out
  inside the bench, with longer wall-clock budget

Key aggregate results (geometric mean over 21 programs that completed all
modes):

- `--sparse-truncate` over default: **1.51×**
- `--vectorize-truncate` over default: **3.17×**
- `--vectorize-truncate` over `--sparse-truncate`: **2.10×** (median 1.40×,
  max 12.66×)

Standout cases: ClinicalTrial 12.66×, Bernoulli 11.10×, RandomWalkUnif10
(timeout → 7 s) 7.24×, RandomWalkGauss10 4.19×. Equivalence: max relative
diff `E[var]` across modes = 2.78e-07.

### 1.5 Documentation

Updated `README.md`, `Manual/ReusabilityGuide.md`, and
`Manual/ReplicabilityGuide.md` with the two new CLI flags, the performance
considerations section, and the full benchmark table.

---

## 2. Open improvements

### 2.1 Architectural debt (highest-leverage to fix)

These items are not user-visible but they gate every future optimization and
the transition to a real product.

#### 2.1.1 CFG with mutable global state

Nodes carry mutable attributes: `node.dist`, `node.p`, `node.trunc`,
`node.list_dist`. One call to `start_SOGA(cfg, …)` mutates `cfg` in place. A
second call on the same CFG produces stale data — this is the root cause of
the `DigitRecognition.soga` `TypeError` on in-process repeated runs (state
leak in `data[node.idx][0]` for the loop counter).

Recommended fix: separate the static CFG from a per-run `ExecutionContext`
that holds the mutable state. CFG becomes immutable and reusable; running
the same model on different inputs becomes cheap.

Side benefit: precondition for path-level parallelism (workers cannot
safely share mutable nodes).

#### 2.1.2 Module-level mutable state for flags

`USE_SPARSE_TRUNCATE`, `USE_VECTORIZE_TRUNCATE`, and `pool` are module
globals in `libSOGAtruncate.py`. Not thread-safe, not reproducible in
parallel test execution, awkward to debug.

Recommended fix: pass a configuration object through `start_SOGA` and the
dispatcher functions; no module globals.

#### 2.1.3 No clean Python API

The only entry point is the CLI. Programmatic use requires
`subprocess.run`. There is no `import soga; result = soga.analyze(program)`
surface.

Recommended fix: design a Python API around `Model.compile(source)` →
`Result = Model.run(options)`. The CLI becomes a thin wrapper. This is the
single most impactful product-transformation change.

#### 2.1.4 Multiprocessing wrapper is imposed

`SOGA.SOGA()` always spawns a `Process` with `Queue` for the timeout. Adds
~1 s of startup overhead per invocation. Wasted on batch use, on benchmark
harnesses, on integration tests.

Recommended fix: timeout becomes optional; default is in-process execution.
The Process+Queue wrapping is opt-in via `--timeout`.

#### 2.1.5 `from X import *` cascade

`libSOGA.py` does `from libSOGAtruncate import *` which transitively imports
`libSOGAshared.*`. Hides dependencies, breaks tooling, makes refactoring
fragile.

Recommended fix: explicit imports, `__all__` per module.

#### 2.1.6 ANTLR parser re-executed at each truncate

Every `observe(LBC)` re-runs the ANTLR parser on the LBC string. Profiling
on `BayesPointMachine.soga` (`experiments/feasibility_dead_var_pruning_2026-05-20/profiles/V1.cumulative.txt`)
showed ~88 % of SOGA runtime in ANTLR parsing for small-d programs.

Recommended fix: precompile each LBC at CFG construction time and attach
the resulting function to the node. Same for `update_rule` ASGMT parsing.
Expected speedup: 3–5× on programs with d small (where parsing dominates).

#### 2.1.7 Hard-coded distribution primitives

`gauss`, `bernoulli`, `uniform`, `beta`, `laplace`, `exprnd` are wired in.
Adding a new distribution requires editing grammar + preprocessor + tests.

Recommended fix: distribution registry with a plugin interface. New
distributions register themselves with a name, a parameter spec, and an EM
fitter.

#### 2.1.8 Sparse error model

Parse errors surface as raw ANTLR tracebacks. Σ non-PSD produces a warning
on stderr and silently rebuilds. Infeasible observes (`P → 0`) drop
components without explanation.

Recommended fix: exception hierarchy (`SogaParseError`,
`SogaNumericalError`, `SogaInfeasibleObserveError`, …) with actionable
messages and optional structured diagnostics.

### 2.2 Performance optimizations (algorithmic / numerical)

In rough priority order. All but #1 require code work, not just CLI flags.

1. **Vectorize `update_rule`**. Same per-component-loop pattern as
   `truncate()`. Speedup proportional to n_comp. Expected geometric mean
   1.5–3× on top of the current vectorized truncate.
2. **Vectorize `merge` and `classic_prune`**. `classic_prune` has an O(n_comp²)
   distance-matrix step (`libSOGAmerge.py:81-109`) that can be batched.
3. **Auto-prune insertion in `produce_cfg`**. Walk the CFG, estimate the
   peak n_comp per branch, insert `prune(K)` automatically when above a
   threshold. This is the only known fix for `ClickGraph.soga` and similar
   programs that time out due to combinatorial component explosion (not a
   numerical-kernel issue — see §3.1).
4. **Cache `asgmt_parse` / `trunc_parse` results**. Memoize by expression
   string. Cheap to add and gives material speedup on programs that revisit
   the same expression inside loops.
5. **Precompile LBC and ASGMT expressions at CFG construction time** (see
   §2.1.6). Strictly subsumes #4.
6. **Path-level parallelism** (branch-segment workers). The right
   granularity for multiprocessing — IPC overhead amortizes over a whole
   path segment between two merges. Complementary to vectorization, not a
   replacement. Implementation requires §2.1.1 done first.
7. **Sparsity of Σ in truncate** (extend `select_indices`). Already used
   locally inside `compute_moments`; not exploited in the outer A·Σ·Aᵀ
   computation of the classic path. Largely subsumed by the new vectorized
   path.
8. **GPU backend** (cupy / JAX). The vectorized truncate is the foundation.
   Direct port of the `(n_comp, d, d)` tensor ops to GPU.
9. **Vectorize `eq_func` further** (extension of `_eq_truncate_vectorized_impl`,
   currently identical math to per-component but already in batch).

### 2.3 Reliability / numerical robustness

1. **Fix the `DigitRecognition.soga` `TypeError`**. Root cause: `data[node.idx][0]`
   for loop counters is mutated and not reset between consecutive in-process
   runs. Block fixed by §2.1.1 (immutable CFG → per-run ExecutionContext).
2. **Tighten PSD enforcement**. `make_psd` currently emits a print() warning
   and substitutes negative eigenvalues silently. Should optionally raise
   `SogaNumericalError` and report the offending location.
3. **Better handling of `P → 0` components**. The Kalman paper
   (Perälä & Ali-Löytty 2008, §3.1) gives a closed-form limit for `α → 0`
   that we currently skip; SOGA returns `pi = 0` and drops the component.
   The paper's limit collapses the component to the constraint manifold
   instead, which is more accurate when the truncation is just rare.
4. **Hard timeout in core, not only in benchmark scripts**. We used SIGALRM
   inside `bench_all_canonical.py`; the same hardening should live in
   `start_SOGA` so a stuck program does not hang a host process.

### 2.4 Packaging / product transformation

Items in this section turn SOGA from "a script" into "a library people can
adopt".

1. **`pyproject.toml`** with setuptools build backend.
   `pip install -e .` should work from a clean checkout.
2. **`__version__`** in `src/__init__.py`. Semantic versioning.
3. **CI/CD**. GitHub Actions: lint (`ruff`), tests (`pytest`), bench
   regression on `feat/*` branches.
4. **Type hints** on the public API surface. The core
   (`libSOGAtruncate`, `libSOGA`) is the priority.
5. **Logging instead of `print`**. Levels, structured fields.
6. **Slim `requirements.txt`**. `pymc3`, `mathics-omnibus`, `pandas`,
   `scikit-learn` are used only for baseline experiments and visualization;
   they should be extras (`pip install soga[experiments]`), not core deps.
7. **Docker image** with only the core deps (numpy / scipy / sympy /
   antlr4-python3-runtime / mpmath / psutil). The existing `Dockerfile` can
   be simplified.
8. **Sphinx or MkDocs documentation site** with auto-generated API
   reference and a tutorial notebook.

### 2.5 Domain extensions (research)

1. **Reliability analysis use case** (GPU kernel FI). Investigated in this
   session; conclusion in §3.5. Could become a specialization track if
   significant DSL extensions (matrix primitives, bit-level ops, marginal
   mode) are added.
2. **Markov-style sequential models** as a published benchmark family.
   These programs (random walks, HMMs) exercise the truncate path heavily
   and showcase the largest speedups from the optimizations in §1.
3. **Bayesian Point Machine at varying N** as a scaling-study benchmark.
   `experiments/feasibility_dead_var_pruning_2026-05-20/` includes a
   generator (`gen_programs.py`) for the Pattern B family.

---

## 3. Directions explored and rejected, with reasoning

These were live design decisions during the session. Recording them
prevents re-litigation later.

### 3.1 ClickGraph and `RandomWalkUnif10` timeouts

We measured both programs timing out at `--vectorize-truncate` too.
Diagnosis:

- **ClickGraph**: no `prune()` in the loop body. Each iteration multiplies
  n_comp by ~8 net (after observes filter). After 5 iterations, peak
  n_comp ~ 10⁶ at internal points; every truncate at that point exceeds
  the wall budget. **Not a SOGA-kernel issue; the program lacks a prune
  hint**. Confirmed by `ClickGraphPrune.soga` (same program + explicit
  `prune(K)`) completing in 847 ms.
- **RandomWalkUnif10**: timed out only in classic mode. Sparse path
  completes in 52 s; vectorize path completes in 7.3 s. So the
  optimization unlocks the program; structurally it has n_comp ≈ 271764
  in the final mixture.
- **DigitRecognition**: `TypeError` on the second in-process run because of
  state leak in `data[node.idx][0]` (loop counter mutated but not reset).
  Independent of mode; pre-existing SOGA bug.

Recommended fix-route, not implemented: auto-prune insertion (§2.2.3) for
ClickGraph; immutable CFG (§2.1.1) for DigitRecognition.

### 3.2 Automatic dead-variable pruning

Investigated as a candidate optimization
(`experiments/feasibility_dead_var_pruning_2026-05-20/`). Hand-rewrite study
on `BayesPointMachine.soga` plus synthetic Pattern B and Pattern C scaling
families. Findings:

- For BPM-like patterns the speedup of *pure* static liveness is bounded
  asymptotically by ~4×, because the future-used variables remain live at
  every `observe[k]` until they are themselves observed. Loop-fusion-style
  rewriting (single `tmp` instead of an array) achieves more but is no
  longer pure dead-var pruning.
- For Markov-like patterns (random walk with per-step observes) the
  speedup is O(T³) in principle, but in practice the per-call Python
  overhead caps the realized gain at ~1.8× even at d = 151.
- At realistic SOGA dimensionalities (d ≤ 30) the dead-var-pruning win is
  dominated by the sparse / vectorize wins already shipped in §1.

Conclusion: not worth implementing as a primary optimization on the
current benchmark suite. Re-evaluate if d > 100 becomes a routine target.

### 3.3 Sparse-aware "Markov-blanket of α" inside truncate

Hypothesis: the rotation matrix A in `find_basis(α)` is intrinsically
block-sparse (identity on the d-k coordinates where α is zero), so the
A·Σ·Aᵀ and back-projection products should be O(k²·d) instead of O(d³).

Result: confirmed by the standalone prototype
(`experiments/sparse_truncate_prototype_2026-05-20/prototype.py`); 22×
per-truncate speedup at d = 1000. This became the published "sparse
truncate" path. *Not rejected — promoted to §1.*

### 3.4 Kalman-paper RPT vs BPT (Perälä & Ali-Löytty 2008)

The paper proposes two methods: Recursive PDF Truncation (RPT) applying
constraints one at a time, and Box PDF Truncation (BPT) applying them
together via Gram–Schmidt. Their own experiments (Table 1 of the paper)
show RPT beats BPT in accuracy (~4× lower MSE) and in computation cost.

Verdict: SOGA already implements RPT (this is exactly the per-`observe`
truncate). BPT offers no benefit for SOGA's DSL. The paper is useful as an
external validation of the sparse-truncate math (Algorithm 1 of the paper
= the rank-1 update we ship), but it does not propose a new optimization
to adopt.

### 3.5 SOGA as a reliability-analysis tool for GPU kernels

Investigated against the project sketched in `Input_type_new.pdf`.
Findings:

- The SOGA DSL cannot natively express matrix primitives, bit-level
  operations, or GPU thread parallelism; modelling even a 2MM 4×4 kernel
  requires unrolling and an n_comp explosion (4³² inputs from independent
  GMs) before computation begins.
- Bit-flip fault models do not fit Gaussian-mixture propagation
  naturally; they would require a categorical with ~32 values per
  injection site.
- A specialized "SOGA-Reliability" engine (native matmul primitives,
  marginal-mode propagation, bit-level ops, GPU vectorization, aggressive
  auto-prune) would be a 6–9 month engineering effort.

Verdict: not the right tool for this problem at scale. A narrower
application — using SOGA as a sub-component to compute the input-value
distribution per thread analytically while keeping the
thread-input → resilience curve empirical — is feasible but only
incrementally novel.

### 3.6 `parallel_truncate` (current implementation)

The existing `--parallel N` flag (libSOGAtruncate.parallel_truncate) uses
`multiprocessing.Pool.map` per component. Diagnosed as
counter-productive at SOGA-realistic dimensions:

- Per-component serialization + IPC: ~200 µs
- Per-component numeric work (sparse path): ~50 µs

Overhead exceeds the work for d ≤ 30 in essentially every benchmark.
Numpy BLAS already releases the GIL and parallelizes via SIMD/multi-core,
so the `mp.Pool` layer adds cost without adding parallelism.

Verdict: the correct unit for cross-component parallelism is *not* the
component but the *branch-segment* (path between a `test` node and the
next `merge`). Together with vectorization at component level (already
shipped), this would deliver a complete parallelism story. Implementation
gated on §2.1.1 (immutable CFG).

---

## 4. Proposed roadmap

Ordered by leverage. Each phase is independent enough to ship as a
milestone.

**Phase A — Architectural foundations** (~3 weeks)

- §2.1.1 Immutable CFG / ExecutionContext
- §2.1.5 Replace `from X import *` with explicit imports
- §2.4.1 `pyproject.toml`
- §2.4.2 `__version__`
- §2.4.3 CI/CD with the existing test suite

**Phase B — Clean Python API** (~2 weeks, depends on A)

- §2.1.3 `Model.compile / Model.run / Result`
- §2.1.4 Optional multiprocessing
- §2.1.2 Configuration object replacing module globals
- §2.1.8 Exception hierarchy

**Phase C — Performance round 2** (~4 weeks, mostly independent of A/B)

- §2.2.1 Vectorize `update_rule`
- §2.2.3 Auto-prune insertion (fixes ClickGraph class of programs)
- §2.2.5 Precompile LBC and ASGMT expressions
- §2.3.1 Fix DigitRecognition `TypeError`

**Phase D — Productisation** (~3 weeks, depends on A+B)

- §2.4.5 Logging instead of print
- §2.4.6 Slim `requirements.txt`, optional extras
- §2.4.8 Documentation site (Sphinx or MkDocs)
- §2.4.4 Type hints on public API
- §2.4.7 Slim Docker image

**Phase E — Advanced parallelism** (~4 weeks, depends on A)

- §2.2.6 Branch-segment path-level parallelism
- §2.2.8 GPU backend (cupy/JAX) for the vectorized truncate / update

**Phase F — Optional research extensions** (open-ended)

- §2.5.1 Reliability analysis specialization (only if there is a use case
  with a believable user base)
- §2.5.2 Markov benchmarks published as a stress family
- §3.3 expansion: sparse-aware exploiting structural sparsity of α in
  multi-coefficient LBCs

---

## References

- Perälä, T. & Ali-Löytty, S. *Kalman-type Positioning Filters with Floor
  Plan Information*. MoMM 2008. (RPT/BPT validation; Algorithm 1 of the
  paper = rank-1 update we ship as `--sparse-truncate`.)
- `experiments/feasibility_dead_var_pruning_2026-05-20/REPORT.md` —
  dead-variable-pruning feasibility study.
- `experiments/sparse_truncate_prototype_2026-05-20/REPORT.md` —
  sparse-truncate prototype validation (7200-sample equivalence).
- `experiments/sparse_truncate_prototype_2026-05-20/BENCH_3WAY_REPORT.md`
  — 3-way A/B/C benchmark across the canonical SOGA suite.
- `experiments/sparse_truncate_prototype_2026-05-20/results/bench_3way.csv`
  — raw timings.
