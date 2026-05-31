# PolyBench 2MM — Kernel Reference for the Register-Level Replica Experiment

**Plan**: `plan/2026-05-28-polybench-2mm-register-level-replica.md`
**Date**: 2026-05-29
**Source**: PolyBench/C 4.2.1 (GitHub mirror `MatthiasJReisinger/PolyBenchC-4.2.1`, stamped 2016-05-10), `linear-algebra/kernels/2mm/2mm.c`. Retrieved verbatim per `paper-replicator` memo (`/tmp/polybench_2mm_memo.md`, 2026-05-27).

> **HONESTY DISCLAIMER (mandatory, per plan Constraint §8)**: the register-level
> Python Monte Carlo simulator in this experiment is **NOT an NVBit-FI proxy**. SASS
> pipeline / warp-scheduling / register-staging effects are not modellable in Python.
> Any comparison with Lishan Yang's GPU measurements is **qualitative, shape-only**.

---

## 1. Canonical kernel (PolyBench/C 4.2.1)

`2mm` computes `D := alpha*A*B*C + beta*D` as two chained matmuls:

```c
/* Phase 1: tmp = alpha * A * B */
for (i = 0; i < _PB_NI; i++)
  for (j = 0; j < _PB_NJ; j++) {
    tmp[i][j] = 0.0;
    for (k = 0; k < _PB_NK; ++k)
      tmp[i][j] += alpha * A[i][k] * B[k][j];
  }

/* Phase 2: D = beta * D + tmp * C */
for (i = 0; i < _PB_NI; i++)
  for (j = 0; j < _PB_NL; j++) {
    D[i][j] *= beta;
    for (k = 0; k < _PB_NJ; ++k)
      D[i][j] += tmp[i][k] * C[k][j];
  }
```

Default scalars: `alpha = 1.5`, `beta = 1.2`. Default `DATA_TYPE` is `double`.
Rectangular dims `ni × nj × nk × nl`. Init is fully deterministic (modular arithmetic,
no RNG):

```c
A[i][j] = (DATA_TYPE)((i*j+1) % ni) / ni;
B[i][j] = (DATA_TYPE)(i*(j+1) % nj) / nj;
C[i][j] = (DATA_TYPE)((i*(j+3)+1) % nl) / nl;
D[i][j] = (DATA_TYPE)(i*(j+2) % nk) / nk;
```

---

## 2. Deviations adopted for THIS experiment (all documented, all intentional)

| # | PolyBench default | This experiment | Rationale |
|---|---|---|---|
| D1 | `alpha = 1.5, beta = 1.2` | **`alpha = 1, beta = 0`** | `beta=0` drops the `D *= beta` term and the `+beta*D` self-dependence → `D = A*B*C`, a clean two-matmul chain. `alpha=1` drops 2 scalar mults/cell with no methodological loss (plan Alt D). |
| D2 | rectangular `ni×nj×nk×nl` | **square `N×N`, `N ∈ {4, 8}`** | scope ceiling per `soga-internal-expert` memo (N=8 per-cell, K=100, ~169s). |
| D3 | `DATA_TYPE = double`, values `∈ [0,1)` | **`int32`** | int32 is the cleaner fault domain (no mantissa-bit ambiguity); matches Lishan's "Int, 2mm" domain (plan Alt C). |
| D4 | deterministic modular init for **all** matrices | **A, C deterministic; B = flat `v`** | User decision 2026-05-29: A, C are fixed "kernel/weight" matrices; B is the **swept input** (all entries = v). Mirrors `lishan_resilience` (fixed kernel, swept input) and gives R7's "negative input value v" a literal meaning. |
| D5 | init divides by a dim constant (`/ni`) | **drop the denominator** | keeps A, C in `int32` (plan §44 / `paper-replicator` memo). |

### Concrete init used here (square `N`, int32)

```
A[i][j] = (i*j + 1)       % N      # deterministic kernel matrix
C[i][j] = (i*(j+3) + 1)   % N      # deterministic kernel matrix
B[i][j] = v                        # FLAT swept input, v ∈ {-1, 0, 1, 6}
```

(`B`'s PolyBench modular formula is overridden by the flat-v sweep; `D`'s init is
irrelevant because `beta = 0` overwrites it.)

### Resulting computation (square N, int32 wraparound)

```
tmp[i][j] = Σ_k A[i][k] * B[k][j] = v * Σ_k A[i][k]      (B flat ⇒ column-independent)
D[i][j]   = Σ_k tmp[i][k] * C[k][j]
```

All arithmetic is `int32` two's-complement with wraparound (`numpy.int32` semantics,
verified explicitly in `mc_register.py`). Wraparound is classified as **SDC**, never
OTR (Typhoon convention; `int32` has no Inf/NaN).

---

## 3. FMA fault-site enumeration (register-level model)

The kernel performs **`2·N³` accumulator additions** (FMA sites):

- **Phase 1**: `N²` output cells × `N` inner steps = `N³` sites
- **Phase 2**: `N²` output cells × `N` inner steps = `N³` sites

Global site index `s ∈ {0, …, 2N³−1}` enumerated in execution order
(`i` outer, `j` middle, `k` inner; phase 1 then phase 2). A register-level fault:

1. picks `s` uniform in `{0, …, 2N³−1}`,
2. picks bit `b` uniform in `{0, …, 31}`,
3. XORs bit `b` into the **int32 accumulator state immediately after the add at site `s`**,
4. lets the computation continue (corruption propagates).

| N | sites `2N³` | `p_per_step ≈ 1/(2N³)` |
|---|---|---|
| 2 | 16  | 6.25e-2 |
| 4 | 128 | 7.8e-3  |
| 8 | 1024| 9.8e-4  |

This matches the prototype (`lishan_prototype_2x2_scalar.soga`), which flipped the
accumulator sign between the two FMAs of a single 2×2 inner product.

### Fault propagation scope (per `soga-internal-expert` memo Q4)

- A **Phase-2** fault (accumulator for `D[i][j]`) affects **exactly one** output cell ⇒ per-cell union bound is **exact**.
- A **Phase-1** fault (accumulator for `tmp[i][j]`) propagates to `D[i][:]` (all `N` cells in row `i`) ⇒ per-cell union bound **overcounts** by `O(p²)` (validated empirically in iter 3).

---

## 4. Fault categorical (int32 bit-flip → additive delta)

A bit-`b` flip on an int32 register that currently holds value `x`:

- if bit `b` of `x` is `0`: delta = `+2^b`
- if bit `b` of `x` is `1`: delta = `−2^b`

For bit 31 (sign bit), the flip toggles the sign contribution `±2^31` (two's complement).

- **iter 2 (preliminary)**: 33-class positive-only model `{0, +2^0, …, +2^31}` — **directionally biased** (R7): assumes the pre-flip bit was always 0, undercounting SDC at negative `v`. Flagged preliminary.
- **iter 3 (final)**: 65-class signed model `{0, ±2^0, …, ±2^31}` — promoted **unconditionally** (R7), the correct symmetric model.

The MC simulator (`mc_register.py`) is **exact** — it XORs the real bit, so it
automatically realises the correct signed delta. The categorical above is only the
SOGA-side analytical approximation that must converge to the MC.

---

## 5. What Lishan's work does (scope guard, per `lishan-typhoon-expert` memo)

- SUGAR (SIGMETRICS 2021, doi:10.1145/3447375) does **not** use 2MM (closest: GEMM, single matmul).
- Typhoon (SIGMETRICS SRC 2021, short paper) uses MVT, 2DCONV, GEMM, **3MM** — still not 2MM. "Int, 2mm" numbers are from an **unpublished working note** ⇒ qualitative-only.
- Typhoon fault model: NVBit-FI on real hardware, exactly 1 bit-flip per kernel, uniform over dynamic SASS instructions and bits 0–31. Integer matrix operations only; floats / negatives / mixtures explicitly out of scope.

We therefore validate SOGA against **our own** `mc_register.py` (controlled ground
truth), and treat any Lishan overlay as shape-only context.
