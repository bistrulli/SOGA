# REPORT — PolyBench 2MM Register-Level Fault Injection, Analytically via SOGA

**Plan**: `plan/2026-05-28-polybench-2mm-register-level-replica.md`
**Run**: `results/orchestrate_2026-05-29_07-40-52`
**Branch**: `feat/lishan-resilience-poc`
**Dates**: 2026-05-29 → 2026-05-31

> **HONESTY DISCLAIMER (mandatory)**: the register-level Python Monte Carlo here is
> **NOT an NVBit-FI proxy**. SASS/warp/register-staging effects are not modelled. Any
> comparison with Lishan Yang's GPU "Int, 2mm" data is **qualitative, shape-only**.

---

## TL;DR

1. **SOGA analytically reproduces register-level fault injection on the int32 2MM
   EXACTLY** (modulo int32 wraparound). A threshold-free check of 2,560 (N=4) and
   18,432 (N=8) per-(site,bit) injections finds the SOGA engine's propagated outcome
   distribution contains the actual int32 MC outcome in **100% of non-wrapping cases**;
   every mismatch (32 at N=4, 672 at N=8) is exactly an int32-wraparound case (R5),
   none unexplained.
2. **We reproduce the SHAPE of Lishan's resilience-vs-input-norm curve**: fault
   susceptibility `(1−MSK)` is maximal at norm 0 and decreases **monotonically** with
   the input Frobenius norm — matching Typhoon's "resilience ≈ monotonic in Frobenius
   norm" hypothesis. The monotone shape is qualitatively consistent across the tolerance
   `eps` (quantitative spread ~5 pp at large norm — reported, not hidden).
3. **A key design correction (iter 1)**: the plan's `eps=0` SDC criterion is
   **degenerate** — exhaustively proven input-value-independent (every fault changes the
   int32 output; `MSK≡0`). v-dependence requires a relative tolerance `eps>0`, which is
   adopted as a **declared modeling assumption** (not a fitted parameter).

**Verdict**: **GO on the methodology** (SOGA = analytical register-level FI for this
kernel). **QUALIFIED on the Lishan shape match** — the monotone norm-resilience shape
is reproduced, but via relative-error masking, a mechanism not proven identical to
Lishan's physical masking (R8 remains open).

---

## 1. What changed vs the plan, and why (epistemic honesty)

| Plan said | We did | Reason |
|---|---|---|
| `eps=0` (int32, "any change = SDC") | relative `eps>0`, swept {1,4,16} | `eps=0` is **degenerate**: exhaustive enumeration (4096 / 32768 injections) gives `MSK=0` for **all** v at both N. The output difference `delta*C mod 2^32` is independent of v, so the resilience curve is flat. v-dependence requires relative masking. (`iter_1_finding.md`) |
| absolute MSK, 4-point grid {-1,0,1,6} | normalized susceptibility vs a 17-point **norm sweep** | user goal (2026-05-29): reproduce the **shape**, not absolute values. v=±1 are identical (function of \|v\|), so 4 points cannot trace a curve. |
| `n=10^4` Monte Carlo | **exact enumeration** | the N∈{4,8} fault spaces (4096 / 32768 register injections) are fully enumerable → **zero sampling noise**, strictly stronger than sampling. |
| per-cell union bound to per-kernel (R2) | per-kernel computed **exactly** (any-cell-changes) | the union bound is unnecessary; R2 is moot. |
| 33-class then 65-class (R7) | **65-class signed directly** | the 33-class positive-only model is directionally biased (R7); for the `|shift|>thr` metric the sign is irrelevant, so 65-class is both correct and sufficient. |

These are documented in `config.json`; the design blocker is in
`results/orchestrate_2026-05-29_07-40-52/iter_1_finding.md`.

## 2. Setup

- Kernel: PolyBench/C 4.2.1 `2mm`, square `N×N`, `alpha=1, beta=0`, **int32** two's-complement.
  `A[i][j]=(i*j+1)%N`, `C[i][j]=(i*(j+3)+1)%N` (fixed kernels); **`B = flat v`** (swept input).
  `tmp=A·B`, `D=tmp·C`. (`POLYBENCH_REFERENCE.md`)
- Fault model: single bit-flip, uniform over `2N³` int32 accumulator FMA sites × 32 bits;
  65-class signed categorical `{0, ±2^b}`.
- SDC: `|D_pert − D_base| > eps·|D_base|` (threshold 0 at base 0). MSK = 1−SDC; OTR = 0 (int32).
- MC: **exact enumeration** (`mc_register.py`, `mc_input_side.py`, `resilience_curve.py`).
- SOGA: per-cell `.soga` with one 65-class fault, run through the **engine**; the
  propagated component means give the outcome shifts (`generate_soga.py`, `soga_curve.py`).

## 3. Result A — SOGA propagation is exact (threshold-free, non-circular)

`validate_soga_vs_mc.py` checks, for every (cell, fault-location, bit), that the engine's
predicted outcome set contains the actual int32 MC outcome — **no `eps`, no shared metric**:

| N | checks | exact match | wrap-only mismatch (R5) | unexplained |
|---|---|---|---|---|
| 4 | 2,560 | 2,528 (98.75%) | 32 | **0** |
| 8 | 18,432 | 17,760 (96.35%) | 672 | **0** |

Every mismatch is a case where `delta·C[k][s] ≥ 2^31` wraps mod 2³² (engine = real
arithmetic, MC = int32). This is the **genuinely non-circular** validation that SOGA's
analytical fault propagation equals register-level FI.

## 4. Result B — the resilience-vs-norm SHAPE (the deliverable)

`resilience_curve.py` (MC, exact) and `soga_curve.py` (engine-sourced) both yield
susceptibility `1−MSK` **monotonically decreasing** in the input Frobenius norm
`||B||_F = |v|·N`, maximal at norm 0 — the Typhoon/Lishan shape. See
`figures/soga_vs_mc_register_vs_input_side.png`.

- **SOGA vs MC_register** (`compare_3way_register.py`): per-kernel `max|SOGA−MC| = 0.0`;
  per-cell `max = 4.9e-4` (= the int32-wrap cases, R5); normalized-shape Spearman = 1.0.
  *(The susceptibility metric is the same functional applied to both outcome sets, so
  this agreement follows from Result A and is reported as derived, not independent.)*
- **eps-robustness**: monotone shape holds for all eps∈{1,4,16}; normalized
  susceptibility at max norm spans ~0.69–0.74 (≈5 pp). Reported honestly.
- **Input-side vs register**: input-side is **more** susceptible per-kernel
  (mean `+0.048` at N=4, `+0.103` at N=8) — a single B-cell fault is read by all N Phase-1 inner products
  in its column, reaching N× more accumulator paths than one register fault. (This is the
  opposite of the plan's hypothesis that register-level would dominate — an honest finding.)

## 5. Comparison with Lishan (shape-only, qualitative)

We reproduce the **qualitative** monotone norm-vs-resilience relationship Typhoon
hypothesizes. We do **NOT** claim her absolute values, do **NOT** overlay her
(unpublished, working-note) "Int, 2mm" numbers, and do **NOT** claim NVBit-FI
reproduction. The normalization that would cancel structural masking `c` (plan Option B)
is moot here because our arithmetic susceptibility is already the full curve.

## 6. Open risks / limitations (honest)

- **R8 (unresolved)**: our shape arises from **relative-error masking** (low-order-bit
  flips negligible vs output magnitude). Lishan's integer masking may be structural/
  architectural and need not share this mechanism. A matching shape does **not** prove a
  matching cause.
- **eps is a modeling assumption** for integers (no intrinsic tolerance). We mitigate by
  showing the shape is qualitatively eps-stable, but absolute MSK is eps-dependent.
- **R5**: real-arithmetic engine vs int32 wraparound — quantified above (<0.1% per-cell).
- **flat-v ⇒ monotone only**: non-monotonic resilience (cf. prior input-distribution work)
  would require a distribution sweep (varying zero-fraction / structure), out of scope here.

## 7. Artifacts

```
POLYBENCH_REFERENCE.md  kernel2mm.py  mc_register.py  mc_input_side.py
resilience_curve.py  generate_soga.py  soga_curve.py  validate_soga_vs_mc.py
compare_3way_register.py  plot_comparison.py  config.json  HASHES.txt
results/{resilience_curves,soga_curve,three_way_summary}.csv
figures/soga_vs_mc_register_vs_input_side.png
programs/Example/lishan_polybench_2mm_N{4,8}_v1_cell00.soga
```

## 8. Recommendation for Option B (N=32)

The analytical-propagation result is **strong and exact** and scales (engine runs are
v-independent and O(N³) per size). Before scaling to N=32: (a) resolve R8 by obtaining
one calibration curve from Lishan to test the relative-error-masking hypothesis against a
structural-masking model; (b) add an input-**distribution** sweep to probe non-monotonic
regimes. Absent (a), the N=32 result would extend the methodology but not settle the
physical-cause question.
