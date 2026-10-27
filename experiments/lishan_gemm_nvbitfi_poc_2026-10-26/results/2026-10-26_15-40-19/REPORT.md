# GEMM register-FI: SOGA prediction vs emulated NVBitFI (2026-10-26_15-40-19)

n=512, unroll=4, G_GP + FLIP_SINGLE_BIT, gemm.cu check (0.05 %), small campaign=1000, large=30000, seed=20261026.
ASSUMED per-thread mix: 2699 G_GP sites, arithmetic share 76.0% (replace with NVBitFI profiler counts).

## 1. Headline at the gemm.cu criterion (eps = 5e-4)

| A distribution | ‖A‖_F | SOGA MSK (arith) | emu MSK arith (large) | emu 1000-run ALL MSK / SDC / DUE |
|---|---|---|---|---|
| U[0,1] | 295.7 | **67.4%** | 67.2% ± 0.6% | 50.7% / 33.4% / 15.9% (±3.1%) |
| N(0,1/3) | 295.9 | **53.3%** | 54.1% ± 0.6% | 43.4% / 42.2% / 14.4% (±3.1%) |
| Sparse10% | 295.6 | **82.8%** | 82.6% ± 0.5% | 62.6% / 22.5% / 14.9% (±3.1%) |
| U[0,10] | 2959.3 | **67.1%** | 67.0% ± 0.6% | 50.5% / 33.8% / 15.7% (±3.1%) |

## 2. Per arithmetic class (eps = 5e-4): SOGA vs large campaign

| class | U[0,1] | N(0,1/3) | Sparse10% | U[0,10] |
|---|---|---|---|---|
| LDG.c | 0.941 vs 1.000 (n=16) | 0.645 vs 0.778 (n=9) | 0.786 vs 0.750 (n=16) | 0.969 vs 0.900 (n=10) |
| FMUL.beta | 0.837 vs 1.000 (n=9) | 0.641 vs 0.818 (n=11) | 0.762 vs 0.750 (n=8) | 0.889 vs 0.929 (n=14) |
| LDG.a | 0.751 vs 0.745 (n=5590) | 0.568 vs 0.583 (n=5681) | 0.935 vs 0.939 (n=5703) | 0.740 vs 0.741 (n=5747) |
| LDG.b | 0.751 vs 0.743 (n=5727) | 0.569 vs 0.583 (n=5874) | 0.962 vs 0.965 (n=5642) | 0.751 vs 0.751 (n=5614) |
| FMUL.alpha | 0.734 vs 0.734 (n=5701) | 0.564 vs 0.574 (n=5732) | 0.961 vs 0.958 (n=5677) | 0.732 vs 0.732 (n=5696) |
| FFMA | 0.459 vs 0.468 (n=5773) | 0.430 vs 0.421 (n=5581) | 0.457 vs 0.447 (n=5764) | 0.458 vs 0.456 (n=5720) |

## 3. Option B: MSK(eps) of the arithmetic sites, SOGA vs large campaign

| eps | U[0,1] | N(0,1/3) | Sparse10% | U[0,10] |
|---|---|---|---|---|
| 1e-06 | 0.374 vs 0.373 | 0.251 vs 0.259 | 0.734 vs 0.731 | 0.372 vs 0.375 |
| 1e-05 | 0.476 vs 0.477 | 0.355 vs 0.363 | 0.770 vs 0.766 | 0.473 vs 0.478 |
| 0.0001 | 0.582 vs 0.582 | 0.459 vs 0.466 | 0.804 vs 0.801 | 0.585 vs 0.585 |
| 0.0005 | 0.674 vs 0.672 | 0.533 vs 0.541 | 0.828 vs 0.826 | 0.671 vs 0.670 |
| 0.001 | 0.717 vs 0.716 | 0.565 vs 0.576 | 0.839 vs 0.836 | 0.711 vs 0.711 |
| 0.01 | 0.856 vs 0.855 | 0.689 vs 0.698 | 0.876 vs 0.873 | 0.826 vs 0.824 |
| 0.1 | 0.885 vs 0.884 | 0.831 vs 0.833 | 0.921 vs 0.918 | 0.868 vs 0.867 |

## 4. Cost

The CPU emulator is our validation ORACLE (it needs the kernel semantics and an assumed instruction mix), not the GPU campaign. GPU campaign wall time is NOT measured here: it is (#injections) x (one instrumented run of the binary).

| A distribution | SOGA prediction (s, all eps, 0 injections) | CPU oracle (1000+30000 emulated injections, s) |
|---|---|---|
| U[0,1] | 4.3 | 0.5 |
| N(0,1/3) | 0.6 | 0.5 |
| Sparse10% | 3.5 | 0.5 |
| U[0,10] | 3.3 | 0.5 |

## 5. Assumptions (stated, not verified)

- Per-thread G_GP mix derived from gemm.cu (accumulator in a register, unroll 4, 64-bit pointer increments as lo/hi): replace with NVBitFI profiler counts. Per-class rates do not depend on the mix; ALL does.
- Structural sites (index / address / loop counter): buffers 2^32-aligned, out-of-buffer access => DUE, index fault => SDC. Illustrative only; SOGA does not model them, the GPU campaign measures them.
- SOGA: input priors as 8-component GMs (EM preprocessor), one-term law from the engine, 512-term sums moment-matched to one Gaussian, float32 flips exact on a deterministic quadrature. B and C0 treated as U[0,1] priors (fixed instance in the campaign, averaged over 262k cells).
