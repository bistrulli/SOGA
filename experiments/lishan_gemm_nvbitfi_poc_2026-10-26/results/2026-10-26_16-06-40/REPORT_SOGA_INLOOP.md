# Fault injector inside SOGA: GEMM, NVBitFI G_GP FLIP_SINGLE_BIT (2026-10-26_16-06-40)

n=512; FFMA k-grid stride 16 (33 programs per eps); oracle = CPU emulator, 30000 injections; 6 worker processes.

## eps = 0.0005

| A | LDG.a SOGA / emu | LDG.b SOGA / emu | FMUL.alpha SOGA / emu | FFMA SOGA / emu | ARITH SOGA / emu | Δ |
|---|---|---|---|---|---|---|
| N(0,1/3) | 0.591 / 0.583 | 0.591 / 0.583 | 0.589 / 0.574 | 0.446 / 0.421 | **0.554** / 0.540 | +1.4 pp |

## Solve time (SOGA, all eps)

| A | CPU seconds, sum over programs | programs |
|---|---|---|
| N(0,1/3) | 2925 | 36 |

Wall clock, all distributions, 6 processes: 509 s.
