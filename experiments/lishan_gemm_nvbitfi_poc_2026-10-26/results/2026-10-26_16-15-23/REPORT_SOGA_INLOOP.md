# Fault injector inside SOGA: GEMM, NVBitFI G_GP FLIP_SINGLE_BIT (2026-10-26_16-15-23)

n=512; FFMA k-grid stride 16 (33 programs per eps); oracle = CPU emulator, 30000 injections; 6 worker processes.

## eps = 0.0005

| A | LDG.a SOGA / emu | LDG.b SOGA / emu | FMUL.alpha SOGA / emu | FFMA SOGA / emu | ARITH SOGA / emu | Δ |
|---|---|---|---|---|---|---|
| U[0,1] | 0.751 / 0.745 | 0.751 / 0.743 | 0.735 / 0.734 | 0.461 / 0.468 | **0.675** / 0.673 | +0.2 pp |
| Sparse10% | 0.935 / 0.939 | 0.962 / 0.965 | 0.962 / 0.958 | 0.459 / 0.447 | **0.829** / 0.827 | +0.2 pp |
| U[0,10] | 0.741 / 0.741 | 0.751 / 0.751 | 0.733 / 0.732 | 0.461 / 0.456 | **0.671** / 0.670 | +0.1 pp |

## Solve time (SOGA, all eps)

| A | CPU seconds, sum over programs | programs |
|---|---|---|
| U[0,1] | 216 | 36 |
| Sparse10% | 879 | 36 |
| U[0,10] | 208 | 36 |

Wall clock, all distributions, 6 processes: 224 s.
