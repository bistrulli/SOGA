# Does the Frobenius norm predict the resilience of floating-point GEMM?

## Question

Typhoon found that, for integer matrix kernels, the masked rate follows the Frobenius norm of the
input. Does this still hold for floating-point inputs?

## What we did

We wrote PolyBench-GPU `gemm.cu` (float, n = 512, default α and β), together with an
NVBitFI-style fault (one random instruction, one random bit of its destination register) and the
`gemm.cu` correctness check, as a single SOGA program. SOGA computes the probability that the
fault is masked directly from the distribution of the input matrix A, without any fault
injection.

Only the arithmetic instructions are modelled: the loads of A and B, the FMUL by α, and the FFMA
accumulator. Faults on address, index and loop-control registers are not modelled.

## SOGA predictions

B and C are U[0,1] in all cases; only A changes. Masked means that the output passes the
`gemm.cu` check (relative error ≤ 0.05%).

| A | ‖A‖F | Predicted masked rate (arithmetic instructions) |
|---|---|---|
| U[0,1] | 296 | 67.5% |
| N(0, 1/3) | 296 | 55.4% |
| Sparse: 90% zeros, 10% U[0,√10] | 296 | 82.9% |
| U[0,10] | 2956 | 67.1% |

Per instruction:

| A | FFMA | FMUL | loads |
|---|---|---|---|
| U[0,1] | 46% | 74% | 75% |
| N(0, 1/3) | 45% | 59% | 59% |
| Sparse | 46% | 96% | 94–96% |
| U[0,10] | 46% | 73% | 74–75% |

What SOGA predicts:

- **The same norm gives different resilience.** The first three inputs have the same norm and
  give 55%, 67% and 83%.
- **A 10× larger norm changes nothing.** U[0,10] gives the same rate as U[0,1].
- **The differences come from the loads and the FMUL.** The FFMA accumulator is about 46% in
  every case.

If the GPU confirms these numbers, the norm is not a sufficient predictor for floating-point
GEMM. In that case the distribution of the input determines the resilience, and SOGA computes it
without a campaign.

These numbers were fixed before any GPU run.

## Experiments to run on the GPU

### 1. Program

Use `gpu/gemm_soga.cu` and `gpu/soga_inputs.h`. Copy both next to the original `gemm.cu` and
`gemm.cuh` (PolyBench-GPU `CUDA/GEMM/`) and build them the same way as `gemm.cu`. The kernel,
the CPU reference and the check are the original ones. Only three things change:

| | Original `gemm.cu` | `gemm_soga.cu` |
|---|---|---|
| A | `i*j/NI` | drawn from the distribution named on the command line, seed 20261027 |
| B, C | `i*j/NI` | U[0,1], seed 20261026, the same in all four campaigns |
| Timing prints | on | off, so stdout is the same in every fault-free run |
| Side file | none | `soga_maxrelerr.txt`: maximum relative error, number of NaN/Inf outputs |

Unchanged: float, NI = NJ = NK = 512, α = 32412, β = 2123, thread block 32 × 8, check threshold
0.05%. The random numbers come from a fixed generator (splitmix64), so the inputs are
bit-identical on any machine.

| Command | A | ‖A‖F of the generated A |
|---|---|---|
| `./gemm_soga u01` | U[0,1] | 295.6 |
| `./gemm_soga normal` | N(0, 1/3) | 296.0 |
| `./gemm_soga sparse` | 0 with probability 0.9, else U[0,√10] | 295.8 |
| `./gemm_soga u010` | U[0,10] | 2955.7 |

SOGA treats the entries of A, B and C as random with these distributions, and the faulty output
element as random too. So the prediction does not depend on these particular seeds: another seed
should give the same masked rate within the confidence interval.

### 2. Runs, for each of the four inputs

| Run | NVBitFI setting | Injections | Required |
|---|---|---|---|
| Profiler | `gemm_kernel` | 1 | yes |
| Standard campaign | `G_GP`, `FLIP_SINGLE_BIT` | 1000 | yes |
| Floating-point registers only | `G_FP32`, `FLIP_SINGLE_BIT` | 1000 | optional |
| Load registers only | `G_LD`, `FLIP_SINGLE_BIT` | 1000 | optional |

The outcome is classified as you usually do: masked, SDC, or other (crash or hang).

### 3. What to record

- For each injection: the opcode, the bit, and the outcome. These are already in the NVBitFI logs.
- For each campaign: the GPU model and the total wall time, profiler included.
- Optional: for each injection, the content of `soga_maxrelerr.txt`. Keep this file out of the
  SDC check. It lets us compare tolerances other than 0.05%.

### 4. How to compare

1. From the `G_GP` campaign, keep only the injections on FFMA, FMUL and the floating-point loads.
   Their masked fraction is the number to compare with the first table.
2. Compare the `G_FP32` and `G_LD` campaigns with the per-instruction table.
3. If the profiler shows a different instruction mix from the one we assumed, we reweight the
   per-instruction predictions. No new SOGA run is needed.

The prediction is confirmed if:

1. each predicted value lies inside the 95% confidence interval of the GPU measurement;
2. the ordering Sparse > U[0,1] ≈ U[0,10] > N(0, 1/3) is reproduced.

### 5. Statistics

With the instruction mix we assumed, about 760 of the 1000 `G_GP` injections land on arithmetic
instructions, which gives about ±3.5 pp. The predicted differences between inputs with the same
norm (12 to 28 pp) are therefore visible with a standard campaign. The equality of U[0,1] and
U[0,10] is tested only to about ±5 pp.
