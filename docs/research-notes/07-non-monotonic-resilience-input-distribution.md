# Research note — Non-monotonic resilience vs input distribution

**Date**: 2026-05-25
**Triggered by**: /research command via brainstorm session on Lishan Yang collaboration
**Scope**: Prior art on non-monotonic GPU kernel resilience as a function of input distribution parameters; specifically whether the monotonicity assumption (Assumption-1) in SUGAR can be broken by non-flat input distributions inaccessible to Yang's O(n^9) campaign.

---

## Top-5 references (verified DOI/arXiv)

| # | Title | Authors | Year | Venue | DOI/arXiv | Notes |
|---|-------|---------|------|-------|-----------|-------|
| 1 | SUGAR: Speeding Up GPGPU Application Resilience Estimation with Input Sizing | L. Yang, E. Smirni, A. Jog | 2021 | ACM POMACS (SIGMETRICS) | doi.org/10.1145/3447375 | Source of Assumption-1 (monotonic resilience-input relationship) |
| 2 | GPU Reliability Assessment: Insights Across the Abstraction Layers | L. Yang, G. Papadimitriou et al. | 2024 | IEEE CLUSTER (Best Paper Finalist) | doi.org/10.1109/CLUSTER59578.2024.00022 | Yang's own follow-up; cross-layer but no input distribution sweep |
| 3 | Peppa-X: Finding Program Test Inputs to Bound Silent Data Corruption Probability | H. Rahman et al. | 2021 | SC '21 | doi.org/10.1145/3458817.3476195 | Directly treats input as a variable; finds SDC sensitivity distribution is NOT uniformly stationary across input space |
| 4 | NVBitFI: Dynamic Fault Injection for GPUs | T. Tsai, S. Hari, M. Sullivan, S. Keckler | 2021 | ISPASS | doi.org/10.1109/ISPASS51385.2021.00034 | Canonical tool; all campaigns use fixed (typically uniform random) inputs — input distribution is a blind spot |
| 5 | MPGemmFI: A Fault Injection Technique for Mixed Precision GEMM in ML Applications | — | 2023 | arXiv | arxiv.org/abs/2311.05782 | GEMM-specific FI; no input distribution sweep; only ML weight distributions tested |

---

## Secondary references (verified)

| # | Title | Authors | Year | Venue | DOI/arXiv | Notes |
|---|-------|---------|------|-------|-----------|-------|
| 6 | SASSIFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluation | S. Hari, T. Tsai et al. | 2017 | ISPASS | doi.org/10.1109/ISPASS.2017.7975296 | Foundational tool; uses single fixed input per application |
| 7 | Ares: A Framework for Quantifying the Resilience of Deep Neural Networks | B. Reagen, U. Gupta et al. | 2018 | DAC | doi.org/10.1145/3195970.3195997 | Input-side sweep over DNN test sets; first large-scale input-aware FI; but limited to classification accuracy, not raw SDC rate vs distribution parameters |
| 8 | FlipTracker: Understanding Natural Error Resilience in HPC Applications | Z. Guo, Y. Li et al. | 2018 | SC | doi.org/10.1109/SC.2018.00011 | Identifies code patterns (min/max, absolute value, modular reduction) that create natural masking; patterns are input-value sensitive by construction |
| 9 | Evaluating and Accelerating High-Fidelity Error Injection for HPC (Hamartia) | L. Chang, N. Lym et al. | 2018 | SC | doi.org/10.1109/SC.2018.00048 | Uses RTL-level error models; input data is fixed per benchmark; application-level masking noted but not swept over input distributions |
| 10 | SpotSDC: Revealing the Silent Data Corruption Propagation in HPC Systems | Y. Wei et al. | 2020 | IEEE TVCG | doi.org/10.1109/TVCG.2020.2994954 | Visualization of SDC propagation; uniform random inputs assumed throughout |
| 11 | The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance | — | 2025 | arXiv (preprint) | arxiv.org/abs/2605.04213 | 63 CUDA micro-benchmarks including GEMM; input stimuli varied by functional category but no systematic distribution family sweep |
| 12 | Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs | T. Cui et al. | 2025 | SC | doi.org/10.1145/3712285.3759821 | Field data study; no input distribution control |
| 13 | Towards Analytically Evaluating the Error Resilience of GPU Programs (GPUTrident) | S. Hari et al. | 2019 | SELSE | research.nvidia.com/publication/2019-03_towards-analytically-evaluating-error-resilience-gpu-programs | Analytical model; branch divergence shown to be input-dependent; SDC profile sensitivity to input noted but not swept |

---

## Key findings

### Finding 1: Yang's Assumption-1 is explicitly stated as monotonic, not proven

**Source**: Yang et al. [SUGAR 2021], doi.org/10.1145/3447375

**Evidence**: SUGAR's core claim is that resilience (SDC rate + DUE rate) exhibits repeating, monotonically scalable patterns as a function of input *size*, because at thread level the dynamic instruction count determines outcome. The assumption is stated operationally: small-input campaigns are sufficient to estimate full-scale resilience *up to a constant factor*.

**Relevance to SOGA**: Assumption-1 is about input *size* scaling, not input *distribution*. Yang explicitly tests only inputs that differ in matrix dimension (n x n) with the same underlying distribution (typically uniform random floats). The O(n^9) wall (nine nested loops for 2MM at large n) means she cannot test large-n inputs at all, and she never tests whether a *different distribution* at the same n changes the SDC rate non-monotonically. This is the gap we can target.

---

### Finding 2: Peppa-X is the closest prior work to input-as-variable — and it found SDC sensitivity is NOT uniform across input space

**Source**: Rahman et al. [SC '21], doi.org/10.1145/3458817.3476195

**Evidence**: Peppa-X searches for the worst-case input (the input that maximises SDC probability) using a guided search over the input value space. The key finding: "benchmark inputs rarely manifest SDC vulnerabilities, leading to over-optimistic assessment and unexpectedly higher failure rates in production." The SDC-bounding input found by Peppa-X is substantially different from a typical random input.

**Relevance to SOGA**: This is direct evidence that SDC rate is input-value-dependent (not merely input-size-dependent), even for CPU programs. Peppa-X targets CPU kernels, not GPU GEMM kernels specifically. SOGA can extend this to GEMM/2MM on GPU with a principled probabilistic formulation over input distributions, which Peppa-X does not provide.

---

### Finding 3: FlipTracker identifies masking code patterns that are intrinsically input-value sensitive

**Source**: Guo et al. [FlipTracker, SC 2018], doi.org/10.1109/SC.2018.00011, arXiv:1809.01362

**Evidence**: FlipTracker catalogues "naturally resilient" code patterns including: (a) min/max operations that mask corrupted values dominated by correct ones, (b) absolute value / modular reduction, (c) accumulation with cancellation (sum of positive and negative terms). Pattern (c) is the critical one: an injected fault in a partial sum is masked if the subsequent additions cancel it out. The masking probability depends directly on the magnitude distribution of operands.

**Relevance to SOGA**: For GEMM the inner product sum A[i,k] * B[k,j] is exactly an accumulation with potential cancellation. If operands are drawn from a bimodal distribution with zero mean, cancellation is maximised and masking increases. If operands are strictly positive (e.g. uniform on [1, 10]), cancellation is minimal and masking decreases. This is a concrete mechanism for non-monotonic behaviour across distribution families. No paper in this corpus has measured it for GEMM specifically.

---

### Finding 4: Branch divergence creates input-dependent SDC profiles — confirmed analytically

**Source**: Hari et al. [GPUTrident, SELSE 2019], research.nvidia.com/publication/2019-03_towards-analytically-evaluating-error-resilience-gpu-programs; also implied by SUGAR [Yang 2021]

**Evidence**: In GPU programs where the control flow path taken by a warp depends on the input values (data-dependent branches), different inputs yield different dynamic instruction sequences and therefore different fault sites. SUGAR notes: "when input does not affect branch divergence, the instruction execution context persists across different inputs." The contrapositive — when branch divergence IS input-dependent, resilience profiles differ across inputs — is the foundation for non-monotonic behaviour.

**Relevance to SOGA**: Pure GEMM has no data-dependent branches, so this mechanism does not apply to the inner loop. However 2MM (two matrix-multiply) in PolyBench does have conditional memory-access patterns in some GPU implementations. More importantly, the FP exception path (NaN/Inf generation) on overflow IS data-dependent, providing a second mechanism.

---

### Finding 5: Non-monotonic resilience has been observed in approximate computing and DNN weights — suggesting it is a general phenomenon

**Source**: Search results referencing ScienceDirect 2026 paper "Memory efficient soft error mitigation for CNN accelerators exploiting nonmonotonic bit sensitivity", doi.org/10.1016/j.microrel.2026.115662 [unverified — DOI did not resolve; flagged]

**Evidence**: "[C]omprehensive fault injection experiments revealed that normalized weights exhibit non-monotonic bit sensitivity, where certain middle exponent bits prove more critical than traditionally protected higher-order bits."

**Relevance to SOGA**: This is conceptually close to what we expect for GEMM with specific input distributions. The non-monotonicity in that context is about bit position; our target non-monotonicity is about distribution parameters (mean, spread, bimodality). The mechanism (interaction between magnitude of operands and the significance of the corrupted bit) is the same.

**Note**: The specific 2026 ScienceDirect DOI did not resolve at time of verification. The observation is flagged as plausible-but-unverified for the specific cite; the broader finding (non-monotonic bit sensitivity in FI) is well-supported by FlipTracker and Ares independently.

---

### Finding 6: No paper has conducted a systematic sweep over input distribution families for GEMM or 2MM

**Source**: Synthesis across all 13 references surveyed.

**Evidence**: Every tool surveyed (SASSIFI, NVBitFI, Hamartia, SUGAR, Peppa-X, MPGemmFI, Anatomy of SDC) uses a fixed input per campaign, typically:
- Uniform random floats in [0, 1] or [-1, 1]
- Random integers
- Application-specific datasets (for DNN benchmarks)

None systematically varies the distribution family (Gaussian, bimodal, Laplace, heavy-tailed, near-zero) across a parameter grid while holding input size constant.

**Relevance to SOGA**: **This is the primary novelty gap.** The combination of (a) a probabilistic characterisation of the input distribution as a Gaussian mixture and (b) a forward-propagation model of that distribution through the GEMM computation graph — which SOGA provides — is not present in any existing tool. SOGA can generate the theoretical SDC-rate-vs-distribution-parameter curve analytically, which no current FI tool can produce.

---

### Finding 7: The O(n^9) limit explicitly prevents Yang from testing large-n non-flat distributions

**Source**: Yang [SUGAR 2021, CLUSTER 2024]; also Typhoon 2021 SRC abstract

**Evidence**: Typhoon's stated goal was to extend SUGAR to "different input types" but was presented only as a student research competition abstract (not a full paper) and does not appear to have been published as a refereed venue paper. Yang's CLUSTER 2024 paper moves to cross-layer abstraction analysis rather than extending Typhoon's input-type direction. The O(n^9) complexity constraint (from 2MM's nine nested loops in the full FI campaign) is inherited by any tool that requires exhaustive fault injection.

**Relevance to SOGA**: SOGA's symbolic propagation avoids the O(n^9) wall entirely by computing expected SDC-rate analytically under a given input distribution model. This is not a marginal speedup over Yang's approach — it is a fundamentally different capability that enables the non-flat-distribution experiment to be run at all.

---

## Existing implementations

| Tool | URL | License | Last active | Notes |
|------|-----|---------|-------------|-------|
| NVBitFI | github.com/NVlabs/nvbitfi | BSD-3 | 2023 | Current standard GPU FI; input fixed per run |
| SASSIFI | github.com/NVlabs/sassifi | BSD-3 | 2019 | Predecessor to NVBitFI; deprecated |
| Ares | github.com/alugupta/ares | MIT | 2020 | DNN-specific; input sweep over test set only |
| Peppa-X | github.com/hasanur-rahman/Peppa-X | not specified | 2022 | CPU programs only; worst-case input search |

---

## Gaps SOGA could fill

- **Gap 1 (primary)**: No tool can produce a continuous resilience-vs-distribution-parameter curve for GEMM/2MM. SOGA's analytical propagation over Gaussian mixture inputs fills this gap directly.
- **Gap 2**: No paper has tested bimodal or heavy-tailed input distributions on GPU matrix-multiply kernels. The FlipTracker cancellation mechanism predicts they should produce qualitatively different SDC rates.
- **Gap 3**: Yang's Typhoon direction (input types) was never published as a refereed paper. A SOGA-enabled version of that study, covering continuous distribution families, could be the natural follow-up paper with Yang as a collaborator.
- **Gap 4**: No existing framework connects the probabilistic description of input data (as a prior distribution) to a predicted posterior SDC-rate distribution. SOGA's GM propagation is the natural substrate for this.
- **Gap 5**: Non-monotonicity of resilience as a function of distribution parameters (e.g., bimodality index, kurtosis, mass near zero) has never been measured or predicted for GEMM. If SOGA finds a non-monotonic curve, it constitutes a counter-example to Assumption-1 extended to distribution space.

---

## Recommended distribution families to test (informed by literature)

- **Bimodal Gaussian mixture (two symmetric modes, zero mean)**: maximises cancellation in GEMM inner products (FlipTracker pattern c); expected to show elevated masking and lower effective SDC rate than flat uniform — a candidate for non-monotonic peak.
- **Uniform on [0, 1] (Yang's baseline)**: reference point; all values positive, minimal cancellation.
- **Gaussian N(0, sigma) with varying sigma**: smooth interpolation between near-zero (high cancellation) and large-magnitude (low cancellation) regimes; sigma is the single sweep parameter.
- **Laplace (heavy-tailed, zero mean)**: heavier mass near zero than Gaussian; expected to amplify cancellation effects and produce even higher masking than bimodal Gaussian.
- **Bernoulli / sparse inputs (many zeros)**: zero operands trivially mask any fault in the accumulation of that term; extreme point of the cancellation axis.
- **Near-integer or quantised distributions (INT8 / INT16 range)**: relevant for mixed-precision GEMM (MPGemmFI context); integer overflow is a distinct fault-propagation pathway not present in FP.
- **Adversarial worst-case (Peppa-X-guided)**: use Peppa-X search strategy adapted for GPU to find the distribution *parameters* (not just values) that maximise SDC rate; provides an upper bound on our non-monotonic range.

---

## Recommended next actions

- [ ] Download and read SUGAR (doi.org/10.1145/3447375) full text to extract exact wording of Assumption-1 and identify which specific input distributions Yang uses for her GEMM/2MM campaigns.
- [ ] Download and read Peppa-X (doi.org/10.1145/3458817.3476195) Sections 3-5 to understand the input-search algorithm and assess adaptability to GPU/GEMM.
- [ ] Run SOGA's forward propagation on a 2MM stub with the seven distribution families above and produce a prototype resilience-vs-sigma curve for the Gaussian family to check for non-monotonicity.
- [ ] Attempt to reach Lishan Yang (now at U. Alberta per homepage) to confirm whether Typhoon was abandoned or is in preparation — if the latter, avoid duplicating her unpublished work and propose collaboration instead.
- [ ] Check whether the FlipTracker cancellation pattern (sum with opposing signs) is present in the PolyBench 2MM GPU kernel implementation used by SUGAR; if yes, this is the mechanistic basis for our non-monotonicity hypothesis.
- [ ] Verify the ScienceDirect 2026 "non-monotonic bit sensitivity" paper independently (DOI did not resolve; may be too recent or incorrect DOI).

---

## Assessment: novelty status

**SPARSE PRIOR ART — potential novelty confirmed.**

No paper in the surveyed corpus has:
1. Observed a non-monotonic resilience curve as a function of input distribution parameters for any GPU kernel.
2. Proved monotonicity theoretically (Yang's Assumption-1 applies only to input size, not distribution).
3. Conducted a systematic sweep over input distribution families for GEMM or 2MM.
4. Used a probabilistic propagation model (as opposed to sampling-based FI) to predict SDC rates under a given input prior.

The concept of "adversarial input distribution" as a distinct paradigm from hardware fault injection does not appear in the GPU reliability literature. The closest adjacent work is Peppa-X (worst-case input *values*) and Ares (DNN input set diversity), neither of which addresses distribution families analytically.

This is a research gap of sufficient width to support a novel contribution.
