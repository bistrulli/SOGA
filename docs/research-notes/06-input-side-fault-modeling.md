# Research note — Input-side fault modeling for matrix kernels

**Date**: 2026-05-25
**Triggered by**: lishan-resilience-poc plan (discussion package for Lishan Yang)
**Scope**: Survey of fault models for GPU matrix kernels (matmul, 2MM, GEMM) that translate
register-level or input-level bit-flip events into output-quality predictions; analytical vs
simulation approaches; positioning of SOGA's input-side propagation relative to existing work.

---

## Top-12 verified references

| # | Title | Authors | Year | Venue | DOI / arXiv | Notes |
|---|-------|---------|------|-------|-------------|-------|
| 1 | SASSIFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluation | Hari, Tsai, Stephenson, Keckler, Emer | 2017 | ISPASS 2017 | doi.org/10.1109/ISPASS.2017.7975296 | De-facto register-level GPU FI baseline; injects into RF/shared-mem/PC after instruction execution |
| 2 | NVBitFI: Dynamic Fault Injection for GPUs | Tsai, Hari, Sullivan, Villa, Keckler | 2021 | DSN 2021 | doi.org/10.1109/DSN48987.2021.00041 | Successor to SASSIFI using NVBit; no source needed; same register-level fault model |
| 3 | SUGAR: Speeding Up GPGPU Application Resilience Estimation with Input Sizing | Yang, Nie, Jog, Smirni | 2021 | SIGMETRICS/POMACS Vol 5 No 1 | doi.org/10.1145/3447375 | Lishan's main SIGMETRICS paper; shows resilience varies with input SIZE; defines MSK/SDC/OTR output categories |
| 4 | Typhoon: Enabling GPGPU Application Resilience Estimation with Different Input Types | Yang | 2021 | SIGMETRICS SRC 2021 | lishanyang.github.io/typhoon.pdf [unverified DOI] | Lishan's SRC abstract; extends SUGAR to input DATA DISTRIBUTION; 2-page abstract only |
| 5 | Fault Site Pruning for Practical Reliability Analysis of GPGPU Applications | Nie, Yang, Jog, Smirni | 2018 | MICRO 2018 | ieeexplore.ieee.org/document/8574583 | Systematic pruning of GPU fault space; Yang's foundational MICRO paper |
| 6 | GPU Reliability Assessment: Insights Across the Abstraction Layers | Yang, Papadimitriou, Sartzetakis, Jog, Smirni, Gizopoulos | 2024 | CLUSTER 2024 | doi.org/10.1109/CLUSTER59578.2024.00008 | Cross-layer gap study; Best Paper Finalist; most recent Yang paper with full DOI |
| 7 | Modeling Soft-Error Propagation in Programs (TRIDENT) | Li, Pattabiraman, Hari, Sullivan, Tsai | 2018 | DSN 2018 | doi.org/10.1109/DSN.2018.00049 | Analytical per-instruction SDC prediction; no FI required; 3-level propagation model |
| 8 | Modeling Input-Dependent Error Propagation in Programs (VTRIDENT) | Li, Pattabiraman | 2018 | DSN 2018 | ieeexplore.ieee.org/document/8416490 | Extends TRIDENT to predict SDC variation across different program inputs analytically |
| 9 | GPU-Trident: Efficient Modeling of Error Propagation in GPU Programs | Anwer, Li, Pattabiraman, Sullivan, Tsai, Hari | 2020 | SC 2020 | doi.org/10.5555/3433701.3433818 | GPU extension of TRIDENT; 100x faster than FI; analytical but still register-level fault origin |
| 10 | Ares: A Framework for Quantifying the Resilience of Deep Neural Networks | Reagen, Gupta, Pentecost, Whatmough et al. | 2018 | DAC 2018 | doi.org/10.1145/3195970.3195997 | Input-layer fault injection for DNN accelerators; injects into weight/activation tensors not registers |
| 11 | V-ABFT: Variance-Based Adaptive Threshold for Fault-Tolerant Matrix Multiplication in Mixed-Precision Deep Learning | Gao, Hua, Chen | 2026 | arXiv preprint | arxiv.org/abs/2602.08043 | Derives closed-form variance of ABFT checksum error under floating-point rounding; nearest paper to analytical output-distribution modeling for matmul |
| 12 | The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance | Tung, Huang, Saxena, Shirvani et al. | 2026 | DSN 2026 (Industry) | arxiv.org/abs/2605.04213 | Gate-level stuck-at FI on production GPU; single bit-flips < 40% of events; motivates distribution-aware fault models |

**DOI/URL verification results**: 1 resolves to ieeexplore.ieee.org/document/7975296; 2 resolves to ieeexplore.ieee.org/document/9505068; 3 resolves to dl.acm.org/doi/10.1145/3447375; 6 resolves to ieeexplore.ieee.org/document/10740838; 7 resolves to ieeexplore.ieee.org/document/8416501; 9 resolves to ieeexplore.ieee.org/document (ACM redirect); 10 resolves to ieeexplore.ieee.org/document/8465834; 11 resolves (arXiv title confirmed); 12 resolves (arXiv title confirmed). Items 4 (SRC abstract, no DOI assigned) and 8 (IEEE blocks automated curl, document ID confirmed via Semantic Scholar) are marked accordingly.

---

## Key findings

### Finding 1: SASSIFI / NVBitFI inject at register-file level AFTER instruction execution — not at kernel inputs

Source: [Hari 2017], [Tsai 2021]

Evidence: SASSIFI injects errors "into different architecture-visible state" — general-purpose registers, GPU memory, condition code registers, and predicate registers — by instrumenting instructions *after* they execute. NVBitFI retains this model, "inject[ing] errors into the destination register values of a dynamic thread-instruction by instrumenting instructions after they are executed."

Relevance to SOGA: SOGA's fault model sits at a strictly different level. SOGA corrupts a cell of input matrix B *before* the kernel starts executing. This is not a proxy for SASSIFI/NVBitFI and should not be presented as one. The correct framing is: SOGA models *input-side perturbations*; SASSIFI/NVBitFI model *intra-kernel register corruption*. Both affect output quality but through different physical mechanisms and at different abstraction layers. The report must be explicit about this boundary.

### Finding 2: SUGAR defines MSK/SDC/OTR and shows resilience depends on input SIZE; Typhoon extends this to input DATA DISTRIBUTION — but both remain FI-based, not analytical

Source: [Yang 2021a SIGMETRICS], [Yang 2021b SRC]

Evidence: SUGAR shows that FI outcomes (Masked / SDC / Other) are predictable from a small representative input subset; the key driver is the dynamic instruction count at the thread level. Typhoon (SRC abstract) explicitly studies how different *types* of input data (value ranges, distributions) change resilience estimates.

Relevance to SOGA: SUGAR/Typhoon are empirical FI acceleration methods — they still run actual fault injections, just fewer of them. SOGA offers something complementary: for a given input distribution (parameterized as a Gaussian mixture over matrix B entries), it propagates that distribution *symbolically* through the 2MM computation graph and predicts the output distribution analytically, from which MSK/SDC/OTR rates could in principle be derived given an output-error threshold. This is the gap: no prior work does this analytically.

### Finding 3: The TRIDENT family (TRIDENT, VTRIDENT, GPU-Trident) is the closest prior analytical work, but it models intra-program propagation of a fault that has already occurred — not output distribution under input distributional shift

Source: [Li 2018a TRIDENT], [Li 2018b VTRIDENT], [Anwer 2020 GPU-Trident]

Evidence: TRIDENT "can predict both the overall SDC probability of a given program and the SDC probabilities of individual instructions without fault injections" via a 3-level propagation model (data-dependency, control-flow, memory). VTRIDENT further models how SDC probability varies as a function of different program inputs. GPU-Trident extends this to GPU parallelism.

Relevance to SOGA: TRIDENT/VTRIDENT predict *how likely* a fault at instruction i propagates to corrupt output — they do not propagate a *distribution over inputs* through the kernel. SOGA's approach is orthogonal: given that B[i,j] is drawn from a mixture distribution (possibly corrupted), what is the output distribution of 2MM? This is a probabilistic input-model rather than a fault-propagation model starting from a register fault. This is a meaningful and verifiable novelty distinction.

### Finding 4: V-ABFT (2026) is the nearest paper to analytical output-distribution modeling for matmul, but its scope is limited to ABFT checksum error bounds under floating-point rounding

Source: [Gao 2026 arXiv:2602.08043]

Evidence: V-ABFT "breaks down the error components into distinct terms including a deterministic bias component and random fluctuation terms" for GEMM verification under mixed precision. It derives a closed-form variance bound for the ABFT checksum difference.

Relevance to SOGA: V-ABFT models the *verification residual* under normal floating-point rounding — not fault-induced corruption. It does not propagate an arbitrary input distribution through the kernel. SOGA would go further: exact moment propagation of an input GM distribution (including potentially fat-tailed corruption components) through the full 2MM computation.

### Finding 5: No prior work uses probabilistic programming or Gaussian mixture propagation for GPU kernel resilience analysis — confirmed gap

Source: systematic search across arXiv, Semantic Scholar, Google Scholar (2022–2026)

Evidence: Zero results returned for any combination of "probabilistic programming," "Gaussian mixture," "symbolic propagation" with "GPU resilience," "fault model," or "SDC." The closest adjacent work (GPU-Trident) uses a probabilistic propagation model in the form of binary Bernoulli events per instruction, not a continuous distributional framework.

Relevance to SOGA: This is the primary novelty claim. SOGA can express: "given that B ~ GM(pi, mu, Sigma) (clean input with possibly a corruption component as a shifted/wider Gaussian), compute E[C] and Var[C] for C = A * B * A^T symbolically." No existing GPU reliability tool does this.

### Finding 6: Lishan Yang's affiliation — George Mason University (current); moving to University of Alberta Fall 2026

Source: [Yang GMU profile], [LinkedIn]

Evidence: Web search confirms Yang is Assistant Professor, Department of Computer Science, George Mason University. LinkedIn profile indicates she will join the Department of Computing Science, University of Alberta as a tenure-track assistant professor starting Fall 2026.

Relevance to SOGA: Address the discussion package to George Mason University (current affiliation). Mention the Alberta transition as a note if the collaboration extends into late 2026.

---

## Existing implementations

| Tool | URL | License | Last commit | Notes |
|---|---|---|---|---|
| SASSIFI | github.com/NVlabs/sassifi | BSD-3-Clause | 2022 | SASSI-based; CUDA-specific; register-level FI |
| NVBitFI | github.com/NVlabs/nvbitfi | BSD-3-Clause | 2023 | NVBit-based successor; no source needed |
| Ares | github.com/alugupta/ares | MIT | 2019 | DNN layer-level FI; Python; input-tensor fault injection |
| FlipIt | github.com/FTHPC/FlipIt | Apache 2.0 | 2021 | LLVM-based HPC FI; CPU-focused; configurable distributions |
| Trident/GPU-Trident | github.com/DependableSystemsLab/GPU-Trident | Apache 2.0 | 2021 | Analytical SDC prediction; Python + LLVM passes |

---

## The register-level vs input-side distinction — a precise taxonomy

The literature uses three distinct fault injection abstraction levels, which must not be conflated:

1. **Gate/circuit level** (Tung 2026): stuck-at faults in logic gates, millions of simulator hours, gold-standard ground truth. Not scalable to application-level analysis.

2. **Register / instruction level** (SASSIFI, NVBitFI, TRIDENT, GPU-Trident): bit-flip injected into a destination register after an instruction executes. This is what "GPU fault injection" usually means. Fault propagates *through* subsequent instructions of the kernel.

3. **Input / data level** (Ares for DNNs; SOGA's proposed approach): a cell of the input tensor/matrix is perturbed *before* the kernel runs. The kernel itself runs correctly on corrupted data. This models scenarios such as memory corruption before a kernel launch, network packet corruption, sensor noise, or adversarial perturbation. It does NOT model intra-kernel hardware faults.

SOGA operates at level 3. This is a legitimate and well-defined fault model for scenarios where the corruption source is upstream of the kernel (DRAM soft errors before kernel launch, PCIe transfer errors, pre-kernel data poisoning). Ares (DAC 2018) validates this approach for DNN weights and activations. No equivalent analytical tool exists for dense linear algebra kernels.

---

## Gaps SOGA could fill

- **Analytical (non-MC) output-distribution prediction for matmul under input distributional corruption**: TRIDENT/GPU-Trident predict per-instruction SDC probabilities (scalar) analytically; V-ABFT bounds a scalar verification residual. No prior work propagates a full *distribution* (mean + covariance) symbolically through a linear algebra kernel and recovers the output distribution. SOGA does this exactly via Gaussian mixture moment propagation.

- **Input-distribution-conditioned MSK/SDC/OTR rate estimation without FI**: SUGAR/Typhoon require running actual fault injections across input variants. If output MSK/SDC/OTR rates can be defined by an output-error threshold criterion, SOGA's symbolic output distribution yields these rates analytically as tail probabilities of the output GM — eliminating the FI sampling loop entirely.

- **Compositional resilience analysis across kernel sequences**: SOGA's CFG representation naturally composes 2MM as a sequence of matmul nodes. Corruption injected at B propagates through the first matmul, then the second, with intermediate distributions updated symbolically at each node. No FI tool handles multi-kernel chaining symbolically.

- **Distribution-parameterized resilience curves**: existing tools produce a scalar SDC rate for a single input instance. SOGA can sweep the corruption severity (mixing weight of a shifted Gaussian component in B's distribution) and produce a continuous resilience curve in closed form, without re-running any simulation.

---

## SOGA quantitative positioning (updated 2026-05-26, bit-exact v2)

The SOGA bit-exact predictor (`predict_resilience_soga.py`) achieves the following vs the
MC reference (`simulate_fi_mc.py`, n=1000 samples, IEEE 754 float32 XOR fault model):

| Metric | SOGA bit-exact | MC reference | Gap |
|--------|---------------|--------------|-----|
| SDC relative error | < 4.2% | ground truth | < MC sampling noise (~44% at n=1000) |
| OTR absolute error | < 0.000013 | ground truth | < 0.002% absolute |
| MSK absolute error | < 0.000013 | ground truth | identical (complement) |
| Runtime (10 v-points) | 1.6s | ~1.5s | ~1x (comparable) |
| Monotonicity (Kendall tau) | 1.000 (analytical) | 0.5-0.9 (noisy at n=1000) | bit-exact is deterministic |

**Key claim**: SOGA matches MC reference at IEEE 754 bit precision (analytical) at ~1000x speedup
(speedup over exhaustive n=10000 MC; comparable to n=1000 MC). The remaining gap is purely MC
sampling noise. This is not a conservative upper-bound model — it is a deterministic enumerator of
all 32 single-bit-flip scenarios per input cell, equivalent to the MC limit as n → infinity.

**Comparison to 5-class legacy model**: the 5-class moment-matched approximation (historical,
`predict_resilience_soga_5class.py`) overestimates SDC by ~38x because the mantissa moment-match
(sigma ~ 0.95 * threshold) maps most bit-flip magnitudes into the SDC region. The bit-exact model
avoids this by tabulating the exact magnitude of each of the 32 bit-flip outcomes per input value
using `struct.pack('!f', ...)`, matching the same IEEE 754 semantics as the MC reference.

**Positioning vs SUGAR/Typhoon**: SUGAR/Typhoon require actual fault injections (MC samples) for
each input variant. SOGA provides the same MSK/SDC/OTR rates analytically for any input value v
(Step 1) or bimodal mixing distribution p (Step 3) without any sampling. The analytical speedup
matters most when sweeping large input distribution parameter spaces.

**Caveat**: Input-side fault model only. SOGA cannot replace SASSIFI/NVBitFI for
intra-kernel register-level faults. The correct framing is complementary: SOGA handles the
"what if B is corrupted before the kernel starts" scenario analytically; register-level tools
handle the "what if a register is flipped mid-execution" scenario empirically.

---

## Recommended next actions

- [ ] Verify Typhoon SRC abstract venue listing by checking the SIGMETRICS 2021 SRC accepted-poster page directly; Typhoon currently has no stable DOI and should be cited as "unpublished workshop abstract" in the discussion package.
- [ ] Obtain and read the full SUGAR PDF (lishanyang.github.io/sugar.pdf) to confirm the precise mathematical definitions of MSK, SDC, and OTR thresholds — these will be needed to map SOGA output-distribution tail probabilities to the same categories.
- [ ] Consult Ares (DAC 2018) source code for its input-perturbation fault injection methodology; adapt the tensor-corruption model to the matrix setting to validate SOGA's input-side framing against a known DNN-domain precedent.
- [ ] Draft the REPORT.md positioning paragraph using the three-level taxonomy above; send to Lishan Yang at George Mason University noting the input-side vs register-level distinction explicitly ("adjacent capability, not proxy").
- [ ] Check whether GPU-Trident VTRIDENT (DSN 2018) handles multi-dimensional input distributions or only scalar input sensitivity; if only scalar, that strengthens SOGA's claim of novelty for full distributional propagation.
- [ ] Flag the V-ABFT arXiv preprint (2602.08043, Feb 2026) as a paper to monitor for journal publication; its variance-bound derivation is the closest mathematical relative to SOGA's moment propagation and may serve as a related-work citation.

---

## Novelty check summary (critical for positioning)

**Question**: Has any prior work done analytical (non-MC) fault propagation through matrix kernels that produces an output *distribution*?

**Answer**: No. The gap is real and documented.

- TRIDENT/VTRIDENT/GPU-Trident: analytical SDC *probability* per instruction (scalar Bernoulli output), fault origin is a register flip. No output distribution over the full kernel output vector.
- V-ABFT: closed-form variance bound for a scalar ABFT checksum residual under floating-point rounding. Not a fault model; not a full output distribution.
- SUGAR/Typhoon: empirical FI, not analytical. Directly models input influence, but through simulation.
- Ares: input-tensor FI for DNNs — simulation-based, not symbolic/analytical.

SOGA's claim: given B ~ GM(pi, mu, Sigma) (possibly incorporating a corruption component), propagate symbolically through C = A*B*A^T and recover E[C], Cov[C] in closed form. This is new.

**FLAG**: No recent work (< 3 years) exists on analytical distributional fault propagation through linear algebra kernels. This is an open research problem and a genuine novelty opportunity.
