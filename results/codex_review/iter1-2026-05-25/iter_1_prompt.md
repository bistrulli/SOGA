# Iter 1 Review Prompt

You are reviewing the following PLAN for the SOGA project (a probabilistic programming language using Gaussian Mixture symbolic execution).

Evaluate it against these criteria for a PLAN checklist:
1. Completeness: are all sub-tasks atomic and verifiable?
2. Novelty: does the plan explain what is NEW vs prior art / baselines?
3. Methodology: is the approach mathematically sound and reproducible?
4. Feasibility: are estimates realistic (time, complexity)?
5. Baseline coverage: are PSI / Stan / AQUA / BLOG comparisons included where applicable?

SPECIFIC FOCUS AREAS (answer each):
(a) Statistical fidelity of 5-class fault model vs SASSIFI bit-flip distribution — are the per-class probabilities 1/32, 4/32, 4/32, 7/32, 16/32 a faithful aggregate of uniform-over-32-bits? Does the partitioning of bits (sign=31, high-exp=27-30, low-exp=23-26, high-mantissa=16-22, low-mantissa=0-15) match the IEEE 754 float32 layout?
(b) Validity of analytical SDC/MSK/OTR computation: is moment-matching truly exact for all 5 classes? Is the K=26 per-output-cell symmetry reduction correct?
(c) Experimental design soundness for monotonicity test — is 21 p-points sufficient? Is bimodal Bernoulli the right family? Is the Gaussian approximation to bimodal Bernoulli valid for p in [0,1] full range?
(d) Honest framing of adjacent claim — is the input-side vs register-level disclaimer consistently applied? Are the citations to SUGAR/Typhoon/Ares accurate?
(e) Numerical recipe soundness — log-space overflow detection (H1), logsumexp for tail aggregation (H2), float64 internal / float32 only for OTR (H6)
(f) Scope discipline — any creeping requirements?
(g) Risk register completeness — any missing risks?

Artifact:
---
[See iter_1_artifact.md in this directory]
---

Output format (strict):
VERDICT: <APPROVE | APPROVE_WITH_CHANGES | REJECT>
FINDINGS:
- <finding 1>
- <finding 2>
RECOMMENDATIONS:
- <action 1>
