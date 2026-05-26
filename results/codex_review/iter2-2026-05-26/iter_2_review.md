# Iter 2 Review — Claude self-review (codex-cli downgrade)

**Mode**: claude-self-review (codex-cli exited code 1 — Azure provider auth failure)
**Date**: 2026-05-26
**Reviewer role**: Independent reviewer. Do NOT consider yourself the author. Be skeptical.

---

## Targeted verification: F1-F9

### F1 (HIGH) — OTR aggregation semantics

STATUS: VERIFIED

- R0.3 task present (line 198): explicitly tasks an agent to read `simulate_fi_mc.py` and document per-execution vs per-cell-averaged semantics in `lib/DESIGN_BIT_EXACT.md § OTR_SEMANTICS`.
- R2.2b task present (line 217): requires alignment of analytical OTR formula to MC semantics, with both cases spelled out (per-execution and per-cell-averaged).
- "⚠️ CRITICAL — OTR aggregation semantics" section present in Approach (lines 112-123): clearly describes the mathematical distinction and the two-case formula.
- R-BE11 present in risk register (line 360): correctly marked HIGH with NEW R0.3 + R2.2b mitigation.
- Acceptance criterion added (line 302): "OTR aggregation semantics aligned with MC reference (R0.3 + R2.2b; closes Codex iter 1 F1)".

F1 fully applied. No gaps.

---

### F2 (MEDIUM) — Inf input guard contradiction

STATUS: PARTIALLY VERIFIED — residual issue in R-BE2 (LOW severity)

The scalar code block (lines 83-92) correctly guards `np.isnan(v)` only:
```python
if np.isnan(v):
    return 0.0, True
```
Comment on line 79-80 explicitly states Inf input flows through, individual bits handled per IEEE 754. Correct.

The vectorized code (lines 100-106) correctly guards `is_nan_in = np.isnan(v32)` for inputs, with `is_special` for outputs using `~np.isfinite(v_post)`. Correct.

Line 89 `is_special = not np.isfinite(v_post)` is a check on the POST-flip output value, NOT the input guard. This is correct behavior — a flipped bit may produce +Inf/-Inf/NaN as the output, which is a special case for the delta computation. This is NOT a false alarm; it is intentionally checking whether the result of flipping the bit is a special float.

HOWEVER: R-BE2 mitigation text (line 351) still reads:
> "Numerical-stability Bug A fix: explicit `not np.isfinite(v)` guard at entry; R1.4 NaN edge case test"

This is the OLD description of the fix before F2 was applied. It now incorrectly describes the guard as `not np.isfinite(v)` (which was the bug) rather than `np.isnan(v)` (which is the fix). An implementer reading R-BE2 in isolation would implement the wrong guard. This is a documentation inconsistency of LOW severity — the code block in the Approach section is correct, but the risk register mitigation contradicts it.

---

### F3 (MEDIUM) — NON-MONOTONE τ threshold

STATUS: PARTIALLY VERIFIED — inconsistency between Approach section and R4.3 task (MEDIUM severity)

Approach section (lines 151-161) correctly revised:
- Uses `p_kendall` as primary statistic (not bare τ threshold)
- Decision rule: MONOTONE if `p_kendall < 0.0167 AND |τ| > 0.7`
- NON-MONOTONE if `p_kendall > 0.05 AND n_wrong ≥ 10`

HOWEVER: R4.3 sub-task (line 244) was NOT updated:
> "Robust threshold rule: MONOTONE if τ>0.9 AND n_wrong≤2; NON-MONOTONE if τ<0.7 OR (n_wrong≥5 AND binom_p<0.01); AMBIGUOUS else; plateau-degenerate exempt"

This text uses the OLD thresholds (τ<0.7, n_wrong≥5) that F3 and F4 were supposed to fix. The fix was applied to the Approach section but NOT to the R4.3 task text. An implementer following the R4.3 checklist item would implement the wrong thresholds. This is a MEDIUM severity inconsistency because it directly governs test implementation.

---

### F4 (MEDIUM) — n_wrong ≥ 5 redundancy

STATUS: PARTIALLY VERIFIED — same inconsistency as F3

Approach section (line 159): correctly uses `n_wrong ≥ 10`.

R4.3 (line 244): still uses `n_wrong≥5`. The fix was applied in the Approach section but NOT in the R4.3 task text.

Same root cause as the F3 issue above: the R4.3 bullet was not synchronized with the Approach section revision.

---

### F5 (MEDIUM) — Bonferroni correction

STATUS: VERIFIED

Approach section (lines 156, 159): `α_per_test = 0.05/3 ≈ 0.0167` explicitly documented; threshold `p_kendall < 0.0167` applied in MONOTONE decision rule.
R-BE13 (line 362): "R4.3 Bonferroni α=0.0167 per curve (per Codex iter 1 F5)" confirmed.

The R4.3 task text is inconsistent (see F3/F4 above) but does not contradict Bonferroni explicitly — it just uses different thresholds entirely. F5 is verified in the Approach section; the gap is the R4.3 sync issue shared with F3/F4.

---

### F6 (MEDIUM) — Default --mode backward-compat

STATUS: VERIFIED

R2.5 task (lines 223-224) now includes:
- (a) runtime banner: `"[MODE] Using bit_exact fault model (refinement primary; --mode 5_class for legacy)"`
- (b) DESIGN_BIT_EXACT.md + CHANGELOG documentation
- (c) `config_version: 2` bump; old configs default to `1` with deprecation warning
- Explicitly labeled: "BACKWARD-COMPATIBILITY NOTICE (per Codex iter 1 F6)" and "DELIBERATE behavioral change"

R-BE12 (line 361): added and marked MEDIUM with R2.5 mitigation. Correct.

---

### F7 (LOW) — R-BE11/12/13 missing from risk register

STATUS: VERIFIED

- R-BE11 (line 360): OTR semantics mismatch — HIGH severity, NEW R0.3 + R2.2b mitigation. Present.
- R-BE12 (line 361): Default --mode behavioral change — MEDIUM severity, R2.5 mitigation. Present.
- R-BE13 (line 362): Multiple-testing Bonferroni — MEDIUM severity, R4.3 α=0.0167 mitigation. Present.

All three new risks present and correctly described.

---

### F8 (LOW) — Pearson criterion for OTR

STATUS: VERIFIED

Acceptance criteria (lines 299-304) now explicitly:
- "Pearson > 0.95 between SOGA bit-exact and MC for **MSK and SDC curves only**"
- Separate line: "Step 1 OTR: max abs err < 1% (Pearson not applicable due to sparseness)"

The distinction is clearly drawn and the rationale (sparseness makes Pearson degenerate) is stated.

---

### F9 (LOW) — Endianness description

STATUS: VERIFIED

No code change required (cosmetic). The vectorized code correctly uses `np.asarray(v_array, dtype=np.float32)` without explicit `'<f4'` endianness specifier (line 100). Cross-review history notes "no code change needed (cosmetic)" (line 377). Appropriate.

---

## Summary of residual issues

| Issue | Severity | Description |
|-------|----------|-------------|
| R4.3 threshold inconsistency (F3+F4) | MEDIUM | R4.3 task text still uses old thresholds (τ<0.7, n_wrong≥5). Approach section is correct. Implementer following R4.3 checklist gets wrong thresholds. |
| R-BE2 mitigation text (F2) | LOW | R-BE2 still says "explicit `not np.isfinite(v)` guard at entry" — should say "np.isnan(v) only". Contradicts the Approach code block. |

---

VERDICT: APPROVE_WITH_CHANGES
FINDINGS:
- F1 (HIGH): VERIFIED — R0.3, R2.2b, CRITICAL section, R-BE11, acceptance criterion all present and consistent.
- F2 (MEDIUM): VERIFIED for code blocks. RESIDUAL LOW: R-BE2 mitigation text (line 351) still says `not np.isfinite(v)` (old guard) instead of `np.isnan(v)` (correct fix). Contradicts the Approach section code.
- F3 (MEDIUM): PARTIALLY VERIFIED. Approach section (lines 151-161) correctly uses p_kendall as primary. R4.3 task text (line 244) NOT UPDATED — still uses old thresholds `τ<0.7 OR (n_wrong≥5 AND binom_p<0.01)`. Implementer will implement wrong thresholds.
- F4 (MEDIUM): PARTIALLY VERIFIED. Same root cause as F3. Approach section uses n_wrong≥10 (correct). R4.3 still uses n_wrong≥5 (old).
- F5 (MEDIUM): VERIFIED — Bonferroni α=0.0167 in Approach section and R-BE13. R4.3 sync gap shared with F3/F4 but F5 itself is not the primary problem in that line.
- F6 (MEDIUM): VERIFIED — R2.5 has all three mitigations (banner, CHANGELOG, config_version bump). R-BE12 added.
- F7 (LOW): VERIFIED — R-BE11, R-BE12, R-BE13 all present and correctly described.
- F8 (LOW): VERIFIED — acceptance criteria clearly separates Pearson (MSK/SDC) from abs-err (OTR).
- F9 (LOW): VERIFIED — no code change needed; noted appropriately.
RECOMMENDATIONS:
- Fix R4.3 task text (line 244): replace threshold rule with Approach-section version: "MONOTONE if p_kendall < 0.0167 AND |τ| > 0.7; NON-MONOTONE if p_kendall > 0.05 AND n_wrong ≥ 10; AMBIGUOUS else; plateau-degenerate exempt"
- Fix R-BE2 mitigation text (line 351): replace "explicit `not np.isfinite(v)` guard at entry" with "np.isnan(v) only guard at entry (Inf flows through for per-bit IEEE 754 evaluation)"
