# Analytical Ground Truth — Prototype 2×2 Scalar Matmul with Intermediate Fault

**Plan**: `plan/2026-05-28-prototype-step-by-step-matmul.md`
**Program**: `programs/Example/lishan_prototype_2x2_scalar.soga`
**Date**: 2026-05-28
**Derivation source**: `gaussian-mixture-expert` memo (Q2), independently re-derived here.

---

## Setup

Matrix A first row `[a00, a01] = [1, 1]`. Only output cell `D[0, 0]` is computed.

Input distributions: `B[0, 0] ~ N(v, σ_b²)`, `B[1, 0] ~ N(v, σ_b²)`, independent. We use `σ_b² = 10⁻¹²` (effectively a point mass, but nonzero to avoid Tallis degeneracy).

Fault injection: a Bernoulli random variable `fault ~ Bern(p)` is sampled. When `fault = 1`, the accumulator state immediately after FMA #1 is sign-flipped (`acc ← −acc`). FMA #2 (the second add) then runs normally on the (possibly flipped) accumulator.

---

## Step-by-step propagation

### Step 1: `acc = 0`

Deterministic: `acc ~ N(0, 0)`.

### Step 2: `acc = acc + b00`

Affine update with constant `a00 = 1`. Since `acc` was a point at 0, after the add:

```
acc ~ N(v, σ_b²)
```

### Step 3: Bernoulli branch — `fault = Bern(p)`

Branch creates two execution paths:

- **No-fault path** (probability `1 − p`): `acc` unchanged → `N(v, σ_b²)`
- **Fault path** (probability `p`): `acc ← 0 − acc` — affine with `A = −1`, `b = 0`. By the affine rule, mean negates, variance unchanged: → `N(−v, σ_b²)`

After the implicit merge node at `end if`, the joint distribution of `acc` is the 2-component Gaussian Mixture:

```
acc ~ (1 − p) · N(v, σ_b²)  +  p · N(−v, σ_b²)
```

### Step 4: `acc = acc + b10`

Affine update applied **per-component** (this is what we are testing in the prototype):

- No-fault component: `N(v, σ_b²) + N(v, σ_b²) = N(2v, 2σ_b²)`
- Fault component: `N(−v, σ_b²) + N(v, σ_b²) = N(0, 2σ_b²)`

Final distribution of `acc` (= `d00`):

```
d00 ~ (1 − p) · N(2v, 2σ_b²)  +  p · N(0, 2σ_b²)
```

---

## Moment formulas

### Expected value

```
E[d00] = (1 − p) · 2v  +  p · 0  =  2v(1 − p)
```

### Variance

Using `Var[X] = E[X²] − E[X]²`:

```
E[d00²] = (1 − p) · ((2v)² + 2σ_b²) + p · (0² + 2σ_b²)
        = (1 − p) · (4v² + 2σ_b²) + p · 2σ_b²
        = 4v²(1 − p) + 2σ_b²

(E[d00])² = 4v²(1 − p)²

Var[d00] = 4v²(1 − p) + 2σ_b² − 4v²(1 − p)²
         = 4v²(1 − p)[1 − (1 − p)] + 2σ_b²
         = 4v² · p · (1 − p) + 2σ_b²
```

### SDC probability

Define SDC region: `|d00 − E_baseline| > ε · |E_baseline|` where `E_baseline = 2v` (the no-fault expected value).

For p ≪ 1 and σ_b ≪ v, the fault component is centered at 0 with std `√(2)·σ_b`. Distance from baseline = `2v`. Tail mass outside `[2v(1−ε), 2v(1+ε)]` of a Gaussian centered at 0 with std `√2·σ_b` is essentially the full mass (probability 1):

```
Pr(SDC) = p · 1  +  (1−p) · Pr(|N(2v, 2σ_b²) − 2v| > 2v·ε)
        ≈ p
```

The second term is `2 · Φ(−ε·2v / √(2σ_b²))`. With `ε = 10⁻³`, `v = 1`, `σ_b² = 10⁻¹²`, the argument is `0.002/√(2e-12) = 1.4e6` standard deviations → effectively zero. So:

```
Pr(SDC) ≈ p   (to <10⁻⁹ precision)
```

---

## Numerical grid (v, p)

`σ_b² = 10⁻¹²` for all entries.

| v   | p     | E[d00]  | Var[d00] | Pr(SDC) |
|-----|-------|---------|----------|---------|
| 0.5 | 0.001 | 0.999   | 0.000999 | 0.001   |
| 0.5 | 0.01  | 0.99    | 0.0099   | 0.01    |
| 0.5 | 0.05  | 0.95    | 0.0475   | 0.05    |
| 1.0 | 0.001 | 1.998   | 0.003996 | 0.001   |
| 1.0 | 0.01  | 1.98    | 0.0396   | 0.01    |
| 1.0 | 0.05  | 1.9     | 0.19     | 0.05    |
| 2.0 | 0.001 | 3.996   | 0.015984 | 0.001   |
| 2.0 | 0.01  | 3.96    | 0.1584   | 0.01    |
| 2.0 | 0.05  | 3.8     | 0.76     | 0.05    |

**Spot-check against SOGA empirical run (v=1, p=0.01)**:
- SOGA: E[d00] = 1.98, Var[d00] = 0.0396, n_components = 2
- Analytical (this doc): E[d00] = 1.98, Var[d00] = 0.0396, n_components = 2
- **Match at machine precision.** ✓

---

## Edge cases

### v = 0 (zero crossing, NOT in main grid but tested at edge)

- No-fault: `N(0, 2σ_b²)`
- Fault: `N(0, 2σ_b²)`
- Components coincide → effective 1 Gaussian `N(0, 2σ_b²)`
- E[d00] = 0, Var[d00] = 2σ_b² ≈ 0
- Pr(SDC): baseline = 0, eps·|baseline| = 0 → degenerate. Use absolute threshold `ε` (per `numerical-stability-expert` Q3).

### p = 0 (no fault)

- Single Gaussian `N(2v, 2σ_b²)`
- E[d00] = 2v, Var[d00] = 2σ_b²
- Pr(SDC) = 2·Φ(−ε·2v/√(2σ_b²)) ≈ 0

### p = 1 (always fault)

- Single Gaussian `N(0, 2σ_b²)`
- E[d00] = 0, Var[d00] = 2σ_b²
- Pr(SDC) = 1 (all mass at 0, baseline is 2v ≠ 0 for v ≠ 0)
