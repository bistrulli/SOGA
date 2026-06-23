"""Beta validation (self-contained, no Yang/GPU needed): does an ANALYTICAL
prediction of float32 register-FI resilience, computed from the INPUT PRIOR
alone (zero fault injection, zero input sampling), match Monte-Carlo ground
truth — AND does it falsify Typhoon's norm-as-sufficient-statistic?

Kernel: a single inner product / accumulation  out = sum_k x_k  (one GEMM cell).
Fault model: pick term k ~ U{0..n-1}, bit b ~ U{0..31}, single-bit-flip x_k.
The output error equals delta_b(x_k) = flip(x_k,b) - x_k (linear kernel), so the
golden recomputation only matters through the accumulator magnitude |out|.

Classes (vs golden out):
  OTR : flipped term is Inf/NaN  (exponent/sign flips that overflow)
  MSK : |error| <= eps*|out|     (rounding/tolerance-masked)
  SDC : otherwise

Two priors with the SAME expected Frobenius norm AND the SAME accumulator law
out ~ N(0, norm^2):
  balanced     : every x_k ~ N(0, norm^2 / n)
  concentrated : x_0 ~ N(0, big_frac*norm^2);  x_{1..n-1} ~ N(0, (1-big_frac)*norm^2/(n-1))
Both give E[||x||^2] = norm^2 and Var(out) = norm^2 by construction, so any
difference in SDC/MSK is due ONLY to the per-term magnitude distribution -> a
clean falsification of "the Frobenius norm is a sufficient statistic for SDC".

ANALYTICAL prediction (the SOGA-spirit object, no injection):
  P(class) = E_{v ~ flipped-term marginal, b ~ U} [ classify ]
  - OTR(v,b): deterministic indicator that flip(v,b) is non-finite.
  - given a finite flip with |delta| = T_eps := |delta_b(v)| / eps, MSK holds iff
    |out| = |v + r| >= T_eps, with r = sum_{j!=k} x_j ~ N(0, norm^2 - Var(x_k))
    independent of v:  P_r(|v+r| >= T) = [1-Phi((T-v)/s_r)] + Phi((-T-v)/s_r).
  The outer E over v is a fine 1-D quadrature against the prior density; the
  flipped-term marginal is a 1-component (balanced) or 2-component (concentrated)
  Gaussian mixture -> the prediction is a GM tail integral.

Scope (honest): the closed form targets the relative-tolerance regime eps>0
(per the int32-eps0-degenerate finding, eps=0 bit-exact masking is order- and
accumulator-history-dependent and is NOT modeled here). MC uses true float32
re-accumulation for every (k,b) and is the independent oracle.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from float_example import classify, dot_f32, flip_bit_f32

_erf = np.vectorize(math.erf)


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


@dataclass(frozen=True)
class Prior:
    """A zero-mean Gaussian-mixture prior over the n input terms.

    component_var[i] is the variance of term i; the terms are independent.
    By construction sum(component_var) == norm**2 (matched Frobenius norm) and
    Var(out) == norm**2 (matched accumulator).
    """

    name: str
    component_var: np.ndarray  # shape (n,)
    norm: float

    @property
    def n(self) -> int:
        return int(self.component_var.shape[0])

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        sd = np.sqrt(self.component_var)
        return (rng.standard_normal((n_samples, self.n)) * sd).astype(np.float32)


def make_priors(n: int, norm: float, big_frac: float) -> tuple[Prior, Prior]:
    balanced_var = np.full(n, norm ** 2 / n, dtype=np.float64)
    conc_var = np.empty(n, dtype=np.float64)
    conc_var[0] = big_frac * norm ** 2
    conc_var[1:] = (1.0 - big_frac) * norm ** 2 / (n - 1)
    return (
        Prior("balanced", balanced_var, norm),
        Prior("concentrated", conc_var, norm),
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo ground truth (the independent oracle: true float32 re-accumulation)
# --------------------------------------------------------------------------- #
def mc_rates(prior: Prior, eps: float, rng: np.random.Generator,
             n_samples: int) -> dict[str, float]:
    X = prior.sample(rng, n_samples)
    n = prior.n
    counts = {"MSK": 0, "SDC": 0, "OTR": 0}
    total = 0
    for s in range(n_samples):
        x = X[s]
        golden = float(dot_f32(x))
        for k in range(n):
            xk = float(x[k])
            for b in range(32):
                xf = x.copy()
                xf[k] = np.float32(flip_bit_f32(xk, b))
                out = float(dot_f32(xf))
                counts[classify(out, golden, eps)] += 1
                total += 1
    return {c: counts[c] / total for c in counts} | {"_total": total}


# --------------------------------------------------------------------------- #
# Analytical prediction (no injection, no input sampling)
# --------------------------------------------------------------------------- #
def _component_rates(var_k: float, var_r: float, eps: float,
                     n_quad: int) -> dict[str, float]:
    """Class probabilities for flipping a single term v ~ N(0, var_k), with the
    rest-sum r ~ N(0, var_r), averaged over bit b ~ U{0..31}, by 1-D quadrature."""
    sd_k = math.sqrt(var_k)
    sd_r = math.sqrt(var_r) if var_r > 0 else 0.0
    # fine grid over the flipped-term marginal (piecewise-smooth integrand:
    # delta_b(v) jumps at powers of two, so use a dense uniform grid).
    grid = np.linspace(-8.0 * sd_k, 8.0 * sd_k, n_quad)
    dens = np.exp(-0.5 * (grid / sd_k) ** 2) / (sd_k * math.sqrt(2.0 * math.pi))
    w = dens / dens.sum()  # normalized quadrature weights over v

    msk = np.zeros(n_quad)
    otr = np.zeros(n_quad)
    for b in range(32):
        vflip = np.array([flip_bit_f32(float(v), b) for v in grid], dtype=np.float64)
        finite = np.isfinite(vflip)
        otr += (~finite).astype(np.float64)
        delta = np.where(finite, vflip - grid, 0.0)
        T = np.abs(delta) / eps  # need |out|=|v+r| >= T for MSK
        if sd_r > 0:
            p_mask = (1.0 - _phi((T - grid) / sd_r)) + _phi((-T - grid) / sd_r)
        else:  # degenerate: out == v
            p_mask = (np.abs(grid) >= T).astype(np.float64)
        msk += np.where(finite, p_mask, 0.0)
    msk /= 32.0
    otr /= 32.0
    p_otr = float((w * otr).sum())
    p_msk = float((w * msk).sum())
    p_sdc = max(0.0, 1.0 - p_msk - p_otr)
    return {"MSK": p_msk, "SDC": p_sdc, "OTR": p_otr}


def analytic_rates(prior: Prior, eps: float, n_quad: int = 4001) -> dict[str, float]:
    """Average the per-term-identity component rates with k ~ U{0..n-1}."""
    n = prior.n
    norm2 = prior.norm ** 2
    acc = {"MSK": 0.0, "SDC": 0.0, "OTR": 0.0}
    for k in range(n):
        var_k = float(prior.component_var[k])
        var_r = max(0.0, norm2 - var_k)
        rk = _component_rates(var_k, var_r, eps, n_quad)
        for c in acc:
            acc[c] += rk[c] / n
    return acc


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16, help="inner-product length")
    ap.add_argument("--norm", type=float, default=4.0, help="Frobenius norm (matched)")
    ap.add_argument("--big-frac", type=float, default=0.95,
                    help="fraction of norm^2 carried by the concentrated big term")
    ap.add_argument("--samples", type=int, default=1500, help="MC input samples per prior")
    ap.add_argument("--n-quad", type=int, default=4001, help="quadrature nodes (analytic)")
    ap.add_argument("--seed", type=int, default=20260623)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--csv", type=str, default="", help="optional results.csv path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    balanced, concentrated = make_priors(args.n, args.norm, args.big_frac)

    print("Beta input-prior validation  —  analytical (no FI) vs Monte-Carlo")
    print(f"n={args.n}  norm={args.norm}  big_frac={args.big_frac}  "
          f"samples={args.samples}  n_quad={args.n_quad}  seed={args.seed}")
    print(f"matched: E[||x||^2]={args.norm**2:.3f} for both priors; "
          f"Var(out)={args.norm**2:.3f} for both (identical accumulator law)")
    print("=" * 88)
    hdr = (f"{'prior':>12} {'eps':>7} | {'MSK mc':>7} {'MSK an':>7} | "
           f"{'SDC mc':>7} {'SDC an':>7} | {'OTR mc':>7} {'OTR an':>7} | {'maxΔ':>6}")
    print(hdr)
    print("-" * 88)

    rows: list[dict] = []
    worst = 0.0
    sep_mc: dict[float, float] = {}
    sep_an: dict[float, float] = {}
    for eps in args.eps_grid:
        sdc_by_prior_mc: dict[str, float] = {}
        sdc_by_prior_an: dict[str, float] = {}
        for prior in (balanced, concentrated):
            mc = mc_rates(prior, eps, rng, args.samples)
            an = analytic_rates(prior, eps, args.n_quad)
            maxd = max(abs(mc[c] - an[c]) for c in ("MSK", "SDC", "OTR"))
            worst = max(worst, maxd)
            sdc_by_prior_mc[prior.name] = mc["SDC"]
            sdc_by_prior_an[prior.name] = an["SDC"]
            print(f"{prior.name:>12} {eps:>7.0e} | "
                  f"{mc['MSK']*100:>6.2f}% {an['MSK']*100:>6.2f}% | "
                  f"{mc['SDC']*100:>6.2f}% {an['SDC']*100:>6.2f}% | "
                  f"{mc['OTR']*100:>6.2f}% {an['OTR']*100:>6.2f}% | {maxd*100:>5.2f}")
            rows.append({"prior": prior.name, "eps": eps,
                         **{f"{c}_mc": mc[c] for c in ("MSK", "SDC", "OTR")},
                         **{f"{c}_an": an[c] for c in ("MSK", "SDC", "OTR")},
                         "max_abs_err": maxd, "mc_total": mc["_total"]})
        sep_mc[eps] = sdc_by_prior_mc["balanced"] - sdc_by_prior_mc["concentrated"]
        sep_an[eps] = sdc_by_prior_an["balanced"] - sdc_by_prior_an["concentrated"]
        print("-" * 88)

    # Binomial 95% CI half-width on the per-(sample,k,b) trials (worst-case p=0.5).
    trials = args.samples * args.n * 32
    ci = 1.96 * 0.5 / math.sqrt(trials)
    print(f"\nMC trials per cell = {trials:,}  (95% CI half-width <= {ci*100:.2f}%)")
    print(f"worst |analytic - MC| across grid = {worst*100:.2f}%")
    print("\nTyphoon falsification  (SDC_balanced - SDC_concentrated, same norm):")
    for eps in args.eps_grid:
        print(f"  eps={eps:>7.0e} | MC sep = {sep_mc[eps]*100:+6.2f} pp | "
              f"analytic sep = {sep_an[eps]*100:+6.2f} pp")
    max_sep_mc = max(abs(v) for v in sep_mc.values())

    method_ok = worst <= max(0.02, 3 * ci)          # within ~3 CI or 2pp
    falsified = max_sep_mc > 10 * ci                 # separation >> noise
    print("\n" + "=" * 88)
    print(f"VERDICT method-validated: {'PASS' if method_ok else 'FAIL'} "
          f"(worst err {worst*100:.2f}% vs tol {max(2.0, 300*ci):.2f}%)")
    print(f"VERDICT Typhoon-falsified: {'YES' if falsified else 'NO'} "
          f"(max |SDC separation| {max_sep_mc*100:.2f} pp, same Frobenius norm)")

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
