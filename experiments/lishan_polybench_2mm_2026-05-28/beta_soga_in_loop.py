"""Beta validation, SOGA-IN-THE-LOOP: re-run the beta experiment but let the
REAL SOGA engine propagate the fault, instead of the hand-rolled analytical
quadrature in `beta_input_prior_validation.py`. If the engine reproduces the
Monte-Carlo ground truth, claims (1) Typhoon-falsified and (2) analytical
resilience are validated WITH SOGA in the loop — and the float fault-encoding
built here is the reusable artifact for gamma (the full 2mm).

Pipeline per (prior, hit-term identity):
  1. Characterize the fault CHANNEL: enumerate every single-bit-flip on one term
     v ~ N(0, sigma_hit^2); finite flips give relative perturbations r = (v'-v)/v
     (magnitude-independent for sign/mantissa), non-finite flips give the OTR mass.
     Bin the finite r into a zero-variance categorical  r ~ gm([w],[r],[0]).
  2. Let SOGA propagate it through the inner product, via a .soga program:
         x0 = gauss(0, s0); ...; xn1 = gauss(0, sn1);
         out = x0; out = out + x1; ...;        # affine accumulator
         err = xhit * r;                       # RV x RV product (mul_func, exact)
     SOGA returns the JOINT GM over (out, err); crucially it tracks the
     covariance cov(out,err) = sigma_hit^2 * r_k (the hit term is part of out),
     i.e. the engine itself captures the masking coupling rho = sigma_hit/norm.
  3. Post-hoc, per GM component, integrate the correlated bivariate
     P(|err| <= eps|out|) -> MSK; SDC = rest of the finite mass; then scale by
     (1 - P_OTR) and add P_OTR as the two-tier OTR (Bernoulli) channel.
  4. Average over which term is hit (k ~ U); compare SOGA vs MC vs analytic.

Only modeling approximation: r is taken INDEPENDENT of the exact x_hit value
(we propagate r's marginal). MC keeps the exact per-sample coupling, so
SOGA == MC validates exactly that approximation + the engine's propagation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

from beta_input_prior_validation import (Prior, analytic_rates, make_priors,
                                         mc_rates)
from float_example import flip_bit_f32

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str((EXP_DIR.parent.parent / "src").resolve()))

_SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf)


def _phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) / _SQRT2))


def _dec(x: float) -> str:
    """Plain positional decimal (SOGA NUM token has no exponent form)."""
    return np.format_float_positional(float(x), trim="-")


# --------------------------------------------------------------------------- #
# 1. Fault-channel characterization (one-time per term identity, NOT kernel MC)
# --------------------------------------------------------------------------- #
def characterize_channel(sigma_hit: float, rng: np.random.Generator,
                         n_v: int, log2_bin: float
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Enumerate every (value, bit) single flip on a term v ~ N(0, sigma_hit^2).
    Returns (weights, ratios) of a binned zero-variance categorical of the finite
    relative perturbation r=(v'-v)/v, plus the OTR probability (non-finite flips)."""
    vs = (rng.standard_normal(n_v) * sigma_hit).astype(np.float32)
    ratios: list[float] = []
    n_otr = 0
    n_tot = 0
    for v in vs:
        fv = float(v)
        if fv == 0.0:
            continue
        for b in range(32):
            vp = flip_bit_f32(fv, b)
            n_tot += 1
            if not np.isfinite(vp):
                n_otr += 1
            else:
                ratios.append((vp - fv) / fv)
    p_otr = n_otr / n_tot
    r = np.asarray(ratios, dtype=np.float64)
    # bin in signed log2-magnitude space to keep the categorical small
    w_each = (1.0 - p_otr) / len(r)
    keys = np.where(r == 0.0, 0,
                    np.sign(r).astype(int) *
                    (1 + np.round(np.log2(np.abs(r) + 1e-300) / log2_bin).astype(int)))
    atoms: dict[int, list[float]] = {}
    for key, val in zip(keys.tolist(), r.tolist()):
        atoms.setdefault(key, []).append(val)
    weights = np.array([len(v) * w_each for v in atoms.values()])
    means = np.array([float(np.mean(v)) for v in atoms.values()])
    return weights, means, p_otr


# --------------------------------------------------------------------------- #
# 2. SOGA engine: build + run the inner-product program with the fault product
# --------------------------------------------------------------------------- #
def run_soga_full(program_text: str):
    from sogaPreprocessor import compile2SOGA
    from producecfg import produce_cfg
    from libSOGA import start_SOGA

    tmp = Path("/tmp") / "soga_beta_inloop.soga"
    tmp.write_text(program_text)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
        compiled = compile2SOGA(str(tmp))
        cfg = produce_cfg(compiled)
        out = start_SOGA(cfg, useR=False)
    log = err.getvalue()
    for m in ("mismatched", "no viable alternative", "expecting",
              "token recognition error", "extraneous input", "missing"):
        if m in log:
            raise RuntimeError(f"SOGA parse error:\n{log}\n--- program ---\n{program_text}")
    return out


def build_program(prior: Prior, hit: int, weights: np.ndarray,
                  means: np.ndarray) -> str:
    sd = np.sqrt(prior.component_var)
    lines = [f"/* beta SOGA-in-loop: n={prior.n} hit-term={hit} */"]
    for k in range(prior.n):
        lines.append(f"x{k} = gauss(0, {_dec(sd[k])});")
    lines.append("out = x0;")
    for k in range(1, prior.n):
        lines.append(f"out = out + x{k};")
    wstr = ", ".join(_dec(w) for w in weights)
    mstr = ", ".join(_dec(m) for m in means)
    sstr = ", ".join("0" for _ in means)
    lines.append(f"r = gm([{wstr}], [{mstr}], [{sstr}]);")
    lines.append(f"err = x{hit} * r;")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 3. Post-hoc correlated-bivariate masking integral, per GM component
# --------------------------------------------------------------------------- #
def comp_msk(var_o: float, var_e: float, cov: float, eps: float,
             n_quad: int = 2001) -> float:
    """P(|err| <= eps|out|) for (out,err) ~ N(0, [[var_o,cov],[cov,var_e]])."""
    if var_e <= 1e-300:                       # err == 0 -> always masked
        return 1.0
    sd_o = math.sqrt(var_o)
    s2 = var_e - cov * cov / var_o            # conditional variance err|out
    if s2 <= 1e-12 * var_e:                   # perfectly correlated: err = k*out
        return 1.0 if abs(cov / var_o) <= eps else 0.0
    s = math.sqrt(s2)
    o = np.linspace(-8.0 * sd_o, 8.0 * sd_o, n_quad)
    dens = np.exp(-0.5 * (o / sd_o) ** 2)
    w = dens / dens.sum()
    m = cov / var_o * o                        # conditional mean err|out
    t = eps * np.abs(o)
    p = _phi((t - m) / s) - _phi((-t - m) / s)
    return float((w * p).sum())


def soga_rates_for_hit(prior: Prior, hit: int, eps_grid: list[float],
                       rng: np.random.Generator, n_v: int, log2_bin: float
                       ) -> dict[float, dict[str, float]]:
    sigma_hit = math.sqrt(float(prior.component_var[hit]))
    weights, means, p_otr = characterize_channel(sigma_hit, rng, n_v, log2_bin)
    dist = run_soga_full(build_program(prior, hit, weights, means))
    io = dist.var_list.index("out")
    ie = dist.var_list.index("err")
    comps = [(float(dist.gm.pi[k]),
              float(dist.gm.sigma[k][io, io]),
              float(dist.gm.sigma[k][ie, ie]),
              float(dist.gm.sigma[k][io, ie])) for k in range(dist.gm.n_comp())]
    out: dict[float, dict[str, float]] = {}
    for eps in eps_grid:
        msk_finite = sum(w * comp_msk(vo, ve, cv, eps) for (w, vo, ve, cv) in comps)
        # weights already sum to (1 - p_otr); msk/sdc live in the finite mass
        msk = msk_finite
        sdc = max(0.0, (1.0 - p_otr) - msk_finite)
        out[eps] = {"MSK": msk, "SDC": sdc, "OTR": p_otr}
    return out


def soga_rates(prior: Prior, eps_grid: list[float], rng: np.random.Generator,
               n_v: int, log2_bin: float) -> dict[float, dict[str, float]]:
    """Average over which term is hit (k ~ U{0..n-1})."""
    acc = {eps: {"MSK": 0.0, "SDC": 0.0, "OTR": 0.0} for eps in eps_grid}
    # group identical term identities by variance to avoid redundant engine runs
    seen: dict[float, dict[float, dict[str, float]]] = {}
    for hit in range(prior.n):
        key = round(float(prior.component_var[hit]), 12)
        if key not in seen:
            seen[key] = soga_rates_for_hit(prior, hit, eps_grid, rng, n_v, log2_bin)
        for eps in eps_grid:
            for c in ("MSK", "SDC", "OTR"):
                acc[eps][c] += seen[key][eps][c] / prior.n
    return acc


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--norm", type=float, default=4.0)
    ap.add_argument("--big-frac", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=1500, help="MC input samples")
    ap.add_argument("--n-v", type=int, default=4000, help="values for channel characterization")
    ap.add_argument("--log2-bin", type=float, default=0.5, help="ratio binning resolution")
    ap.add_argument("--seed", type=int, default=20260623)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    balanced, concentrated = make_priors(args.n, args.norm, args.big_frac)

    print("Beta SOGA-IN-THE-LOOP  —  SOGA engine vs Monte-Carlo vs analytic")
    print(f"n={args.n}  norm={args.norm}  big_frac={args.big_frac}  "
          f"samples={args.samples}  n_v={args.n_v}  seed={args.seed}")
    print("=" * 96)
    print(f"{'prior':>12} {'eps':>7} | {'SDC mc':>7} {'SDC soga':>9} {'SDC an':>7} | "
          f"{'MSK mc':>7} {'MSK soga':>9} | {'OTR mc':>7} {'OTR soga':>9} | {'|soga-mc|':>9}")
    print("-" * 96)

    worst = 0.0
    sep_mc: dict[float, float] = {}
    sep_soga: dict[float, float] = {}
    for prior in (balanced, concentrated):
        soga = soga_rates(prior, args.eps_grid, rng, args.n_v, args.log2_bin)
        for eps in args.eps_grid:
            mc = mc_rates(prior, eps, rng, args.samples)
            an = analytic_rates(prior, eps)
            sg = soga[eps]
            d = max(abs(sg[c] - mc[c]) for c in ("MSK", "SDC", "OTR"))
            worst = max(worst, d)
            print(f"{prior.name:>12} {eps:>7.0e} | "
                  f"{mc['SDC']*100:>6.2f}% {sg['SDC']*100:>8.2f}% {an['SDC']*100:>6.2f}% | "
                  f"{mc['MSK']*100:>6.2f}% {sg['MSK']*100:>8.2f}% | "
                  f"{mc['OTR']*100:>6.2f}% {sg['OTR']*100:>8.2f}% | {d*100:>8.2f}")
            sep_mc.setdefault(eps, 0.0)
            sep_soga.setdefault(eps, 0.0)
            sep_mc[eps] += (1 if prior is balanced else -1) * mc["SDC"]
            sep_soga[eps] += (1 if prior is balanced else -1) * sg["SDC"]
        print("-" * 96)

    print(f"\nworst |SOGA-engine - MC| across grid = {worst*100:.2f}%")
    print("\nTyphoon falsification (SDC_balanced - SDC_concentrated, same Frobenius norm):")
    for eps in args.eps_grid:
        print(f"  eps={eps:>7.0e} | MC sep = {sep_mc[eps]*100:+6.2f} pp | "
              f"SOGA sep = {sep_soga[eps]*100:+6.2f} pp")
    max_sep_soga = max(abs(v) for v in sep_soga.values())

    method_ok = worst <= 0.03
    falsified = max_sep_soga > 0.02
    print("\n" + "=" * 96)
    print(f"VERDICT SOGA-in-loop validated: {'PASS' if method_ok else 'FAIL'} "
          f"(worst |SOGA-MC| {worst*100:.2f}% vs 3.0% tol)")
    print(f"VERDICT Typhoon-falsified by SOGA: {'YES' if falsified else 'NO'} "
          f"(max |SOGA SDC separation| {max_sep_soga*100:.2f} pp, same norm)")


if __name__ == "__main__":
    main()
