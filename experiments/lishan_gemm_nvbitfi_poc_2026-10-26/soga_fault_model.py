"""The NVBitFI fault injector INSIDE a SOGA program.

For one GEMM output cell and one fault-site class, `build_program` writes a .soga
program with three blocks:

  1. KERNEL  - golden computation of the cell from the input priors:
         a, b ~ GM priors;  t = ALPHA*a;  T = t*b;  R = rest of the dot product;  C = T + R
  2. FAULT   - NVBitFI FLIP_SINGLE_BIT on the destination register h of the class
     (h = a for LDG.a, b for LDG.b, t for FMUL.alpha, acc_k for FFMA):
       * `sel`  : selector over the binades [2^e, 2^(e+1)) of h (exact discrete branch),
                  then `if` truncations put h in that binade -> its IEEE-754 exponent is known;
       * `kind` : which bit (1/32 each): mantissa bit b => additive error 2^(e-23+b);
                  sign / exponent bit j => multiplicative error |M| (1-2^-2^j, 2^2^j - 1, 2);
                  exponent reaching 255 => NaN (passes the gemm.cu check => masked);
       * `f`    : the error atoms; the output error is d = f*g (additive) or d = f*T (mult),
                  where g is what multiplies h downstream (b, ALPHA*b, t or 1).
  3. CHECK   - gemm.cu outcome: masked iff |d| <= eps*|C|, as SOGA `if` truncations;
               `sc` splits |C| into its two signs (disjoint, exact).
SOGA's output gives E[masked] = P(MSK) for that class. No sampling, no injection.

Weights: `sel` and `sc` are uniform selectors; `masked` is set to (#sel * #sc) so that
E[masked] undoes the selector weights exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from gemm_model import ALPHA, BETA, InputDist
from soga_predict import GM, U01_PARTS, prior_gm, run_soga

M_CLAMP = 1e12          # any |M| this large is SDC unless the hit value is ~0
MASS_MIN = 1e-5         # binades below this probability mass are dropped
SPLIT_K, SPLIT_RHO = 24, 0.25   # Gaussian splitting of single wide Gaussians (see split_gauss)


def num(x: float) -> str:
    """Shortest round-trip positional decimal (SOGA NUM has no exponent form)."""
    return np.format_float_positional(float(x), trim="-")


def gm_text(w, mu, sd) -> str:
    return (f"gm([{', '.join(num(v) for v in w)}], [{', '.join(num(v) for v in mu)}], "
            f"[{', '.join(num(v) for v in sd)}])")


def scaled(g: GM, c: float) -> GM:
    return GM(g.w, g.mu * c, g.sd * abs(c))


def gm_cdf(g: GM, x: float) -> float:
    out = 0.0
    for w, mu, sd in zip(g.w, g.mu, g.sd):
        out += w * (float(x >= mu) if sd == 0 else float(ndtr((x - mu) / sd)))
    return out


def binades(h: GM) -> tuple[list[tuple[int, int, float]], int]:
    """Binades (sign, e, mass) of h with mass >= MASS_MIN, and the zero-branch exponent
    e0: |h| < 2^e0 is treated as h = 0 (zero / denormal-like register)."""
    kept = []
    for s in (1, -1):
        for e in range(-60, 130):
            lo, hi = 2.0 ** e, 2.0 ** (e + 1)
            m = (gm_cdf(h, hi) - gm_cdf(h, lo)) if s > 0 else (gm_cdf(h, -lo) - gm_cdf(h, -hi))
            if m >= MASS_MIN:
                kept.append((s, e, m))
    e0 = min(e for _, e, _ in kept)
    # everything below the smallest kept binade goes to the zero branch
    return [(s, e, m) for s, e, m in kept], e0


def bit_atoms(e: int) -> tuple[list[float], list[float], int]:
    """Error atoms of a single-bit flip of a float32 with exponent e (normal number).
    Returns (additive kappas for the 23 mantissa bits, multiplicative |M| atoms, #NaN bits)."""
    kappa = [2.0 ** (e - 23 + b) for b in range(23)]
    mult = [2.0]                                   # sign bit: v -> -v, |d| = 2|v|
    n_nan = 0
    E = e + 127
    for j in range(8):
        if (E >> j) & 1:
            mult.append(1.0 - 2.0 ** (-(2 ** j)))  # exponent shrinks
        elif E + 2 ** j == 255:
            n_nan += 1                             # exponent 255, mantissa != 0 -> NaN -> masked
        else:
            mult.append(min(2.0 ** (2 ** j) - 1.0, M_CLAMP))
    return kappa, mult, n_nan


def split_gauss(m: float, s: float, k: int = SPLIT_K, rho: float = SPLIT_RHO) -> GM:
    """Gaussian splitting (Gaussian-sum approximation): N(m, s^2) as k narrow components of
    sd rho*s on a grid, weights ~ pdf; mean and variance preserved exactly."""
    z = np.linspace(-4.0, 4.0, k)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    z /= math.sqrt(float(np.sum(w * z ** 2)))      # weighted grid variance = 1
    return GM(w, m + s * math.sqrt(1.0 - rho ** 2) * z, np.full(k, rho * s))


def split_gm(g: GM) -> GM:
    """Split every non-degenerate wide component (only plain Gaussian priors are single wide ones)."""
    if len(g.w) > 1:
        return g
    return split_gauss(float(g.mu[0]), float(g.sd[0])) if g.sd[0] > 0 else g


ZERO_KAPPA = [2.0 ** (2 ** j - 127) for j in range(8)]   # flipping exponent bit j of +0.0


@dataclass(frozen=True)
class ClassSpec:
    kernel: str          # SOGA statements defining h, g, T, C
    h: str               # register hit by the fault
    g: str | None        # downstream multiplier (None => 1)
    h_gm: GM             # law of h (for binade selection only)
    g_signed: bool       # True if g may be negative
    c_positive: bool     # True if C < 0 has negligible mass (skip the sc split)


def class_spec(dist: InputDist, cls: str, n: int, k: int | None, k_gm: int, seed: int
               ) -> ClassSpec:
    gm_a = split_gm(prior_gm(dist.soga_parts, k_gm, seed))
    gm_b = prior_gm(U01_PARTS, k_gm, seed)
    ma, va = gm_a.moments()
    mb, vb = gm_b.moments()
    mc, vc = gm_b.moments()                       # C0 ~ U[0,1]
    alpha, beta = float(ALPHA), float(BETA)
    mt = alpha * ma * mb                          # one term T = ALPHA*a*b (a, b independent)
    vt = alpha ** 2 * ((va + ma ** 2) * (vb + mb ** 2) - (ma * mb) ** 2)
    a_signed = gm_cdf(gm_a, 0.0) > 1e-6 and dist.name.startswith("N(")
    if cls == "FFMA":
        m_h = beta * mc + (k + 1) * mt
        s_h = math.sqrt(beta ** 2 * vc + (k + 1) * vt)
        rest = n - 1 - k
        m_r, s_r = rest * mt, math.sqrt(rest * vt)
        h_gm = split_gauss(m_h, s_h)
        kern = [f"h = {gm_text(h_gm.w, h_gm.mu, h_gm.sd)};",
                f"R = gauss({num(m_r)}, {num(s_r)});" if rest > 0 else "R = 0;",
                "C = h + R;", "T = h;"]
        c_mean, c_sd = m_h + m_r, math.sqrt(s_h ** 2 + s_r ** 2)
        return ClassSpec("\n".join(kern), "h", None, h_gm, False, c_mean - 9 * c_sd > 0)
    m_r = beta * mc + (n - 1) * mt
    s_r = math.sqrt(beta ** 2 * vc + (n - 1) * vt)
    kern = [f"a = {gm_text(gm_a.w, gm_a.mu, gm_a.sd)};",
            f"b = {gm_text(gm_b.w, gm_b.mu, gm_b.sd)};",
            f"t = {num(alpha)} * a;", "T = t * b;",
            f"R = gauss({num(m_r)}, {num(s_r)});", "C = T + R;"]
    c_pos = (m_r + mt) - 9 * math.sqrt(s_r ** 2 + vt) > 0
    if cls == "LDG.a":
        kern.append(f"g = {num(alpha)} * b;")
        return ClassSpec("\n".join(kern), "a", "g", gm_a, False, c_pos)
    if cls == "FMUL.alpha":
        return ClassSpec("\n".join(kern), "t", "b", scaled(gm_a, alpha), False, c_pos)
    if cls == "LDG.b":
        return ClassSpec("\n".join(kern), "b", "t", gm_b, a_signed, c_pos)
    raise ValueError(f"class {cls} not modeled in SOGA")


def _check_block(spec: ClassSpec, d: str, sign: int, eps: float, weight: int) -> list[str]:
    """gemm.cu check: masked iff |d| <= eps*|C|, with |d| = sign*d known in this branch."""
    op = "-" if sign > 0 else "+"                     # eps*C - |d|, |d| = sign*d
    pos = [f"if {num(eps)}*C {op} {d} >= 0 {{", f"    masked = {weight};",
           "} else {", "    masked = 0;", "} end if;"]
    if spec.c_positive:
        return pos
    neg = [f"if -{num(eps)}*C {op} {d} >= 0 {{", f"    masked = {2 * weight};",
           "} else {", "    masked = 0;", "} end if;"]
    pos[1] = f"    masked = {2 * weight};"
    return ["sc = gm([0.5, 0.5], [0, 1], [0, 0]);", "if sc == 0 {", *pos, "} else {",
            *neg, "} end if;"]


def _binade_body(spec: ClassSpec, s_h: int, s_g: int, e: int | None, eps: float,
                 weight: int) -> list[str]:
    """Statements for h in binade e with sign s_h (e=None: zero branch)."""
    if e is None:
        kappa, mult, n_nan = ZERO_KAPPA, [], 0
        n_masked_free = 32 - len(kappa)              # mantissa -> denormal, sign -> -0
    else:
        kappa, mult, n_nan = bit_atoms(e)
        n_masked_free = n_nan
    kinds_w, kinds_v = [], []
    if kappa:
        kinds_w.append(len(kappa) / 32)
        kinds_v.append(0)
    if mult:
        kinds_w.append(len(mult) / 32)
        kinds_v.append(1)
    if n_masked_free:
        kinds_w.append(n_masked_free / 32)
        kinds_v.append(2)
    L = [f"kind = {gm_text(kinds_w, kinds_v, [0] * len(kinds_w))};"]
    # additive: d = kappa * g  (|d| = kappa*|g|)
    add_d = "f" if spec.g is None else "d"
    add = [f"f = {gm_text([1 / len(kappa)] * len(kappa), kappa, [0] * len(kappa))};"]
    if spec.g is not None:
        add.append(f"d = f * {spec.g};")
    add += _check_block(spec, add_d, s_g, eps, weight)
    # multiplicative: d = |M| * T  (|d| = |M|*|T|, sign of T = s_h*s_g)
    mul = []
    if mult:
        mul = [f"f = {gm_text([1 / len(mult)] * len(mult), mult, [0] * len(mult))};",
               "d = f * T;"] + _check_block(spec, "d", s_h * s_g, eps, weight)
    branches = []
    if kappa:
        branches.append((0, add))
    if mult:
        branches.append((1, mul))
    if n_masked_free:
        branches.append((2, [f"masked = {weight};"]))
    # chain: if kind == v0 {..} else { if kind == v1 {..} else {..} }
    def chain(bs):
        v, body = bs[0]
        if len(bs) == 1:
            return body
        return [f"if kind == {v} {{", *body, "} else {", *chain(bs[1:]), "} end if;"]
    return L + chain(branches)


def build_program(dist: InputDist, cls: str, n: int, eps: float, k: int | None = None,
                  k_gm: int = 8, seed: int = 20261026) -> tuple[str, dict]:
    spec = class_spec(dist, cls, n, k, k_gm, seed)
    bins, e0 = binades(spec.h_gm)
    g_signs = (1, -1) if spec.g_signed else (1,)
    # SOGA truncates on half-spaces only, and a second truncation acts on the moment-matched
    # result of the first. So each binade is the DIFFERENCE of two half-line events,
    #   P(h in [lo,hi) and masked) = P(h >= lo and masked) - P(h >= hi and masked),
    # i.e. two worlds with masked = +W and -W, each with ONE truncation on h.
    halves = []                                    # (sign of h, exponent e, condition, +-1)
    for s, e, _ in bins:
        lo, hi = 2.0 ** e, 2.0 ** (e + 1)
        if s > 0:
            halves += [(s, e, f"{spec.h} >= {num(lo)}", 1), (s, e, f"{spec.h} >= {num(hi)}", -1)]
        else:
            halves += [(s, e, f"{spec.h} <= -{num(lo)}", 1), (s, e, f"{spec.h} <= -{num(hi)}", -1)]
    z = num(2.0 ** e0)                             # zero branch: |h| < 2^e0
    halves += [(1, None, f"{spec.h} >= -{z}", 1), (1, None, f"{spec.h} >= {z}", -1)]
    W = len(halves)
    L = ["/* GEMM cell, NVBitFI FLIP_SINGLE_BIT on register " + spec.h + f" ({cls}), "
         f"A ~ {dist.name}, eps={eps} */", spec.kernel, "masked = 0;"]

    def worlds_block(s_g: int) -> list[str]:
        out = [f"sel = {gm_text([1 / W] * W, list(range(W)), [0] * W)};"]
        blocks = []
        for idx, (s, e, cond, sign) in enumerate(halves):
            body = _binade_body(spec, s, s_g, e, eps, sign * W)
            blocks.append((idx, [f"if {cond} {{", *body, "} else {", "masked = 0;", "} end if;"]))

        def chain(bs):
            v, body = bs[0]
            if len(bs) == 1:
                return body
            return [f"if sel == {v} {{", *body, "} else {", *chain(bs[1:]), "} end if;"]
        return out + chain(blocks)

    if len(g_signs) == 1:
        L += worlds_block(1)
    else:
        L += [f"if {spec.g} >= 0 {{", *worlds_block(1), "} else {", *worlds_block(-1),
              "} end if;"]
    meta = {"n_worlds": W, "n_binades": len(bins), "coverage": sum(m for _, _, m in bins) + gm_cdf(spec.h_gm, 2.0 ** e0)
            - gm_cdf(spec.h_gm, -(2.0 ** e0)), "lines": len(L)}
    return "\n".join(L) + "\n", meta


def p_masked_soga(dist: InputDist, cls: str, n: int, eps: float, k: int | None = None
                  ) -> tuple[float, dict]:
    prog, meta = build_program(dist, cls, n, eps, k)
    out = run_soga(prog)
    ix = out.var_list.index("masked")
    pi = np.asarray(out.gm.pi, float)
    mu = np.asarray([m[ix] for m in out.gm.mu], float)
    meta["n_comp"] = len(pi)
    meta["mass"] = float(pi.sum())
    return float(np.sum(pi * mu)), meta
