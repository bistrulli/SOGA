"""SOGA prediction of GEMM register-FI outcomes from the INPUT PRIOR alone.

Zero fault injections, zero input sampling. Per arithmetic site class:
  1. Priors as Gaussian mixtures: SOGA preprocessor (EM fit of uniform -> GM),
     Dirac / Gaussian parts written directly as gm(...) components.
  2. SOGA engine propagates  t = ALPHA*a ; T = t*b  (exact product moments),
     giving the law of one GEMM term T = ALPHA*a_k*b_kj.
  3. The 512-term accumulator is aggregated by moment matching to one Gaussian
     (what SOGA's prune(1) does on a sum of iid GM terms):
        acc_k ~ N(BETA*E[c0] + (k+1)E[T], BETA^2 Var[c0] + (k+1)Var[T]).
  4. Fault functional: for every node of a deterministic quadrature over the GM of
     the hit register value and every bit 0..31, the exact float32 flip gives the
     output perturbation delta; masked iff |C| >= |delta|/eps (gemm.cu check),
     NaN => masked (gemm.cu quirk), Inf => SDC. The rest-of-sum is Gaussian, so
     P(masked | node, bit) is closed form (two normal CDFs).
Structural classes (index/address/counter) are NOT predicted: they are the part
the GPU campaign measures and SOGA does not model.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from gemm_model import (ALPHA, ARITH_CLASSES, BETA, InputDist, class_counts,
                        flip_f32, mul32)

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str((EXP_DIR.parent.parent / "src").resolve()))
GM_CACHE = EXP_DIR / "gm_cache.json"
U01_PARTS = ((1.0, "uniform", (0.0, 1.0)),)


def _dec(x: float) -> str:
    """Plain positional decimal (SOGA NUM token has no exponent form)."""
    return np.format_float_positional(float(x), trim="-", precision=12)


@dataclass(frozen=True)
class GM:
    w: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    def moments(self) -> tuple[float, float]:
        m = float(np.sum(self.w * self.mu))
        return m, float(np.sum(self.w * (self.sd ** 2 + self.mu ** 2)) - m * m)

    def soga(self) -> str:
        return (f"gm([{', '.join(_dec(x) for x in self.w)}], "
                f"[{', '.join(_dec(x) for x in self.mu)}], "
                f"[{', '.join(_dec(x) for x in self.sd)}])")


# --------------------------------------------------------------------------- #
# 1. Priors -> GM via the SOGA preprocessor (seeded, cached)
# --------------------------------------------------------------------------- #
def _uniform_gm(lo: float, hi: float, k: int, seed: int) -> GM:
    key = f"uniform([{lo},{hi}],{k})@{seed}"
    cache = json.loads(GM_CACHE.read_text()) if GM_CACHE.exists() else {}
    if key not in cache:
        from sogaPreprocessor import compileUniform
        np.random.seed(seed)  # the preprocessor samples from the global numpy RNG
        with contextlib.redirect_stdout(io.StringIO()):
            text = compileUniform(input_prog=f"x = uniform([{lo}, {hi}], {k});")
        m = re.search(r"gm\(\[(.*?)\],\[(.*?)\],\[(.*?)\]\)", text)
        if m is None:
            raise RuntimeError(f"cannot parse preprocessor output: {text}")
        cache[key] = [[float(v) for v in g.split(",")] for g in m.groups()]
        GM_CACHE.write_text(json.dumps(cache, indent=1))
    w, mu, sd = (np.asarray(v) for v in cache[key])
    return GM(w, mu, sd)


def prior_gm(parts: tuple[tuple[float, str, tuple[float, ...]], ...], k: int,
             seed: int) -> GM:
    ws, mus, sds = [], [], []
    for weight, kind, p in parts:
        if kind == "dirac":
            g = GM(np.array([1.0]), np.array([p[0]]), np.array([0.0]))
        elif kind == "gauss":
            g = GM(np.array([1.0]), np.array([p[0]]), np.array([p[1]]))
        elif kind == "uniform":
            g = _uniform_gm(p[0], p[1], k, seed)
        else:
            raise ValueError(f"unknown prior part {kind}")
        ws.append(weight * g.w)
        mus.append(g.mu)
        sds.append(g.sd)
    return GM(np.concatenate(ws), np.concatenate(mus), np.concatenate(sds))


# --------------------------------------------------------------------------- #
# 2. SOGA engine: law of one term T = ALPHA * a * b
# --------------------------------------------------------------------------- #
def run_soga(program: str):
    from libSOGA import start_SOGA
    from producecfg import produce_cfg
    from sogaPreprocessor import compile2SOGA

    log = io.StringIO()
    with tempfile.NamedTemporaryFile("w", suffix=".soga", delete=False) as fh:
        fh.write(program)
        path = fh.name
    with contextlib.redirect_stderr(log), contextlib.redirect_stdout(log):
        out = start_SOGA(produce_cfg(compile2SOGA(path)), useR=False)
    Path(path).unlink()
    for marker in ("mismatched", "no viable alternative", "token recognition error",
                   "extraneous input", "missing"):
        if marker in log.getvalue():
            raise RuntimeError(f"SOGA parse error:\n{log.getvalue()}\n{program}")
    return out


def term_moments(gm_a: GM, gm_b: GM) -> tuple[float, float]:
    prog = (f"a = {gm_a.soga()};\nb = {gm_b.soga()};\n"
            f"t = {_dec(float(ALPHA))} * a;\nT = t * b;\n")
    dist = run_soga(prog)
    ix = dist.var_list.index("T")
    pi = np.asarray(dist.gm.pi, float)
    mu = np.asarray([m[ix] for m in dist.gm.mu], float)
    var = np.asarray([s[ix, ix] for s in dist.gm.sigma], float)
    mean = float(np.sum(pi * mu))
    return mean, float(np.sum(pi * (var + mu ** 2)) - mean ** 2)


# --------------------------------------------------------------------------- #
# 3-4. Fault functional by deterministic quadrature
# --------------------------------------------------------------------------- #
def gm_nodes(g: GM, n_per: int, width: float = 6.5) -> tuple[np.ndarray, np.ndarray]:
    xs, ws = [], []
    for w, mu, sd in zip(g.w, g.mu, g.sd):
        if sd == 0.0:
            xs.append(np.array([mu]))
            ws.append(np.array([w]))
            continue
        x = np.linspace(mu - width * sd, mu + width * sd, n_per)
        d = np.exp(-0.5 * ((x - mu) / sd) ** 2)
        xs.append(x)
        ws.append(w * d / d.sum())
    return np.concatenate(xs), np.concatenate(ws)


def gauss_nodes(m: float, s: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return gm_nodes(GM(np.array([1.0]), np.array([m]), np.array([s])), n_nodes, 7.0)


def p_masked(delta: np.ndarray, center: np.ndarray, m_r: float, s_r: float,
             eps: float) -> np.ndarray:
    """P_R(|center + R| >= |delta|/eps), R ~ N(m_r, s_r^2); NaN delta => masked."""
    with np.errstate(over="ignore", invalid="ignore"):
        tau = np.abs(delta) / eps
        if s_r == 0.0:
            p = (np.abs(center + m_r) >= tau).astype(float)
        else:
            p = 1.0 - (ndtr((tau - center - m_r) / s_r) - ndtr((-tau - center - m_r) / s_r))
    return np.where(np.isnan(delta), 1.0, p)


BITS = np.arange(32, dtype=np.uint32)


def _bitflip_delta(v32: np.ndarray) -> np.ndarray:
    """(32, len(v)) array of flip(v,b) - v in float64 (Inf/NaN propagate)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return flip_f32(v32[None, :], BITS[:, None]).astype(np.float64) - v32.astype(np.float64)[None, :]


@dataclass
class Prediction:
    msk: dict[str, dict[float, float]]   # class -> eps -> P(MSK)
    weights: dict[str, int]

    def arith(self, eps: float) -> float:
        tot = sum(self.weights[c] for c in ARITH_CLASSES)
        return sum(self.weights[c] * self.msk[c][eps] for c in ARITH_CLASSES) / tot


def predict(dist: InputDist, n: int, unroll: int, eps_grid: list[float],
            k_gm: int = 8, seed: int = 12345, n_per: int = 60, n_1d: int = 1201,
            k_stride: int = 8) -> Prediction:
    gm_a = prior_gm(dist.soga_parts, k_gm, seed)
    gm_b = prior_gm(U01_PARTS, k_gm, seed)
    gm_c = gm_b                                  # C0 ~ U[0,1], same GM
    mu_t, var_t = term_moments(gm_a, gm_b)
    mu_c, var_c = gm_c.moments()
    beta = float(BETA)

    # rest-of-sum laws (moment-matched sums of iid terms)
    m_rest_minus_k = beta * mu_c + (n - 1) * mu_t
    s_rest_minus_k = float(np.sqrt(beta ** 2 * var_c + (n - 1) * var_t))
    m_all_terms, s_all_terms = n * mu_t, float(np.sqrt(n * var_t))

    a_x, a_w = gm_nodes(gm_a, n_per)
    b_x, b_w = gm_nodes(gm_b, n_per)
    c_x, c_w = gm_nodes(gm_c, n_per)
    a32, b32, c32 = a_x.astype(np.float32), b_x.astype(np.float32), c_x.astype(np.float32)
    t32 = mul32(ALPHA, a32)
    T = t32.astype(np.float64)[:, None] * b32.astype(np.float64)[None, :]      # (Na, Nb)
    w2 = a_w[:, None] * b_w[None, :]

    with np.errstate(over="ignore", invalid="ignore"):
        d_ldg_a = (mul32(ALPHA, flip_f32(a32[None, :], BITS[:, None])).astype(np.float64)[:, :, None]
                   * b32.astype(np.float64)[None, None, :]) - T[None]
        d_fmul = _bitflip_delta(t32)[:, :, None] * b32.astype(np.float64)[None, None, :]
        d_ldg_b = t32.astype(np.float64)[None, :, None] * _bitflip_delta(b32)[:, None, :]
        acc0 = mul32(BETA, c32)
        d_ldg_c = mul32(BETA, flip_f32(c32[None, :], BITS[:, None])).astype(np.float64) \
            - acc0.astype(np.float64)[None, :]
        d_fmul_beta = _bitflip_delta(acc0)

    ks = sorted(set(range(0, n, k_stride)) | {n - 1})
    msk: dict[str, dict[float, float]] = {c: {} for c in ARITH_CLASSES}
    for eps in eps_grid:
        for cls, d in (("LDG.a", d_ldg_a), ("FMUL.alpha", d_fmul), ("LDG.b", d_ldg_b)):
            p = p_masked(d, T[None], m_rest_minus_k, s_rest_minus_k, eps)
            msk[cls][eps] = float(np.sum(p.mean(axis=0) * w2))
        for cls, d in (("LDG.c", d_ldg_c), ("FMUL.beta", d_fmul_beta)):
            p = p_masked(d, acc0.astype(np.float64)[None, :], m_all_terms, s_all_terms, eps)
            msk[cls][eps] = float(np.sum(p.mean(axis=0) * c_w))
        acc_p = []
        for k in ks:
            m_v = beta * mu_c + (k + 1) * mu_t
            s_v = float(np.sqrt(beta ** 2 * var_c + (k + 1) * var_t))
            v_x, v_w = gauss_nodes(m_v, s_v, n_1d)
            v32 = v_x.astype(np.float32)
            rest = n - 1 - k
            p = p_masked(_bitflip_delta(v32), v32.astype(np.float64)[None, :],
                         rest * mu_t, float(np.sqrt(rest * var_t)), eps)
            acc_p.append(float(np.sum(p.mean(axis=0) * v_w)))
        msk["FFMA"][eps] = float(np.mean(acc_p))
    return Prediction(msk, class_counts(n, unroll))
