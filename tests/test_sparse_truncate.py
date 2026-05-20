"""
Tests for the sparse-aware truncate optimization in libSOGAtruncate.

Verifies that USE_SPARSE_TRUNCATE=True produces output identical (up to
floating-point tolerance) to the classic SVD-rotation path, including
edge cases (delta variables, zero-probability truncation, aux gm() vars).
"""

import os
import sys
import subprocess
import numpy as np
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from libSOGAshared import Dist, GaussianMix, prob_tol  # noqa: E402
from libSOGAtruncate import (  # noqa: E402
    USE_SPARSE_TRUNCATE,
    set_sparse_truncate,
    ineq_func,
    _ineq_func_classic,
    _ineq_func_sparse,
    _truncated_normal_moments_1d,
)


def _make_random_comp(d, n_aux, rng):
    """Build a random Dist with PSD Sigma over d original vars,
    plus n_aux auxiliary "virtual" dims kept off the cov (handled scalar-wise)."""
    A = rng.standard_normal((d, d))
    Sigma = A @ A.T + 0.1 * np.eye(d)
    mu = rng.standard_normal(d)
    var_list = [f"x{i}" for i in range(d)]
    gm = GaussianMix([1.0], [mu], [Sigma])
    return Dist(var_list, gm), mu, Sigma


class _FakeTruncRule:
    """Lightweight stand-in for TruncRule, exposing only the attributes that
    _ineq_func_{classic,sparse} read."""

    def __init__(self, coeff, const, type_, aux_pis=None, aux_means=None, aux_covs=None):
        self.coeff = list(coeff)
        self.const = float(const)
        self.type = type_
        self.aux_pis = aux_pis or []
        self.aux_means = aux_means or []
        self.aux_covs = aux_covs or []


def _max_abs_diff_gm(gm_a, gm_b):
    """Compare the truncated GaussianMix outputs from the two paths.
    Components must be in the same order (aux combination loop is deterministic)."""
    assert len(gm_a.pi) == len(gm_b.pi), "component count differs"
    max_dpi = max(abs(a - b) for a, b in zip(gm_a.pi, gm_b.pi))
    max_dmu = 0.0
    max_dsig = 0.0
    for ma, mb in zip(gm_a.mu, gm_b.mu):
        max_dmu = max(max_dmu, float(np.max(np.abs(np.asarray(ma) - np.asarray(mb)))))
    for sa, sb in zip(gm_a.sigma, gm_b.sigma):
        max_dsig = max(max_dsig, float(np.max(np.abs(np.asarray(sa) - np.asarray(sb)))))
    return max_dpi, max_dmu, max_dsig


@pytest.mark.parametrize("d", [5, 10, 30])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("direction", [">", "<"])
def test_equivalence_random_no_aux(d, k, direction):
    """Classic and sparse paths must agree on random PSD inputs without aux vars."""
    rng = np.random.default_rng(seed=(d * 100 + k * 10 + (1 if direction == ">" else 0)))
    n_samples = 20
    for _ in range(n_samples):
        comp, mu, Sigma = _make_random_comp(d, n_aux=0, rng=rng)
        coeff = np.zeros(d)
        positions = rng.choice(d, size=k, replace=False)
        for p in positions:
            v = rng.standard_normal()
            coeff[p] = v if abs(v) > 0.1 else 0.5
        mu_s = coeff.dot(mu)
        sigma_s = np.sqrt(max(coeff.dot(Sigma.dot(coeff)), 1e-10))
        c = float(mu_s + rng.uniform(-1.5, 1.5) * sigma_s)
        rule = _FakeTruncRule(coeff=coeff.tolist(), const=c, type_=direction)
        out_classic = _ineq_func_classic(rule, comp)
        out_sparse = _ineq_func_sparse(rule, comp)
        dpi, dmu, dsig = _max_abs_diff_gm(out_classic, out_sparse)
        # Tolerance scales mildly with d due to accumulated rounding in matmuls
        assert dpi < 1e-10, f"pi mismatch: {dpi}"
        assert dmu < 1e-8 * d, f"mu mismatch: {dmu}"
        assert dsig < 1e-8 * d * d, f"sigma mismatch: {dsig}"


def test_equivalence_with_aux_gm():
    """LBC containing gm() literals: extra aux components are folded into mu_s/var_s."""
    rng = np.random.default_rng(seed=12345)
    d = 6
    for _ in range(20):
        comp, mu, Sigma = _make_random_comp(d, n_aux=0, rng=rng)
        # Pretend the LBC referenced two extra gm()s: e.g. observe(x0 + 2*x3 + gm + gm > 0)
        # so coeff is length d + 2.
        coeff = np.zeros(d + 2)
        coeff[0] = 1.0
        coeff[3] = 2.0
        coeff[d] = 1.0       # first aux
        coeff[d + 1] = -1.0  # second aux
        aux_pis = [[1.0], [1.0]]
        aux_means = [[0.5], [-0.3]]
        aux_covs = [[0.04], [0.09]]
        c = float(rng.uniform(-2, 2))
        rule = _FakeTruncRule(
            coeff=coeff.tolist(), const=c, type_=">",
            aux_pis=aux_pis, aux_means=aux_means, aux_covs=aux_covs,
        )
        out_classic = _ineq_func_classic(rule, comp)
        out_sparse = _ineq_func_sparse(rule, comp)
        dpi, dmu, dsig = _max_abs_diff_gm(out_classic, out_sparse)
        assert dpi < 1e-10
        assert dmu < 1e-6
        assert dsig < 1e-6


def test_delta_var_substitution():
    """A variable with sigma[i,i] < delta_tol is treated as a Dirac and substituted
    into the constant. Both paths must agree."""
    d = 4
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    Sigma = np.diag([1.0, 1e-15, 0.5, 0.5])  # var[1] is delta
    Sigma[0, 2] = Sigma[2, 0] = 0.3
    comp = Dist([f"x{i}" for i in range(d)], GaussianMix([1.0], [mu], [Sigma]))
    coeff = [1.0, 1.0, 0.0, 0.0]  # uses both x0 (random) and x1 (delta)
    c = 2.5
    rule = _FakeTruncRule(coeff=coeff, const=c, type_=">")
    out_classic = _ineq_func_classic(rule, comp)
    out_sparse = _ineq_func_sparse(rule, comp)
    dpi, dmu, dsig = _max_abs_diff_gm(out_classic, out_sparse)
    assert dpi < 1e-10
    assert dmu < 1e-8
    assert dsig < 1e-8


def test_zero_probability_edge():
    """When the truncation event has effectively zero mass, both paths return pi=0
    and leave the component (mu, Sigma) unchanged."""
    d = 3
    mu = np.zeros(d)
    Sigma = np.eye(d) * 0.01  # tight Gaussian near origin
    comp = Dist([f"x{i}" for i in range(d)], GaussianMix([1.0], [mu], [Sigma]))
    coeff = [1.0, 0.0, 0.0]
    c = 100.0  # x0 > 100 is essentially impossible for x0 ~ N(0, 0.01)
    rule = _FakeTruncRule(coeff=coeff, const=c, type_=">")
    out_classic = _ineq_func_classic(rule, comp)
    out_sparse = _ineq_func_sparse(rule, comp)
    # Both should give pi=0 for this combination
    assert sum(out_classic.pi) < prob_tol
    assert sum(out_sparse.pi) < prob_tol


def test_truncated_normal_helper_matches_scipy():
    """Sanity check the 1D Mills-ratio helper against scipy.stats.truncnorm."""
    from scipy.stats import truncnorm
    rng = np.random.default_rng(0)
    for _ in range(50):
        mu_s = float(rng.standard_normal())
        var_s = float(rng.uniform(0.1, 4.0))
        sigma = np.sqrt(var_s)
        c = float(mu_s + rng.uniform(-1.5, 1.5) * sigma)
        # Lower truncation (>)
        m_hat, v_hat, P = _truncated_normal_moments_1d(mu_s, var_s, c, ">")
        a_std = (c - mu_s) / sigma
        ref = truncnorm(a=a_std, b=np.inf, loc=mu_s, scale=sigma)
        assert abs(m_hat - ref.mean()) < 1e-8
        assert abs(v_hat - ref.var()) < 1e-8
        # Upper truncation (<)
        m_hat, v_hat, P = _truncated_normal_moments_1d(mu_s, var_s, c, "<")
        ref = truncnorm(a=-np.inf, b=a_std, loc=mu_s, scale=sigma)
        assert abs(m_hat - ref.mean()) < 1e-8
        assert abs(v_hat - ref.var()) < 1e-8


def test_dispatcher_routes_correctly():
    """ineq_func should call _classic by default and _sparse when flag is on."""
    rng = np.random.default_rng(0)
    d = 5
    comp, mu, Sigma = _make_random_comp(d, n_aux=0, rng=rng)
    coeff = [0.0, 1.0, 0.0, 0.0, 0.0]
    rule = _FakeTruncRule(coeff=coeff, const=float(mu[1]), type_=">")

    set_sparse_truncate(False)
    out_default = ineq_func(rule, comp)
    out_classic = _ineq_func_classic(rule, comp)
    dpi, dmu, dsig = _max_abs_diff_gm(out_default, out_classic)
    assert dpi == 0 and dmu == 0 and dsig == 0  # exact same code path

    set_sparse_truncate(True)
    out_sparse_via_dispatcher = ineq_func(rule, comp)
    out_sparse_direct = _ineq_func_sparse(rule, comp)
    dpi, dmu, dsig = _max_abs_diff_gm(out_sparse_via_dispatcher, out_sparse_direct)
    assert dpi == 0 and dmu == 0 and dsig == 0

    set_sparse_truncate(False)  # restore default


# ----- End-to-end sanity: a known analytical posterior must match in both modes -----

PROGRAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "programs", "SOGA"))


def _run_cli(program, sparse=False, var_names=None):
    """Run SOGA.py via subprocess and parse E[var] from stdout."""
    cmd = [
        sys.executable, "SOGA.py",
        "-f", os.path.join(PROGRAMS_DIR, program),
    ]
    if sparse:
        cmd.append("--sparse-truncate")
    if var_names:
        cmd.extend(["-v"] + list(var_names))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SRC, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"SOGA exited {result.returncode}: {result.stderr}")
    means = {}
    for line in result.stdout.splitlines():
        if line.startswith("E[") and ":" in line:
            name = line[2:line.index("]")]
            value = float(line.split(":")[-1].strip())
            means[name] = value
    return means


@pytest.mark.parametrize("program,vars_to_check", [
    ("Bernoulli.soga", None),
    ("BayesPointMachine.soga", ["w[0]", "w[1]", "w[2]"]),
    ("TrueSkills.soga", ["skillA", "skillB", "skillC"]),
])
def test_end_to_end_equivalence(program, vars_to_check):
    """Two runs (classic vs sparse) of the same program must produce the same posterior means."""
    if not os.path.exists(os.path.join(PROGRAMS_DIR, program)):
        pytest.skip(f"{program} not present")
    means_classic = _run_cli(program, sparse=False, var_names=vars_to_check)
    means_sparse = _run_cli(program, sparse=True, var_names=vars_to_check)
    assert means_classic.keys() == means_sparse.keys()
    for var, val_c in means_classic.items():
        val_s = means_sparse[var]
        # 5 decimals of agreement (SOGA CLI rounds to 5 decimals before printing)
        assert abs(val_c - val_s) < 1e-4, f"{program}: {var} differs (classic={val_c}, sparse={val_s})"
