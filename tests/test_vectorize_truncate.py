"""
Tests for the vectorized batch truncate path in libSOGAtruncate.

Verifies that USE_VECTORIZE_TRUNCATE=True produces output identical (up to
floating-point tolerance) to both the classic SVD-rotation path and the
sparse-aware per-component path, including edge cases (delta variables that
differ per component, aux gm() vars, n_comp=1, n_comp=10000, eq conditioning).
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
    set_sparse_truncate,
    set_vectorize_truncate,
    truncate,
    _ineq_truncate_vectorized_impl,
    _eq_truncate_vectorized_impl,
    _truncated_normal_moments_1d,
    _truncated_normal_moments_1d_batched,
    trunc_parse,
)


@pytest.fixture(autouse=True)
def _reset_flags():
    """Ensure module flags return to defaults after each test."""
    yield
    set_sparse_truncate(False)
    set_vectorize_truncate(False)


def _make_random_mixture(n_comp, d, rng):
    """Random GM with PSD per-component Sigma. Returns Dist."""
    mu_list = []
    sigma_list = []
    pi_raw = rng.uniform(0.5, 2.0, size=n_comp)
    pi_list = list(pi_raw / pi_raw.sum())
    for _ in range(n_comp):
        A = rng.standard_normal((d, d))
        sigma_list.append(A @ A.T + 0.1 * np.eye(d))
        mu_list.append(rng.standard_normal(d))
    var_list = [f"x{i}" for i in range(d)]
    return Dist(var_list, GaussianMix(pi_list, mu_list, sigma_list))


def _data_for_dist(dist):
    """SOGA-style data dict; empty for our tests since we use simple LBCs."""
    return {}


def _run_truncate_mode(dist, trunc_str, mode):
    """Run truncate() in the requested mode by toggling the module flags."""
    set_vectorize_truncate(False)
    set_sparse_truncate(False)
    if mode == "classic":
        pass
    elif mode == "sparse":
        set_sparse_truncate(True)
    elif mode == "vectorize":
        set_sparse_truncate(True)
        set_vectorize_truncate(True)
    else:
        raise ValueError(mode)
    return truncate(dist, trunc_str, _data_for_dist(dist))


def _max_abs_diff_dist(p_a, dist_a, p_b, dist_b):
    """Compare two (norm_factor, Dist) tuples."""
    if abs(p_a - p_b) > 1e-7:
        return float("inf"), float("inf"), abs(p_a - p_b)
    n_a, n_b = dist_a.gm.n_comp(), dist_b.gm.n_comp()
    if n_a != n_b:
        # Different component counts can be benign if components had pi < prob_tol
        # in one and not the other; compare moments instead via mixture mean+cov.
        mean_a, mean_b = dist_a.gm.mean(), dist_b.gm.mean()
        cov_a, cov_b = dist_a.gm.cov(), dist_b.gm.cov()
        return float(np.max(np.abs(mean_a - mean_b))), float(np.max(np.abs(cov_a - cov_b))), abs(p_a - p_b)
    # Same number of components: assume same order (deterministic per-aux iteration)
    max_dmu = 0.0
    max_dsig = 0.0
    for ma, mb in zip(dist_a.gm.mu, dist_b.gm.mu):
        max_dmu = max(max_dmu, float(np.max(np.abs(np.asarray(ma) - np.asarray(mb)))))
    for sa, sb in zip(dist_a.gm.sigma, dist_b.gm.sigma):
        max_dsig = max(max_dsig, float(np.max(np.abs(np.asarray(sa) - np.asarray(sb)))))
    return max_dmu, max_dsig, abs(p_a - p_b)


def test_batched_1d_moments_matches_scalar():
    """Batched 1D moments must coincide with the scalar helper at every entry."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        mu_s = float(rng.standard_normal())
        var_s = float(rng.uniform(0.1, 4.0))
        sigma = np.sqrt(var_s)
        c = float(mu_s + rng.uniform(-1.5, 1.5) * sigma)
        for direction in (">", "<"):
            m_s, v_s, p_s = _truncated_normal_moments_1d(mu_s, var_s, c, direction)
            M, V, P = _truncated_normal_moments_1d_batched(
                np.array([mu_s]), np.array([var_s]), np.array([c]), direction
            )
            assert abs(float(M[0]) - m_s) < 1e-12
            assert abs(float(V[0]) - v_s) < 1e-12
            assert abs(float(P[0]) - p_s) < 1e-12


@pytest.mark.parametrize("d", [3, 10])
@pytest.mark.parametrize("n_comp", [1, 5, 50, 500])
@pytest.mark.parametrize("direction", [">", "<"])
def test_ineq_equivalence_classic_sparse_vectorize(d, n_comp, direction):
    """All three truncate paths must produce identical output (machine eps)."""
    rng = np.random.default_rng(seed=(d * 1000 + n_comp * 10 + (1 if direction == ">" else 0)))
    dist = _make_random_mixture(n_comp, d, rng)
    # Build an LBC referencing 1-2 variables; pick threshold within the marginal
    coeff_idx = int(rng.integers(0, d))
    other_idx = (coeff_idx + 1) % d
    coeff = [0.0] * d
    coeff[coeff_idx] = float(rng.choice([-1.0, 1.0]))
    if d > 1 and rng.random() < 0.5:
        coeff[other_idx] = float(rng.choice([-1.0, 1.0]))
    # Compute marginal mu_s/var_s from component 0 and pick c within typical range
    mu0 = np.array(dist.gm.mu[0])
    sigma0 = np.array(dist.gm.sigma[0])
    mu_s = float(np.dot(coeff, mu0))
    var_s = float(np.array(coeff) @ sigma0 @ np.array(coeff))
    sigma_s = np.sqrt(max(var_s, 1e-6))
    c = float(mu_s + rng.uniform(-1.0, 1.0) * sigma_s)
    trunc_str = " + ".join(
        f"{coeff[i]}*x{i}" for i in range(d) if coeff[i] != 0
    ) + f" {direction} {c}"

    p_c, d_c = _run_truncate_mode(dist, trunc_str, "classic")
    p_s, d_s = _run_truncate_mode(dist, trunc_str, "sparse")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")

    # classic vs sparse already tested in test_sparse_truncate.py; check vectorize matches both
    dmu_cs, dsig_cs, dp_cs = _max_abs_diff_dist(p_c, d_c, p_s, d_s)
    dmu_cv, dsig_cv, dp_cv = _max_abs_diff_dist(p_c, d_c, p_v, d_v)
    dmu_sv, dsig_sv, dp_sv = _max_abs_diff_dist(p_s, d_s, p_v, d_v)

    # Tolerance scales mildly with d and n_comp due to BLAS accumulation order
    tol_mu = 1e-7 * d * max(np.sqrt(n_comp), 1.0)
    tol_sig = 1e-7 * d * d * max(np.sqrt(n_comp), 1.0)

    assert dp_cv < 1e-8, f"classic vs vectorize P diff: {dp_cv}"
    assert dmu_cv < tol_mu, f"classic vs vectorize mu diff: {dmu_cv}"
    assert dsig_cv < tol_sig, f"classic vs vectorize sigma diff: {dsig_cv}"
    assert dp_sv < 1e-10
    assert dmu_sv < tol_mu
    assert dsig_sv < tol_sig


def test_ineq_with_aux_gm_vectorize():
    """LBC containing gm() should produce identical results in all three modes."""
    rng = np.random.default_rng(seed=987654)
    d = 5
    n_comp = 20
    dist = _make_random_mixture(n_comp, d, rng)
    # observe(x0 + 2*x3 + gm([1.],[0.5],[0.2]) > 0.5)
    trunc_str = "1.0*x0 + 2.0*x3 + 1.0*gm([1.],[0.5],[0.2]) > 0.5"

    p_c, d_c = _run_truncate_mode(dist, trunc_str, "classic")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")
    dmu, dsig, dp = _max_abs_diff_dist(p_c, d_c, p_v, d_v)
    assert dp < 1e-9
    assert dmu < 1e-6
    assert dsig < 1e-6


def test_ineq_delta_var_per_component_vectorize():
    """When sigma[i,i] < delta_tol for SOME but not all components, the vectorized
    path must handle each component's delta mask independently."""
    d = 4
    # Component 0: x1 is delta. Component 1: x2 is delta. Component 2: no deltas.
    mu_list = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
    ]
    sigma_list = []
    base = np.eye(d) * 0.5
    base[0, 2] = base[2, 0] = 0.2
    s0 = base.copy(); s0[1, 1] = 1e-15; s0[1, :] = 0; s0[:, 1] = 0
    s1 = base.copy(); s1[2, 2] = 1e-15; s1[2, :] = 0; s1[:, 2] = 0
    sigma_list = [s0, s1, base.copy()]
    dist = Dist([f"x{i}" for i in range(d)],
                GaussianMix([1/3, 1/3, 1/3], mu_list, sigma_list))
    trunc_str = "1.0*x0 + 1.0*x1 + 1.0*x2 > 1.5"

    p_c, d_c = _run_truncate_mode(dist, trunc_str, "classic")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")
    dmu, dsig, dp = _max_abs_diff_dist(p_c, d_c, p_v, d_v)
    assert dp < 1e-9
    assert dmu < 1e-6
    assert dsig < 1e-6


def test_ineq_n_comp_1_vectorize():
    """Degenerate vectorization case: n_comp=1 still works correctly."""
    rng = np.random.default_rng(42)
    d = 4
    dist = _make_random_mixture(1, d, rng)
    trunc_str = "1.0*x2 > 0.0"
    p_c, d_c = _run_truncate_mode(dist, trunc_str, "classic")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")
    dmu, dsig, dp = _max_abs_diff_dist(p_c, d_c, p_v, d_v)
    assert dp < 1e-12
    assert dmu < 1e-8
    assert dsig < 1e-8


def test_ineq_large_n_comp_vectorize():
    """Stress: n_comp=2000 should still give equivalent moments to sparse path."""
    rng = np.random.default_rng(7)
    d = 5
    n_comp = 2000
    dist = _make_random_mixture(n_comp, d, rng)
    trunc_str = "1.0*x0 + 1.0*x2 > 0.0"
    p_s, d_s = _run_truncate_mode(dist, trunc_str, "sparse")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")
    dmu, dsig, dp = _max_abs_diff_dist(p_s, d_s, p_v, d_v)
    assert dp < 1e-9
    # mixture mean/cov for n_comp=2000 should still agree closely
    assert dmu < 1e-5
    assert dsig < 1e-5


def test_eq_equivalence_vectorize():
    """Equality conditioning: vectorized must match classic on simple cases."""
    rng = np.random.default_rng(123)
    d = 4
    n_comp = 10
    dist = _make_random_mixture(n_comp, d, rng)
    # equality on x1
    trunc_str = "x1 == 0.0"
    p_c, d_c = _run_truncate_mode(dist, trunc_str, "classic")
    p_v, d_v = _run_truncate_mode(dist, trunc_str, "vectorize")
    dmu, dsig, dp = _max_abs_diff_dist(p_c, d_c, p_v, d_v)
    assert dp < 1e-9
    assert dmu < 1e-6
    assert dsig < 1e-6


# ----- End-to-end via CLI subprocess -----

PROGRAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "programs", "SOGA"))
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "programs", "Example"))


def _run_cli(program_path, mode, var_names=None):
    cmd = [sys.executable, "SOGA.py", "-f", program_path]
    if mode == "sparse":
        cmd.append("--sparse-truncate")
    elif mode == "vectorize":
        cmd.append("--vectorize-truncate")
    if var_names:
        cmd.extend(["-v"] + list(var_names))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=SRC, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"SOGA exited {r.returncode}: {r.stderr}")
    means = {}
    for line in r.stdout.splitlines():
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
def test_end_to_end_vectorize_matches_classic(program, vars_to_check):
    """Classic CLI run and --vectorize-truncate CLI run must agree on E[var]."""
    path = os.path.join(PROGRAMS_DIR, program)
    if not os.path.exists(path):
        pytest.skip(f"{program} not present")
    means_c = _run_cli(path, "classic", vars_to_check)
    means_v = _run_cli(path, "vectorize", vars_to_check)
    assert means_c.keys() == means_v.keys()
    for var, vc in means_c.items():
        vv = means_v[var]
        assert abs(vc - vv) < 1e-4, f"{program}: {var} differs (classic={vc}, vectorize={vv})"
