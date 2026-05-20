"""
Standalone prototype: sparse-aware truncate vs SOGA's current rotation method.

Implementation #1: SOGA-style — full d x d rotation via find_basis (SVD)
                   + A * Sigma * A.T + inv(A) + back-projection. Faithful
                   replication of libSOGAtruncate.py:ineq_func (1D inequality).

Implementation #2: Sparse-aware — rank-1 conditional Gaussian update.
                   No rotation. Uses g = Sigma @ alpha and updates mu/Sigma
                   via the closed-form formulas derived from the law of total
                   variance.

This prototype:
  - Verifies that the two implementations produce IDENTICAL output on
    random inputs (PSD Sigma, sparse alpha, random c giving |z|<1.5).
  - Measures empirical speedup at varying d and k.
  - Does NOT modify SOGA source code.
"""

import numpy as np
from scipy.stats import norm
import time
import csv
import os
from statistics import mean, stdev


# === 1D truncated Gaussian moments (closed form) ===

def truncated_normal_moments_1d(mu_s, var_s, c, direction):
    """Returns (m_hat, v_hat, P) for s ~ N(mu_s, var_s) truncated to direction at c.

    direction: '>' for s > c, '<' for s < c.
    """
    if var_s <= 0:
        return mu_s, 0.0, 1.0 if (direction == '>' and mu_s > c) or (direction == '<' and mu_s < c) else 0.0
    sigma = np.sqrt(var_s)
    z = (c - mu_s) / sigma
    if direction == '>':
        denom = 1.0 - norm.cdf(z)
        if denom < 1e-300:
            return mu_s, var_s, 0.0
        lam = norm.pdf(z) / denom
    else:  # '<'
        denom = norm.cdf(z)
        if denom < 1e-300:
            return mu_s, var_s, 0.0
        lam = -norm.pdf(z) / denom
    m_hat = mu_s + sigma * lam
    v_hat = var_s * (1.0 - lam * (lam - z))
    P = denom
    return m_hat, v_hat, P


# === Implementation #1: SOGA-style (faithful) ===

def find_basis(alpha):
    """Replicates libSOGAtruncate.py:find_basis."""
    alpha = np.array(alpha, dtype=float)
    u, s, v = np.linalg.svd([alpha])
    alpha1 = v[:, 1:]
    A = np.vstack((alpha.reshape(1, alpha.shape[0]), alpha1.transpose()))
    return A


def truncate_soga_style(mu, Sigma, alpha, c, direction):
    """
    Faithful replication of SOGA's ineq_func for the 1D inequality case
    (no auxiliary variables). Steps:

        1) normalize alpha
        2) A = find_basis(alpha)          ; d x d via SVD
        3) transl_mu    = A @ mu          ; O(d^2)
        4) transl_sigma = A @ Sigma @ A.T ; O(d^3)
        5) 1D truncated moments along axis 0
        6) conditional rank-1 update in rotated space
        7) A_inv = inv(A)                 ; O(d^3)
        8) back-project mu', Sigma'       ; O(d^3)
    """
    d = len(mu)
    alpha = np.asarray(alpha, dtype=float)

    # (1) normalize
    n = np.linalg.norm(alpha)
    alpha_n = alpha / n
    c_n = c / n

    # (2) build rotation
    A = find_basis(alpha_n)

    # (3-4) rotate
    transl_mu = A.dot(mu)
    transl_sigma = A.dot(Sigma).dot(A.T)

    # (5) 1D moments along axis 0
    s_mu = transl_mu[0]
    s_var = transl_sigma[0, 0]
    m_hat, v_hat, P = truncated_normal_moments_1d(s_mu, s_var, c_n, direction)
    if P < 1e-300:
        return mu.copy(), Sigma.copy(), 0.0

    # (6) conditional moments in rotated space (rank-1 update along axis 0)
    g_rot = transl_sigma[:, 0]
    v_rot = transl_sigma[0, 0]
    new_transl_mu = transl_mu + (g_rot / v_rot) * (m_hat - s_mu)
    new_transl_sigma = transl_sigma - np.outer(g_rot, g_rot) / v_rot * (1.0 - v_hat / v_rot)

    # (7-8) back-project
    A_inv = np.linalg.inv(A)
    new_mu = A_inv.dot(new_transl_mu)
    new_sigma = A_inv.dot(new_transl_sigma).dot(A_inv.T)

    return new_mu, new_sigma, P


# === Implementation #2: Sparse-aware (rank-1 in original space) ===

def truncate_sparse_aware(mu, Sigma, alpha, c, direction):
    """
    Direct rank-1 conditional Gaussian update. No rotation, no inversion.

        g     = Sigma @ alpha       ; O(d^2) dense (could be O(k*d) if alpha sparse)
        var_s = alpha @ g           ; O(d) dense, O(k) sparse
        m_hat, v_hat, P  = 1D truncated moments
        mu'    = mu    + (g/var_s) * (m_hat - mu_s)
        Sigma' = Sigma - (g g^T / var_s) * (1 - v_hat/var_s)   ; O(d^2)
    """
    alpha = np.asarray(alpha, dtype=float)
    mu_s = float(alpha.dot(mu))
    g = Sigma.dot(alpha)
    var_s = float(alpha.dot(g))
    if var_s <= 0:
        return mu.copy(), Sigma.copy(), 0.0
    m_hat, v_hat, P = truncated_normal_moments_1d(mu_s, var_s, c, direction)
    if P < 1e-300:
        return mu.copy(), Sigma.copy(), 0.0
    new_mu = mu + (g / var_s) * (m_hat - mu_s)
    new_sigma = Sigma - np.outer(g, g) / var_s * (1.0 - v_hat / var_s)
    return new_mu, new_sigma, P


# === Sparse-aware that exploits k (only k columns of Sigma) ===

def truncate_sparse_aware_k(mu, Sigma, alpha, c, direction):
    """Same as sparse_aware but Sigma @ alpha is computed only on non-zero positions of alpha."""
    alpha = np.asarray(alpha, dtype=float)
    nz = np.where(alpha != 0)[0]
    mu_s = float(alpha[nz].dot(mu[nz]))
    g = Sigma[:, nz].dot(alpha[nz])  # O(k * d)
    var_s = float(alpha[nz].dot(g[nz]))
    if var_s <= 0:
        return mu.copy(), Sigma.copy(), 0.0
    m_hat, v_hat, P = truncated_normal_moments_1d(mu_s, var_s, c, direction)
    if P < 1e-300:
        return mu.copy(), Sigma.copy(), 0.0
    new_mu = mu + (g / var_s) * (m_hat - mu_s)
    new_sigma = Sigma - np.outer(g, g) / var_s * (1.0 - v_hat / var_s)
    return new_mu, new_sigma, P


# === Random input generators ===

def gen_random_psd(d, rng):
    A = rng.standard_normal((d, d))
    return A @ A.T + 0.1 * np.eye(d)


def gen_random_mu(d, rng):
    return rng.standard_normal(d)


def gen_sparse_alpha(d, k, rng):
    positions = rng.choice(d, size=k, replace=False)
    alpha = np.zeros(d)
    for p in positions:
        v = rng.standard_normal()
        while abs(v) < 0.1:
            v = rng.standard_normal()
        alpha[p] = v
    return alpha


def gen_random_c(mu, Sigma, alpha, rng):
    """Pick c so that |z| < 1.5 (truncation has non-trivial probability)."""
    mu_s = alpha.dot(mu)
    var_s = float(alpha.dot(Sigma.dot(alpha)))
    sigma_s = np.sqrt(max(var_s, 1e-10))
    z = rng.uniform(-1.5, 1.5)
    return mu_s + z * sigma_s


# === Correctness verification ===

def verify_correctness(N=200, d_values=(5, 10, 30, 100, 300), k_values=(1, 2, 3, 5), log_path=None, master_seed=42):
    lines = ["=== CORRECTNESS VERIFICATION ===", f"N={N} samples per (d, k, direction)", ""]
    all_pass = True
    max_dmu = 0.0
    max_dsig = 0.0
    max_dP = 0.0
    rng_master = np.random.default_rng(master_seed)
    for d in d_values:
        for k in k_values:
            if k > d:
                continue
            for direction in ['>', '<']:
                dmu_list, dsig_list, dP_list = [], [], []
                for i in range(N):
                    seed = rng_master.integers(0, 2**31 - 1)
                    rng = np.random.default_rng(seed)
                    Sigma = gen_random_psd(d, rng)
                    mu = gen_random_mu(d, rng)
                    alpha = gen_sparse_alpha(d, k, rng)
                    c = gen_random_c(mu, Sigma, alpha, rng)
                    mu_a, S_a, P_a = truncate_soga_style(mu, Sigma, alpha, c, direction)
                    mu_b, S_b, P_b = truncate_sparse_aware(mu, Sigma, alpha, c, direction)
                    mu_c, S_c, P_c = truncate_sparse_aware_k(mu, Sigma, alpha, c, direction)
                    # all three should agree
                    dmu_list.append(max(np.max(np.abs(mu_a - mu_b)), np.max(np.abs(mu_a - mu_c))))
                    dsig_list.append(max(np.max(np.abs(S_a - S_b)), np.max(np.abs(S_a - S_c))))
                    dP_list.append(max(abs(P_a - P_b), abs(P_a - P_c)))
                mmu, msig, mP = max(dmu_list), max(dsig_list), max(dP_list)
                max_dmu = max(max_dmu, mmu)
                max_dsig = max(max_dsig, msig)
                max_dP = max(max_dP, mP)
                # tolerance scales with d (numerical errors from matrix ops)
                tol_mu = 1e-7 * d
                tol_sig = 1e-7 * d * d
                tol_P = 1e-10
                ok = mmu < tol_mu and msig < tol_sig and mP < tol_P
                if not ok:
                    all_pass = False
                status = "PASS" if ok else "FAIL"
                lines.append(f"d={d:4d}  k={k:2d}  dir={direction}  max|Δμ|={mmu:.2e}  max|ΔΣ|={msig:.2e}  max|ΔP|={mP:.2e}  [{status}]")
    lines.append("")
    lines.append("=== OVERALL ===")
    lines.append(f"max |Δμ| overall: {max_dmu:.2e}")
    lines.append(f"max |ΔΣ| overall: {max_dsig:.2e}")
    lines.append(f"max |ΔP| overall: {max_dP:.2e}")
    lines.append(f"VERDICT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    text = "\n".join(lines)
    print(text)
    if log_path:
        with open(log_path, "w") as f:
            f.write(text + "\n")
    return all_pass


# === Timing benchmark ===

def time_fn(fn, args, n_calls):
    # warmup
    for _ in range(2):
        fn(*args)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n_calls)
    return min(times)


def benchmark_timings(d_values, k_values, log_path=None, master_seed=7):
    print("\n=== TIMING BENCHMARK (best-of-5 batches) ===")
    print(f"{'d':>6s} {'k':>3s} {'SOGA(ms)':>11s} {'Sparse(ms)':>12s} {'SparseK(ms)':>13s} {'Speedup':>10s} {'SpeedupK':>10s}")
    rng_master = np.random.default_rng(master_seed)
    rows = []
    for d in d_values:
        for k in k_values:
            if k > d:
                continue
            # adaptive n_calls
            if d <= 30:
                n_calls = 200
            elif d <= 100:
                n_calls = 50
            elif d <= 300:
                n_calls = 10
            else:
                n_calls = 3
            seed = rng_master.integers(0, 2**31 - 1)
            rng = np.random.default_rng(seed)
            Sigma = gen_random_psd(d, rng)
            mu = gen_random_mu(d, rng)
            alpha = gen_sparse_alpha(d, k, rng)
            c = gen_random_c(mu, Sigma, alpha, rng)
            t_soga = time_fn(truncate_soga_style, (mu, Sigma, alpha, c, '>'), n_calls)
            t_sparse = time_fn(truncate_sparse_aware, (mu, Sigma, alpha, c, '>'), n_calls)
            t_sparse_k = time_fn(truncate_sparse_aware_k, (mu, Sigma, alpha, c, '>'), n_calls)
            su = t_soga / t_sparse
            su_k = t_soga / t_sparse_k
            print(f"{d:>6d} {k:>3d} {t_soga*1000:>9.3f}   {t_sparse*1000:>10.3f}   {t_sparse_k*1000:>11.3f}   {su:>7.2f}x   {su_k:>7.2f}x")
            rows.append({
                "d": d, "k": k,
                "t_soga_ms": t_soga * 1000,
                "t_sparse_ms": t_sparse * 1000,
                "t_sparse_k_ms": t_sparse_k * 1000,
                "speedup_sparse": su,
                "speedup_sparse_k": su_k,
                "n_calls": n_calls,
            })
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 70)
    print("Sparse-aware truncate prototype")
    print("=" * 70)

    correct = verify_correctness(
        N=200,
        d_values=(5, 10, 30, 100, 300),
        k_values=(1, 2, 3, 5),
        log_path=os.path.join(results_dir, "correctness.txt"),
    )
    if not correct:
        print("\n!! CORRECTNESS FAILED — not running timing.")
        return

    rows = benchmark_timings(
        d_values=(5, 10, 30, 100, 300, 1000),
        k_values=(1, 3),
    )
    csv_path = os.path.join(results_dir, "timing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved timing CSV to {csv_path}")


if __name__ == "__main__":
    main()
