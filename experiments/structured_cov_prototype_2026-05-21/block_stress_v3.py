"""
Block stress v3 — SOGA-realistic update pattern with rank-confined g.

Key insight: in a real SOGA truncate, the rank-1 update is

    Σ' = Σ − factor · g gᵀ      where g = Σ · α

So g LIVES IN THE COLUMN SPACE OF Σ. If Σ has rank k (which the empirical
sigma analysis shows is ~2 throughout SOGA benchmarks), then each rank-1
modification stays in that same k-dim subspace. The true rank of Σ does
NOT grow under SOGA's actual operations, regardless of how many truncates
fire.

This test reproduces that pattern: g is drawn from `Σ · α` where α is a
sparse coefficient vector (mimicking an observe's LBC). Compactification
with k_max = 4 should then track the true low-rank structure without
unbounded drift.

We compare:
  - "naive" stress: g drawn from N(0, I/d) — worst case, no structure
  - "realistic" stress: g = Σ · α with α sparse (1-3 non-zero) — SOGA-like
"""

import os
import csv
import numpy as np


def gen_initial_lowrank(d, k_true, rng):
    """Σ = D + U Uᵀ with U d × k_true. The 'true rank' of the off-diagonal
    part stays bounded at k_true."""
    D = rng.uniform(0.1, 1.0, size=d)
    U = rng.standard_normal((d, k_true)) * 0.3
    Sd = np.diag(D) + U @ U.T
    return D, U, Sd


def update_step(D, U_struct, dense_ref, k_max, rng, mode, d):
    """Perform one rank-1 update in both the structured and dense forms."""
    if mode == "naive":
        g = rng.standard_normal(d) / np.sqrt(d)
    elif mode == "realistic":
        # Sparse alpha (1-3 non-zero entries), then g = Σ · α
        nnz = rng.integers(1, 4)
        idx = rng.choice(d, size=nnz, replace=False)
        alpha_vec = np.zeros(d)
        alpha_vec[idx] = rng.standard_normal(nnz)
        g = dense_ref @ alpha_vec
        # Normalize so the update magnitude is comparable
        gnorm = np.linalg.norm(g)
        if gnorm > 0:
            g = g / gnorm
    else:
        raise ValueError(mode)

    # SOGA-like alpha: small subtractive
    factor = rng.uniform(0.05, 0.2)
    alpha = -factor

    # Update dense ref
    dense_ref_new = dense_ref + alpha * np.outer(g, g)
    # Update structured: append signed column
    col = np.sqrt(abs(alpha)) * g
    U_new = np.hstack([U_struct, col[:, None]])
    return D, U_new, dense_ref_new


def compactify_simple(D, U, k_max):
    if U.shape[1] <= k_max:
        return D, U
    P, sigma, Qt = np.linalg.svd(U, full_matrices=False)
    U_new = P[:, :k_max] * sigma[:k_max]
    tail = (P[:, k_max:] ** 2) @ (sigma[k_max:] ** 2)
    return D + tail, U_new


def quadratic(D, U, v):
    return float(np.sum(D * v * v) + np.sum((U.T @ v) ** 2))


def run(mode, n_steps, d=200, k_max=4, k_true=2, seed=1):
    rng = np.random.default_rng(seed)
    D, U, dense_ref = gen_initial_lowrank(d, k_true, rng)
    test_vecs = [rng.standard_normal(d) for _ in range(3)]
    history = []
    psd_violations = 0
    for step in range(n_steps):
        D, U, dense_ref = update_step(D, U, dense_ref, k_max, rng, mode, d)
        if U.shape[1] > 2 * k_max:
            D, U = compactify_simple(D, U, k_max)
        if np.any(D < -1e-10):
            psd_violations += 1
        if (step + 1) % 50 == 0:
            rel = 0.0
            for v in test_vecs:
                q_struct = quadratic(D, U, v)
                q_dense = float(v @ dense_ref @ v)
                rel = max(rel, abs(q_struct - q_dense) / max(abs(q_dense), 1e-9))
            history.append({"step": step + 1, "rel_quad": rel})
    final_rel = 0.0
    for v in test_vecs:
        q_struct = quadratic(D, U, v)
        q_dense = float(v @ dense_ref @ v)
        final_rel = max(final_rel, abs(q_struct - q_dense) / max(abs(q_dense), 1e-9))
    return {
        "mode": mode, "n_steps": n_steps, "d": d, "k_max": k_max, "k_true": k_true,
        "psd_violations": psd_violations,
        "max_drift": max(h["rel_quad"] for h in history) if history else final_rel,
        "final_drift": final_rel,
        "u_cols_final": U.shape[1],
    }


def main():
    print("=" * 90)
    print("Block stress v3 — naive (random g) vs SOGA-realistic (g = Σ·α, low-rank-confined)")
    print("=" * 90)
    print(f"\nd = 200, k_max = 4, true rank of Σ initially = 2\n")
    print(f"{'Mode':<14s} {'n_steps':>8s} {'PSD viol':>9s} {'max drift':>12s} {'final drift':>13s}")
    print("-" * 75)
    rows = []
    for mode in ("naive", "realistic"):
        for n_steps in (200, 500, 1000, 5000):
            r = run(mode, n_steps=n_steps)
            print(f"{mode:<14s} {n_steps:>8d} {r['psd_violations']:>9d} "
                  f"{r['max_drift']:>12.2e} {r['final_drift']:>13.2e}")
            rows.append(r)

    here = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(here, "results", "block_stress_v3.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved: {out_csv}")

    # Compute the realistic-vs-naive drift ratio
    realistic_drifts = [r["max_drift"] for r in rows if r["mode"] == "realistic"]
    naive_drifts = [r["max_drift"] for r in rows if r["mode"] == "naive"]
    ratio = sum(naive_drifts) / sum(realistic_drifts) if realistic_drifts else float("inf")
    print(f"\n=== Verdict ===")
    print(f"Realistic drift / Naive drift ratio: {1/ratio:.3f}  "
          f"(realistic mode is {ratio:.1f}x less drifted)")


if __name__ == "__main__":
    main()
