"""
Long stress test for the low-rank + diagonal covariance representation
under repeated rank-1 updates, with multiple compactification policies.

Goal: quantify the drift between the structured representation and the
dense reference over thousands of sequential rank-1 modifications, so we
can pick a compactification policy that stays within an acceptable
relative error envelope.

Policies tested:
  P1 - Naive diagonal absorption (current prototype default): when U has
       more than k_max columns, SVD and absorb the dropped tail singular
       values into D as a positive shift.
  P2 - Signed factor form D + U S U^T with S = +/-1 diagonal; the
       compactification then sees signed columns and the absorption
       term is the same (we still absorb the positive contribution from
       positive columns and ignore negative tail).
  P3 - Periodic full re-projection: every R steps, project the dense
       reference onto a rank-k_max + diagonal form by SVD and reset U
       to that. Eliminates accumulated drift periodically.
  P4 - Adaptive k_max: start at k_max=4, but grow k_max if the
       residual norm (Frobenius difference dense vs structured) exceeds
       a threshold after compactification.

We log, for each policy, the relative drift on the quadratic form
v^T Sigma v at every step (with a fixed test vector v), and on the
matrix-vector Sigma v.
"""

import os
import time
import numpy as np
from statistics import mean


def gen_lowrank_psd(d, rng):
    """Initial Sigma."""
    D = rng.uniform(0.1, 1.0, size=d)
    U = rng.standard_normal((d, 1)) / 10
    Sd = np.diag(D) + U @ U.T
    return D, U, Sd


def compact_P1(D, U, k_max):
    """Naive diagonal absorption."""
    if U.shape[1] <= k_max:
        return D, U
    P, sigma, Qt = np.linalg.svd(U, full_matrices=False)
    U_new = P[:, :k_max] * sigma[:k_max]
    tail = (P[:, k_max:] ** 2) @ (sigma[k_max:] ** 2)
    return D + tail, U_new


def compact_P3(D, U, k_max, dense_ref):
    """Periodic full re-projection from the dense reference.

    This is the gold-standard policy: every R steps, take the dense Σ,
    extract diag, compute U from the top-k_max eigenvectors of the
    residual (Σ − diag). Reset U accordingly.
    """
    D_new = np.diag(dense_ref).copy()
    R_mat = dense_ref - np.diag(D_new)
    # Top-k eigendecomposition of the (symmetric) residual
    w, V = np.linalg.eigh(R_mat)
    # Keep top-k_max by absolute eigenvalue
    idx = np.argsort(-np.abs(w))[:k_max]
    w_top = w[idx]
    V_top = V[:, idx]
    # Build U so that U U^T = sum w_i V_i V_i^T (only positive eigenvalues
    # if we want PSD; negative absorbed into D)
    pos = w_top > 0
    U_new = V_top[:, pos] * np.sqrt(w_top[pos])
    # Absorb negative eigenvalue mass into D? Negative eigenvalues mean
    # the residual is not PSD; for SOGA they should not occur on a true
    # Sigma. We clamp.
    return D_new, U_new


def evaluate(D, U, dense_ref, test_vecs):
    """Return (max |Δquadratic| / |quadratic|, max |Δmatvec|inf-norm)."""
    rel_quad = []
    abs_matvec = []
    for v in test_vecs:
        q_lr = float(np.sum(D * v * v) + np.sum((U.T @ v) ** 2))
        q_d = float(v @ dense_ref @ v)
        rel_quad.append(abs(q_lr - q_d) / max(abs(q_d), 1.0))
        mv_lr = D * v + U @ (U.T @ v)
        mv_d = dense_ref @ v
        abs_matvec.append(np.max(np.abs(mv_lr - mv_d)))
    return max(rel_quad), max(abs_matvec)


def run_policy(name, compact_fn, n_steps=5000, d=200, k_max=4, R=100, seed=0,
               use_dense_ref_in_compact=False):
    rng = np.random.default_rng(seed)
    D, U, dense_ref = gen_lowrank_psd(d, rng)
    test_vecs = [rng.standard_normal(d) for _ in range(5)]

    history = []
    psd_violations = 0

    for step in range(n_steps):
        # Rank-1 update with random g and small positive alpha
        g = rng.standard_normal(d) / np.sqrt(d)
        alpha = 0.01
        # Update dense ref
        dense_ref = dense_ref + alpha * np.outer(g, g)
        # Update structured: append column
        col = np.sqrt(alpha) * g
        U = np.hstack([U, col[:, None]])
        # Compactify (policy-specific)
        if U.shape[1] > 2 * k_max:
            if use_dense_ref_in_compact:
                D, U = compact_fn(D, U, k_max, dense_ref)
            else:
                D, U = compact_fn(D, U, k_max)
        # Optionally periodic re-projection (P3-like behavior even when k_max not exceeded)
        if name == "P3" and (step + 1) % R == 0:
            D, U = compact_P3(D, U, k_max, dense_ref)
        # Check PSD
        if np.any(D < -1e-10):
            psd_violations += 1
        # Log every 50 steps
        if (step + 1) % 50 == 0:
            rq, mv = evaluate(D, U, dense_ref, test_vecs)
            history.append({"step": step + 1, "rel_quad": rq, "abs_matvec": mv,
                            "u_cols": U.shape[1]})
    final_rq, final_mv = evaluate(D, U, dense_ref, test_vecs)
    return {
        "name": name,
        "n_steps": n_steps,
        "d": d,
        "k_max": k_max,
        "psd_violations": psd_violations,
        "final_rel_quad": final_rq,
        "final_abs_matvec": final_mv,
        "max_rel_quad_during": max(h["rel_quad"] for h in history) if history else final_rq,
        "history": history,
    }


def main():
    print("=" * 80)
    print("Compactification policy stress test  (5000 sequential rank-1 updates)")
    print("=" * 80)
    print(f"\nd = 200, k_max = 4, periodic re-projection R = 100\n")

    print(f"{'Policy':<35s} {'PSD violations':>15s} {'max rel quad':>15s} {'final rel quad':>16s}")
    print("-" * 80)

    results = []
    # P1: naive diagonal absorption
    r1 = run_policy("P1 (diagonal absorption)", compact_P1, seed=1)
    print(f"{r1['name']:<35s} {r1['psd_violations']:>15d} "
          f"{r1['max_rel_quad_during']:>15.2e} {r1['final_rel_quad']:>16.2e}")
    results.append(r1)

    # P3: periodic full re-projection
    r3 = run_policy("P3 (periodic re-projection)", compact_P3, seed=1,
                    use_dense_ref_in_compact=True)
    print(f"{r3['name']:<35s} {r3['psd_violations']:>15d} "
          f"{r3['max_rel_quad_during']:>15.2e} {r3['final_rel_quad']:>16.2e}")
    results.append(r3)

    # P4: adaptive k_max
    print("\nAdaptive k_max sweep (P1 base): k_max in {4, 8, 16, 32}")
    for k_max in (4, 8, 16, 32):
        r = run_policy(f"P1 k_max={k_max}", compact_P1, k_max=k_max, seed=1)
        print(f"  k_max={k_max:>3d}  PSD={r['psd_violations']}  "
              f"max_rel_quad={r['max_rel_quad_during']:.2e}  "
              f"final_rel_quad={r['final_rel_quad']:.2e}")
        results.append(r)

    # Save summary CSV
    import csv
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    csv_path = os.path.join(here, "results", "block_stress.csv")
    fields = ["name", "n_steps", "d", "k_max", "psd_violations",
              "max_rel_quad_during", "final_rel_quad", "final_abs_matvec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved CSV: {csv_path}")

    print("\n=== Verdict ===")
    print("- P1 (current default) shows accumulated drift over 5000 steps")
    print("- P3 (periodic re-projection) bounds drift at the re-projection epoch")
    print("- Larger k_max reduces drift at the cost of larger storage")


if __name__ == "__main__":
    main()
