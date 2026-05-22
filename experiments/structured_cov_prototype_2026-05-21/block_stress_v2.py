"""
Refined block stress test for the structured covariance representation
under realistic SOGA-like update sequences.

Improvements over `block_stress.py`:
  - Signed factor form  Σ = D + Φ S Φᵀ  with S diagonal of ±1
    (handles rank-1 subtractions naturally — these arise from every
    SOGA truncate as `Σ' = Σ − α g gᵀ`).
  - SOGA-realistic update pattern: alternates additive updates (mimicking
    `update_rule` adding component variance) with subtractive updates
    (mimicking `truncate` reducing variance), with α drawn from the
    same scale as a real truncate factor_cov.
  - Three policies compared:
      P_signed       — signed factor with SVD-based compactification
      P_adaptive     — like P_signed but grows k_max when drift > 1e-4
      P_orthonorm    — like P_signed + periodic Gram-Schmidt of Φ
                       to keep the Woodbury inverse well-conditioned
  - Sweeps n_steps in {500, 1000, 2000, 5000} so we can see the
    drift-vs-program-length curve.
"""

import os
import time
import csv
import numpy as np


# ----- core ops on signed factor form Σ = D + Φ S Φᵀ -----

def gen_initial(d, rng):
    D = rng.uniform(0.1, 1.0, size=d)
    Phi = (rng.standard_normal((d, 1)) / 10).astype(float)
    S = np.array([+1.0])
    Sigma_dense = np.diag(D) + Phi @ np.diag(S) @ Phi.T
    return D, Phi, S, Sigma_dense


def quadratic_signed(D, Phi, S, v):
    """v^T (D + Φ S Φᵀ) v."""
    Phi_v = Phi.T @ v
    return float(np.sum(D * v * v) + float((S * Phi_v * Phi_v).sum()))


def matvec_signed(D, Phi, S, v):
    """(D + Φ S Φᵀ) v."""
    return D * v + Phi @ (S * (Phi.T @ v))


def update_rank1_signed(D, Phi, S, g, alpha):
    """Σ += α g gᵀ;  positive or negative α handled by sign of S."""
    col = np.sqrt(abs(alpha)) * g
    sign = +1.0 if alpha >= 0 else -1.0
    Phi_new = np.hstack([Phi, col[:, None]])
    S_new = np.concatenate([S, np.array([sign])])
    return D, Phi_new, S_new


def compactify_signed(D, Phi, S, k_max):
    """SVD-based compactification of Φ keeping the columns with the
    largest *signed* contribution. The remaining tail is absorbed into
    D as a *positive* diagonal shift only (we cannot absorb a negative
    rank-1 into D safely; positive contributions are kept on Φ_+ side).
    Returns updated (D, Φ, S) with at most k_max columns.
    """
    m = Phi.shape[1]
    if m <= k_max:
        return D, Phi, S
    # We project onto top-k columns by Frobenius contribution, with sign tracking.
    # For simplicity: SVD on Φ * sqrt(S_abs) (positive scaling), keep top-k_max,
    # and re-attach the sign per column via S of the closest original column.
    # In practice the signed compactification is straightforward when S is
    # block-uniform (all +1 in update_rule, all -1 in truncate); here we
    # implement the simple version that handles mixed S by absorbing the
    # positive tail into D.
    # Step 1: separate Φ into positive and negative sides
    pos_idx = (S > 0)
    neg_idx = (S < 0)
    Phi_p = Phi[:, pos_idx]
    Phi_n = Phi[:, neg_idx]
    # Compact each side independently
    def compact_side(Mat, budget):
        if Mat.shape[1] <= budget:
            return Mat, np.zeros(Mat.shape[0])
        P, sigma, Qt = np.linalg.svd(Mat, full_matrices=False)
        head = P[:, :budget] * sigma[:budget]
        tail_var = (P[:, budget:] ** 2) @ (sigma[budget:] ** 2)
        return head, tail_var
    half = k_max // 2 if k_max >= 2 else 1
    p_budget = max(1, min(Phi_p.shape[1], half))
    n_budget = max(1, min(Phi_n.shape[1], half))
    if Phi_p.shape[1] > 0:
        Phi_p_new, tail_p = compact_side(Phi_p, p_budget)
    else:
        Phi_p_new = np.zeros((D.shape[0], 0)); tail_p = np.zeros(D.shape[0])
    if Phi_n.shape[1] > 0:
        Phi_n_new, tail_n = compact_side(Phi_n, n_budget)
    else:
        Phi_n_new = np.zeros((D.shape[0], 0)); tail_n = np.zeros(D.shape[0])
    # Absorb positive tail into D (always safe)
    D_new = D + tail_p
    # For negative tail, we cannot absorb safely (would make D smaller). Drop it.
    # The resulting structured form is a slight upper bound on the true Σ.
    Phi_new = np.hstack([Phi_p_new, Phi_n_new])
    S_new = np.concatenate([np.ones(Phi_p_new.shape[1]), -np.ones(Phi_n_new.shape[1])])
    return D_new, Phi_new, S_new


def orthonormalize(Phi, S):
    """Re-orthogonalize Φ columns using Gram-Schmidt while preserving
    Φ S Φᵀ. We work on each sign-block independently."""
    if Phi.shape[1] == 0:
        return Phi, S
    pos = (S > 0)
    neg = (S < 0)
    cols = []
    signs = []
    for mask, sign in [(pos, +1.0), (neg, -1.0)]:
        if not mask.any():
            continue
        sub = Phi[:, mask]
        Q, R = np.linalg.qr(sub)
        # The product Φ S Φᵀ for this side = sub · sign · subᵀ = (Q R)·sign·(Q R)ᵀ
        # We re-express as Q' · sign · Q'ᵀ with Q' = Q R (rank-revealing); keep Q' as new columns.
        cols.append(Q @ R)
        signs.extend([sign] * sub.shape[1])
    Phi_new = np.hstack(cols)
    S_new = np.array(signs)
    return Phi_new, S_new


# ----- realistic update generator -----

def soga_like_update(d, rng, dense_ref):
    """Returns (g, alpha) for the next update step. Alternates between
    update_rule-style additive (α > 0) and truncate-style subtractive
    (α > 0 but applied with negative sign).
    """
    g = rng.standard_normal(d) / np.sqrt(d)
    # Mimic SOGA's factor_cov scale: (1 - v_hat/var_s)/var_s with var_s ~ 0.1-2, v_hat <= var_s
    var_s_typical = abs(g @ dense_ref @ g)
    if var_s_typical < 1e-9:
        var_s_typical = 1e-9
    # for truncate-like subtraction, alpha = something in [0, 1/var_s_typical]
    # for update-like addition, alpha is the variance added by a new noise term
    # Use alternating sign each step
    is_subtraction = rng.random() < 0.6  # 60% subtractive (more truncates than updates in obs-heavy programs)
    if is_subtraction:
        scale = rng.uniform(0.1, 0.5)
        alpha = -scale / var_s_typical
    else:
        alpha = rng.uniform(0.01, 0.05)
    return g, alpha


# ----- policies -----

def run_policy_signed(name, n_steps, d=200, k_max=4, seed=1,
                     re_orthonorm_every=None, adaptive_k=False, drift_threshold=1e-4):
    rng = np.random.default_rng(seed)
    D, Phi, S, dense_ref = gen_initial(d, rng)
    test_vecs = [rng.standard_normal(d) for _ in range(3)]
    drift_history = []
    psd_violations = 0
    k_grew = 0
    current_k_max = k_max
    for step in range(n_steps):
        g, alpha = soga_like_update(d, rng, dense_ref)
        # Update dense ref
        dense_ref = dense_ref + alpha * np.outer(g, g)
        # Update structured (signed)
        D, Phi, S = update_rank1_signed(D, Phi, S, g, alpha)
        # Compactify
        if Phi.shape[1] > 2 * current_k_max:
            D, Phi, S = compactify_signed(D, Phi, S, current_k_max)
        if re_orthonorm_every and (step + 1) % re_orthonorm_every == 0:
            Phi, S = orthonormalize(Phi, S)
        # PSD check on D (positive D required for Woodbury stability)
        if np.any(D < -1e-10):
            psd_violations += 1
        # Drift check
        if (step + 1) % 50 == 0:
            rel = 0.0
            for v in test_vecs:
                q_struct = quadratic_signed(D, Phi, S, v)
                q_dense = float(v @ dense_ref @ v)
                rel = max(rel, abs(q_struct - q_dense) / max(abs(q_dense), 1.0))
            drift_history.append({"step": step + 1, "rel_quad": rel, "k": current_k_max,
                                  "n_cols": Phi.shape[1]})
            # Adaptive k_max
            if adaptive_k and rel > drift_threshold and current_k_max < 32:
                current_k_max *= 2
                k_grew += 1
    # Final drift
    rel_final = 0.0
    for v in test_vecs:
        q_struct = quadratic_signed(D, Phi, S, v)
        q_dense = float(v @ dense_ref @ v)
        rel_final = max(rel_final, abs(q_struct - q_dense) / max(abs(q_dense), 1.0))
    max_drift = max(h["rel_quad"] for h in drift_history) if drift_history else rel_final
    return {
        "policy": name,
        "n_steps": n_steps,
        "d": d,
        "initial_k_max": k_max,
        "final_k_max": current_k_max,
        "k_grew_times": k_grew,
        "psd_violations": psd_violations,
        "final_drift": rel_final,
        "max_drift_during": max_drift,
        "history": drift_history,
    }


def main():
    print("=" * 90)
    print("Refined block stress — signed factor form + SOGA-realistic updates")
    print("=" * 90)
    print(f"\n d = 200, k_max = 4 (default), seed = 1\n")

    rows = []
    print(f"{'Policy':<28s} {'n_steps':>8s} {'final k':>8s} {'PSD viol':>9s} {'max drift':>11s} {'final drift':>13s}")
    print("-" * 90)

    for n_steps in (500, 1000, 2000, 5000):
        for name, kwargs in [
            ("P_signed", dict()),
            ("P_signed+orthonorm200", dict(re_orthonorm_every=200)),
            ("P_signed+adaptive_k", dict(adaptive_k=True)),
            ("P_signed+ortho+adapt", dict(re_orthonorm_every=200, adaptive_k=True)),
        ]:
            r = run_policy_signed(name, n_steps=n_steps, **kwargs)
            print(f"{name:<28s} {n_steps:>8d} {r['final_k_max']:>8d} "
                  f"{r['psd_violations']:>9d} {r['max_drift_during']:>11.2e} {r['final_drift']:>13.2e}")
            rows.append(r)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "block_stress_v2.csv")
    fields = ["policy", "n_steps", "d", "initial_k_max", "final_k_max", "k_grew_times",
              "psd_violations", "final_drift", "max_drift_during"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved: {out_csv}")

    # Verdict
    print("\n=== Verdict ===")
    # Find best policy by max-drift at n_steps=5000
    best = min((r for r in rows if r["n_steps"] == 5000), key=lambda r: r["max_drift_during"])
    print(f"  Best policy at 5000 steps: {best['policy']}  max_drift={best['max_drift_during']:.2e}  "
          f"final_k_max={best['final_k_max']}")


if __name__ == "__main__":
    main()
