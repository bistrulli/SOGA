"""
Standalone math prototype: structured (low-rank + diagonal) vs dense covariance.

Validates the math of `Sigma = D + U U^T` and compares with dense numpy
operations on the kernel ops used inside SOGA (matvec, quadratic form,
inverse, determinant, rank-1 update). The goal is to confirm
equivalence (machine epsilon) and measure the per-operation speedup at
varying (d, k) BEFORE touching the SOGA source.

No SOGA imports. Pure numpy.
"""

import time
import numpy as np
from statistics import mean


# ---------- structured covariance operations ----------

def matvec_dense(Sigma, v):
    return Sigma @ v


def matvec_lr(D, U, v):
    """Sigma @ v  where Sigma = diag(D) + U U^T."""
    return D * v + U @ (U.T @ v)


def quadratic_dense(Sigma, v):
    return float(v @ Sigma @ v)


def quadratic_lr(D, U, v):
    """v^T Sigma v."""
    return float(np.sum(D * v * v) + np.sum((U.T @ v) ** 2))


def inv_dense(Sigma):
    return np.linalg.inv(Sigma)


def inv_lr(D, U):
    """Sigma^{-1} via Woodbury identity, returned as dense d x d.

    For SOGA we usually do NOT need the dense inverse — we apply it to a
    vector. The dense inverse here is for equivalence-testing only.
    """
    Dinv = 1.0 / D
    Dinv_U = (Dinv[:, None] * U)
    K = np.linalg.inv(np.eye(U.shape[1]) + U.T @ Dinv_U)
    return np.diag(Dinv) - Dinv_U @ K @ Dinv_U.T


def inv_apply_lr(D, U, v):
    """Sigma^{-1} @ v via Woodbury, without materializing Sigma^{-1}."""
    Dinv = 1.0 / D
    Dinv_v = Dinv * v
    Dinv_U = Dinv[:, None] * U
    K = np.linalg.inv(np.eye(U.shape[1]) + U.T @ Dinv_U)
    return Dinv_v - Dinv_U @ (K @ (U.T @ Dinv_v))


def logdet_dense(Sigma):
    sign, ld = np.linalg.slogdet(Sigma)
    return float(ld) if sign > 0 else float("nan")


def logdet_lr(D, U):
    """log det(diag(D) + U U^T) via Matrix Determinant Lemma."""
    ld_D = float(np.sum(np.log(D)))
    K = np.eye(U.shape[1]) + U.T @ (U / D[:, None])
    sign, ld_K = np.linalg.slogdet(K)
    return ld_D + float(ld_K) if sign > 0 else float("nan")


def rank1_update_dense(Sigma, g, alpha):
    """Sigma + alpha * g g^T (alpha may be negative)."""
    return Sigma + alpha * np.outer(g, g)


def rank1_update_lr(D, U, g, alpha):
    """Append sqrt(|alpha|) * g as a new column of U; carry sign via S diagonal.

    For signed factor form D + U S U^T, we keep S = diag(s) with s_i in {+1, -1}.
    Here we return U' and S' (extending by one column).
    """
    if alpha >= 0:
        col = np.sqrt(alpha) * g
        sign = +1.0
    else:
        col = np.sqrt(-alpha) * g
        sign = -1.0
    U_new = np.hstack([U, col[:, None]])
    return D, U_new, sign


def compactify_lr(D, U, S, k_max):
    """If U has more than k_max columns, SVD-truncate keeping the top-k_max.

    S is a vector of signs (one per column of U); negative columns enter the
    truncation as if they had imaginary contribution — for simplicity here we
    only handle the all-positive case (sufficient to demonstrate the speedup;
    the signed case is handled in the implementation plan).
    """
    if U.shape[1] <= k_max:
        return D, U, S
    # All-positive S only:
    P, sigma, Qt = np.linalg.svd(U, full_matrices=False)
    U_new = P[:, :k_max] * sigma[:k_max]
    # Trace-preserving diagonal absorption of the dropped tail
    if len(sigma) > k_max:
        tail = (P[:, k_max:] ** 2) @ (sigma[k_max:] ** 2)  # (d,)
        D = D + tail
    return D, U_new, np.ones(k_max)


# ---------- input generators ----------

def gen_lowrank_psd(d, k, rng):
    """Generate Sigma = D + U U^T with random PSD D > 0 and random U (d x k)."""
    D = rng.uniform(0.1, 1.0, size=d)
    U = rng.standard_normal((d, k)) / np.sqrt(k)
    Sigma_dense = np.diag(D) + U @ U.T
    return D, U, Sigma_dense


# ---------- equivalence tests ----------

def equivalence_test(n_samples=1000, d_values=(5, 30, 100, 500), k_values=(1, 2, 5, 10)):
    print("\n=== Equivalence test (dense vs lowrank) ===")
    print(f"{'d':>5s} {'k':>4s} {'matvec':>14s} {'quad':>14s} {'inv':>14s} {'logdet':>14s}")
    rows = []
    for d in d_values:
        for k in k_values:
            if k > d:
                continue
            rng = np.random.default_rng(d * 1000 + k)
            diffs_matvec, diffs_quad, diffs_inv, diffs_logdet = [], [], [], []
            for _ in range(max(50, n_samples // (d_values.index(d) + 1))):
                D, U, Sd = gen_lowrank_psd(d, k, rng)
                v = rng.standard_normal(d)
                # matvec
                a = matvec_dense(Sd, v)
                b = matvec_lr(D, U, v)
                diffs_matvec.append(np.max(np.abs(a - b)))
                # quadratic
                a = quadratic_dense(Sd, v)
                b = quadratic_lr(D, U, v)
                diffs_quad.append(abs(a - b))
                # inv
                a = inv_dense(Sd)
                b = inv_lr(D, U)
                diffs_inv.append(np.max(np.abs(a - b)))
                # logdet
                a = logdet_dense(Sd)
                b = logdet_lr(D, U)
                diffs_logdet.append(abs(a - b))
            mmv, mq, mi, ml = max(diffs_matvec), max(diffs_quad), max(diffs_inv), max(diffs_logdet)
            print(f"{d:>5d} {k:>4d}  {mmv:>11.2e}   {mq:>11.2e}   {mi:>11.2e}   {ml:>11.2e}")
            rows.append({"d": d, "k": k, "matvec": mmv, "quad": mq, "inv": mi, "logdet": ml})
    return rows


# ---------- timing tests ----------

def time_one(fn, args, n_calls):
    # warmup
    fn(*args); fn(*args)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn(*args)
        elapsed = (time.perf_counter() - t0) / n_calls
        best = min(best, elapsed)
    return best


def timing_test(d_values=(30, 100, 300, 1000), k_values=(1, 2, 5, 10)):
    print("\n=== Timing test (best-of-5, dense vs lowrank) ===")
    print(f"{'d':>5s} {'k':>4s}  {'op':<10s} {'dense (us)':>14s} {'lowrank (us)':>16s} {'speedup':>10s}")
    rows = []
    for d in d_values:
        rng = np.random.default_rng(d * 7)
        for k in k_values:
            if k >= d:
                continue
            D, U, Sd = gen_lowrank_psd(d, k, rng)
            v = rng.standard_normal(d)
            # n_calls adaptive
            n = 5000 if d <= 100 else (500 if d <= 300 else 50)
            for op_name, dense_fn, dense_args, lr_fn, lr_args in [
                ("matvec",   matvec_dense,   (Sd, v),    matvec_lr,    (D, U, v)),
                ("quad",     quadratic_dense,(Sd, v),    quadratic_lr, (D, U, v)),
                ("inv_apply",lambda S,v: inv_dense(S) @ v, (Sd, v),    inv_apply_lr,(D, U, v)),
                ("logdet",   logdet_dense,   (Sd,),      logdet_lr,    (D, U)),
            ]:
                t_d = time_one(dense_fn, dense_args, n)
                t_l = time_one(lr_fn, lr_args, n)
                speedup = t_d / t_l if t_l > 0 else float("inf")
                print(f"{d:>5d} {k:>4d}  {op_name:<10s}  {t_d*1e6:>10.2f}     {t_l*1e6:>12.2f}    {speedup:>7.2f}x")
                rows.append({"d": d, "k": k, "op": op_name, "dense_us": t_d * 1e6, "lr_us": t_l * 1e6, "speedup": speedup})
    return rows


# ---------- rank-1 update stress test ----------

def stress_rank1_updates(n_updates=200, d=100, k_max=4):
    print(f"\n=== Rank-1 update stress: d={d}, k_max={k_max}, {n_updates} updates ===")
    rng = np.random.default_rng(42)
    D = rng.uniform(0.1, 1.0, size=d)
    U = rng.standard_normal((d, 1)) / 10
    Sd = np.diag(D) + U @ U.T
    S = np.ones(1)
    max_diff_history = []
    psd_violations = 0
    for step in range(n_updates):
        g = rng.standard_normal(d) / np.sqrt(d)
        alpha = 0.05  # positive updates only for now
        Sd = rank1_update_dense(Sd, g, alpha)
        D, U, S_new = rank1_update_lr(D, U, g, alpha)
        S = np.hstack([S, S_new])
        if U.shape[1] > 2 * k_max:
            D, U, S = compactify_lr(D, U, S, k_max)
        # Check equivalence on a fixed test vector
        v = rng.standard_normal(d)
        diff = abs(quadratic_dense(Sd, v) - quadratic_lr(D, U, v))
        scale = max(quadratic_dense(Sd, v), 1.0)
        max_diff_history.append(diff / scale)
        # PSD check
        if np.any(D < -1e-10):
            psd_violations += 1
    print(f"  max relative |Δquadratic|: {max(max_diff_history):.2e}")
    print(f"  mean relative |Δquadratic|: {mean(max_diff_history):.2e}")
    print(f"  PSD violations: {psd_violations}/{n_updates}")
    return {"max_diff": max(max_diff_history), "mean_diff": mean(max_diff_history), "psd_violations": psd_violations}


# ---------- main ----------

def main():
    print("=" * 78)
    print("Structured-covariance (D + U U^T) standalone math prototype")
    print("=" * 78)

    eq_rows = equivalence_test()

    print("\nVerdict (equivalence):")
    max_diff_all = max(max(r["matvec"], r["quad"], r["inv"], r["logdet"]) for r in eq_rows)
    print(f"  max |Δ| overall: {max_diff_all:.2e}  ({'PASS' if max_diff_all < 1e-6 else 'FAIL'})")

    timing_rows = timing_test()

    print("\nVerdict (timing):")
    matvec_speedups = [r["speedup"] for r in timing_rows if r["op"] == "matvec"]
    inv_speedups = [r["speedup"] for r in timing_rows if r["op"] == "inv_apply"]
    print(f"  matvec speedup range: {min(matvec_speedups):.2f}x – {max(matvec_speedups):.2f}x")
    print(f"  inv_apply speedup range: {min(inv_speedups):.2f}x – {max(inv_speedups):.2f}x")

    stress = stress_rank1_updates()
    print(f"\nVerdict (rank-1 stress): {'PASS' if stress['max_diff'] < 1e-6 and stress['psd_violations'] == 0 else 'WARN'}")

    # Persist
    import csv, os
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "equivalence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq_rows[0].keys()))
        w.writeheader(); w.writerows(eq_rows)
    with open(os.path.join(here, "results", "timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        w.writeheader(); w.writerows(timing_rows)
    print(f"\nSaved CSV to {here}/results/")


if __name__ == "__main__":
    main()
