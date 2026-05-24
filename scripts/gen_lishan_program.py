#!/usr/bin/env python3
"""
scripts/gen_lishan_program.py — Generate Lishan-class .soga programs.

Usage:
    python3 scripts/gen_lishan_program.py --size 4  --output programs/Example/lishan_2mm_4x4.soga
    python3 scripts/gen_lishan_program.py --size 8  --output programs/Example/lishan_2mm_8x8.soga
    python3 scripts/gen_lishan_program.py --size 16 --output programs/Example/lishan_2mm_16x16.soga
    python3 scripts/gen_lishan_program.py --size 32 --output programs/Example/lishan_2mm_32x32.soga

Lishan-class model:
    C = A_kernel @ X_input + N_fault

where:
    A_kernel  : deterministic data matrix (identity by default; can be random with --seed)
    X_input   : matrix-Gaussian prior, MN(0, I_m, sigma_x^2 * I_n)
    N_fault   : isotropic fault injection noise, MN(0, sigma_f^2 * I_m, I_n)

DSL syntax notes:
  - All numeric values must be fully-expanded integer or float literals
    (no Python expressions like 0.5*eye(4) are allowed in the .soga DSL).
  - Row-major mlist encoding: [[r0_v0, ..., r0_vN-1], [r1_v0, ...], ...]
  - matrix_gm(M, U, V) — mean M, row-cov U, col-cov V (KRONECKER_CONVENTION=V_outer_U)

Plan reference: §M6.1-M6.4 of plan/2026-05-22-matrix-gm-lishan.md
"""

import argparse
import sys
from pathlib import Path


def _fmt_scalar(v: float) -> str:
    """Format a scalar for inline numeric literals."""
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    return f"{v:.10g}"


def _fmt_matrix(M, indent: int = 4) -> str:
    """Format a 2D list of floats as a DSL mlist literal (row-major, one row per line)."""
    rows = []
    for row in M:
        row_str = "[" + ", ".join(_fmt_scalar(v) for v in row) + "]"
        rows.append(row_str)
    sep = ",\n" + " " * indent
    return "[\n" + " " * indent + sep.join(rows) + "\n" + " " * (indent - 4) + "]"


def _eye(n: float, scale: float = 1.0):
    """Return scaled identity as a 2D list."""
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def generate_lishan_program(
    size: int,
    sigma_x: float = 1.0,
    sigma_f: float = 0.5,
    seed: int = 42,
    kernel_identity: bool = True,
) -> str:
    """Generate the .soga source for a Lishan-class 2MM+FI benchmark.

    Parameters
    ----------
    size : int
        Matrix dimension (size x size).
    sigma_x : float
        Standard deviation of X_input column factor (V = sigma_x^2 * I_n).
        Row factor U = I_m (isotropic across rows).
    sigma_f : float
        Standard deviation of fault injection noise (U = sigma_f^2 * I_m, V = I_n).
    seed : int
        Random seed for generating a random kernel (used only when kernel_identity=False).
    kernel_identity : bool
        If True, A_kernel = I_n (simplest case — matches PoC Demo 3 with A=I).
        If False, use a random orthogonal matrix from the seed (for non-trivial benchmarks).
    """
    import numpy as np

    m = size
    n = size

    # A_kernel: deterministic data matrix
    if kernel_identity:
        A = np.eye(m)
    else:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
        A = Q

    # X_input: zero mean, U = I_m, V = sigma_x^2 * I_n
    M_x = [[0.0] * n for _ in range(m)]
    U_x = _eye(m, scale=1.0)           # row cov = I_m
    V_x = _eye(n, scale=sigma_x ** 2)  # col cov = sigma_x^2 * I_n

    # N_fault: zero mean, U = sigma_f^2 * I_m, V = I_n
    M_n = [[0.0] * n for _ in range(m)]
    U_n = _eye(m, scale=sigma_f ** 2)  # row cov = sigma_f^2 * I_m
    V_n = _eye(n, scale=1.0)           # col cov = I_n

    # Convert A to list-of-lists for DSL formatting
    A_list = A.tolist()

    # Analytical expected output moments for the comment header:
    # E[C] = A @ E[X] + E[N] = 0
    # Var(C[i,j]) = A_row_i^T @ (U_x) @ A_row_i * V_x[j,j] + U_n[i,i] * V_n[j,j]
    #             = ||a_i||^2 * sigma_x^2 + sigma_f^2
    # For A = I_m: = 1.0 * sigma_x^2 + sigma_f^2
    A_np = A
    var_c00 = float(A_np[0, :] @ np.eye(m) @ A_np[0, :]) * sigma_x ** 2 + sigma_f ** 2

    # For identity kernel, the program simplifies: A @ X = X → C = X + N
    # The SOGA DSL 'data' keyword only supports flat 1D lists; 2D data is not
    # supported in v1.  Non-identity kernel programs are generated via matmul
    # of matrix_gm with a deterministic init (set as matrix_gm with zero var).
    # For the identity case (kernel_identity=True) we use the simpler form.
    if kernel_identity:
        assignment_line = f"C = Xinput + Nfault;"
        kernel_comment = f"/* Akernel = I_{m} (identity, Akernel@Xinput = Xinput) */"
    else:
        # Non-identity: store the kernel as a matrix_gm with zero variance
        # (a Dirac at A), then C = A_matrix @ X_input + N_fault
        # In v1 DSL, non-identity kernels require encoding A as a matrix variable
        # initialised to a matrix_gm with zero covariance.  This is a planned
        # extension for v2; raise an error here.
        raise NotImplementedError(
            "Non-identity kernel programs require v2 matmul support.  "
            "Use --kernel-identity (default) for v1 programs."
        )

    lines = [
        f"/* Lishan-class {m}x{n} 2MM+FI: C = Akernel @ Xinput + Nfault",
        f" * Akernel: {'identity' if kernel_identity else 'random orthogonal (seed=' + str(seed) + ')'}",
        f" * Xinput ~ MN(0, I{m}, {sigma_x**2:.4g}*I{n})",
        f" * Nfault ~ MN(0, {sigma_f**2:.4g}*I{m}, I{n})",
        f" * Analytical: E[C]=0, Var(C[0,0])={var_c00:.6g}",
        f" * Plan reference: M6 of plan/2026-05-22-matrix-gm-lishan.md",
        f" * Note: IDV tokens do not support underscores in v1 grammar.",
        f" */",
        f"",
        f"matrix[{m}][{n}] Xinput;",
        f"matrix[{m}][{n}] Nfault;",
        f"matrix[{m}][{n}] C;",
        f"",
        f"/* Xinput prior: zero mean, U=I{m} (iso), V={sigma_x**2:.4g}*I{n} (iso) */",
        f"Xinput = matrix_gm(",
        f"  {_fmt_matrix(M_x, indent=4)},",
        f"  {_fmt_matrix(U_x, indent=4)},",
        f"  {_fmt_matrix(V_x, indent=4)}",
        f");",
        f"",
        f"/* Nfault: sigma_f^2={sigma_f**2:.4g} isotropic in row factor (Lishan FI model) */",
        f"Nfault = matrix_gm(",
        f"  {_fmt_matrix(M_n, indent=4)},",
        f"  {_fmt_matrix(U_n, indent=4)},",
        f"  {_fmt_matrix(V_n, indent=4)}",
        f");",
        f"",
        f"{kernel_comment}",
        f"{assignment_line}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate Lishan-class .soga programs")
    parser.add_argument("--size", type=int, required=True,
                        help="Matrix dimension (generates size x size program)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .soga file path")
    parser.add_argument("--sigma-x", type=float, default=1.0,
                        help="Std dev of X_input column factor (default 1.0)")
    parser.add_argument("--sigma-f", type=float, default=0.5,
                        help="Std dev of fault injection noise (default 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (used only when --random-kernel is set)")
    parser.add_argument("--random-kernel", action="store_true",
                        help="Use a random orthogonal kernel (default: identity)")
    args = parser.parse_args()

    program = generate_lishan_program(
        size=args.size,
        sigma_x=args.sigma_x,
        sigma_f=args.sigma_f,
        seed=args.seed,
        kernel_identity=not args.random_kernel,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(program)
    print(f"Generated {args.size}x{args.size} Lishan program -> {args.output}")
    print(f"  sigma_x={args.sigma_x}, sigma_f={args.sigma_f}")


if __name__ == "__main__":
    main()
