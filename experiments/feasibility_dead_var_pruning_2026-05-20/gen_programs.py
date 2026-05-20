"""
Synthetic program generator for scaling study.

Pattern B (BPM-like): K weights + N observations.
    - V1: array[N] mu, all mu's stay in joint (d_total = K+N)
    - V2_tmp: single tmp reassigned (d_total = K+1)
    NOTE: V2_tmp is the fused+pruned variant. Pure dead-var pruning on V1
    is theoretically bounded ~4× regardless of N.

Pattern C (Markov sequential): T-step random walk with observations.
    - V1: array[T+1] state, all states stay in joint (d_total = T+1)
    - V2_tmp: single tmp reassigned (d_total = 1)
    For Pattern C, V2_tmp ≈ pure dead-var pruning (the previous state is
    naturally dead once the next is computed).

All programs share `gm([1.],[0.],[0.1])` Gaussian noise (matching BPM).
Deterministic via seed.
"""

import os
import random


def gen_pattern_B(K: int, N: int, seed: int = 0, out_dir: str = ".") -> tuple[str, str]:
    """Generate V1 (array mu) and V2_tmp (single tmp) variants for Pattern B.
    Returns paths to the two files."""
    rng = random.Random(seed)
    # Generate K feature arrays of length N (integer values in [1, 10])
    feats = [[rng.randint(1, 10) for _ in range(N)] for _ in range(K)]
    # Generate a "true" w to pick consistent observation signs
    true_w = [rng.gauss(0, 1) for _ in range(K)]
    # Sign of expected mu_i under the true w (no noise)
    signs = []
    for i in range(N):
        e_mu = sum(feats[k][i] * true_w[k] for k in range(K))
        signs.append(">" if e_mu > 0 else "<")

    # === V1: array[N] mu, both loops unrolled into for blocks ===
    lines = []
    for k in range(K):
        lines.append(f"data feat{k+1} = {feats[k]};")
    lines.append("")
    lines.append(f"array[{K}] w;")
    lines.append(f"array[{N}] mu;")
    lines.append("")
    lines.append(f"for i in range({K}) {{")
    lines.append("    w[i] = gm([1.], [0.], [1.]);")
    lines.append("} end for;")
    lines.append("")
    # Assign mu[i] (we need to unroll because grammar doesn't support generic
    # expressions in index; but mu[i] inside a for-loop over i works)
    lines.append(f"for i in range({N}) {{")
    lin = " + ".join(f"feat{k+1}[i]*w[{k}]" for k in range(K))
    lines.append(f"    mu[i] = {lin};")
    lines.append("    mu[i] = gm([1.], [0.], [0.1]) + mu[i];")
    lines.append("} end for;")
    lines.append("")
    # Unrolled observes
    for i in range(N):
        lines.append(f"observe(mu[{i}] {signs[i]} 0);")
    v1_text = "\n".join(lines) + "\n"

    # === V2_tmp: single tmp reassigned, unrolled ===
    lines = []
    for k in range(K):
        lines.append(f"data feat{k+1} = {feats[k]};")
    lines.append("")
    lines.append(f"array[{K}] w;")
    lines.append("")
    lines.append(f"for i in range({K}) {{")
    lines.append("    w[i] = gm([1.], [0.], [1.]);")
    lines.append("} end for;")
    lines.append("")
    for i in range(N):
        lin = " + ".join(f"feat{k+1}[{i}]*w[{k}]" for k in range(K))
        lines.append(f"tmp = {lin};")
        lines.append("tmp = gm([1.], [0.], [0.1]) + tmp;")
        lines.append(f"observe(tmp {signs[i]} 0);")
        lines.append("")
    v2_text = "\n".join(lines) + "\n"

    v1_path = os.path.join(out_dir, f"B_K{K}N{N}_V1.soga")
    v2_path = os.path.join(out_dir, f"B_K{K}N{N}_V2.soga")
    with open(v1_path, "w") as f:
        f.write(v1_text)
    with open(v2_path, "w") as f:
        f.write(v2_text)
    return v1_path, v2_path


def gen_pattern_C(T: int, seed: int = 0, out_dir: str = ".") -> tuple[str, str]:
    """Generate V1 (array state) and V2_tmp (single tmp) for Pattern C (Markov).

    Chain: state[t] = state[t-1] + ε_t,  ε_t ~ N(0, 1).
    Observation: state[t] > -100  (loose, ~always satisfied so truncate runs
    but mass survives — we are stress-testing the d³ matrix work, not
    forcing degenerate posteriors).
    """
    # === V1: array[T+1] state, fully unrolled (grammar doesn't allow
    # state[t-1] inside a for-loop body) ===
    lines = []
    lines.append(f"array[{T+1}] state;")
    lines.append("")
    lines.append("state[0] = gm([1.], [0.], [1.]);")
    for t in range(1, T + 1):
        lines.append(f"state[{t}] = state[{t-1}] + gm([1.], [0.], [1.]);")
        lines.append(f"observe(state[{t}] > -100);")
    v1_text = "\n".join(lines) + "\n"

    # === V2_tmp: single tmp ===
    lines = []
    lines.append("tmp = gm([1.], [0.], [1.]);")
    for t in range(1, T + 1):
        lines.append("tmp = tmp + gm([1.], [0.], [1.]);")
        lines.append("observe(tmp > -100);")
    v2_text = "\n".join(lines) + "\n"

    v1_path = os.path.join(out_dir, f"C_T{T}_V1.soga")
    v2_path = os.path.join(out_dir, f"C_T{T}_V2.soga")
    with open(v1_path, "w") as f:
        f.write(v1_text)
    with open(v2_path, "w") as f:
        f.write(v2_text)
    return v1_path, v2_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(here, "inputs")
    os.makedirs(inputs, exist_ok=True)

    # Pattern B
    K = 3
    B_sizes = [6, 12, 25, 50, 75]
    for N in B_sizes:
        v1, v2 = gen_pattern_B(K, N, seed=42, out_dir=inputs)
        print(f"B K={K} N={N:3d}  d_V1={K+N:3d}  d_V2={K+1}  →  {os.path.basename(v1)}, {os.path.basename(v2)}")

    # Pattern C
    C_sizes = [5, 10, 20, 40, 60]
    for T in C_sizes:
        v1, v2 = gen_pattern_C(T, seed=42, out_dir=inputs)
        print(f"C T={T:3d}        d_V1={T+1:3d}  d_V2=1    →  {os.path.basename(v1)}, {os.path.basename(v2)}")


if __name__ == "__main__":
    main()
