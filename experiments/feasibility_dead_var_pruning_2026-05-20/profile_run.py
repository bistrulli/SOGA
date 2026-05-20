"""
Feasibility study — Method B (profile real benchmark).

Runs BayesPointMachine through SOGA in-process under cProfile to identify
which functions dominate the runtime. Output: pstats dump + top-N text reports.

Usage:
    python profile_run.py <input.soga> <output_prefix>
"""

import cProfile
import pstats
import sys
import os
from io import StringIO

# Make src/ importable
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA

import numpy as np
import random
random.seed(0)
np.random.seed(0)


def run_soga(input_path):
    compiled = compile2SOGA(input_path)
    cfg = produce_cfg(compiled)
    out = start_SOGA(cfg, useR=False, parallel=None)
    return out


def main():
    input_path = sys.argv[1]
    out_prefix = sys.argv[2]

    profiler = cProfile.Profile()
    profiler.enable()
    result = run_soga(input_path)
    profiler.disable()

    # Dump raw stats
    profiler.dump_stats(out_prefix + ".pstats")

    # Top 30 by cumulative time
    buf = StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(30)
    with open(out_prefix + ".cumulative.txt", "w") as f:
        f.write(buf.getvalue())

    # Top 30 by total time (own time)
    buf = StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    ps.print_stats(30)
    with open(out_prefix + ".tottime.txt", "w") as f:
        f.write(buf.getvalue())

    # Summary
    print("=== RESULT ===")
    print(f"n_comp = {result.gm.n_comp()}")
    print(f"d      = {len(result.var_list)}")
    means = result.gm.mean()
    for var, val in zip(result.var_list, means):
        print(f"E[{var}] = {val:.5f}")
    print(f"\nProfile dumps written to: {out_prefix}.{{pstats,cumulative.txt,tottime.txt}}")


if __name__ == "__main__":
    main()
