"""
Isolated re-test of the 3 programs that failed in bench_all_canonical.py
(ClickGraph timeout, RandomWalkUnif10 timeout, DigitRecognition TypeError).

Each program is tried INDEPENDENTLY in both modes, so a failure in one
mode does not prevent the other from being measured. Uses subprocess +
process-level timeout via SOGA's existing multiprocessing wrapper, which
is the hardest kill we can get without altering core SOGA code.

Methodology:
  - Invoke `python SOGA.py -f <prog>` and `python SOGA.py -f <prog> --sparse-truncate`
  - Cap each at 180s wall-clock (subprocess timeout)
  - Parse the printed "Runtime: X s" if SOGA completes
  - Categorize: completed | timeout | error
"""

import os
import sys
import subprocess
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")

# Programs that failed in the comprehensive bench
PROGRAMS = [
    ("ClickGraph",        os.path.join(ROOT, "programs/SOGA/ClickGraph.soga")),
    ("RandomWalkUnif10",  os.path.join(ROOT, "programs/SOGA/RandomWalkUnif10.soga")),
    ("DigitRecognition",  os.path.join(ROOT, "programs/SOGA/DigitRecognition.soga")),
]

WALL_TIMEOUT_S = 180.0    # outer subprocess timeout
SOGA_INNER_T   = 150      # inner SOGA -t timeout (seconds, integer)


def run_one(soga_path: str, sparse: bool) -> dict:
    """Return dict with keys: status, soga_runtime_s, n_comp, d, error, stdout_tail."""
    cmd = [VENV_PY, "SOGA.py", "-f", soga_path, "-t", str(SOGA_INNER_T)]
    if sparse:
        cmd.append("--sparse-truncate")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd, cwd=SRC, capture_output=True, text=True, timeout=WALL_TIMEOUT_S
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {"status": "wall-timeout", "wall_s": WALL_TIMEOUT_S}

    out = r.stdout
    # Detect SOGA's internal timeout printout
    if "Warning: SOGA Timeout occurred" in out:
        return {"status": "soga-timeout", "wall_s": wall, "stdout_tail": out[-500:]}

    runtime = None
    n_comp, d = None, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Runtime:"):
            # "Runtime: 12.345 s"
            try:
                runtime = float(s.split(":")[1].strip().split()[0])
            except Exception:
                pass
        elif s.startswith("c:"):
            try:
                n_comp = int(s.split(":")[1].strip())
            except Exception:
                pass
        elif s.startswith("d:"):
            try:
                d = int(s.split(":")[1].strip())
            except Exception:
                pass

    if r.returncode != 0 or runtime is None:
        # Look for a traceback fragment
        err_tail = (r.stderr or out)[-500:]
        return {"status": "error", "wall_s": wall, "exit": r.returncode, "stderr_tail": err_tail}

    return {
        "status": "ok",
        "wall_s": wall,
        "soga_runtime_s": runtime,
        "n_comp": n_comp,
        "d": d,
    }


def main():
    print(f"{'Program':<22s} {'Mode':>8s}  {'Status':<15s}  {'SOGA runtime':>14s}  {'n_comp':>8s}  {'d':>4s}")
    print("-" * 88)
    rows = []
    for label, path in PROGRAMS:
        for sparse in (False, True):
            mode = "sparse" if sparse else "classic"
            r = run_one(path, sparse)
            r["program"] = label
            r["mode"] = mode
            rows.append(r)
            if r["status"] == "ok":
                print(
                    f"{label:<22s} {mode:>8s}  {'completed':<15s}  "
                    f"{r['soga_runtime_s']:>11.3f} s  {r.get('n_comp',''):>8}  {r.get('d',''):>4}"
                )
            elif r["status"] == "wall-timeout":
                print(f"{label:<22s} {mode:>8s}  {'wall-timeout':<15s}  {WALL_TIMEOUT_S:>11.1f} s  (subprocess killed)")
            elif r["status"] == "soga-timeout":
                print(f"{label:<22s} {mode:>8s}  {'soga-timeout':<15s}  {SOGA_INNER_T:>11d} s  (SOGA -t fired)")
            else:
                print(f"{label:<22s} {mode:>8s}  {r['status']:<15s}  (exit={r.get('exit','?')})")
                if r.get("stderr_tail"):
                    for ln in r["stderr_tail"].splitlines()[-3:]:
                        print(f"    {ln}")

    # Pairwise summary
    print("\n=== Pairwise comparison (classic vs sparse) ===")
    by_prog = {}
    for r in rows:
        by_prog.setdefault(r["program"], {})[r["mode"]] = r
    for prog, pair in by_prog.items():
        c = pair.get("classic", {})
        s = pair.get("sparse", {})
        print(f"\n{prog}:")
        print(f"  classic: {c.get('status')}  runtime={c.get('soga_runtime_s')}")
        print(f"  sparse : {s.get('status')}  runtime={s.get('soga_runtime_s')}")
        if c.get("status") == "ok" and s.get("status") == "ok":
            spd = c["soga_runtime_s"] / s["soga_runtime_s"]
            print(f"  --> both completed; speedup = {spd:.2f}x")
        elif c.get("status") != "ok" and s.get("status") == "ok":
            print(f"  --> classic FAILED but sparse COMPLETED (sparse made the difference)")
        elif c.get("status") == "ok" and s.get("status") != "ok":
            print(f"  --> sparse FAILED but classic COMPLETED (regression!)")
        else:
            print(f"  --> both modes fail (structural issue independent of the optimization)")

if __name__ == "__main__":
    main()
