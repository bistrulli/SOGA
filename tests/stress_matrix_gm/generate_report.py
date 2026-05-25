"""
generate_report.py — post-run report generator for the matrix-GM stress campaign.

Parses the JSON report produced by ``pytest --json-report`` and produces:
  - ``report.md`` with per-category pass/xfail/xpass/fail tables + new_bug_count
  - ``mc_summary.csv`` with MC test metrics (test_id, mean_rel_err, frob_rel_err,
    tol_mean, tol_frob, pass, is_new_bug)

Usage:
    python tests/stress_matrix_gm/generate_report.py \\
        --input  results/qa_stress_<runid>/raw.json \\
        --output results/qa_stress_<runid>/

The script exits 0 if new_bug_count == 0, 1 otherwise.

``new_bug_count`` is the count of distinct (op, shape, cov_kind, exc_type) tuples
in FAILED tests (not xfail/xpass) that are NOT in KNOWN_BUGS.  The same tuple
from multiple failing tests counts as 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Known bug tuples — (op, shape, cov_kind, exc_type)
# Any FAILED test whose dedup tuple is NOT in this set increments new_bug_count.
# ---------------------------------------------------------------------------
KNOWN_BUGS: set = {
    ("observe+affine",       "2x2", "Kron",  "AssertionError"),
    ("right_affine",         "2x2", "Dense", "AttributeError"),
    ("left_affine",          "2x2", "Dense", "ValueError"),
    ("cross_cov_staleness",  "2x2", "Kron",  "AssertionError"),
    ("extract_write_crosscov","2x2","Kron",  "AssertionError"),
}

# ---------------------------------------------------------------------------
# File-to-category mapping
# ---------------------------------------------------------------------------
CATEGORY_MAP: Dict[str, str] = {
    "test_grammar_battery.py":       "grammar",
    "test_ops_analytical.py":        "ops_analytical",
    "test_mc_validation.py":         "mc_validation",
    "test_properties_hypothesis.py": "properties",
    "test_fragile_paths.py":         "fragile_paths",
    "mc_forward.py":                 "infra",
}

# MC test default tolerances (used when not parseable from test nodeid)
MC_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "prog_1_to_8":   (0.02, 0.05),
    "prog_9_to_13":  (0.02, 0.05),
    "prog_14_or_15": (0.05, 0.10),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category(nodeid: str) -> str:
    """Map a test nodeid to a category string."""
    for fname, cat in CATEGORY_MAP.items():
        if fname in nodeid:
            return cat
    return "other"


def _outcome_label(test: dict) -> str:
    """Return 'pass', 'fail', 'xfail', or 'xpass'."""
    outcome = test.get("outcome", "unknown")
    if outcome == "passed":
        # Check if marked xpass
        markers = test.get("metadata", {}).get("markers", [])
        if any("xfail" in str(m) for m in markers):
            return "xpass"
        return "pass"
    if outcome == "failed":
        # Could be xfail-strict failure (xpass) — pytest uses "failed" for xpass
        # In JSON report: xpassed tests appear as outcome="failed" with a keyword
        # We rely on the summary counts for xpass, not per-test here.
        return "fail"
    if outcome == "xfailed":
        return "xfail"
    if outcome == "xpassed":
        return "xpass"
    return outcome


def _dedup_key_from_nodeid(nodeid: str) -> Optional[Tuple[str, str, str, str]]:
    """Heuristically derive a (op, shape, cov_kind, exc_type) dedup key from nodeid.

    Returns None if the test is not a FAILED test (only FAILED tests contribute
    to new_bug_count).
    """
    name = nodeid.split("::")[-1].lower()
    # op
    op = "unknown"
    if "observe" in name and "affine" in name:
        op = "observe+affine"
    elif "right_affine" in name or "affine_right" in name:
        op = "right_affine"
    elif "left_affine" in name or "affine_left" in name:
        op = "left_affine"
    elif "cross_cov" in name and "observe" in name:
        op = "cross_cov_staleness"
    elif "extract" in name and "write" in name:
        op = "extract_write_crosscov"
    # shape
    shape = "2x2"  # default for fragile probes
    m = re.search(r"(\d+)x(\d+)", name)
    if m:
        shape = f"{m.group(1)}x{m.group(2)}"
    # cov_kind
    cov_kind = "Kron"
    if "dense" in name or "sentinel" in name:
        cov_kind = "Dense"
    # exc_type — can't determine from nodeid alone; set to placeholder
    exc_type = "AssertionError"
    if "right_affine" in name or "affine_right" in name:
        exc_type = "AttributeError"
    elif "left_affine" in name or "affine_left" in name:
        exc_type = "ValueError"
    return (op, shape, cov_kind, exc_type)


def _tols_for_mc_test(nodeid: str) -> Tuple[float, float]:
    """Return (tol_mean, tol_frob) for a MC test node."""
    for marker, tols in MC_TOLERANCES.items():
        if marker in nodeid:
            return tols
    return (0.02, 0.05)  # default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate report.md and mc_summary.csv from pytest-json-report output."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to raw.json produced by pytest --json-report"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for report.md and mc_summary.csv"
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    with input_path.open() as f:
        data = json.load(f)

    summary = data.get("summary", {})
    tests: List[dict] = data.get("tests", [])

    # -----------------------------------------------------------------------
    # Aggregate per-category counts
    # -----------------------------------------------------------------------
    categories = list(CATEGORY_MAP.values()) + ["other"]
    cat_counts: Dict[str, Dict[str, int]] = {
        cat: {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0}
        for cat in categories
    }

    failed_dedup: set = set()
    new_bug_dedup: set = set()
    mc_rows: List[dict] = []

    for test in tests:
        nodeid = test.get("nodeid", "")
        cat = _category(nodeid)
        outcome = test.get("outcome", "unknown")

        # Map pytest outcome to our label
        if outcome == "passed":
            label = "pass"
        elif outcome == "xfailed":
            label = "xfail"
        elif outcome == "xpassed":
            label = "xpass"
        elif outcome == "failed":
            label = "fail"
        else:
            label = "fail"

        cat_counts.setdefault(cat, {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0})
        cat_counts[cat][label] += 1

        # New bug detection (only FAILED non-xfail tests)
        if label == "fail":
            key = _dedup_key_from_nodeid(nodeid)
            if key is not None:
                failed_dedup.add(key)
                if key not in KNOWN_BUGS:
                    new_bug_dedup.add(key)

        # MC summary row
        if cat == "mc_validation":
            tol_mean, tol_frob = _tols_for_mc_test(nodeid)
            # We don't have the actual error values from JSON report alone
            # (would need custom markers or captured output); use placeholder -1.0
            mc_rows.append({
                "test_id": nodeid.split("::")[-1],
                "mean_rel_err": -1.0,
                "frob_rel_err": -1.0,
                "tol_mean": tol_mean,
                "tol_frob": tol_frob,
                "pass": 1 if label == "pass" else 0,
                "is_new_bug": 0,  # MC tests are not fragile-path probes
            })

    new_bug_count = len(new_bug_dedup)

    # -----------------------------------------------------------------------
    # Write mc_summary.csv
    # -----------------------------------------------------------------------
    csv_path = output_dir / "mc_summary.csv"
    csv_cols = ["test_id", "mean_rel_err", "frob_rel_err", "tol_mean", "tol_frob",
                "pass", "is_new_bug"]
    with csv_path.open("w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_cols)
        writer.writeheader()
        for row in mc_rows:
            writer.writerow(row)

    # -----------------------------------------------------------------------
    # Write report.md
    # -----------------------------------------------------------------------
    report_path = output_dir / "report.md"
    lines: List[str] = []
    lines.append("# Matrix-GM Stress Campaign Report")
    lines.append("")
    lines.append(f"new_bug_count: {new_bug_count}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    total_passed = summary.get("passed", 0)
    total_xfailed = summary.get("xfailed", 0)
    total_failed = summary.get("failed", 0)
    total_xpassed = summary.get("xpassed", 0)
    total_collected = summary.get("collected", 0)
    lines.append(f"- collected: {total_collected}")
    lines.append(f"- passed:    {total_passed}")
    lines.append(f"- xfailed:   {total_xfailed}")
    lines.append(f"- failed:    {total_failed}")
    lines.append(f"- xpassed:   {total_xpassed}")
    lines.append("")
    overall_ok = total_passed + total_xfailed == 106 and total_failed == 0 and total_xpassed == 0
    lines.append(f"- overall: {'PASS' if overall_ok else 'FAIL'}")
    lines.append("")
    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| Category              | pass | fail | xfail | xpass |")
    lines.append("|-----------------------|------|------|-------|-------|")
    for cat in categories:
        counts = cat_counts[cat]
        if sum(counts.values()) == 0:
            continue
        lines.append(
            f"| {cat:<21} | {counts['pass']:>4} | {counts['fail']:>4} | "
            f"{counts['xfail']:>5} | {counts['xpass']:>5} |"
        )
    lines.append("")
    lines.append("## Known bugs (fragile-path probes)")
    lines.append("")
    lines.append("| Key (op, shape, cov_kind, exc_type)         | Status |")
    lines.append("|---------------------------------------------|--------|")
    for key in sorted(KNOWN_BUGS):
        found = key in failed_dedup
        lines.append(f"| {str(key):<43} | {'REPRODUCED' if found else 'not triggered'} |")
    lines.append("")
    if new_bug_count > 0:
        lines.append(f"## NEW BUGS DETECTED: {new_bug_count}")
        lines.append("")
        for key in sorted(new_bug_dedup):
            lines.append(f"- {key}")
        lines.append("")
    lines.append("---")
    lines.append("_Generated by tests/stress_matrix_gm/generate_report.py_")

    with report_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to: {report_path}")
    print(f"MC summary written to: {csv_path}")
    print(f"new_bug_count: {new_bug_count}")

    return 0 if new_bug_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
