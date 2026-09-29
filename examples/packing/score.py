"""Score a candidate for the packing example, by the contract ``CommandEvaluator`` reads.

Run in a candidate's workspace, where the candidate is ``solution.py``::

    python score.py

It runs ``solution.py``, checks what it printed, and writes ``eval/score.json`` holding
``{"score": <smallest pairwise distance>, "correct": true}``. An answer it cannot score —
it crashed, hung, printed something that is not ten points in the unit square — writes
``eval/error.txt`` with the reason instead, which is what the discovery agent reads next.

The candidate runs in a child process with a timeout, so a candidate that loops forever or
prints without end costs thirty seconds and not the run. This is the *user's* scorer: it is
trusted code, and running the candidate is what it is for. It is not a sandbox, and model-
written *policy* code never comes near it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

POINTS = 10
TIMEOUT_SECONDS = 30
# The most of a candidate's output this reads: ten points is a few hundred bytes.
OUTPUT_LIMIT = 1_000_000


def min_distance(points: list[tuple[float, float]]) -> float:
    """The smallest distance between any two of ``points``."""
    return min(
        math.dist(first, second)
        for index, first in enumerate(points)
        for second in points[index + 1 :]
    )


def parse(output: str) -> list[tuple[float, float]]:
    """``points`` from what a candidate printed, or ``ValueError`` saying what is wrong."""
    try:
        payload = json.loads(output)
    except ValueError as exc:
        raise ValueError(f"the output is not JSON: {exc}") from exc
    if not isinstance(payload, list) or len(payload) != POINTS:
        got = len(payload) if isinstance(payload, list) else type(payload).__name__
        raise ValueError(f"expected a list of {POINTS} points, got {got}")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(payload):
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in point)
        ):
            raise ValueError(f"point {index} must be a pair of numbers, got {point!r}")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"point {index} = ({x}, {y}) is outside the unit square")
        points.append((x, y))
    return points


def evaluate(program: Path, timeout: float = TIMEOUT_SECONDS) -> float:
    """Run ``program`` and score what it prints, or raise ``ValueError`` saying why not."""
    try:
        completed = subprocess.run(
            [sys.executable, str(program)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"solution.py timed out after {timeout:g}s") from None
    if completed.returncode != 0:
        tail = completed.stderr.strip()[-500:]
        raise ValueError(f"solution.py exited with status {completed.returncode}: {tail}")
    return min_distance(parse(completed.stdout[:OUTPUT_LIMIT]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score solution.py in the current directory: the smallest pairwise "
        "distance of the ten points it prints (larger is better)."
    )
    parser.add_argument("--program", type=Path, default=Path("solution.py"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_SECONDS,
        help=f"seconds the candidate may run (default: {TIMEOUT_SECONDS})",
    )
    args = parser.parse_args(argv)

    reports = Path("eval")
    reports.mkdir(exist_ok=True)
    try:
        score = evaluate(args.program, args.timeout)
    except ValueError as exc:
        (reports / "error.txt").write_text(f"{exc}\n", encoding="utf-8")
        return 0
    (reports / "score.json").write_text(
        json.dumps({"score": score, "correct": True}) + "\n", encoding="utf-8"
    )
    print(f"smallest pairwise distance: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
