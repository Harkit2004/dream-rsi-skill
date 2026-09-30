"""A stand-in for a coding-agent CLI, so no test calls a model (issue #70).

Gemini CLI, Claude Code and OpenCode each have a non-interactive mode that takes a
prompt, works in the current directory, and exits. This does the same with none of
the model: it reads the prompt it is handed — as the last argument, or on stdin
with ``--stdin`` — finds the paths the prompt names, records what it could see
there, and writes the files a discovery agent would.

The paths are read out of the prompt, as a real agent would. A test template lists
them as ``KEY=value`` lines; the paper's own prompt, which the default template
renders, lists them in one sentence — ``Variables (`node`, `history`, `baseline`,
`program`, `problem`) are filled in ...`` — and both are understood.

``--mode`` picks what it does, so one script covers the failures too:

* ``write`` (default): a proposal and a program, both naming the seed.
* ``plan``: a program the toy task can score, ``PLAN = (width, depth)``.
* ``fail``: prints to stderr and exits 3.  ``write-then-fail``: writes a good
  attempt and *then* exits 4.  ``hang``: never returns.  ``flood``: prints without
  end.
* ``no-program``, ``empty-program``, ``unchanged``, ``no-proposal``: the ways an
  attempt can leave the workspace short of what was asked for.

What it saw goes to a JSON file in ``$STAND_IN_LOG`` when that is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BACKTICKED = re.compile(r"`([^`]*)`")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="write")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args()
    prompt = sys.stdin.read() if args.stdin else (args.prompt or "")

    fields = _fields(prompt)
    node = Path(fields["NODE"])
    seed = fields.get("SEED") or os.environ.get("DREAM_RSI_SEED") or str(os.getpid())
    program = node / fields["PROGRAM"]
    _log(
        {
            "argv": sys.argv[1:],
            "seed_env": os.environ.get("DREAM_RSI_SEED"),
            "cwd": os.getcwd(),
            "prompt": prompt,
            "fields": fields,
            "problem": Path(fields["PROBLEM"]).read_text(encoding="utf-8"),
            "history": _history(Path(fields["HISTORY"]), fields["PROGRAM"]),
            "baseline_exists": Path(fields["BASELINE"]).is_dir(),
        }
    )

    if args.mode == "fail":
        print("boom: the stand-in was told to fail", file=sys.stderr)
        return 3
    if args.mode == "hang":
        time.sleep(600)
        return 0
    if args.mode == "flood":
        chunk = "x" * 65536
        while True:
            sys.stdout.write(chunk)
    if args.mode == "unchanged":
        return 0
    if args.mode != "no-proposal":
        (node / "proposal.md").write_text(f"proposal for seed {seed}\n", encoding="utf-8")
    if args.mode == "no-program":
        return 0
    program.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "empty-program":
        program.write_text("", encoding="utf-8")
    elif args.mode == "plan":
        number = int(seed)
        program.write_text(f"PLAN = ({number % 5}, {(number // 5) % 5})\n", encoding="utf-8")
    else:
        program.write_text(f"program for seed {seed}\n", encoding="utf-8")
    if args.mode == "write-then-fail":
        print("crashed after writing", file=sys.stderr)
        return 4
    return 0


def _fields(prompt: str) -> dict[str, str]:
    """The paths the prompt names: ``KEY=value`` lines, or the paper's ``Variables (...)`` line."""
    keyed = {}
    for line in prompt.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.isupper() and key.isalpha():
            keyed[key] = value.strip()
    if "NODE" in keyed:
        return keyed
    names = ("NODE", "HISTORY", "BASELINE", "PROGRAM", "PROBLEM")
    for line in prompt.splitlines():
        if line.startswith("Variables ("):
            # The five variables, in the paper's order; the same line goes on to
            # backtick `attempt_*/` and the node directory again.
            return dict(zip(names, BACKTICKED.findall(line)[: len(names)], strict=True))
    raise SystemExit(f"the prompt names no paths: {prompt[:200]!r}")


def _history(directory: Path, program: str) -> dict[str, dict[str, object]]:
    """Every attempt directory the prompt pointed at, and what is in it."""
    seen: dict[str, dict[str, object]] = {}
    for attempt in sorted(directory.glob("attempt_*")):
        proposal, code = attempt / "proposal.md", attempt / program
        score, error = attempt / "eval" / "score.json", attempt / "eval" / "error.txt"
        seen[attempt.name] = {
            "proposal": proposal.read_text(encoding="utf-8") if proposal.is_file() else None,
            "program": code.read_text(encoding="utf-8") if code.is_file() else None,
            "score": json.loads(score.read_text(encoding="utf-8")) if score.is_file() else None,
            "error": error.read_text(encoding="utf-8") if error.is_file() else None,
        }
    return seen


def _log(record: dict[str, object]) -> None:
    directory = os.environ.get("STAND_IN_LOG")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"call_{time.time_ns()}_{os.getpid()}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(main())
