"""A stand-in for a coding-agent CLI in the policy-development role (issue #71).

The adapter hands the CLI a short instruction — as the last argument, or on stdin
with ``--stdin`` — and puts the real prompt in ``prompt.md`` and the policy being
revised in ``policy.py``, in the current directory. This reads them, records what
it saw, and writes ``revised_policy.py`` the way a model told to would. No model is
called.

``--mode`` picks what it does, so one script covers the failures too:

* ``revise`` (default): writes the contents of the file ``--revision`` names.
* ``prose``: writes something that is not Python.
* ``fail``: prints to stderr and exits 3.  ``write-then-fail``: writes the revision
  and *then* exits 4.  ``hang``: never returns.
* ``no-file``: exits 0 having written nothing.  ``empty``: writes an empty file.

What it saw goes to a JSON file in ``$STAND_IN_LOG`` when that is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="revise")
    parser.add_argument("--revision")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("instruction", nargs="?")
    args = parser.parse_args()
    instruction = sys.stdin.read() if args.stdin else (args.instruction or "")

    prompt = Path("prompt.md")
    policy = Path("policy.py")
    _log(
        {
            "argv": sys.argv[1:],
            "instruction": instruction,
            "cwd": os.getcwd(),
            "prompt": prompt.read_text(encoding="utf-8") if prompt.is_file() else None,
            "policy": policy.read_text(encoding="utf-8") if policy.is_file() else None,
            "files": sorted(entry.name for entry in Path.cwd().iterdir()),
        }
    )

    if args.mode == "fail":
        print("boom: the stand-in was told to fail", file=sys.stderr)
        return 3
    if args.mode == "hang":
        time.sleep(600)
        return 0
    if args.mode == "no-file":
        return 0
    output = Path("revised_policy.py")
    if args.mode == "empty":
        output.write_text("", encoding="utf-8")
    elif args.mode == "prose":
        output.write_text("I would suggest widening the batch and stopping earlier.\n")
    else:
        output.write_text(Path(args.revision).read_text(encoding="utf-8"), encoding="utf-8")
    if args.mode == "write-then-fail":
        print("crashed after writing", file=sys.stderr)
        return 4
    return 0


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
