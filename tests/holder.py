"""A real run to hold a run directory, and the helpers that look at one from outside.

Shared by the tests of the lock (issue #72) and of the report (issue #73), which
both need a run that is genuinely mid-cycle: the CLI, with the toy discovery agent
swapped for one that announces itself and then hangs. ``main`` is the real one, so
the holder takes the lock exactly where a real run does and stays inside a cycle for
as long as it lives.
"""

from __future__ import annotations

import select
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Generous next to what the toy task spends: nothing here times the machine, it is
# there so a holder that never came up fails the test instead of hanging it.
TIMEOUT_SECONDS = 120

HOLDER = textwrap.dedent(
    """
    import sys
    import time

    import dream_rsi.run as run


    class Hangs:
        def __init__(self, script):
            pass

        def propose(self, context):
            print("started", flush=True)
            time.sleep(3600)


    run.ToySearchAgent = Hangs
    raise SystemExit(run.main(sys.argv[1:]))
    """
)


def cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """The real command, run to completion."""
    return subprocess.run(
        [sys.executable, "-m", "dream_rsi.run", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )


@contextmanager
def holding(tmp_path: Path, *arguments: str) -> Iterator[subprocess.Popen[str]]:
    """A run that is mid-cycle, and is killed however the block ends."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], TIMEOUT_SECONDS)
        assert ready, "the holding run never reached its first attempt"
        assert process.stdout.readline().strip() == "started"
        yield process
    finally:
        process.kill()
        process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def contents(directory: Path) -> dict[str, bytes | None]:
    """Every path under ``directory`` with its bytes, so "unchanged" means unchanged."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }
