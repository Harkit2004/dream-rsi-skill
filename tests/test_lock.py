"""At most one process ever writes to a run directory (issue #72).

Under a harness the common case is a run that outlasts one tool call, so the agent
launches it detached and then reruns the command to see how it is going — which
started a second writer. Both read the same cycle records, decided the same cycle
was next, and wrote into the same cycle directory and pool.

The holder in these tests is a *real* run: the CLI with a discovery agent that
announces itself and then hangs, so the lock is held by whatever ``run_cycles``
takes and not by a helper the tests wrote. A ``run_cycles`` that forgot to lock
would therefore fail them.
"""

from __future__ import annotations

import select
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from dream_rsi.lock import RunInUseError, RunLock

# Generous next to what the toy task spends: nothing here times the machine, it is
# there so a holder that never came up fails the test instead of hanging it.
TIMEOUT_SECONDS = 120

# The CLI, with the toy discovery agent swapped for one that says it has started
# and then never returns. ``main`` is the real one, so the holder takes the lock
# exactly where a real run does, and stays inside a cycle for as long as it lives.
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


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dream_rsi.run", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )


@contextmanager
def _holding(tmp_path: Path, *arguments: str) -> Iterator[subprocess.Popen[str]]:
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


def _contents(directory: Path) -> dict[str, bytes | None]:
    """Every path under ``directory`` with its bytes, so "unchanged" means unchanged."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }


def test_a_second_run_on_a_held_directory_is_refused_before_it_touches_anything(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"

    with _holding(tmp_path, "--cycles", "1", str(directory)) as holder:
        before = _contents(directory)

        second = _cli("--cycles", "1", str(directory))

        assert second.returncode != 0
        assert "in use" in second.stderr
        # The pid is written for this message alone, so what is asserted is that the
        # refusal says who to look at, not that anything depends on it.
        assert str(holder.pid) in second.stderr
        assert "Traceback" not in second.stderr
        assert _contents(directory) == before


def test_a_directory_whose_holder_was_killed_can_be_run_again_and_resumes(
    tmp_path: Path,
) -> None:
    """The OS releases a dead holder's lock, so there is no stale lock to clean up."""
    directory = tmp_path / "run"
    first = _cli("--cycles", "1", str(directory))
    assert first.returncode == 0, first.stderr

    with _holding(tmp_path, "--cycles", "2", str(directory)) as holder:
        holder.send_signal(signal.SIGKILL)
        holder.wait()

    resumed = _cli("--cycles", "2", str(directory))

    assert resumed.returncode == 0, resumed.stderr
    # Cycle 0 was read back from its record and cycle 1, which the killed run left
    # without one, was done from the top.
    assert resumed.stdout.startswith("run: 2 cycle(s)")


def test_a_lock_is_released_when_its_holder_lets_go(tmp_path: Path) -> None:
    with RunLock(tmp_path), pytest.raises(RunInUseError, match="in use"), RunLock(tmp_path):
        pass

    # Released by leaving the block, whether or not it was left by an exception.
    with RunLock(tmp_path):
        pass


def test_the_lock_module_imports_where_fcntl_does_not_exist() -> None:
    """``fcntl`` is POSIX-only, and a top-level import would break the package on Windows (#67)."""
    hide_fcntl = "import sys; sys.modules['fcntl'] = None; import dream_rsi.run, dream_rsi.lock"

    completed = subprocess.run(
        [sys.executable, "-c", hide_fcntl],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
