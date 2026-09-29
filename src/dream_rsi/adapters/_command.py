"""Running an external command with a timeout and bounded output.

Three adapters drive a command the user chose — a scorer
(:mod:`~dream_rsi.adapters.command_evaluator`), a discovery agent
(:mod:`~dream_rsi.adapters.command_agent`) and a policy-development agent
(:mod:`~dream_rsi.adapters.command_developer`) — and each needs the same three
guarantees from the process underneath, which are easy to get subtly wrong and
worth getting right once:

* **A timeout that ends the whole command.** The command gets its own process
  group, and the group is killed when the command ends — on a timeout, and after a
  clean exit too — so a benchmark or a coding-agent CLI that started children does
  not leave them running after the loop has stopped waiting for it.
* **Output that is neither unbounded nor a way to hang.** The command's stdout and
  stderr go to a file, not a pipe. A pipe is held open by everything the command
  started, so a child left behind would keep the caller waiting past the kill, and
  a pipe buffers all of what a chatty command prints in memory besides. Only the
  tail is read back: a failure is read to find out why the command stopped.
* **No shell.** The command is an argv list, so nothing the candidate, the prompt
  or the workspace contains is ever interpreted as one.

POSIX, like the sandbox (:mod:`dream_rsi.sandbox`): a process group and
``SIGKILL`` are looked up when a command is run, so importing this on another
platform still works and the loop refuses to start there by name (issue #67).
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

__all__ = ["TAIL_BYTES", "CommandOutcome", "run_command", "tail_of"]

# How much of a command's output goes into a failure record: the tail, since that
# is where a command says why it stopped. The same bound ``sandbox`` puts on a
# failed candidate's output.
TAIL_BYTES = 4096


@dataclass(frozen=True)
class CommandOutcome:
    """How a command ended: its exit status, and the tail of what it printed.

    ``status`` is ``None`` for a command that outlived its timeout and was killed.
    ``output`` is stdout and stderr together, in the order they were written, so
    the tail reads as one log.
    """

    status: int | None
    output: str


def run_command(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float,
    stdin: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandOutcome:
    """Run ``argv`` in ``cwd`` for at most ``timeout`` seconds.

    ``stdin`` is fed from a file, for the reason output goes to one: a pipe the
    command never reads would block the writer. ``OSError`` from starting the
    command — not on the path, not executable — is left to the caller, which knows
    what a command that cannot start means for its own role.
    """
    with tempfile.TemporaryFile() as sink, tempfile.TemporaryFile() as feed:
        if stdin is not None:
            feed.write(stdin.encode("utf-8"))
            feed.flush()
            feed.seek(0)
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL if stdin is None else feed,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status: int | None
        try:
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = None
        finally:
            # Also on an interrupt: a command outliving the loop that started it is
            # the failure this guards against, and the group is the only handle on
            # everything it started.
            _kill_group(process)
        return CommandOutcome(status=status, output=_tail(sink))


def tail_of(text: str) -> str:
    """The last :data:`TAIL_BYTES` characters of ``text``, stripped."""
    return text[-TAIL_BYTES:].strip()


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """End everything ``process`` started, and reap ``process`` itself.

    Whether or not the command itself is still running: one that exited cleanly can
    have left a child behind, and a process nothing is waiting for is a leak. A
    group with nothing left in it is not an error.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def _tail(sink: IO[bytes]) -> str:
    """The last :data:`TAIL_BYTES` of what the command wrote to ``sink``."""
    sink.seek(0, os.SEEK_END)
    size = sink.tell()
    sink.seek(max(0, size - TAIL_BYTES))
    return sink.read().decode("utf-8", errors="replace").strip()
