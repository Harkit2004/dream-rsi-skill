"""One writer per run directory (issue #72).

A run directory has exactly one history and every guarantee around it — the
manifest a resume is held to (issue #45), the record that says a cycle finished,
the flushed writes that keep a finished cycle across a power loss (issue #47) —
assumes a single process is appending to it. Two are not merely redundant: both
read the same records, decide the same cycle is next, and write into the same
cycle directory and pool.

Under an agent harness that is the ordinary way to end up with two. A real run
outlasts one tool call, so the agent launches it detached and then reruns the
command to see how it is going. So :class:`RunLock` holds an exclusive
``flock`` on a file inside the directory for as long as a run is writing to it,
and a second process is refused before it touches anything.

**The OS owns the lock, not this module.** ``flock`` is released when the holder
exits by any route — a crash, ``SIGKILL``, a power loss and reboot — so there is
no stale-lock cleanup to write and none that could go wrong. The holder's pid is
written into the file so a refusal can say who to look at. It is for that
message only: the lock's correctness never reads it, so a stale or missing pid
costs a less helpful sentence and never a wrong decision.

``flock`` is POSIX, like the sandbox (:mod:`dream_rsi.sandbox`), and a platform
without one is refused earlier and by name (issue #67). ``fcntl`` is therefore
imported where it is used, so the package still imports elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

__all__ = ["LOCK_FILENAME", "RunInUseError", "RunLock", "probe"]

LOCK_FILENAME = "run.lock"


class RunInUseError(RuntimeError):
    """Another process is writing to this run directory, so this one may not."""


class RunLock:
    """An exclusive lock on a run directory, held for the length of a ``with`` block.

    Entering creates the directory if it is missing, and the lock file if that is
    missing, and nothing else: an existing directory that is held is refused
    without any of its files being opened for writing.
    """

    def __init__(self, directory: str | Path) -> None:
        self._path = Path(directory) / LOCK_FILENAME
        self._descriptor: int | None = None

    # ``Self`` would be the annotation, and it needs Python 3.11 while this package
    # supports 3.10 (as ``sandbox.SandboxedPolicy.__enter__`` says of itself).
    def __enter__(self) -> RunLock:  # noqa: PYI034
        import fcntl

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # No ``O_TRUNC``: opening a held lock file must not change it, or refusing a
        # second process would have already written to the directory.
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _holder(descriptor)
            os.close(descriptor)
            raise RunInUseError(
                f"{self._path.parent} is in use by another run{holder}: "
                "wait for it to finish, or read its progress without starting a cycle"
            ) from None
        except BaseException:
            os.close(descriptor)
            raise
        # Ours now, so this is the one process allowed to write it.
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        self._descriptor = descriptor
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._descriptor is not None:
            # Closing the descriptor is what releases the lock.
            os.close(self._descriptor)
            self._descriptor = None


def probe(directory: str | Path) -> tuple[bool, int | None]:
    """Whether a process holds ``directory``'s lock, and the pid it wrote, without writing.

    For a report on a run that must never be able to disturb it (issue #73). The lock
    file is opened read-only, and never created: a directory that has none is not
    held; one that cannot be opened for another reason raises :class:`OSError`, because
    not knowing whether it is held is not knowing that it is free. A shared lock is
    tried and dropped at once, which is a lock a running
    writer refuses and another prober does not.

    The one cost of asking: for the moment that shared lock is held, a run that is
    *starting* on this directory would be refused as "in use". That fails closed —
    a run refused is a run not started — and the window is microseconds, which is
    what a probe that never writes costs.
    """
    import fcntl

    try:
        descriptor = os.open(Path(directory) / LOCK_FILENAME, os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        return False, None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, _pid(descriptor)
        return False, None
    finally:
        os.close(descriptor)


def _pid(descriptor: int) -> int | None:
    """The pid the holder wrote into the lock file, or ``None`` if nothing legible.

    Read through the descriptor already open rather than the path: it is the file
    the ``flock`` was about, and reading it changes nothing.
    """
    try:
        text = os.pread(descriptor, 32, 0).decode("ascii", errors="replace").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _holder(descriptor: int) -> str:
    """`` (pid N)`` for the process that wrote the lock file, or nothing legible."""
    pid = _pid(descriptor)
    return "" if pid is None else f" (pid {pid})"
