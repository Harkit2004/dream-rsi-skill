"""Running a candidate policy under limits, because a candidate is model-written.

Dream-RSI's policy is executable Python (§3, §B.2) and step 4 of the loop has an
LLM rewrite it (issue #14). So the dreaming harness evaluates code nobody
reviewed, and CLAUDE.md's first warning is this one: that path stays sandboxed
and resource-capped, and the caps are never relaxed to make a test pass.

The boundary is a process. :class:`SandboxedPolicy` looks like a policy to
``ReplaySimulator.replay`` — it answers ``reset`` and ``select`` — and holds a
child process where the candidate's source was compiled and where every one of
its decisions is actually taken. A hang, a memory bomb, a fork, a socket or a
write outside the scratch directory is therefore something that happens *there*,
and what the harness sees is one raised exception: issue #12's sweep already
records a candidate that raises as one failed cell and carries on, so the
contract this module has to meet is simply to raise :class:`SandboxError`
instead of hanging, crashing the parent or letting the attempt through.

A process is also what makes the *feedback* possible. The child's stdout and
stderr are captured to files for the life of the policy, so when a candidate
falls over, what it printed on the way down and what it raised both survive into
:attr:`SandboxError.stdout` / :attr:`SandboxError.stderr` and into the error text
the harness records — which is what the development agent revises the next
version from (issue #13's scope, issue #14's input).

Three properties are worth stating because they are what a reviewer should check:

**There is no bypass.** :func:`sandboxed_candidate` is the only way source
becomes a :class:`~dream_rsi.dream.PolicyCandidate`, it always starts a child,
and :class:`SandboxLimits` cannot express "no limit" — every field is a positive
number, so there is no flag, no ``None`` and no environment variable that turns
the caps off.

**The sandbox changes no number.** A correct candidate has to replay exactly as
the same strategy does in-process, or the dreaming score becomes a measurement of
the execution path; ``tests/test_sandbox.py`` pins the whole serialised
trajectory of a sandboxed baseline against the in-process one. Nothing
non-deterministic crosses the boundary in either direction: the request carries
the revealed tree, the eligible set, ``W`` and the generator's state, and the
response carries a batch of node ids.

**The parent parses, it does not unpickle.** Requests and responses are
line-delimited JSON, so a candidate that writes whatever it likes down the pipe
can at worst produce a malformed response — which is a failed cell. Handing the
child a pickle channel would instead let it run code in the harness, which is
the one thing a sandbox may not do.

What the child enforces, and the one layer of it that a determined escape can
reach past, is documented in :mod:`dream_rsi._sandbox_child`.
"""

from __future__ import annotations

import json
import os
import random
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dream_rsi._sandbox_child import DEFAULT_POLICY_NAME
from dream_rsi.dream import PolicyCandidate
from dream_rsi.tree import DiscoveryTree

__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_POLICY_NAME",
    "SandboxError",
    "SandboxLimits",
    "SandboxedPolicy",
    "sandboxed_candidate",
]

# Drop the working directory — the scratch directory — off the child's import
# path, so a candidate cannot shadow a module by writing a file it is allowed to
# write. ``-P`` would do it, but only from Python 3.11, and this package
# supports 3.10.
_BOOTSTRAP = (
    "import sys; sys.path[:] = [entry for entry in sys.path if entry]; "
    "from dream_rsi._sandbox_child import main; raise SystemExit(main())"
)

# How much of a failed candidate's output goes into the failure record. The tail
# rather than the head: a record is read to find out why the candidate stopped.
_CAPTURE_LIMIT = 4096

# How much of one response this process will hold while waiting for the end of
# the line. A cap here and not only in the child, because the descriptor the
# protocol answers on is open in the process the candidate runs in and
# ``os.write`` raises no audit event: a candidate can write down it directly and
# never send a newline, and then the deadline bounds only how long this process
# spends allocating. One legal response is a batch of node ids, so a megabyte is
# already orders of magnitude more than a correct candidate produces.
_RESPONSE_LIMIT = 1024 * 1024


@dataclass(frozen=True)
class SandboxLimits:
    """What one candidate policy may spend.

    ``wall_seconds`` bounds a single decision — one ``select``, one ``reset``, or
    compiling the candidate's source — and is the only thing that catches a
    policy that blocks without computing, on a sleep or on a read. It is per
    request rather than per rollout because the harness has no other handle on
    "still making progress".

    ``cpu_seconds`` is cumulative over the policy's whole life, so it also bounds
    a candidate that burns a slice of CPU on every round rather than spinning on
    one, and ``memory_bytes`` caps its address space.

    ``disk_bytes`` caps any one file the child writes. The scratch directory is a
    directory on a real filesystem, and the candidate's own stdout and stderr are
    redirected to files, so without it a policy reaches the host's disk by
    writing — or simply by printing — for as long as it is alive.

    PAPER-GAP: the paper does not discuss executing the policy code it has an LLM
    write — no limits, no isolation, nothing on what happens to a version that
    hangs (§3 describes only the evaluation). The defaults here are set from what
    a legal policy does rather than from the paper: a decision reads the revealed
    prefix and picks a batch, which over the trees in ``tests/fixtures/trees/``
    is microseconds and on a realistic tree is still a walk over a few thousand
    nodes, so seconds of CPU and hundreds of megabytes are already orders of
    magnitude of headroom, and anything beyond them is a candidate doing
    something other than deciding. They are per-candidate configuration, not
    constants: a domain with genuinely expensive decisions raises them for its
    own runs. Revisit if the authors' implementation lands (see
    references/method.md).
    """

    wall_seconds: float = 10.0
    cpu_seconds: int = 5
    memory_bytes: int = 512 * 1024 * 1024
    disk_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        # Not "must be non-negative": a limit of zero is a limit nothing can
        # satisfy, and a negative one is ``RLIM_INFINITY`` spelled by accident.
        if self.wall_seconds <= 0:
            raise ValueError(f"wall_seconds must be positive, got {self.wall_seconds}")
        if self.cpu_seconds <= 0:
            raise ValueError(f"cpu_seconds must be positive, got {self.cpu_seconds}")
        if self.memory_bytes <= 0:
            raise ValueError(f"memory_bytes must be positive, got {self.memory_bytes}")
        if self.disk_bytes <= 0:
            raise ValueError(f"disk_bytes must be positive, got {self.disk_bytes}")


DEFAULT_LIMITS = SandboxLimits()


class SandboxError(RuntimeError):
    """A candidate policy did not produce a decision, and why.

    Raised for every way that can happen — it raised, it was killed, it asked for
    something it may not have, it answered with something that is not a batch —
    because they are one outcome to the harness: this cell has no score. The
    message carries the captured output alongside the reason, since
    ``dream._replay`` records ``f"{type(exc).__name__}: {exc}"`` and that string
    is the feedback issue #14 learns from.
    """

    def __init__(self, reason: str, *, stdout: str = "", stderr: str = "") -> None:
        self.reason = reason
        self.stdout = stdout
        self.stderr = stderr
        parts = [reason]
        if stdout:
            parts.append(f"stdout: {stdout}")
        if stderr:
            parts.append(f"stderr: {stderr}")
        super().__init__(" | ".join(parts))


class SandboxedPolicy:
    """A candidate policy that decides in a child process (issue #13).

    Satisfies ``replay.ReplayPolicy``, so it is handed to
    ``ReplaySimulator.replay`` like any other policy, and it is what
    :func:`sandboxed_candidate` puts behind a
    :class:`~dream_rsi.dream.PolicyCandidate`'s factory: one instance per
    ``(version, world)`` cell, which is also one child process per cell, matching
    the per-rollout policy state §3 resets between pairs.

    The child is started, and the source compiled in it, by ``__init__`` — so
    source that cannot be used raises here, inside the factory call the harness
    already guards. A failure of any kind ends this instance: the child is killed
    and every later call raises, rather than silently continuing a rollout with a
    policy that has lost the state it built up (§3 resets state per policy-world
    pair, not per round).

    Closing it kills the child and removes the scratch directory. Callers that
    can should use it as a context manager; a :func:`weakref.finalize` handles
    the harness, which drops a policy as soon as its cell is scored.
    """

    def __init__(
        self,
        source: str,
        *,
        config: Mapping[str, Any] | None = None,
        limits: SandboxLimits = DEFAULT_LIMITS,
        scratch_root: str | Path | None = None,
        policy_name: str = DEFAULT_POLICY_NAME,
    ) -> None:
        self._limits = limits
        self._dead: str | None = None
        self._home = Path(
            tempfile.mkdtemp(
                prefix="dream-rsi-sandbox-",
                dir=None if scratch_root is None else str(scratch_root),
            )
        )
        self._scratch = self._home / "scratch"
        self._scratch.mkdir()
        self._stdout = self._home / "stdout"
        self._stderr = self._home / "stderr"
        # Set before anything that can fail, so ``close`` works on a half-built
        # policy: a constructor that raised still has a directory to remove, and
        # everything after this point has a child process to kill as well.
        self._finalizer: weakref.finalize | None = None
        try:
            self._process, fds = self._start()
            self._request_write, self._response_read = fds
            self._finalizer = weakref.finalize(self, _reap, self._process, fds, self._home)
            self._request(
                {
                    "op": "init",
                    "source": source,
                    "config": dict(config or {}),
                    "policy_name": policy_name,
                    "scratch": str(self._scratch),
                    "limits": {
                        "cpu_seconds": limits.cpu_seconds,
                        "memory_bytes": limits.memory_bytes,
                        "disk_bytes": limits.disk_bytes,
                    },
                }
            )
        except BaseException:
            # ``__exit__`` never runs for a constructor that raised, and by here
            # there is a directory, and usually a child process, to let go of.
            self.close()
            raise

    @property
    def scratch(self) -> Path:
        """The only directory this policy may write in."""
        return self._scratch

    def reset(self, rng: random.Random | None = None) -> None:
        """Forget one rollout's state, as replay does before the first decision (§3).

        The generator's state travels rather than the seed, which cannot be read
        back out of a :class:`random.Random` — see ``_sandbox_child._rng``.
        """
        self._request({"op": "reset", "state": None if rng is None else rng.getstate()})

    def select(
        self, tree: DiscoveryTree, eligible: Sequence[str], width: int
    ) -> tuple[str, ...]:
        """One decision round, taken in the child (``ReplayPolicy.select``).

        Raises :class:`SandboxError` if the candidate does not come back with a
        batch of node ids — whatever the reason. A batch naming a node the world
        does not hold is *not* that: it is a legal thing for a policy to attempt
        and the driver rejects it with its own message, so the ids pass through
        unchecked here.
        """
        response = self._request(
            {
                "op": "select",
                "tree": tree.to_dict(),
                "eligible": list(eligible),
                "width": int(width),
            }
        )
        batch = response.get("batch")
        if not isinstance(batch, list) or not all(isinstance(item, str) for item in batch):
            raise self._die(f"the policy answered with {_short(batch)} instead of a batch")
        return tuple(batch)

    def close(self) -> None:
        """Kill the child and remove its scratch directory. Idempotent.

        A closed policy is a dead one: the descriptors it decided over are gone,
        so a later call raises :class:`SandboxError` like any other failure
        rather than an ``OSError`` about a bad file descriptor.
        """
        if self._dead is None:
            self._dead = "the sandboxed policy has been closed"
        if self._finalizer is None:
            shutil.rmtree(self._home, ignore_errors=True)
            return
        self._finalizer()

    # ``Self`` would be the annotation, and it needs Python 3.11 while this
    # package supports 3.10 (``pyproject.toml``).
    def __enter__(self) -> SandboxedPolicy:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _start(self) -> tuple[subprocess.Popen[bytes], tuple[int, int]]:
        """Spawn the child with the protocol on two pipes and its output captured.

        The protocol does not travel on stdin/stdout: those are the candidate's,
        and a policy that prints — or writes to file descriptor 1 directly —
        must not be able to corrupt a response by doing so. Output goes to files
        rather than pipes because a pipe nobody is draining fills up and blocks
        the child, which would look exactly like the hang this module exists to
        catch.
        """
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        try:
            process = self._spawn(request_read, response_write)
        except BaseException:
            for fd in (request_read, request_write, response_read, response_write):
                _close(fd)
            raise
        _close(request_read)
        _close(response_write)
        return process, (request_write, response_read)

    def _spawn(self, request_read: int, response_write: int) -> subprocess.Popen[bytes]:
        """Start the interpreter that will hold the candidate."""
        with open(self._stdout, "wb") as out, open(self._stderr, "wb") as err:
            return subprocess.Popen(
                [
                    sys.executable,
                    "-s",  # no user site directory
                    "-B",  # no bytecode: a .pyc is a write outside the scratch dir
                    # Unbuffered, so what a candidate printed is already in the
                    # capture file when it is killed rather than sitting in a
                    # buffer that SIGKILL discards — that text is the feedback.
                    "-u",
                    "-c",
                    _BOOTSTRAP,
                    str(request_read),
                    str(response_write),
                ],
                cwd=str(self._scratch),
                env=self._env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                pass_fds=(request_read, response_write),
                # Its own process group, so one signal reaches anything it
                # managed to start as well as the child itself.
                start_new_session=True,
            )

    def _env(self) -> dict[str, str]:
        """A minimal environment: enough to import this package, and nothing else.

        ``PYTHONHASHSEED`` is fixed because the child is a fresh interpreter and
        a candidate that iterates a set of node ids would otherwise decide
        differently from one process to the next, which working rule 5 forbids of
        anything on the replay path.
        """
        return {
            "PATH": "",
            "HOME": str(self._scratch),
            "TMPDIR": str(self._scratch),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(entry for entry in sys.path if entry),
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and wait out its deadline for the answer."""
        if self._dead is not None:
            raise self._failure(self._dead)
        deadline = time.monotonic() + self._limits.wall_seconds
        self._send(json.dumps(payload) + "\n", deadline)
        response = self._receive(deadline)
        if response.get("ok") is not True:
            error = response.get("error")
            # Clipped: the candidate chooses this text, and a whole response's
            # worth of it would be carried in a report per failed cell.
            raise self._die(_clip(error) if isinstance(error, str) else _short(response))
        return response

    def _send(self, line: str, deadline: float) -> None:
        """Write a request, without letting a child that stopped reading block us."""
        data = line.encode("utf-8")
        while data:
            self._wait(self._request_write, deadline, writing=True)
            try:
                written = os.write(self._request_write, data)
            except OSError as exc:
                raise self._die(f"the sandboxed policy is gone ({exc.strerror or exc})") from None
            data = data[written:]

    def _receive(self, deadline: float) -> dict[str, Any]:
        """Read one response line, or die at the deadline.

        The first line only. A candidate that wrote extra lines down the response
        descriptor itself would have them dropped here, which costs it nothing it
        could not do anyway: the most a response can say is which nodes to open,
        and choosing those is the candidate's job.
        """
        buffer = bytearray()
        while b"\n" not in buffer:
            self._wait(self._response_read, deadline, writing=False)
            chunk = os.read(self._response_read, 65536)
            if not chunk:
                raise self._die(f"the sandboxed policy {self._status()}")
            buffer += chunk
            if len(buffer) > _RESPONSE_LIMIT:
                raise self._die(
                    f"the sandboxed policy answered with more than "
                    f"{_RESPONSE_LIMIT} bytes and no complete line"
                )
        line = bytes(buffer).split(b"\n", 1)[0]
        try:
            response = json.loads(line)
        except ValueError as exc:
            raise self._die(f"the sandboxed policy answered unreadably ({exc})") from None
        if not isinstance(response, dict):
            raise self._die(f"the sandboxed policy answered with {_short(response)}")
        return response

    def _wait(self, fd: int, deadline: float, *, writing: bool) -> None:
        """Block until ``fd`` is ready, or kill the child at the deadline.

        This is the hard wall-clock timeout, and the one limit that catches a
        candidate which is not computing at all — a sleep, or a read on something
        that never answers. The CPU and memory caps are the child's own
        (``_sandbox_child._install_limits``).
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._die(
                    f"the sandboxed policy did not answer within its "
                    f"{self._limits.wall_seconds}s wall-clock limit"
                )
            readable, writable, _ = select.select(
                [] if writing else [fd], [fd] if writing else [], [], remaining
            )
            if readable or writable:
                return

    def _status(self) -> str:
        """How the child ended, for the failure record.

        A negative return code is a signal — how the CPU cap and an out-of-memory
        kill arrive — and the number, not the elapsed time, because this text
        goes into a report that reads the same on every machine.
        """
        code = self._process.poll()
        if code is None:
            return "stopped answering"
        if code < 0:
            return f"was killed by signal {-code}"
        return f"exited with status {code}"

    def _die(self, reason: str) -> SandboxError:
        """Kill the child, remember why, and hand back the error to raise.

        Returned rather than raised so the call sites read ``raise self._die(…)``
        and keep their own tracebacks.
        """
        if self._dead is None:
            self._dead = reason
        failure = self._failure(reason)
        _kill(self._process)
        return failure

    def _failure(self, reason: str) -> SandboxError:
        """The error for ``reason``, carrying whatever the candidate printed."""
        return SandboxError(reason, stdout=_tail(self._stdout), stderr=_tail(self._stderr))


def sandboxed_candidate(
    name: str,
    source: str,
    *,
    config: Mapping[str, Any] | None = None,
    limits: SandboxLimits = DEFAULT_LIMITS,
    scratch_root: str | Path | None = None,
    policy_name: str = DEFAULT_POLICY_NAME,
) -> PolicyCandidate:
    """One dreaming-round candidate whose decisions are taken in a child process.

    This is how model-written source enters a dreaming round (issue #14: "every
    generated candidate goes through #13"), and the only way: the factory the
    harness calls per world starts a fresh :class:`SandboxedPolicy`, and there is
    no argument here — and no environment variable anywhere — that runs the
    source in the harness instead.
    """
    return PolicyCandidate(
        name=name,
        factory=lambda: SandboxedPolicy(
            source,
            config=config,
            limits=limits,
            scratch_root=scratch_root,
            policy_name=policy_name,
        ),
    )


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Stop the child and everything it started, and reap it."""
    if process.poll() is None:
        try:
            # The process group, not the process: it was started with
            # ``start_new_session``, so its pid is its group id.
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            # The group is already gone, or was never ours to signal.
            process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not refusable
        pass


def _reap(
    process: subprocess.Popen[bytes], fds: tuple[int, ...], home: Path
) -> None:
    """Kill the child, close the pipes and remove the scratch directory.

    A module-level function taking no policy, so :func:`weakref.finalize` can
    hold it without keeping the policy alive.
    """
    _kill(process)
    for fd in fds:
        _close(fd)
    shutil.rmtree(home, ignore_errors=True)


def _close(fd: int) -> None:
    """Close a descriptor that may already be closed."""
    try:
        os.close(fd)
    except OSError:
        pass


def _tail(path: Path) -> str:
    """The last of a capture file, decoded permissively; ``""`` if there is none."""
    try:
        captured = path.read_bytes()
    except OSError:
        return ""
    text = captured.decode("utf-8", errors="replace").strip()
    if len(text) <= _CAPTURE_LIMIT:
        return text
    return "…" + text[-_CAPTURE_LIMIT:]


def _clip(text: str) -> str:
    """Text a candidate wrote, at a length a failure record can carry.

    The head, unlike :func:`_tail`: an exception's type and message come first.
    """
    return text if len(text) <= _CAPTURE_LIMIT else text[:_CAPTURE_LIMIT] + "…"


def _short(value: Any) -> str:
    """A value as a failure record can quote it: a candidate chooses its length."""
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "…"
