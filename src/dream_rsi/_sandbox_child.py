"""The inside of the sandbox: one child process holding one candidate policy.

Nothing imports this to call it — :mod:`dream_rsi.sandbox` starts it as a
subprocess and talks to it over two pipes (see that module for the protocol and
for why the boundary is a process at all). It lives in its own module rather
than inside a ``-c`` string so that the code deciding what model-written Python
may do is code someone can read, lint and diff.

Three layers keep a candidate inside its box, applied in this order before a
single line of its source is compiled:

1. **Resource limits** (:mod:`resource`). CPU time is cumulative over the whole
   life of the process, so a policy cannot spread a spin across many decision
   rounds; the address-space cap turns a memory bomb into a ``MemoryError`` in
   the child instead of an OOM kill somewhere else on the machine; and the file
   size cap is what stands between a candidate and the host's disk, since a
   scratch directory it may write in is still a directory on a real filesystem.
2. **An audit hook** (:func:`sys.addaudithook`), which refuses the network,
   starting or signalling processes, importing the modules that would step
   around this hook, writing anywhere but the scratch directory, and reading
   anywhere but the scratch directory, the ``dream_rsi`` package and the Python
   installation the candidate is imported from. An audit hook cannot be removed
   once installed, and it is installed before the candidate exists.
3. **The process boundary itself**, which is what actually contains a hang: the
   parent kills this process group when a request outlives its deadline.

Layer 2 is a rule about *opening*, not about using: a path may be opened for
writing only inside the scratch directory, and for reading only inside the roots
of :func:`_readable_roots`. Reads have to stay open far enough for a candidate to
import its own base class and narrow enough that the recorded world is not
readable — the fixtures sit on disk at a fixed path, and a real simulator pool's
corpus will too, so a candidate that opened a ``tree.json`` could read the
``s_v`` of nodes it never revealed and defeat the prefix-observability §3/§B.2
are built on ("Never use unrevealed scores, a true optimum, hardcoded winning
cell ids"). Issue #39 is that leak, closed here; the false-refusal risk the issue
weighs (a legal candidate recorded as scoring nothing) is what
:func:`_readable_roots` is sized against.

Layer 2 is Python-level and therefore the weakest of the three — code reaching
the C level around it (``ctypes``, which is why it is refused) is not stopped by
a hook. The limits and the boundary do not depend on the candidate's
cooperation, so a candidate that escapes the hook still cannot outlive its CPU
cap, its memory cap or its deadline. A deployment handing this genuinely hostile
code — rather than a model's honest attempt at a search policy — wants an OS
sandbox underneath as well; CLAUDE.md's rule is that none of this is relaxed to
make a test pass.

Layers 1 and 3 are POSIX: process-wide resource limits and killing a process
group, which is what the project's CI runs on.
"""

from __future__ import annotations

import json
import os
import random
import resource
import sys
from typing import Any

from dream_rsi.policy import GridPlan, GridPlanningContext
from dream_rsi.tree import DiscoveryTree

# §B.2: "Keep ``NAME = "OptimalPolicy"`` and implement ``class
# OptimalPolicy(LLMDesignedMethod)``". The module may override it by defining
# ``NAME``, which is the paper's own hook for renaming the class.
DEFAULT_POLICY_NAME = "OptimalPolicy"

# Refused outright: these reach past a Python-level audit hook, so a candidate
# holding one is not bounded by anything above.
_BLOCKED_IMPORTS = frozenset({"ctypes", "_ctypes"})

# Starting a process would put work outside this process group's limits, and
# signalling one is how a child reaches its own parent.
_PROCESS_EVENTS = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.kill",
        "os.killpg",
        "os.posix_spawn",
        "os.spawn",
        "os.startfile",
        "os.system",
        "pty.spawn",
        "subprocess.Popen",
    }
)

# Mutating operations, and which of the event's arguments name a path that has
# to be inside the scratch directory. ``os.chdir`` is here because every
# relative path below is resolved against the working directory.
_PATH_EVENTS: dict[str, tuple[int, ...]] = {
    "os.chdir": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.link": (0, 1),
    "os.mkdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.rmdir": (0,),
    "os.symlink": (0, 1),
    "os.truncate": (0,),
    "os.utime": (0,),
}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class SandboxViolation(Exception):
    """A candidate asked for something the sandbox does not grant.

    Deliberately not an :class:`OSError`: the standard library wraps those into
    its own errors (``urllib`` turns one into a ``URLError``), and the refusal
    has to reach the failure record intact, because that text is what the
    development agent revises the next version from (issue #14).
    """


class _Guard:
    """The audit hook: one call per audited event, for the life of the process."""

    def __init__(self, scratch: str) -> None:
        self._scratch = os.path.realpath(scratch)
        self._roots = _readable_roots(self._scratch)

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            # Every socket operation, not a list of them: a policy decides from
            # the revealed prefix (§B.2) and has nothing to say to the network.
            raise SandboxViolation(f"policy code may not reach the network ({event})")
        if event in _PROCESS_EVENTS:
            raise SandboxViolation(f"policy code may not start or signal a process ({event})")
        if event == "import" and args and args[0] in _BLOCKED_IMPORTS:
            raise SandboxViolation(
                f"policy code may not import {args[0]!r}: it reaches around every limit here"
            )
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            # One event covers both directions (``_is_write``), and the two rules
            # answer different questions: where may a candidate *put* a file, and
            # where may it *take* one from.
            if _is_write(mode, flags):
                self._inside(path)
            else:
                self._read(path)
            return
        for index in _PATH_EVENTS.get(event, ()):
            if index < len(args):
                self._inside(args[index])

    def _read(self, path: Any) -> None:
        """Refuse a read of anything a candidate does not need to decide."""
        resolved = self._resolve(path)
        if resolved is None or any(_under(resolved, root) for root in self._roots):
            return
        raise SandboxViolation(f"policy code may not read outside its sandbox: {resolved}")

    def _inside(self, path: Any) -> None:
        """Refuse a write outside the scratch directory."""
        resolved = self._resolve(path)
        if resolved is None or _under(resolved, self._scratch):
            return
        raise SandboxViolation(
            f"policy code may not write outside its scratch directory: {resolved}"
        )

    def _resolve(self, path: Any) -> str | None:
        """The named path, resolved; ``None`` for a descriptor that is already open.

        An integer names an open descriptor (``os.fdopen``), and whatever opened
        it was audited then: both rules are about opening, not about using what
        is already open.
        """
        if isinstance(path, int):
            return None
        try:
            return os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise SandboxViolation(f"policy code named a path the sandbox cannot check: {exc}")


def _under(path: str, root: str) -> bool:
    """Whether a resolved ``path`` is ``root`` itself or inside it."""
    return path == root or path.startswith(root + os.sep)


def _readable_roots(scratch: str) -> tuple[str, ...]:
    """Where a candidate may open a file to read: three kinds of place, no more.

    A legal candidate imports ``dream_rsi.policy`` and whatever standard library
    it needs (§B.2's skeleton is a subclass of the package's base class), and the
    import machinery reads those as files: the package's own directory, the
    standard library and site directory under the Python installation, and the
    scratch directory a policy may keep notes in.

    Every prefix spelling the interpreter offers is allowed — ``prefix``,
    ``exec_prefix`` and the ``base_`` pair — because a virtual environment serves
    the standard library out of its base installation and its own site-packages
    out of ``prefix``, so allowing only one of them refuses a legal import on
    some installs and not others. The cost weighed the other way (issue #39) is
    a read rule too tight to import: a wrongly-failed candidate is recorded as
    scoring nothing, which is worse than the directory names a wider root lets
    it see.

    PAPER-GAP: the paper is explicit that a policy decides from the revealed
    prefix (§3, §B.2) but says nothing about how the unrevealed part of the tree
    is kept out of a policy's reach — its implementation is unreleased. These
    roots are our choice: the smallest set that keeps every import a policy is
    told to write working. Revisit if the authors' implementation lands (see
    references/method.md).
    """
    roots = {scratch, os.path.dirname(os.path.realpath(__file__))}
    for name in ("prefix", "exec_prefix", "base_prefix", "base_exec_prefix"):
        value = getattr(sys, name, None)
        if value:
            roots.add(os.path.realpath(value))
    return tuple(sorted(roots))


def _is_write(mode: Any, flags: Any) -> bool:
    """Whether an ``open`` event is opening something for writing.

    The event carries the ``mode`` string when it came from :func:`open` and the
    ``flags`` integer when it came from :func:`os.open`, and ``None`` for
    whichever did not apply, so both are checked.
    """
    if isinstance(mode, str) and any(character in mode for character in "wxa+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _install_limits(limits: dict[str, Any], scratch: str) -> None:
    """Apply every layer, before any candidate code exists to see it happen."""
    cpu = int(limits["cpu_seconds"])
    memory = int(limits["memory_bytes"])
    disk = int(limits["disk_bytes"])
    # Soft below hard: the soft limit arrives as SIGXCPU, which a candidate
    # could install a handler for, and the hard limit is then an uncatchable
    # SIGKILL one second later.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    # Bytes on disk, which neither cap above bounds. There are two ways a
    # candidate reaches the host's filesystem: a file it writes in the scratch
    # directory, and the capture files its own stdout and stderr are redirected
    # to, which it can write to for as long as it is alive.
    resource.setrlimit(resource.RLIMIT_FSIZE, (disk, disk))
    # A killed candidate should not leave a core file behind in the scratch dir.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.addaudithook(_Guard(scratch))


def _load_policy(source: str, config: dict[str, Any], policy_name: str) -> Any:
    """Compile a candidate's source and instantiate the policy it defines.

    Everything here is a normal outcome rather than an error in the harness: a
    model writes source that does not compile, or that defines no policy, often
    enough that the message is a product — issue #14 feeds it back.
    """
    namespace: dict[str, Any] = {"__name__": "dream_rsi_policy_candidate"}
    # The point of this module: this is the untrusted code, and it runs with the
    # limits of :func:`_install_limits` already in force. Executing a model's
    # policy is the design (§3, §B.2) — the sandbox is the answer to it, and this
    # is the one place in the package that does it.
    exec(compile(source, "<policy>", "exec"), namespace)  # noqa: S102

    name = namespace.get("NAME", policy_name)
    if not isinstance(name, str):
        raise SandboxViolation(f"NAME must be the name of the policy class, got {name!r}")
    if name not in namespace:
        raise SandboxViolation(f"the policy source defines no {name}")
    defined = namespace[name]
    if not callable(defined):
        raise SandboxViolation(
            f"{name} is not a policy class: it must be callable and its instances "
            "must define select(tree, eligible, width)"
        )
    # §B.2 has a policy read its configuration off ``self.config``, which the
    # ``OptimalPolicy`` base class fills from this one positional argument.
    policy = defined(config)
    if not callable(getattr(policy, "select", None)):
        raise SandboxViolation(f"{name} does not implement select(tree, eligible, width)")
    return policy


class _Session:
    """The candidate, once it has been loaded, across one rollout's decisions."""

    def __init__(self) -> None:
        self._policy: Any = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "init":
            _install_limits(request["limits"], request["scratch"])
            self._policy = _load_policy(
                request["source"],
                dict(request.get("config") or {}),
                request.get("policy_name") or DEFAULT_POLICY_NAME,
            )
            # Whether this candidate wrote §B.2's optional grid plan (issue #21).
            # Reported once, here, because it is a fact about the source and the
            # parent has to be able to offer the hook only where there is one.
            return {"ok": True, "plans": callable(getattr(self._policy, "plan_grid", None))}
        if self._policy is None:
            raise SandboxViolation(f"the sandbox was asked to {operation!r} before it was loaded")
        if operation == "reset":
            reset = getattr(self._policy, "reset", None)
            if callable(reset):
                reset(_rng(request.get("state")))
            return {"ok": True}
        if operation == "select":
            batch = self._policy.select(
                DiscoveryTree.from_dict(request["tree"]),
                tuple(request["eligible"]),
                int(request["width"]),
            )
            return {"ok": True, "batch": [_node_id(node_id) for node_id in batch]}
        if operation == "plan_grid":
            plan = self._policy.plan_grid(GridPlanningContext(**request["context"]))
            return {"ok": True, "plan": _plan(plan)}
        raise SandboxViolation(f"unknown sandbox operation {operation!r}")


def _rng(state: Any) -> random.Random | None:
    """Rebuild the generator replay seeded, from the state the parent sent.

    The state and not the seed, because ``ReplaySimulator.replay`` hands the
    policy a ``random.Random`` it has already constructed and a seed cannot be
    read back out of one. Transferring the state keeps a sampling candidate
    reproducible (working rule 5), which is the whole reason replay passes a
    generator rather than letting a policy make its own.
    """
    if state is None:
        return None
    version, keys, gauss_next = state
    rng = random.Random()
    rng.setstate((int(version), tuple(int(key) for key in keys), gauss_next))
    return rng


def _node_id(node_id: Any) -> str:
    """A batch entry, as the protocol can carry it."""
    if not isinstance(node_id, str):
        raise SandboxViolation(
            f"a batch holds node ids, and this policy selected {node_id!r}"
        )
    return node_id


def _plan(plan: Any) -> dict[str, Any]:
    """A grid plan, as the protocol can carry it (issue #21).

    Checked here, where the object the candidate built still exists: the pipe
    carries JSON, so an answer that is not a plan of three serialisable fields
    would otherwise fail while the response was being written — outside the
    handler's own error path, which would take the child down instead of
    recording a candidate that answered wrongly.
    """
    if not isinstance(plan, GridPlan):
        raise SandboxViolation(
            f"plan_grid must return a GridPlan from dream_rsi.policy, got {plan!r}"
        )
    for name in ("branch_count", "refine_count"):
        value = getattr(plan, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SandboxViolation(f"a GridPlan's {name} must be a whole number, got {value!r}")
    if not isinstance(plan.reason, str):
        raise SandboxViolation(f"a GridPlan's reason must be text, got {plan.reason!r}")
    return {
        "branch_count": plan.branch_count,
        "refine_count": plan.refine_count,
        "reason": plan.reason,
    }


def main() -> int:
    """Answer requests until the parent closes the request pipe."""
    requests = os.fdopen(int(sys.argv[1]), "r", encoding="utf-8")
    responses = os.fdopen(int(sys.argv[2]), "w", encoding="utf-8")
    session = _Session()
    for line in requests:
        if not line.strip():
            continue
        try:
            response = session.handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - a candidate's failure is a result
            # Type and message, and no traceback: a traceback carries the paths
            # this process was started under, and the failure text ends up in a
            # dreaming report that has to read the same on every machine.
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        responses.write(json.dumps(response) + "\n")
        responses.flush()
    return 0
