"""A task: the roles a run needs, and the file a user writes to supply them (issue #68).

The loop is general — :func:`dream_rsi.run.run_cycles` takes a discovery agent, an
evaluator, a policy-development agent and a problem — but an agent driving it from
a harness has a shell, not a Python session, so the only task it could run was the
toy one wired into ``run.main``. ``python -m dream_rsi.run --task PATH`` closes
that: ``PATH`` is a Python file defining ``task()``, which returns a :class:`Task`.

**Why a Python file.** Each of the three roles is a protocol, and a user may need
their own class for any of them, so a config format would have to grow a way to
name classes — which is a Python file with extra steps. It also adds no
dependency, where TOML would need ``tomllib``, which Python 3.10 lacks.

**The task file is trusted, and runs in this process.** It is the user's own code:
they wrote it, they pointed the command at it, and it constructs the agent and the
evaluator that will run here. That is the opposite of *policy* code, which is
model-written and only ever runs in the sandbox (:mod:`dream_rsi.sandbox`). The
two are different on purpose, and nothing should "unify" them by sandboxing this
file (which cannot then build a discovery agent) or by trusting a policy.

**What identifies a task on resume.** The run manifest records the ``problem``
text and the starting ``policy`` and refuses to resume a directory under
different ones (issue #45). So a task's ``problem`` should say what is being
optimised *and which way the score runs*: changing the scorer without changing
the problem text is a change the manifest cannot see.
"""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from dream_rsi.adapters.agent import CodingAgent
from dream_rsi.adapters.evaluator import TaskEvaluator
from dream_rsi.develop import PolicyDeveloper

__all__ = ["DEFAULT_POLICY_SOURCE", "Task", "TaskError", "load_task"]

# π_1 for a run that is not handed one: the §B.2 shape, "keep NAME =
# "OptimalPolicy" and implement class OptimalPolicy(...)", over a baseline the
# package already ships. A real run supplies its own.
#
# Breadth-first deliberately: it is §4's Recursive Fixed Exploration, the
# paper's own controlled baseline — "10 parallel workspaces with up to 11
# refinement steps", a grid opened whatever it finds — so a run that starts
# there and improves on it is the comparison the paper reports. It is also the
# baseline with the most room above it, since Equation 1 charges it for every
# node of that grid. Starting from a policy nothing on offer can beat is a loop
# that runs correctly and demonstrates nothing (issue #20).
DEFAULT_POLICY_SOURCE = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    pass
"""


class TaskError(ValueError):
    """A task file that cannot supply a :class:`Task`, and why."""


@dataclass(frozen=True)
class Task:
    """Everything a run needs to know about the problem it is searching.

    ``agent`` writes candidates, ``evaluator`` scores them and ``developer``
    rewrites the exploration policy from replay feedback; ``problem`` is the text
    the discovery agent is given, and ``policy`` is ``π_1``, the module source the
    first cycle deploys.

    The roles are checked when the task is built, not when the first cycle reaches
    them: a run that would fail on its first agent call should fail before it has
    created a run directory, not after.
    """

    agent: CodingAgent
    evaluator: TaskEvaluator
    developer: PolicyDeveloper
    problem: str
    policy: str = DEFAULT_POLICY_SOURCE

    def __post_init__(self) -> None:
        for role, protocol, method in (
            ("agent", CodingAgent, "propose"),
            ("evaluator", TaskEvaluator, "evaluate"),
            ("developer", PolicyDeveloper, "revise"),
        ):
            if not isinstance(getattr(self, role), protocol):
                raise TypeError(
                    f"task {role} must implement {protocol.__name__} "
                    f"(a {method}() method), got {type(getattr(self, role)).__name__}"
                )
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise TypeError("task problem must be a non-empty string")
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise TypeError("task policy must be a non-empty module source")


def load_task(path: str | Path) -> Task:
    """Import the task file at ``path``, call its ``task()``, and return what it built.

    Every way this can fail is a :class:`TaskError` that names the file and what
    was wrong, so a caller can print one line: the file is missing, it does not
    import, it defines no ``task()``, ``task()`` raises, or it returns something
    that is not a :class:`Task`. The file's own exception is chained, not shown, so
    the message stays one line while ``__cause__`` still carries the traceback.
    """
    path = Path(path)
    module = _import(path)
    factory = getattr(module, "task", None)
    if not callable(factory):
        raise TaskError(
            f"{path} defines no task(): it must define task() returning a dream_rsi.task.Task"
        )
    try:
        built = factory()
    except Exception as exc:
        raise TaskError(f"{path}: task() raised {_describe(exc)}") from exc
    if not isinstance(built, Task):
        raise TaskError(
            f"{path}: task() returned {type(built).__name__}, not a dream_rsi.task.Task"
        )
    return built


def _import(path: Path) -> object:
    """Execute the file as a module and return it, or say why it could not be."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskError(f"{path}: cannot read the task file: {exc.strerror or exc}") from exc

    # A name of its own per file, and registered while it runs: a dataclass in a
    # module written with ``from __future__ import annotations`` looks its own
    # module up in ``sys.modules`` and fails if it is not there.
    digest = hashlib.blake2b(str(path.resolve()).encode("utf-8"), digest_size=6).hexdigest()
    name = f"dream_rsi_task_{digest}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        # Trusted: the user's own file, run here on purpose (see the module docstring).
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    except Exception as exc:
        sys.modules.pop(name, None)
        raise TaskError(f"{path}: importing it raised {_describe(exc)}") from exc
    return module


def _describe(exc: BaseException) -> str:
    """``Type: message`` on one line, with where a syntax error is."""
    if isinstance(exc, SyntaxError):
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    return f"{type(exc).__name__}: {exc}"
