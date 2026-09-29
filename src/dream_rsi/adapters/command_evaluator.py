"""A task evaluator that scores by running the user's own command (issue #69).

Anyone with a task worth searching already has a scorer: a test script, a
benchmark, ``make bench``. The evaluator protocol asks for a Python class, so
:class:`CommandEvaluator` reduces "wire up my scorer" to one line in a task file
(:mod:`dream_rsi.task`).

**The contract a command meets.** It is run with the candidate's workspace as its
working directory, as an argv list — there is no shell, so nothing in the
candidate or the command is interpreted as one — after the candidate has been
written into the workspace under ``artifact_name``. It then reports one of:

* **A score.** It writes ``eval/score.json`` holding ``{"score": <number>,
  "correct": <bool>}`` and exits 0. It may add ``"fail_class"`` (a string, for a
  task-defined kind of failure) and ``"diagnostics"`` (a string the discovery agent
  reads when it resumes from this node). The score is in the task's own units and
  direction, and is mapped onto ``s_v`` by the evaluator's ``direction`` like any
  other (:meth:`~dream_rsi.adapters.evaluator.EvalResult.to_node_fields`).
* **A failure.** It exits non-zero, or writes ``eval/error.txt``. Either becomes
  :meth:`~dream_rsi.adapters.evaluator.EvalResult.failed`, carrying the error
  text, and either wins over a score file the command also wrote: a scorer that
  crashed after writing a score has not reported one. An exit of 0 that writes
  neither file is a failure too, and its diagnostics say so, rather than a score
  of nothing.

Whatever the command printed — the tail of its stdout and stderr, together — is
appended to the diagnostics either way, since that is what the agent reads next.
A command that outlives ``timeout`` is killed with everything it started and is a
failure whose ``fail_class`` is :data:`TIMEOUT`; it does not raise. A score file
that is not JSON, or is missing a field, or holds a non-finite score, is a failed
result and not a crash, which is the guarantee issue #34 put on the tree.

Results the command left behind on a previous attempt are removed first. A
workspace is resumed from its parent's snapshot, so ``eval/score.json`` from the
parent is *there*, and a scorer that wrote nothing this time would otherwise be
read as having scored what its parent did.

PAPER-GAP: §B.1's exploration prompt has the agent read each attempt's
``eval/score.json``, and ``error.txt`` if it failed, but the paper never shows
either file's schema. The keys above are ours. Revisit if the authors'
implementation lands (see references/method.md).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

from dream_rsi.adapters._command import (
    DEFAULT_MAX_OUTPUT_BYTES,
    CommandOutcome,
    is_inside,
    run_command,
    tail_of_file,
)
from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection

__all__ = [
    "ERROR_PATH",
    "EVAL_ERROR",
    "MALFORMED_SCORE",
    "NO_SCORE",
    "OUTPUT_LIMIT",
    "SCORE_PATH",
    "TIMEOUT",
    "CommandEvaluator",
]

# Where a command reports, relative to the workspace. Shared with whatever lays a
# recorded attempt out in the same shape for the discovery agent to read
# (:mod:`dream_rsi.adapters.command_agent`), so both sides agree on the format.
SCORE_PATH = Path("eval") / "score.json"
ERROR_PATH = Path("eval") / "error.txt"

# ``fail_class`` values. The paper names only "ok" (§B.1), and the discovery agent
# reads the rest as text, so these say what happened in a word.
TIMEOUT = "timeout"
EVAL_ERROR = "eval_error"
NO_SCORE = "no_score"
MALFORMED_SCORE = "malformed_score"
OUTPUT_LIMIT = "output_limit"

# PAPER-GAP: the paper says nothing about how long an evaluation may run. Its
# evaluations are compilations, benchmarks and long runs (§1), so a limit tight
# enough to catch a hang has to be generous; ten minutes is a bound on a runaway
# rather than a budget, and a task whose evaluations are longer sets its own.
# Revisit if the authors' implementation lands (see references/method.md).
_DEFAULT_TIMEOUT = 600.0


@dataclass(frozen=True)
class CommandEvaluator:
    """Scores a candidate by running ``command`` in its workspace.

    ``command`` is an argv list, never a string, because there is no shell to
    split it. ``direction`` and ``baseline_score`` are the task's, stated the way
    :class:`~dream_rsi.adapters.evaluator.TaskEvaluator` states them; the command
    reports its score in those units. ``artifact_name`` is the file the candidate
    is written to, relative to the workspace, and it is the name the discovery
    agent writes its program under too.

    The command is run in the workspace, as an argv list with no shell, and
    reports through files there:

    * ``eval/score.json`` holding ``{"score": number, "correct": bool}``, and
      optionally ``"fail_class"`` and ``"diagnostics"`` (strings), with exit 0:
      a scored result.
    * a non-zero exit, or ``eval/error.txt``: a failed result carrying the error
      text. Either wins over a score file. Exit 0 with neither file is a failure.
    * the tail of its stdout and stderr is appended to the diagnostics either way.
    * outliving ``timeout`` is a failure with ``fail_class`` :data:`TIMEOUT`, and
      printing more than ``max_output_bytes`` is one with :data:`OUTPUT_LIMIT`.

    Nothing the command does raises: a bad score file is a failed result too.
    The module docstring has the reasoning.
    """

    command: Sequence[str]
    direction: ScoreDirection
    baseline_score: float | None = None
    timeout: float = _DEFAULT_TIMEOUT
    artifact_name: str = "solution.py"
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if isinstance(self.command, str) or not self.command:
            raise TypeError(
                "command must be a non-empty argv list such as ['python', 'score.py'], "
                f"not {self.command!r}: there is no shell to split a string"
            )
        if not all(isinstance(argument, str) for argument in self.command):
            raise TypeError(f"command must be a list of strings, got {self.command!r}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_output_bytes <= 0:
            raise ValueError(f"max_output_bytes must be positive, got {self.max_output_bytes}")
        name = PurePath(self.artifact_name)
        if not self.artifact_name or name.is_absolute() or ".." in name.parts:
            raise ValueError(
                f"artifact_name must be a path inside the workspace, got {self.artifact_name!r}"
            )

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        """Write ``artifact`` into ``workspace``, run the command there, read what it left."""
        workspace = Path(workspace)
        target = workspace / self.artifact_name
        reports = workspace / SCORE_PATH.parent
        # Everything this writes or removes is checked first: a symlink an earlier
        # attempt left would make it write through to somewhere else, with the
        # harness's permissions rather than the agent's (see ``is_inside``).
        for path in (target, reports):
            if not is_inside(workspace, path):
                return EvalResult.failed(
                    f"{path.relative_to(workspace)} resolves outside the workspace, so the "
                    "harness will not write through it",
                    fail_class=EVAL_ERROR,
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact, encoding="utf-8")
        # Stale results out first (see the module docstring), and the directory the
        # command reports into made, so a command need not create it.
        reports.mkdir(parents=True, exist_ok=True)
        for stale in (workspace / SCORE_PATH, workspace / ERROR_PATH):
            stale.unlink(missing_ok=True)

        try:
            outcome = run_command(
                self.command,
                cwd=workspace,
                timeout=self.timeout,
                max_output_bytes=self.max_output_bytes,
            )
        except OSError as exc:
            # The command could not be started at all — not on the path, not
            # executable — which is the task's wiring and not the candidate.
            return EvalResult.failed(
                f"could not run {self.command[0]!r}: {exc.strerror or exc}", fail_class=EVAL_ERROR
            )
        if outcome.exceeded:
            reason = f"printed more than {self.max_output_bytes} bytes and was stopped"
            return EvalResult.failed(
                reason,
                fail_class=OUTPUT_LIMIT,
                diagnostics=_with_output(reason, outcome.output),
            )
        if outcome.status is None:
            timed_out = f"timed out after {self.timeout:g}s"
            return EvalResult.failed(
                timed_out,
                fail_class=TIMEOUT,
                diagnostics=_with_output(timed_out, outcome.output),
            )
        return self._read(workspace, outcome)

    def _read(self, workspace: Path, outcome: CommandOutcome) -> EvalResult:
        """What the command reported: an error, a score, or nothing at all."""
        output = outcome.output
        error_file = workspace / ERROR_PATH
        if error_file.is_file():
            error = tail_of_file(error_file)
            return EvalResult.failed(
                error or "the command wrote an empty eval/error.txt",
                fail_class=EVAL_ERROR,
                diagnostics=_with_output(error, output),
            )
        if outcome.status != 0:
            error = f"the command exited with status {outcome.status}"
            return EvalResult.failed(
                error, fail_class=EVAL_ERROR, diagnostics=_with_output(error, output)
            )
        score_file = workspace / SCORE_PATH
        if not score_file.is_file():
            error = "the command exited 0 but wrote no eval/score.json"
            return EvalResult.failed(
                error, fail_class=NO_SCORE, diagnostics=_with_output(error, output)
            )
        return _parse_score(score_file, output)


def _parse_score(path: Path, output: str) -> EvalResult:
    """``eval/score.json`` as an :class:`EvalResult`, or a failed one saying what was wrong."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _malformed(f"eval/score.json is not readable JSON: {exc}", output)
    if not isinstance(payload, dict):
        return _malformed("eval/score.json must hold a JSON object", output)

    score = payload.get("score")
    # ``bool`` is an ``int``, and a scorer that wrote ``true`` for its score has not
    # measured anything.
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return _malformed(f'eval/score.json needs a finite number "score", got {score!r}', output)
    try:
        number = float(score)
    except OverflowError:  # a JSON integer has no size limit and a float does
        number = math.inf
    if not math.isfinite(number):
        return _malformed(
            f'eval/score.json needs a finite number "score", got {str(score)[:40]}', output
        )
    correct = payload.get("correct")
    if not isinstance(correct, bool):
        return _malformed(f'eval/score.json needs a boolean "correct", got {correct!r}', output)
    fail_class = payload.get("fail_class", "ok")
    if not isinstance(fail_class, str):
        return _malformed(f'"fail_class" must be a string, got {fail_class!r}', output)
    diagnostics = payload.get("diagnostics", "")
    if not isinstance(diagnostics, str):
        return _malformed(f'"diagnostics" must be a string, got {diagnostics!r}', output)

    return EvalResult(
        score=number,
        correct=correct,
        diagnostics=_with_output(diagnostics, output),
        fail_class=fail_class,
    )


def _malformed(reason: str, output: str) -> EvalResult:
    return EvalResult.failed(
        reason, fail_class=MALFORMED_SCORE, diagnostics=_with_output(reason, output)
    )


def _with_output(text: str, output: str) -> str:
    """``text``, then the command's output under a heading, where it printed any."""
    if not output:
        return text
    return f"{text}\n--- command output (tail) ---\n{output}" if text else output
