"""A discovery agent that drives a coding-agent CLI (issue #70).

Every discovery-agent call the loop makes otherwise goes to a scripted stand-in,
so the run report's "online agent calls" count calls to a script. The paper's
discovery agent was Gemini 3.1 Pro and 3.7 Flash driven "via the Gemini CLI" (§4),
and the harnesses this skill is installed into are exactly such CLIs, each with a
non-interactive mode. :class:`CommandAgent` shells out to one, so a single adapter
covers all of them without the code naming a provider. Examples of ``command`` —
they are examples, not code paths, and nothing here knows what any of them is:

* Gemini CLI: ``["gemini", "--yolo", "-p"]``
* Claude Code: ``["claude", "-p", "--permission-mode", "acceptEdits"]``
* OpenCode: ``["opencode", "run"]``

With ``prompt_via="argument"`` (the default) the prompt is appended as the final
argument, which is what all three take; ``prompt_via="stdin"`` feeds it on stdin
instead. AGENTS.md rule 6 applies: the adapter wraps the CLI and adds nothing to
its command line beyond that prompt. Which flags let a CLI edit files without
asking are the user's ``command`` to say — the loop cannot answer a permission
prompt, and a CLI that stops to ask fails the attempt with what it printed.

**What the command is given.** For one attempt, the command runs with the
attempt's workspace as its working directory, and the prompt
(:file:`prompts/exploration.md` unless ``template`` says otherwise) names:

* ``$node_dir`` — that workspace. The paper: "your own attempt directory".
* ``$problem_file`` — the problem text.
* ``$history_dir`` — one ``attempt_*/`` directory per attempt this one inherits,
  each holding that attempt's ``proposal.md``, its program under ``$eval_program``,
  and the ``eval/score.json`` or ``eval/error.txt`` its evaluation recorded, in the
  format :class:`~dream_rsi.adapters.command_evaluator.CommandEvaluator` reads —
  so what the agent reads and what the scorer writes agree.
* ``$baseline_dir``, ``$eval_program``, ``$direction_guidance`` and ``$seed``.

All of it lives under the workspace, in ``.dream_rsi/``, because a coding-agent CLI
reads freely inside its working directory and asks about anything outside it — and
the loop cannot answer. The history and problem file are removed when the call
ends, so a snapshot of the workspace does not carry copies of them down the tree.

**What comes back.** The agent writes ``proposal.md`` and ``$eval_program`` in the
workspace, and :meth:`CommandAgent.propose` returns them as an
:class:`~dream_rsi.adapters.agent.Artifact`. A command that exits non-zero, times
out, or leaves no new program raises :class:`CommandAgentError` carrying the tail
of what it printed, and the orchestrator records that attempt as a failed node
(``orchestrator._attempt``) — there is one path for a failure, and no empty
artifact ever comes out of it.

PAPER-GAP: §B.1's prompt names ``$direction_guidance`` and ``$baseline_dir`` and
never says what either holds. ``direction_guidance`` is a string the user supplies
(empty by default), and ``baseline_dir`` is a directory the user supplies, or an
empty one — the paper's baseline is a reference solution the task ships with, and
what a task ships is the task's to say. The prompt also never says where the
problem statement is, so the default template opens with one sentence of ours
pointing at ``$problem_file``. Revisit if the authors' implementation lands (see
references/method.md).

PAPER-GAP: the paper's nodes record "the generated artifact", and the tree here
stores only the program. The proposal is kept where the next attempt can read it
without a schema change: in the workspace, which is what a child resumes from, keyed
by how deep the attempt sits on the chain it belongs to.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from string import Template

from dream_rsi.adapters._command import CommandOutcome, run_command, tail_of
from dream_rsi.adapters.agent import AgentContext, Artifact
from dream_rsi.adapters.command_evaluator import ERROR_PATH, SCORE_PATH
from dream_rsi.tree import Node

__all__ = [
    "PROMPT_PATH",
    "PROPOSAL_FILENAME",
    "SEED_ENV",
    "WORK_DIRNAME",
    "CommandAgent",
    "CommandAgentError",
]

# §B.1's exploration prompt, one file so it can be diffed against the paper's.
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "exploration.md"

# What the agent writes, and where this adapter keeps its own files in a workspace.
PROPOSAL_FILENAME = "proposal.md"
WORK_DIRNAME = ".dream_rsi"
_HISTORY = "history"
_BASELINE = "baseline"
_PROBLEM = "problem.md"
_PROPOSALS = "proposals"

# The attempt's seed for CLIs that read one from the environment. Ones that ignore
# it simply ignore it, and ``AgentContext.seed`` says why it is there at all.
SEED_ENV = "DREAM_RSI_SEED"

# PAPER-GAP: the paper says nothing about how long one discovery-agent call may
# take. A coding-agent CLI working on a real task runs for minutes, so a limit that
# catches a hung CLI has to be generous; half an hour bounds a runaway and is not a
# budget, and a task whose attempts are longer sets its own. Revisit if the
# authors' implementation lands (see references/method.md).
_DEFAULT_TIMEOUT = 1800.0

_PROMPT_VIA = ("argument", "stdin")

# What ``_digest`` reports for a file that is not there. Not hexadecimal, so it can
# never equal the digest of a file that is.
_SENTINEL = "missing"

# What the prompt template may refer to: everything ``propose`` fills in.
_VARIABLES = (
    "node_dir",
    "history_dir",
    "baseline_dir",
    "eval_program",
    "problem_file",
    "direction_guidance",
    "seed",
)


class CommandAgentError(RuntimeError):
    """The command did not produce an attempt, and why (with what it printed)."""


@dataclass(frozen=True)
class CommandAgent:
    """Produces each attempt by running ``command`` with the exploration prompt.

    ``command`` is an argv list, never a string, because there is no shell to split
    it. ``template`` is a :class:`string.Template` — the paper's own ``$var``
    syntax — defaulting to §B.1's prompt. ``eval_program`` is the file the agent
    writes its candidate to, relative to the workspace, and must be the
    ``artifact_name`` of the evaluator scoring it.
    """

    command: Sequence[str]
    template: str | None = None
    eval_program: str = "solution.py"
    timeout: float = _DEFAULT_TIMEOUT
    prompt_via: str = "argument"
    direction_guidance: str = ""
    baseline_dir: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.command, str) or not self.command:
            raise TypeError(
                "command must be a non-empty argv list such as ['claude', '-p'], "
                f"not {self.command!r}: there is no shell to split a string"
            )
        if not all(isinstance(argument, str) for argument in self.command):
            raise TypeError(f"command must be a list of strings, got {self.command!r}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.prompt_via not in _PROMPT_VIA:
            raise ValueError(f"prompt_via must be one of {_PROMPT_VIA}, got {self.prompt_via!r}")
        name = PurePath(self.eval_program)
        if not self.eval_program or name.is_absolute() or ".." in name.parts:
            raise ValueError(
                f"eval_program must be a path inside the workspace, got {self.eval_program!r}"
            )
        # Rendered once with placeholders, so a template that names a variable this
        # adapter does not fill fails here and not on the first attempt of a run.
        try:
            _render(self._template_text(), dict.fromkeys(_VARIABLES, "x"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"template cannot be filled: {exc}") from exc

    def propose(self, context: AgentContext) -> Artifact:
        """Run the command once, resuming from ``context.parent``, and read back its attempt."""
        # Absolute, because the command runs in the workspace and a relative path in
        # the prompt would be relative to somewhere else.
        node_dir = Path(context.workspace).resolve()
        work = node_dir / WORK_DIRNAME
        program = node_dir / self.eval_program
        proposal = node_dir / PROPOSAL_FILENAME
        # What a resumed workspace already holds: the parent's own files, which are
        # there whether or not this attempt wrote anything.
        before = (_digest(program), _digest(proposal))

        try:
            history_dir, problem_file, baseline_dir = self._lay_out(context, work)
            prompt = _render(
                self._template_text(),
                {
                    "node_dir": str(node_dir),
                    "history_dir": str(history_dir),
                    "baseline_dir": str(baseline_dir),
                    "eval_program": self.eval_program,
                    "problem_file": str(problem_file),
                    "direction_guidance": self.direction_guidance,
                    "seed": "" if context.seed is None else str(context.seed),
                },
            )
            outcome = self._run(prompt, node_dir, context.seed)
        finally:
            # Whatever ends the call: none of this belongs in the snapshot.
            shutil.rmtree(work / _HISTORY, ignore_errors=True)
            shutil.rmtree(work / _BASELINE, ignore_errors=True)
            (work / _PROBLEM).unlink(missing_ok=True)

        if outcome.status is None:
            raise CommandAgentError(
                _failure(f"the command timed out after {self.timeout:g}s", outcome.output)
            )
        if outcome.status != 0:
            raise CommandAgentError(
                _failure(f"the command exited with status {outcome.status}", outcome.output)
            )
        content = program.read_text(encoding="utf-8") if program.is_file() else ""
        if not content.strip():
            raise CommandAgentError(
                _failure(f"the command wrote no {self.eval_program}", outcome.output)
            )
        if _digest(program) == before[0]:
            # The same program the parent had is not an attempt: there is nothing new
            # to evaluate, and §B.1 says a repeat is exactly what not to propose.
            raise CommandAgentError(
                _failure(f"the command left {self.eval_program} unchanged", outcome.output)
            )
        written = _digest(proposal) != before[1]
        rationale = proposal.read_text(encoding="utf-8") if written else ""

        # Kept for the attempts below this one, which read it from their workspace.
        saved = work / _PROPOSALS / _proposal_name(len(context.history))
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(rationale, encoding="utf-8")
        return Artifact(content=content, proposal=rationale)

    def _template_text(self) -> str:
        return PROMPT_PATH.read_text(encoding="utf-8") if self.template is None else self.template

    def _run(self, prompt: str, node_dir: Path, seed: int | None) -> CommandOutcome:
        environment = dict(os.environ)
        if seed is not None:
            environment[SEED_ENV] = str(seed)
        if self.prompt_via == "stdin":
            argv, stdin = list(self.command), prompt
        else:
            argv, stdin = [*self.command, prompt], None
        try:
            return run_command(
                argv, cwd=node_dir, timeout=self.timeout, stdin=stdin, env=environment
            )
        except OSError as exc:
            raise CommandAgentError(
                f"could not run {self.command[0]!r}: {exc.strerror or exc}"
            ) from exc

    def _lay_out(self, context: AgentContext, work: Path) -> tuple[Path, Path, Path]:
        """Write what the prompt refers to: the history, the problem, and a baseline."""
        history_dir = work / _HISTORY
        shutil.rmtree(history_dir, ignore_errors=True)
        history_dir.mkdir(parents=True)
        for depth, node in enumerate(context.history):
            if node.parent_id is None:
                continue  # the root is the starting workspace, not an attempt
            self._write_attempt(history_dir / f"attempt_{node.id}", node, work, depth)

        problem_file = work / _PROBLEM
        problem_file.write_text(context.problem, encoding="utf-8")

        if self.baseline_dir is not None:
            baseline_dir = Path(self.baseline_dir).resolve()
        else:
            baseline_dir = work / _BASELINE
            baseline_dir.mkdir(parents=True, exist_ok=True)
        return history_dir, problem_file, baseline_dir

    def _write_attempt(self, directory: Path, node: Node, work: Path, depth: int) -> None:
        """One recorded attempt, laid out the way §B.1 says to read one."""
        directory.mkdir(parents=True)
        saved = work / _PROPOSALS / _proposal_name(depth)
        if saved.is_file():
            (directory / PROPOSAL_FILENAME).write_text(
                saved.read_text(encoding="utf-8"), encoding="utf-8"
            )
        if node.artifact is not None:
            program = directory / self.eval_program
            program.parent.mkdir(parents=True, exist_ok=True)
            program.write_text(node.artifact, encoding="utf-8")
        _write_evaluation(directory, node)


def _write_evaluation(directory: Path, node: Node) -> None:
    """``eval/score.json`` or ``eval/error.txt`` from what the node recorded.

    The task's own number, not the canonical ``s_v`` the node stores: the agent
    reads what the scorer printed, in the scorer's units, which is also what
    ``CommandEvaluator`` reads back in.
    """
    diagnostics: Mapping[str, object] = node.diagnostics
    error = diagnostics.get("error")
    raw = diagnostics.get("raw_score", node.score)
    text = str(diagnostics.get("text", ""))
    (directory / SCORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    if error is None and raw is not None:
        payload = {
            "score": raw,
            "correct": bool(diagnostics.get("correct", True)),
            "fail_class": str(diagnostics.get("fail_class", "ok")),
            "diagnostics": text,
        }
        (directory / SCORE_PATH).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    reason = str(error) if error is not None else "the evaluation produced no score"
    body = reason if not text or text == reason else f"{reason}\n\n{text}"
    (directory / ERROR_PATH).write_text(body, encoding="utf-8")


def _render(template: str, values: Mapping[str, str]) -> str:
    """``template`` filled in, strictly: a variable nothing supplies is an error."""
    return Template(template).substitute(values)


def _proposal_name(depth: int) -> str:
    return f"depth_{depth:06d}.md"


def _digest(path: Path) -> str:
    """A fingerprint of a file's bytes, or a sentinel for one that is not there."""
    if not path.is_file():
        return _SENTINEL
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def _failure(reason: str, output: str) -> str:
    """``reason``, then the tail of what the command printed, on the same line of text."""
    tail = tail_of(output)
    return f"{reason}: {tail}" if tail else reason
