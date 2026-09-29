"""A policy-development agent that drives a coding-agent CLI (issue #71).

The same pattern as :mod:`~dream_rsi.adapters.command_agent`, applied to the other
role. :func:`~dream_rsi.develop.develop` renders §B.2's prompt through
:meth:`~dream_rsi.develop.RevisionContext.prompt` and hands it to a
:class:`~dream_rsi.develop.PolicyDeveloper`; until now the only one was
``FakeDeveloper``, which answers from a fixed list, so every policy improvement a
run has ever reported was written in advance. The replay and selection around it
were real, and the revision was not. With a CLI behind this the revision is a
model's, and the dreaming half is the paper's.

**What the command is given.** A fresh scratch directory, which is also its working
directory, holding ``prompt.md`` — the whole of :meth:`RevisionContext.prompt`, with
a closing section saying where to put the answer — and ``policy.py``, the policy
being revised. The command line carries only a short instruction to read the prompt
file, as the last argument or on stdin (``prompt_via``): the prompt runs to the
replay trajectories of every world, which is more than one argument should carry.

**What comes back.** A file, ``revised_policy.py``, and not the command's output:
stdout would have to have its markdown fences and commentary stripped, and a file
either holds the module or does not. The prompt asks for a reply with "nothing else"
because that is how a model answers a bare prompt, so the closing section tells the
CLI to write the file *instead*.

**What this does not do.** It does not check that the file is Python, or that it
defines a policy: :func:`~dream_rsi.develop.validate_source` and the re-ask on
refusal (``RunConfig.attempts``) already do, and doing it twice would be two
answers to one question. It does not run the module either — never imports it, never
``exec``s it — because the revised source only ever runs through the sandbox
(:mod:`dream_rsi.sandbox`), where everything the harness learns about a version is
learned. What it does do is refuse to hand back nothing: a command that exits
non-zero, times out, or leaves no module raises
:class:`~dream_rsi.develop.RevisionFailed` with the tail of what it printed, which
``develop`` treats as it treats an unusable revision. The cycle is not lost, the
incumbent stays selectable, and the retry is shown what went wrong.

Examples of ``command`` — examples, not code paths — are the ones in
:mod:`~dream_rsi.adapters.command_agent`. AGENTS.md rule 6 applies: the adapter adds
nothing to the command line beyond its instruction.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dream_rsi.adapters._command import (
    DEFAULT_MAX_OUTPUT_BYTES,
    is_inside,
    run_command,
    tail_of,
)
from dream_rsi.develop import RevisionContext, RevisionFailed

__all__ = ["OUTPUT_FILENAME", "PROMPT_FILENAME", "CommandDeveloper"]

PROMPT_FILENAME = "prompt.md"
POLICY_FILENAME = "policy.py"
OUTPUT_FILENAME = "revised_policy.py"

# The section closing the prompt file. §B.2's prompt, as this repository words it,
# ends "reply with one Python module and nothing else"; a CLI in a working directory
# is better asked for a file, and this is where that is said.
_WHERE_TO_WRITE = f"""

## Where to write your answer

Write the complete module to `{OUTPUT_FILENAME}` in the current directory, instead of
replying with it: that file is your whole answer, and it must hold the Python module
and nothing else. The policy you are revising is also in `{POLICY_FILENAME}`. Do not
edit that file or any other.
"""

# What goes on the command line. Short on purpose: the prompt is in the file.
_INSTRUCTION = (
    f"Read {PROMPT_FILENAME} in the current directory and follow it. Write the complete "
    f"revised policy module to {OUTPUT_FILENAME}; that file is your whole answer."
)

# PAPER-GAP: the paper says nothing about how long one policy-development call may
# take. It reads replay trajectories and rewrites a module, which a coding-agent CLI
# does in minutes; a limit that catches a hung one has to be generous, and this bounds
# a runaway rather than budgeting a call. Revisit if the authors' implementation lands
# (see references/method.md).
_DEFAULT_TIMEOUT = 1800.0

_PROMPT_VIA = ("argument", "stdin")

# How the answer is opened: never through a link, never waiting on a pipe, and as bytes
# on every platform. Each of these is absent on some platforms, hence ``getattr``.
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
)


@dataclass(frozen=True)
class CommandDeveloper:
    """Asks a coding-agent CLI for the next version of the policy.

    ``command`` is an argv list, never a string, because there is no shell to split
    it. ``prompt_via`` says whether the instruction reaches it as the last argument
    or on stdin. ``max_output_bytes`` bounds what it may print and how large the
    module it writes may be.
    """

    command: Sequence[str]
    timeout: float = _DEFAULT_TIMEOUT
    prompt_via: str = "argument"
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

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
        if self.max_output_bytes <= 0:
            raise ValueError(f"max_output_bytes must be positive, got {self.max_output_bytes}")
        if self.prompt_via not in _PROMPT_VIA:
            raise ValueError(f"prompt_via must be one of {_PROMPT_VIA}, got {self.prompt_via!r}")

    def revise(self, context: RevisionContext) -> str:
        """Run the command once on ``context``, and return the module it wrote."""
        with tempfile.TemporaryDirectory(prefix="dream-rsi-developer-") as scratch:
            directory = Path(scratch)
            (directory / PROMPT_FILENAME).write_text(
                context.prompt() + _WHERE_TO_WRITE, encoding="utf-8"
            )
            (directory / POLICY_FILENAME).write_text(context.source, encoding="utf-8")

            if self.prompt_via == "stdin":
                argv, stdin = list(self.command), _INSTRUCTION
            else:
                argv, stdin = [*self.command, _INSTRUCTION], None
            try:
                outcome = run_command(
                    argv,
                    cwd=directory,
                    timeout=self.timeout,
                    stdin=stdin,
                    max_output_bytes=self.max_output_bytes,
                )
            except OSError as exc:
                raise RevisionFailed(
                    f"could not run {self.command[0]!r}: {exc.strerror or exc}"
                ) from exc

            if outcome.exceeded:
                raise RevisionFailed(_failure("the command printed without end", outcome.output))
            if outcome.status is None:
                raise RevisionFailed(
                    _failure(f"the command timed out after {self.timeout:g}s", outcome.output)
                )
            if outcome.status != 0:
                raise RevisionFailed(
                    _failure(f"the command exited with status {outcome.status}", outcome.output)
                )
            output = directory / OUTPUT_FILENAME
            # The CLI had a shell in this directory, so the answer may be a link to any
            # file the harness can read — which would then be recorded as the policy
            # and shown to the next attempt. A refusal, like any other unusable answer.
            if not is_inside(directory, output):
                raise RevisionFailed(
                    f"{OUTPUT_FILENAME} resolves outside the scratch directory, so the "
                    "harness will not read through it"
                )
            revision = _read_revision(output, self.max_output_bytes)
            if not revision.strip():
                raise RevisionFailed(
                    _failure(f"the command wrote no {OUTPUT_FILENAME}", outcome.output)
                )
            return revision


def _read_revision(path: Path, limit: int) -> str:
    """What ``path`` holds, or ``""`` if it is not there.

    Opened once, and the descriptor is what is read: a check on the *name* followed by
    a read of the name leaves room for whatever the CLI left running to swap it in
    between. A link is refused by the open itself, a pipe is opened without waiting for
    a writer, and at most ``limit`` bytes are read, so a runaway file is refused without
    being loaded. Anything the operating system will not do is a refusal, like any other
    answer that cannot be used.
    """
    try:
        with os.fdopen(os.open(path, _OPEN_FLAGS), "rb") as source:
            data = source.read(limit + 1)
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RevisionFailed(f"could not read {OUTPUT_FILENAME}: {exc.strerror or exc}") from exc
    if len(data) > limit:
        raise RevisionFailed(f"{OUTPUT_FILENAME} is larger than {limit} bytes, so it was not read")
    return data.decode("utf-8", errors="replace")


def _failure(reason: str, output: str) -> str:
    """``reason``, then the tail of what the command printed."""
    tail = tail_of(output)
    return f"{reason}: {tail}" if tail else reason
