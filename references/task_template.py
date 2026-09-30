"""A task file for the Dream-RSI loop: copy it next to your scorer and edit the marked parts.

Run it small first, detached, and poll it (see SKILL.md, "Running it"):

    python -m dream_rsi.run --task task.py --cycles 1 --workers 2 --rounds 3 runs/first

Two things are read from the environment, so nothing here names a provider:

* ``DREAM_RSI_AGENT_CMD`` — the coding-agent CLI in its non-interactive mode, as a
  command line. Use the host's own, and let it edit files without asking (the loop
  cannot answer a permission prompt). For example:

      Claude Code   claude -p --permission-mode acceptEdits
      Gemini CLI    gemini --yolo -p
      OpenCode      opencode run

* ``DREAM_RSI_DEVELOPER_CMD`` — the same, for the agent that rewrites the exploration
  policy. Optional: it defaults to the command above.

If ``DREAM_RSI_AGENT_CMD`` is unset the file refuses to start and says so. There is no
default CLI, because a default would be a provider chosen for the user.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from dream_rsi.adapters.command_agent import CommandAgent
from dream_rsi.adapters.command_developer import CommandDeveloper
from dream_rsi.adapters.command_evaluator import CommandEvaluator
from dream_rsi.adapters.evaluator import ScoreDirection
from dream_rsi.task import Task

HERE = Path(__file__).resolve().parent

# EDIT: what is being optimised, and which way the score runs. A run directory is tied
# to this text, so resuming it under a different problem is refused.
PROBLEM = """\
Describe the task here: what a candidate is, what the scorer measures, and whether
larger or smaller is better.
"""

# EDIT: your scorer, as an argv list (there is no shell). It runs in the candidate's
# workspace, where the candidate has been written to ``solution.py``, and it must write
# ``eval/score.json`` holding {"score": <number>, "correct": <true|false>} — or exit
# non-zero / write ``eval/error.txt`` to say the candidate failed. Give an absolute path
# to a script that lives beside this file, since the working directory is the workspace.
SCORER = [sys.executable, str(HERE / "score.py")]

# EDIT: which way the scorer's number runs.
DIRECTION = ScoreDirection.HIGHER_IS_BETTER  # or ScoreDirection.LOWER_IS_BETTER

# The file the candidate lives in. The scorer reads it and the agent writes it, so the
# two must agree; change it in one place.
PROGRAM = "solution.py"


def _command(variable: str, *, fallback: str | None = None) -> list[str]:
    text = os.environ.get(variable) or (os.environ.get(fallback) if fallback else None)
    if not text:
        raise RuntimeError(
            f"set {variable} to the coding-agent CLI's non-interactive command, "
            "for example: claude -p --permission-mode acceptEdits"
        )
    return shlex.split(text)


def task() -> Task:
    agent = _command("DREAM_RSI_AGENT_CMD")
    developer = _command("DREAM_RSI_DEVELOPER_CMD", fallback="DREAM_RSI_AGENT_CMD")
    return Task(
        agent=CommandAgent(agent, eval_program=PROGRAM),
        evaluator=CommandEvaluator(SCORER, DIRECTION, artifact_name=PROGRAM),
        developer=CommandDeveloper(developer),
        problem=PROBLEM,
    )
