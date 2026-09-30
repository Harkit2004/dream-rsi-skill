"""The packing example as a Dream-RSI task: a coding-agent CLI proposes, ``score.py`` measures.

    export DREAM_RSI_AGENT_CMD="claude -p --permission-mode acceptEdits"   # or your CLI
    python -m dream_rsi.run --task examples/packing/task.py \\
        --cycles 2 --workers 3 --rounds 3 runs/packing

``DREAM_RSI_AGENT_CMD`` is the coding-agent CLI in its non-interactive mode. It drives both
the discovery agent (one attempt at a time) and the policy-development agent (which
rewrites the exploration policy from replay feedback); ``DREAM_RSI_DEVELOPER_CMD`` may name
a different CLI for the second. There is **no default**: if the variable is unset this file
refuses to start and says so, because a default would be a provider chosen for you. Let the
CLI edit files without asking — the loop cannot answer a permission prompt.

PAPER-GAP: §4's discovery runs are 110 to 640 discovery-agent calls each (10 workspaces × 11
steps for Gemini-3.1 Pro, 32 × 20 for Gemini-3.7-Flash). The budgets suggested above, three
workers over three rounds, are at most nine calls a cycle, which is what makes this an
example that costs minutes and not hours. It is a demonstration of the loop, not a
reproduction of §4, and its scores are not comparable to the paper's. Revisit if the
authors' implementation lands (see references/method.md).
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

# The problem statement is the file a person reads, and the text the agent is handed.
PROBLEM = (HERE / "problem.md").read_text(encoding="utf-8")

# The file the candidate lives in: the agent writes it, the scorer reads it.
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
        # Run in each candidate's workspace, so the scorer is named by absolute path.
        evaluator=CommandEvaluator(
            [sys.executable, str(HERE / "score.py")],
            ScoreDirection.HIGHER_IS_BETTER,
            artifact_name=PROGRAM,
        ),
        developer=CommandDeveloper(developer),
        problem=PROBLEM,
    )
