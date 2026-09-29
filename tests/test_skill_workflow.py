"""SKILL.md must give a host agent a workflow it can follow, and its commands must run (issue #75).

Loaded in OpenCode on a task it fits, the skill was read, reasoned in the light of, and
never run: the session contained no ``dream_rsi`` invocation at all. Nothing in
``SKILL.md`` told the agent to, and the only command there was the toy quickstart. So
the file is now a workflow, and what can be tested about prose is tested here: that
every command it shows actually runs, that the exit codes it documents are the ones
the command uses, that the template it points at builds a task, and that it says
plainly near the top that loading it runs nothing.

The commands run **verbatim**, through ``bash``, against a stand-in task with no model
in it; only ``python`` is shimmed so the documented word finds this interpreter.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dream_rsi.run import EXIT_FINISHED, EXIT_REFUSED, EXIT_RUNNING, EXIT_STOPPED
from dream_rsi.task import Task, TaskError, load_task

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "SKILL.md"
TEMPLATE = REPO / "references" / "task_template.py"

# The section a host agent follows, from its heading to the next one.
WORKFLOW = "## Running it"

TIMEOUT_SECONDS = 180

# How long after a launch "not a run directory" is still an acceptable answer to a poll.
STARTUP_SECONDS = 30

# A task with no model in it, standing in for the file a host agent writes from the
# template: the toy roles, under the name the workflow tells it to use.
STAND_IN_TASK = """\
from dream_rsi.adapters.fake_developer import FakeDeveloper
from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator
from dream_rsi.run import DEFAULT_POLICY_SOURCE, TOY_PROBLEM, TOY_REVISIONS, TOY_SCRIPT
from dream_rsi.task import Task


def task():
    return Task(
        agent=ToySearchAgent(script=TOY_SCRIPT),
        evaluator=ToySearchEvaluator(),
        developer=FakeDeveloper(script=TOY_REVISIONS),
        problem=TOY_PROBLEM,
        policy=DEFAULT_POLICY_SOURCE,
    )
"""


def _section(text: str, heading: str) -> str:
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _commands(section: str) -> list[str]:
    """Every command line in the section's fenced ``bash`` blocks, in order."""
    lines: list[str] = []
    for block in re.findall(r"^```bash\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL):
        lines += [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return lines


def _environment(tmp_path: Path) -> dict[str, str]:
    """This interpreter as ``python`` on the path, and nothing else changed.

    A script and not a symlink: a virtual environment's interpreter finds its packages
    from the path it was started as, which a link elsewhere would lose.
    """
    shim = tmp_path / "bin" / "python"
    shim.parent.mkdir()
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"}


def test_every_command_the_workflow_shows_actually_runs(tmp_path: Path) -> None:
    """Issue #75's testable half: the commands, verbatim, against a stand-in task."""
    commands = _commands(_section(SKILL.read_text(encoding="utf-8"), WORKFLOW))
    assert any("--task" in command for command in commands), "no command runs a task"
    assert any("--report" in command for command in commands), "no command polls a run"
    (tmp_path / "task.py").write_text(STAND_IN_TASK, encoding="utf-8")
    environment = _environment(tmp_path)
    background: list[tuple[str, subprocess.Popen[str]]] = []

    for command in commands:
        if command.endswith("&"):
            process = subprocess.Popen(
                ["bash", "-c", command],
                cwd=tmp_path,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            background.append((command, process))
            continue
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        # A poll right after a launch may find the run not started yet, running, or
        # already done: any of the report's four answers is a correct one, and none is
        # a crash. Everything else must simply succeed.
        allowed = (
            {EXIT_FINISHED, EXIT_REFUSED, EXIT_RUNNING, EXIT_STOPPED}
            if "--report" in command
            else {0}
        )
        assert completed.returncode in allowed, f"{command!r}: {completed.stderr}"
        assert "Traceback" not in completed.stderr, f"{command!r}: {completed.stderr}"

    for command, process in background:
        # ``bash -c "… &"`` returns once the job is launched, not once it is done, which
        # is exactly why the workflow polls instead of waiting.
        _, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        assert process.returncode == 0, f"{command!r}: {stderr}"

    # Poll as the workflow says to, until the run is neither starting nor running.
    poll = next(command for command in commands if "--report" in command)
    started = time.monotonic()
    # "Not a run directory" is right for the first moments after a launch and wrong
    # for long: a launch command that never started anything should fail fast.
    deadline = started + TIMEOUT_SECONDS
    while True:
        final = subprocess.run(
            ["bash", "-c", poll],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        now = time.monotonic()
        never_started = final.returncode == EXIT_REFUSED and now > started + STARTUP_SECONDS
        if final.returncode not in (EXIT_RUNNING, EXIT_REFUSED) or never_started or now > deadline:
            break
        time.sleep(0.5)

    assert final.returncode == EXIT_FINISHED, final.stdout + final.stderr
    assert "status: finished" in final.stdout


def test_the_exit_codes_the_workflow_documents_are_the_ones_the_command_uses() -> None:
    text = SKILL.read_text(encoding="utf-8")
    documented = {
        int(code) for code in re.findall(r"^\|\s*`(\d+)`\s*\|", text, flags=re.MULTILINE)
    }

    assert documented == {EXIT_FINISHED, EXIT_RUNNING, EXIT_STOPPED, EXIT_REFUSED}


def test_the_skill_says_before_anything_else_that_loading_it_runs_nothing() -> None:
    """The failure this issue is about: the skill was read, and nothing was run."""
    text = SKILL.read_text(encoding="utf-8")
    opening = text[: text.index("## When this applies")]

    assert "loading this skill runs nothing" in opening.lower()
    assert "dream_rsi" in opening


def test_the_template_builds_a_task_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DREAM_RSI_AGENT_CMD", "some-cli -p --flag")
    monkeypatch.delenv("DREAM_RSI_DEVELOPER_CMD", raising=False)

    task = load_task(TEMPLATE)

    assert isinstance(task, Task)
    assert list(task.agent.command) == ["some-cli", "-p", "--flag"]  # type: ignore[attr-defined]
    # The developer defaults to the same CLI, and can be pointed at another.
    assert list(task.developer.command) == ["some-cli", "-p", "--flag"]  # type: ignore[attr-defined]
    monkeypatch.setenv("DREAM_RSI_DEVELOPER_CMD", "other-cli run")
    assert list(load_task(TEMPLATE).developer.command) == ["other-cli", "run"]  # type: ignore[attr-defined]
    # The agent writes the file the scorer reads.
    assert task.agent.eval_program == task.evaluator.artifact_name  # type: ignore[attr-defined]


def test_the_template_refuses_to_start_without_a_cli_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DREAM_RSI_AGENT_CMD", raising=False)
    monkeypatch.delenv("DREAM_RSI_DEVELOPER_CMD", raising=False)

    with pytest.raises(TaskError, match="DREAM_RSI_AGENT_CMD"):
        load_task(TEMPLATE)
