"""The README quickstart: one command, no model, and a policy that visibly improves (issue #20).

The quickstart is the only part of this repo that someone who has not read the
paper will run, so the thing worth testing is not that the loop works — every
other file here does that — but that *the README is still true*. Documentation
drifts silently: a column gets renamed, a default changes, the command grows a
required flag, and the block in the README goes on looking plausible for months.

So the command and the expected output are read out of ``README.md`` rather than
written here, and the run is a subprocess started from an empty directory: what
is asserted is that the documented command still runs, and that what it prints
is what the README says it prints.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from dream_rsi.run import CYCLE_TEMPLATE, CYCLES_DIRNAME, POLICY_FILENAME, RECORD_FILENAME

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

# The heading the quickstart lives under.
HEADING = "## Quickstart"

# The two columns of the report that measure the machine rather than the run:
# a duration is not reproducible and the README cannot promise one.
WALL_CLOCK = ("online.s", "dream.s")


def _section(heading: str) -> str:
    """The README under ``heading``, up to the next heading of the same level."""
    text = README.read_text(encoding="utf-8")
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _block(section: str, language: str) -> str:
    """The one fenced ``language`` block in ``section``."""
    found = re.findall(rf"^```{language}\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL)
    assert len(found) == 1, f"expected one ```{language} block under {HEADING}, found {len(found)}"
    return found[0]


def _mask(report: str) -> str:
    """A report reduced to what the README can promise: labels, numbers, order.

    Runs of spaces collapse, because the column padding is cosmetic and widens
    with whatever the longest value happens to be, and the two wall-clock
    columns become ``-``, because how long a cycle took is a fact about the
    machine. Everything else — every label, every count, every score, and the
    order they come in — is a function of the toy task and the run's seed, so it
    is something the README is allowed to state and this test can hold it to.
    """
    masked: list[str] = []
    columns: tuple[str, ...] = ()
    for line in report.splitlines():
        fields = line.split()
        if not fields:
            columns = ()
            masked.append("")
            continue
        if fields[0] == "cycle" and fields[-1] == "selected":
            columns = tuple(fields)
        elif columns and len(fields) == len(columns):
            fields = [
                "-" if column in WALL_CLOCK else field
                for column, field in zip(columns, fields)
            ]
        masked.append(" ".join(fields))
    return "\n".join(masked).strip()


def _arguments() -> list[str]:
    """The README's quickstart command, minus the interpreter it names.

    The documented ``python`` is dropped in favour of the interpreter running
    the tests, so the command is exercised against the checkout under test
    rather than against whatever ``python`` happens to be first on this
    machine's path. Everything after it is the README's, argument for argument.
    """
    command = _block(_section(HEADING), "bash").strip()
    assert "\n" not in command, f"the quickstart is one command, found:\n{command}"
    executable, *arguments = command.split()
    assert executable in ("python", "python3"), f"quickstart runs {executable!r}, not python"
    return arguments


def _quickstart(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the README's command from an empty directory, as a reader would."""
    return subprocess.run(
        [sys.executable, *_arguments()],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )


def _cycles(tmp_path: Path) -> Path:
    """The cycle directories the documented command wrote, wherever it puts them.

    Off the command's own last argument, which is the run directory the README
    tells the reader to pass, so moving it in the README moves it here too.
    """
    return tmp_path / _arguments()[-1] / CYCLES_DIRNAME


def test_the_documented_quickstart_command_runs_and_exits_zero(tmp_path: Path) -> None:
    """Issue #20's first "tests first": the command in the README, in CI, exit 0.

    Run from an empty directory and with nothing configured, so a quickstart
    that had quietly come to need an API key, a provider, a network call or a
    file from the checkout fails here — which is the whole point of a command
    offered to someone who has just cloned the repo.
    """
    completed = _quickstart(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert _cycles(tmp_path).is_dir()


def test_the_documented_output_is_what_the_quickstart_actually_prints(tmp_path: Path) -> None:
    """Issue #20's second "tests first": the documented output shape, asserted.

    The whole report, not a line of it: a renamed column, a dropped section, a
    changed default that moves every count, or a reordered report all leave the
    README stating something the command no longer does. Compared with the
    padding collapsed and the two durations masked, because those are facts
    about the machine and not about the run.
    """
    completed = _quickstart(tmp_path)
    documented = _block(_section(HEADING), "text")

    assert completed.returncode == 0, completed.stderr
    assert _mask(completed.stdout) == _mask(documented)


def test_the_quickstart_shows_the_policy_improving(tmp_path: Path) -> None:
    """Issue #20's "done when": a clean checkout plus one command shows a policy improving.

    Two claims, and the README is worth nothing without either. *Improving*: a
    cycle selected a version that beat the one it was deployed with under
    Equation 1, rather than retaining its incumbent for three cycles because the
    toy wiring offers nothing that can win. *Shown*: the report puts the
    resulting source change on screen as a diff, since seeing the policy's code
    change is the thing the quickstart exists to demonstrate.

    Asserted against the cycle directories rather than against literal source,
    so this fails on a report that diffs a cycle against itself, prints the
    chain one cycle out of step, or leaves the diff out — and on toy defaults
    that stop producing an improvement at all.
    """
    completed = _quickstart(tmp_path)
    assert completed.returncode == 0, completed.stderr
    first = _cycles(tmp_path) / CYCLE_TEMPLATE.format(0)
    second = _cycles(tmp_path) / CYCLE_TEMPLATE.format(1)

    selection = json.loads((first / RECORD_FILENAME).read_text(encoding="utf-8"))["selection"]
    assert selection["winner"] != selection["incumbent"]
    assert selection["score"] > selection["incumbent_score"]

    before = (first / POLICY_FILENAME).read_text(encoding="utf-8").splitlines()
    after = (second / POLICY_FILENAME).read_text(encoding="utf-8").splitlines()
    assert before != after

    # Stripped, because the report indents the diff under the cycle it belongs
    # to; what is being checked is which source lines it marks, not the indent.
    printed = [line.strip() for line in completed.stdout.splitlines()]
    for line in set(after) - set(before):
        assert f"+{line}" in printed
    for line in set(before) - set(after):
        assert f"-{line}" in printed
