"""The full RSI driver: T cycles of deploy, record, dream, select, redeploy (issue #16).

§3's outer loop, wired out of pieces that are already tested on their own. What
matters here is therefore the wiring and nothing else: that each cycle's tree
joins the history its own dreaming phase runs over, that the version a cycle
selects is the one the next cycle actually deploys, that a run interrupted
half-way picks up where it stopped without losing or repeating a tree, and that
the whole thing is a function of its seed.

Nothing in this file calls a model: the discovery agent is the scripted toy one
and the development agent is a stub that answers from a list.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.develop import RevisionContext
from dream_rsi.dream import DreamConfig
from dream_rsi.orchestrator import (
    STOP_EMPTY_BATCH,
    STOP_MAX_ROUNDS,
    TREE_FILENAME,
    RolloutConfig,
)
from dream_rsi.run import (
    CYCLE_TEMPLATE,
    CYCLES_DIRNAME,
    POLICY_FILENAME,
    RECORD_FILENAME,
    Run,
    RunConfig,
    run_cycles,
)
from dream_rsi.sandbox import SandboxError, SandboxLimits
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]

# Generous next to what these policies spend — a rollout of eight attempts over
# the toy landscape is milliseconds of child CPU — so nothing here is measuring
# the machine it runs on. No policy in this file writes, bar the one that is
# supposed to be stopped from writing.
TEST_LIMITS = SandboxLimits(
    wall_seconds=5.0,
    cpu_seconds=2,
    memory_bytes=512 * 1024 * 1024,
    disk_bytes=1024 * 1024,
)

PROBLEM = "pick the plan (width, depth) with the best throughput within the cost budget"

# Candidates spanning the toy landscape, so a rollout over them has real
# alternatives to choose between. The agent answers to the attempt's seed, so
# which attempt proposes which plan is fixed (``adapters/toy_search.py``).
SCRIPT = (
    plan_source(0, 1),
    plan_source(1, 1),
    plan_source(2, 2),
    plan_source(3, 3),
    plan_source(2, 1),
    plan_source(1, 2),
    plan_source(0, 2),
    plan_source(4, 0),
)

# π_1: refines level by level and never stops itself, so it runs to the round
# cap and records a tree eight attempts wide and four deep.
BREADTH_SOURCE = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    pass
"""

# The revision the stub answers with: the same strategy that stops after two
# rounds. On the world the incumbent recorded it reaches the same best score
# having revealed half the nodes, so Equation 1's cost term puts it ahead and
# selection has to prefer it — and deployed online it ends on its own empty
# batch instead of the round cap, which is how a test can see which of the two
# drove a rollout.
THRIFTY_SOURCE = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    def reset(self, rng=None):
        super().reset(rng)
        self.spent = 0

    def choose(self, tree, live, width):
        self.spent += 1
        if self.spent > 2:
            return ()
        return super().choose(tree, live, width)
"""


@dataclass
class ScriptedDeveloper:
    """A development agent that answers from a list and keeps what it was asked.

    Deterministic and inert: no model, no network, no clock. The number of
    contexts it collected is how a test sees how many versions a run actually
    developed, which is what tells a resumed run from one that started over.
    """

    script: tuple[str, ...] = (THRIFTY_SOURCE,)
    calls: list[RevisionContext] = field(default_factory=list)

    def revise(self, context: RevisionContext) -> str:
        self.calls.append(context)
        return self.script[(len(self.calls) - 1) % len(self.script)]


def _run(
    directory: Path,
    developer: ScriptedDeveloper,
    *,
    cycles: int = 3,
    policy: str = BREADTH_SOURCE,
    scratch_root: Path | None = None,
) -> Run:
    """A toy run in ``directory``: two versions a cycle, over the toy landscape."""
    return run_cycles(
        agent=ToySearchAgent(script=SCRIPT),
        evaluator=ToySearchEvaluator(),
        developer=developer,
        policy=policy,
        problem=PROBLEM,
        directory=directory,
        config=RunConfig(
            cycles=cycles,
            rollout=RolloutConfig(workers=2, max_rounds=4, max_nodes=16, seed=0),
            dreaming=DreamConfig(width=2),
            # Two: the incumbent and one revision, which is the smallest round
            # that can select anything other than what it started with.
            versions=2,
            limits=TEST_LIMITS,
        ),
        scratch_root=scratch_root,
    )


def _cycle_dir(directory: Path, index: int) -> Path:
    return directory / CYCLES_DIRNAME / CYCLE_TEMPLATE.format(index)


def _records(directory: Path) -> list[str]:
    """Every cycle record a run left behind, as the bytes it wrote them as."""
    return [
        path.read_text(encoding="utf-8")
        for path in sorted((directory / CYCLES_DIRNAME).glob(f"*/{RECORD_FILENAME}"))
    ]


def _trees(directory: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for path in sorted((directory / CYCLES_DIRNAME).glob(f"*/{TREE_FILENAME}"))
    ]


def test_a_three_cycle_run_grows_the_pool_by_one_tree_per_cycle(tmp_path: Path) -> None:
    """Issue #16's first "tests first": three cycles, one new replay world each.

    §3: after the rollout of iteration ``t`` its tree is recorded as ``𝒯_t`` and
    appended, "giving ``ℋ_t = ℋ_{t-1} ∪ {𝒯_t}``", and the offline phase then
    runs over that expanded collection. So cycle ``t`` dreams over ``t + 1``
    worlds and its own is the last of them — a driver that dreamed over the
    history *before* its rollout, or only over its own tree, fails here.
    """
    run = _run(tmp_path, ScriptedDeveloper())

    assert [record.index for record in run.cycles] == [0, 1, 2]
    assert [record.pool for record in run.cycles] == [
        ("cycle_000",),
        ("cycle_000", "cycle_001"),
        ("cycle_000", "cycle_001", "cycle_002"),
    ]
    assert [record.world for record in run.cycles] == ["cycle_000", "cycle_001", "cycle_002"]

    # Every world named is a tree on disk holding the attempts its record claims,
    # so the pool is trees and not just names in a log.
    for record in run.cycles:
        tree = DiscoveryTree.load(_cycle_dir(tmp_path, record.index) / TREE_FILENAME)
        assert record.attempts > 0
        assert len(tree) - 1 == record.attempts


def test_the_next_cycle_deploys_the_version_the_last_one_selected(tmp_path: Path) -> None:
    """§3's step 5 closing into step 1: ``π_{t+1}`` is what cycle ``t`` selected.

    The revision beats the incumbent on the world the incumbent recorded, so the
    selection is a real switch rather than a retention, and the two policies are
    distinguishable by what they do online: the incumbent runs to the round cap,
    the revision stops itself. A driver that redeployed ``π_t`` regardless — the
    easy bug, since the incumbent's source is right there in the loop — records
    the incumbent's stop reason in cycle 1 and fails.
    """
    run = _run(tmp_path, ScriptedDeveloper())
    first, second = run.cycles[0], run.cycles[1]

    assert not first.selection.retained
    assert (_cycle_dir(tmp_path, 1) / POLICY_FILENAME).read_text(encoding="utf-8") == (
        THRIFTY_SOURCE
    )
    assert first.stop_reason == STOP_MAX_ROUNDS
    assert second.stop_reason == STOP_EMPTY_BATCH
    assert second.attempts < first.attempts


def test_a_run_resumed_after_a_crash_keeps_the_cycles_it_finished(tmp_path: Path) -> None:
    """A crash in a later cycle costs that cycle and nothing before it.

    The interruption is the one that actually happens: a cycle whose rollout is
    on disk but whose record was never written. The tree it left behind is
    vouched for by nothing, so the resumed run redoes that cycle — and only that
    cycle. What it ends with has to be what an uninterrupted run ends with: the
    same three worlds, none dropped and none counted twice.
    """
    whole = _run(tmp_path / "whole", ScriptedDeveloper())

    resumed_dir = tmp_path / "resumed"
    _run(resumed_dir, ScriptedDeveloper(), cycles=2)
    (_cycle_dir(resumed_dir, 1) / RECORD_FILENAME).unlink()

    developer = ScriptedDeveloper()
    resumed = _run(resumed_dir, developer)

    # Cycle 0 was read back, not redone: only cycles 1 and 2 asked for a
    # revision. A driver that started from scratch asks three times.
    assert len(developer.calls) == 2
    assert [record.pool for record in resumed.cycles] == [
        record.pool for record in whole.cycles
    ]
    assert _records(resumed_dir) == _records(tmp_path / "whole")
    assert _trees(resumed_dir) == _trees(tmp_path / "whole")


def test_two_runs_under_the_same_seed_record_the_same_cycles(tmp_path: Path) -> None:
    """End-to-end determinism (AGENTS.md rule 5), across two directories.

    Everything a cycle writes is compared, so a driver that let the thread pool's
    completion order, a wall clock, or the run directory's path reach a tree or a
    record fails here rather than in whatever consumes them later.
    """
    _run(tmp_path / "one", ScriptedDeveloper(), cycles=2)
    _run(tmp_path / "two", ScriptedDeveloper(), cycles=2)

    assert _trees(tmp_path / "one") == _trees(tmp_path / "two")
    assert _records(tmp_path / "one") == _records(tmp_path / "two")


def test_the_deployed_policy_is_sandboxed_online_too(tmp_path: Path) -> None:
    """Online deployment runs model-written code, so it goes behind the boundary.

    Dreaming already runs candidates in a child process (issue #13), but the
    winner of a cycle is *deployed*, and the driver is the one place that happens.
    A driver that executed the selected source in the harness — imported it,
    ``exec``-ed it, built the class itself — lets a policy that writes outside
    its scratch directory do exactly that, which is the one thing CLAUDE.md says
    never to relax. Here the write is refused, the cycle fails with it, and the
    file never appears.
    """
    escape = tmp_path / "escaped.txt"
    source = (
        "from pathlib import Path\n"
        "\n"
        "\n"
        "class OptimalPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        f"        Path({str(escape)!r}).write_text('escaped')\n"
        "        return (tree.root_id,)\n"
    )

    with pytest.raises(SandboxError):
        _run(tmp_path / "run", ScriptedDeveloper(), cycles=1, policy=source)

    assert not escape.exists()


def test_the_toy_loop_runs_three_cycles_from_the_command_line(tmp_path: Path) -> None:
    """Issue #16's "done when": ``python -m dream_rsi.run --cycles 3``, no model.

    A subprocess rather than :func:`~dream_rsi.run.main` in-process, because what
    is being checked is that the module is runnable as the issue writes it — the
    default wiring included, since that is where a driver would otherwise need a
    provider to be runnable at all.
    """
    target = tmp_path / "run"
    completed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.run", "--cycles", "3", str(target)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for index in range(3):
        record = json.loads(
            (_cycle_dir(target, index) / RECORD_FILENAME).read_text(encoding="utf-8")
        )
        assert record["index"] == index
        assert len(record["pool"]) == index + 1
