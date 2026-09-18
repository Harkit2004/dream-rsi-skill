"""Record the discovery trees committed under ``trees/`` (issue #6).

Run it to rebuild them — after a ``SCHEMA_VERSION`` bump, or after anything in
the recording path changes::

    python tests/fixtures/record_trees.py            # rewrite trees/ in place
    python tests/fixtures/record_trees.py OUTDIR     # write somewhere else

The fixtures are real rollouts, not hand-written JSON: they come out of
:func:`~dream_rsi.orchestrator.run_rollout` against the deterministic toy task
in ``adapters/toy_search.py``, so what phases 2 and 3 replay has the shape a
recorded rollout actually has. No model is called and nothing is sampled, so
re-recording reproduces the committed bytes exactly — ``tests/test_fixtures.py``
asserts that.

Each recording fixes both halves of what the corpus needs:

* the **policy** fixes the shape — how wide each round is and which leaves it
  extends — through a script of positions in ``A(T)``, whose first entry is
  always the root;
* the **agent script** fixes the outcomes, because the toy agent answers to the
  attempt's seed and the rollout seeds attempt ``n`` with ``n``. So the
  candidate attempt ``n`` proposes is ``script[n % len(script)]``, and a
  recording can place a hard failure exactly where it wants one.

No snapshot store: the toy task's agent and evaluator touch no files, so every
node's ``snapshot_ref`` is null and the fixtures carry no filesystem state. A
replay world is built from recorded outcomes alone (issue #7).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.orchestrator import RolloutConfig, run_rollout
from dream_rsi.tree import DiscoveryTree

DEFAULT_DESTINATION = Path(__file__).parent / "trees"

PROBLEM = "pick the plan (width, depth) with the best throughput within the cost budget"

# An unreadable artifact: the toy evaluator finds no plan in it and records a
# hard failure, which is how a recording grows a dead branch.
NO_PLAN = "# out of ideas, no plan this time\n"


@dataclass
class ScriptedPolicy:
    """Selects, each round, the eligible nodes at the scripted positions.

    Positions into ``A(T)`` rather than node ids, so a recording can be read
    without knowing which id the rollout will mint: position 0 is always the
    root, and the rest are the current leaves in id order. A round the script
    does not cover selects the empty batch, which ends the rollout.
    """

    rounds: tuple[tuple[int, ...], ...]
    _index: int = field(default=0, init=False)

    def select(self, tree: DiscoveryTree, eligible: Sequence[str], width: int) -> Sequence[str]:
        if self._index >= len(self.rounds):
            return ()
        positions = self.rounds[self._index]
        self._index += 1
        return tuple(eligible[position] for position in positions)


@dataclass(frozen=True)
class Recording:
    """One fixture: its directory name and the rollout that produces it."""

    name: str
    workers: int
    rounds: tuple[tuple[int, ...], ...]
    script: tuple[str, ...]


RECORDINGS = (
    # Wide and shallow: two rounds of four attempts, every one of them off the
    # root, so the tree is eight siblings at depth 1. Every candidate is
    # admissible and they span the landscape — the corner the baseline sits on,
    # the local optimum, the valley, the best admissible plan — so a replay test
    # on this tree is choosing between real alternatives and never sees a
    # failure.
    Recording(
        name="wide_shallow",
        workers=4,
        rounds=((0, 0, 0, 0), (0, 0, 0, 0)),
        script=(
            plan_source(0, 0),
            plan_source(1, 1),
            plan_source(2, 2),
            plan_source(4, 2),
            plan_source(2, 4),
            plan_source(3, 3),
            plan_source(0, 4),
            plan_source(4, 0),
        ),
    ),
    # Narrow and deep: one attempt per round, always extending the single leaf
    # (position 1 in A(T) is the deepest node once the chain has started), so
    # the tree is an unbranched chain of six. The scores climb, dip and climb
    # again along it, so a policy reading the chain as a trend is reading
    # something non-monotone.
    Recording(
        name="narrow_deep",
        workers=1,
        rounds=((0,), (1,), (1,), (1,), (1,), (1,)),
        script=(
            plan_source(0, 0),
            plan_source(0, 1),
            plan_source(1, 1),
            plan_source(2, 2),
            plan_source(2, 4),
            plan_source(3, 3),
        ),
    ),
    # A hard-failing branch: three branches off the root, of which the third is
    # unreadable and scores nothing. The next round extends the two live leaves
    # and leaves the dead one alone — positions 1 and 2, the dead branch being
    # position 3 — which is what a policy that reads ``branch_failed_hard`` does
    # (§B.2). One of those extensions is over budget: evaluated successfully,
    # inadmissible, scored 0.0. The last round extends the best branch once more.
    Recording(
        name="failing_branch",
        workers=3,
        rounds=((0, 0, 0), (1, 2), (2,)),
        script=(
            plan_source(1, 1),
            plan_source(2, 2),
            NO_PLAN,
            plan_source(3, 3),
            plan_source(4, 4),
        ),
    ),
)


def record(recording: Recording, destination: Path) -> Path:
    """Record one rollout into ``destination / recording.name``."""
    target = destination / recording.name
    with tempfile.TemporaryDirectory() as workspace:
        # Never read or written — the toy task touches no files — and never
        # recorded, so the tree does not depend on where this ran.
        rollout = run_rollout(
            agent=ToySearchAgent(script=recording.script),
            evaluator=ToySearchEvaluator(),
            policy=ScriptedPolicy(rounds=recording.rounds),
            problem=PROBLEM,
            workspace=Path(workspace),
            config=RolloutConfig(
                workers=recording.workers,
                # One round past the script, so the rollout ends on the policy
                # selecting nothing rather than on the round cap.
                max_rounds=len(recording.rounds) + 1,
                max_nodes=None,
                seed=0,
            ),
        )
    rollout.save(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the committed discovery tree fixtures.")
    parser.add_argument(
        "destination",
        nargs="?",
        default=DEFAULT_DESTINATION,
        type=Path,
        help=f"where to write the fixtures (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args(argv)

    for recording in RECORDINGS:
        target = record(recording, args.destination)
        tree = DiscoveryTree.load(target / "tree.json")
        print(f"{recording.name}: {len(tree) - 1} attempt(s) -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
