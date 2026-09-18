"""The online rollout: branching, ``W`` parallel workers, stopping.

This is the "lightweight orchestration layer that controls branching, parallel
exploration, and stopping while leaving the underlying coding agent unchanged"
(Dream-RSI §1). One rollout runs the paper's decision loop (§3, "Online
rollout"):

* the policy observes the tree built so far and selects a batch ``C`` drawn from
  the eligible set ``A(T) = {r} ∪ {v ∈ T : v is a leaf}``;
* each selected node is handed to a worker, which resumes that node's workspace,
  asks the discovery agent for one candidate and has the evaluator score it;
* every completed attempt is attached as a new child of the node it started
  from, and the next round decides again on the extended tree;
* the rollout ends when the policy selects an empty batch or the budget runs out.

Attaching children never depends on the order the workers happen to finish in,
so the number of workers changes how long a rollout takes and nothing about the
tree it records.

The tree this produces is what a replay world is built from (issue #7), so each
round is also logged: the batch width the policy asked for — Equation 1's ``k``
(issue #8) — the nodes it selected, and the children those selections produced.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dream_rsi.adapters.agent import AgentContext, CodingAgent
from dream_rsi.adapters.evaluator import EvalResult, TaskEvaluator, safe_evaluate
from dream_rsi.adapters.fake_agent import FakeAgent
from dream_rsi.adapters.toy_evaluator import ToyEvaluator
from dream_rsi.tree import DiscoveryTree, Node, eligible_nodes
from dream_rsi.workspace import SnapshotStore

__all__ = [
    "ExplorationPolicy",
    "FirstEligiblePolicy",
    "Rollout",
    "RolloutConfig",
    "RoundRecord",
    "eligible_nodes",
    "main",
    "run_rollout",
]

# Why each rollout stopped. The first two are the paper's own termination rules
# (§3: "the rollout ends when the policy selects an empty batch or completes K₁
# decision rounds"); the last two come with the budget below.
STOP_EMPTY_BATCH = "empty_batch"
STOP_MAX_ROUNDS = "max_rounds"
STOP_MAX_NODES = "max_nodes"
STOP_WALL_CLOCK = "wall_clock"

TREE_FILENAME = "tree.json"
ROUNDS_FILENAME = "rounds.json"
STORE_DIRNAME = "store"

# One directory per attempt, named after the attempt's index in the rollout. The
# paper's exploration prompt has the same shape — "$node_dir is your own attempt
# directory --- exclude it when scanning sibling attempt_*/ dirs" (§B.1).
_ATTEMPT_TEMPLATE = "attempt_{:06d}"


@dataclass(frozen=True)
class RolloutConfig:
    """How wide a rollout runs, and when it stops.

    ``workers`` is the paper's ``W``: how many generation–evaluation requests can
    be in flight at once (§3). It is offered to the policy as the feasible batch
    width and caps the pool that executes the batch.
    """

    workers: int = 1

    # PAPER-GAP: §3 caps a rollout at K₁ decision rounds but never says what K₁
    # is, and names no budget on attempts or on time — the experiments instead
    # report fixed per-round call budgets (§4: 10 workspaces × 11 steps = 110
    # calls for Gemini-3.1 Pro, 32 × 20 = 640 for Gemini-3.7-Flash). We default
    # to 16 rounds and 64 attempts: well under those budgets, so a misconfigured
    # run stops long before it spends a real one, and any serious rollout sets
    # both explicitly. ``max_seconds`` defaults to off because a wall-clock limit
    # makes a rollout depend on how fast the machine was that day; it has to be
    # opted into. Revisit if the authors' implementation lands (see
    # references/method.md).
    max_rounds: int = 16
    max_nodes: int | None = 64
    max_seconds: float | None = None

    # Seeds the per-attempt seeds, so a rollout against a deterministic agent is
    # reproducible (AGENTS.md rule 5).
    seed: int = 0

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1 (§3: W ≥ 1), got {self.workers}")
        if self.max_rounds < 0:
            raise ValueError(f"max_rounds must not be negative, got {self.max_rounds}")
        if self.max_nodes is not None and self.max_nodes < 0:
            raise ValueError(f"max_nodes must not be negative, got {self.max_nodes}")
        if self.max_seconds is not None and self.max_seconds < 0:
            raise ValueError(f"max_seconds must not be negative, got {self.max_seconds}")


@dataclass(frozen=True)
class RoundRecord:
    """One completed decision round.

    ``k`` is the batch width the policy asked for, which is what Equation 1's
    batching term is computed against (issue #8); it is larger than
    ``len(selected)`` on the round where the node budget ran out mid-batch.
    ``selected`` and ``produced`` align pairwise: ``produced[i]`` is the child
    the attempt from ``selected[i]`` created.
    """

    index: int
    k: int
    selected: tuple[str, ...]
    produced: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "k": self.k,
            "selected": list(self.selected),
            "produced": list(self.produced),
        }


@dataclass(frozen=True)
class Rollout:
    """A completed online rollout: the tree it recorded and how it got there."""

    tree: DiscoveryTree
    rounds: tuple[RoundRecord, ...]
    stop_reason: str

    def save(self, directory: str | Path) -> None:
        """Write the tree and the round log into ``directory``.

        The tree goes out through :meth:`DiscoveryTree.save`, so it reloads with
        :meth:`DiscoveryTree.load` and nothing here needs to know its format. The
        round log sits beside it rather than inside it: rounds are a property of
        the rollout that recorded the tree, not of the tree itself, and a replay
        world is built from the tree alone.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.tree.save(target / TREE_FILENAME)
        payload = {
            "stop_reason": self.stop_reason,
            "rounds": [round_.to_dict() for round_ in self.rounds],
        }
        (target / ROUNDS_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@runtime_checkable
class ExplorationPolicy(Protocol):
    """The paper's shared decision interface, as the rollout loop needs it (§3).

    Implementations are structural: anything with this one method drives a
    rollout. The full :class:`OptimalPolicy` surface the policy-development agent
    writes against — ``solve``, the observation signals, ``plan_grid`` — is issue
    #10; this is the part the online loop calls.
    """

    def select(
        self, tree: DiscoveryTree, eligible: Sequence[str], width: int
    ) -> Sequence[str]:
        """Choose the batch to expand next, or nothing to end the rollout.

        ``eligible`` is ``A(T)``, the root followed by the current leaves.
        ``width`` is ``W``. Every returned id must be in ``eligible``; returning
        one twice schedules two attempts from that node in the same round.
        """
        ...


@dataclass(frozen=True)
class FirstEligiblePolicy:
    """Expands the first ``width`` eligible nodes — the root, then oldest leaves.

    A placeholder so the module has something to run end to end; it explores
    nothing in particular. Real policies arrive with issue #10.
    """

    def select(
        self, tree: DiscoveryTree, eligible: Sequence[str], width: int
    ) -> Sequence[str]:
        return tuple(eligible[:width])


def run_rollout(
    *,
    agent: CodingAgent,
    evaluator: TaskEvaluator,
    policy: ExplorationPolicy,
    problem: str,
    workspace: Path,
    snapshots: SnapshotStore | None = None,
    config: RolloutConfig | None = None,
) -> Rollout:
    """Run one online rollout and return the tree it recorded.

    ``workspace`` is the root workspace state: the tree's root records a
    snapshot of it, and every attempt resumes from its own parent's snapshot
    (§3). With a ``snapshots`` store each attempt gets its own directory, so
    siblings cannot see each other's files, and the state it leaves behind is
    recorded as its node's ``snapshot_ref``. Two rollouts must not share a store
    while they are running — attempt directories are named per rollout — but a
    finished rollout's snapshots stay readable from the store afterwards.

    Without a store, ``snapshot_ref`` stays unset and every attempt is handed
    ``workspace`` itself, which is only sound for a task whose agent and
    evaluator touch no files (the toy task in this repo is one).

    ``agent`` and ``evaluator`` are called from up to ``config.workers`` threads
    at once, so an adapter that keeps state has to tolerate that. The two shipped
    in this repo are stateless.
    """
    config = config or RolloutConfig()
    tree = DiscoveryTree.with_root(
        snapshot_ref=None if snapshots is None else snapshots.capture(workspace)
    )
    rounds: list[RoundRecord] = []
    deadline = None if config.max_seconds is None else time.monotonic() + config.max_seconds
    attempts = 0
    stop_reason = STOP_MAX_ROUNDS

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for index in range(config.max_rounds):
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = STOP_WALL_CLOCK
                break
            remaining = None if config.max_nodes is None else config.max_nodes - attempts
            if remaining is not None and remaining <= 0:
                stop_reason = STOP_MAX_NODES
                break

            eligible = eligible_nodes(tree)
            batch = tuple(policy.select(tree, eligible, config.workers))
            _check_batch(batch, eligible)
            if not batch:
                stop_reason = STOP_EMPTY_BATCH
                break

            # Truncated rather than dropped whole: the budget is a ceiling on
            # attempts, and stopping a round short spends exactly what is left
            # instead of leaving it unused. The round still records the width the
            # policy asked for.
            scheduled = batch if remaining is None else batch[:remaining]
            jobs = [
                (
                    _context(tree, parent_id, problem, workspace, config.seed + attempts + offset),
                    tree.node(parent_id).snapshot_ref,
                    _ATTEMPT_TEMPLATE.format(attempts + offset),
                )
                for offset, parent_id in enumerate(scheduled)
            ]
            # Results come back in submission order, so the children are attached
            # in the order the policy selected their parents however the workers
            # interleave.
            outcomes = list(
                pool.map(lambda job: _run_attempt(agent, evaluator, snapshots, *job), jobs)
            )

            produced = []
            for parent_id, (artifact, result, snapshot_ref) in zip(
                scheduled, outcomes, strict=True
            ):
                node = tree.add_child(
                    parent_id,
                    artifact=artifact,
                    observations=_observations(result),
                    snapshot_ref=snapshot_ref,
                    **result.to_node_fields(evaluator.direction),
                )
                produced.append(node.id)
            attempts += len(produced)
            rounds.append(RoundRecord(index, len(batch), tuple(scheduled), tuple(produced)))

    return Rollout(tree=tree, rounds=tuple(rounds), stop_reason=stop_reason)


def _check_batch(batch: Sequence[str], eligible: Sequence[str]) -> None:
    """Reject a batch that is not an action the decision interface allows.

    PAPER-GAP: §3 defines the action as a set ``C ⊆ A(T)`` with ``|C| ≤ W``,
    which cannot express two attempts from the same parent in one round — yet
    the experiments run 10 and 32 parallel workspaces off a single root (§4),
    and the replay cost model in §B.2 prices a batch of size ``k`` at
    ``ceil(k / W)`` sequential rounds, which only says anything when ``k > W``.
    We therefore take the batch as a sequence: repeats are allowed, and each one
    opens its own branch, so a rollout can reach full width in one round. ``W``
    stays the number of workers, and a batch wider than ``W`` simply queues.
    Revisit if the authors' implementation lands (see references/method.md).
    """
    allowed = set(eligible)
    unknown = [node_id for node_id in batch if node_id not in allowed]
    if unknown:
        raise ValueError(
            f"policy selected node(s) outside A(T): {', '.join(sorted(set(unknown)))}; "
            f"eligible are {', '.join(eligible)}"
        )


def _context(
    tree: DiscoveryTree, parent_id: str, problem: str, workspace: Path, seed: int
) -> AgentContext:
    """What the agent is given for one attempt resuming from ``parent_id`` (§3).

    ``workspace`` is the rollout's root directory; :func:`_run_attempt` replaces
    it with this attempt's own checkout when the rollout has a snapshot store.
    """
    history = _chain(tree, parent_id)
    observations = tuple(obs for node in history for obs in node.observations)
    return AgentContext(
        problem=problem,
        workspace=workspace,
        history=history,
        observations=observations,
        seed=seed,
    )


def _chain(tree: DiscoveryTree, node_id: str) -> tuple[Node, ...]:
    """The root → ``node_id`` path: the inherited history of an attempt from it."""
    chain = []
    current: str | None = node_id
    while current is not None:
        node = tree.node(current)
        chain.append(node)
        current = node.parent_id
    return tuple(reversed(chain))


def _run_attempt(
    agent: CodingAgent,
    evaluator: TaskEvaluator,
    snapshots: SnapshotStore | None,
    context: AgentContext,
    base_ref: str | None,
    name: str,
) -> tuple[str | None, EvalResult, str | None]:
    """One attempt in its own workspace, and the snapshot it leaves behind (§3).

    The workspace starts as the parent's saved state, belongs to this attempt
    alone for as long as it runs, and is captured before it is removed — a
    failed attempt included, because the state a failure left behind is the
    outcome its children would resume from, and the paper's policies classify
    failed branches rather than never seeing them (§B.2).

    A store that cannot record a snapshot stops the rollout rather than
    recording a node: that is a broken harness, not a failed attempt.
    """
    if snapshots is None:
        return (*_attempt(agent, evaluator, context), None)
    with snapshots.checkout(base_ref, name) as workspace:
        outcome = _attempt(agent, evaluator, replace(context, workspace=workspace))
        return (*outcome, snapshots.capture(workspace))


def _attempt(
    agent: CodingAgent, evaluator: TaskEvaluator, context: AgentContext
) -> tuple[str | None, EvalResult]:
    """One generation–evaluation attempt, which never raises.

    A worker that dies takes one node down, not the rollout: the attempt is
    recorded with no score and the tree keeps the failure, which is itself a
    replayable outcome — the paper's policies classify failed branches rather
    than never seeing them (§B.2). Evaluator crashes are already handled by
    :func:`safe_evaluate`; this adds the generation half.
    """
    try:
        artifact = agent.propose(context)
    except Exception as exc:  # noqa: BLE001 - the whole point is to not propagate
        return None, EvalResult.failed(f"agent raised {type(exc).__name__}: {exc}")
    return artifact.content, safe_evaluate(evaluator, artifact.content, context.workspace)


def _observations(result: EvalResult) -> tuple[str, ...]:
    """What this attempt adds to the context its children inherit.

    PAPER-GAP: §3 has a node record "evaluation diagnostics" and has the agent
    resume a parent with its "accumulated observations", without saying how the
    two relate. We record the diagnostics line as the node's one observation, so
    the chain a child inherits is the diagnostics of the attempts above it. The
    derived signals a policy reads — ``branch_promising`` and the rest (§B.2) —
    are computed from recorded scores in issue #11, not stored here. Revisit if
    the authors' implementation lands (see references/method.md).
    """
    return (result.diagnostics,) if result.diagnostics else ()


def main(argv: Sequence[str] | None = None) -> int:
    """Run a toy rollout and write it to disk, so the loop is runnable as it lands."""
    parser = argparse.ArgumentParser(description="Run a Dream-RSI rollout on the toy task.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="runs/toy",
        type=Path,
        help=f"where to write {TREE_FILENAME} and {ROUNDS_FILENAME} (default: runs/toy)",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel workers W (default: 4)")
    parser.add_argument("--rounds", type=int, default=6, help="decision rounds (default: 6)")
    args = parser.parse_args(argv)

    workspace = args.directory / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    result = run_rollout(
        agent=FakeAgent(),
        evaluator=ToyEvaluator(),
        policy=FirstEligiblePolicy(),
        problem="write the shortest program that defines solve(values)",
        workspace=workspace,
        snapshots=SnapshotStore(args.directory / STORE_DIRNAME),
        config=RolloutConfig(workers=args.workers, max_rounds=args.rounds),
    )
    result.save(args.directory)
    print(
        f"recorded {len(result.tree) - 1} attempt(s) over {len(result.rounds)} round(s), "
        f"stopped on {result.stop_reason}: {args.directory / TREE_FILENAME}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
