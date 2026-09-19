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

A policy that plans its own grid — §B.2's optional ``plan_grid`` (issue #21) — is
asked for one before the first round, and the plan then bounds how many branches
the rollout may open, how deep it may refine one, and how wide a round is offered
to run: "the runtime grid is the hard bound: controller thresholds may use less,
but can never create branches or attempts beyond the effective plan."

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dream_rsi.adapters.agent import AgentContext, CodingAgent
from dream_rsi.adapters.evaluator import EvalResult, TaskEvaluator, safe_evaluate
from dream_rsi.adapters.fake_agent import FakeAgent
from dream_rsi.adapters.toy_evaluator import ToyEvaluator
from dream_rsi.cost import OnlineCost
from dream_rsi.policy import GridPlan, GridPlanningContext
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

    # PAPER-GAP: §B.2 has the runner validate a policy's grid plan against
    # ``context.hard_max_branch_count`` and ``context.hard_max_refine_count``
    # without saying what either is. We take §4's own largest reported grid — 32
    # parallel workspaces with up to 20 refinement steps — as the ceiling, so a
    # plan this runner honours is one the paper's own experiments would have run,
    # and anything wider or deeper is a policy that has to say so explicitly by
    # raising the cap. They bound the plan only: a rollout without one is bounded
    # by ``max_rounds`` and ``max_nodes`` as before. Revisit if the authors'
    # implementation lands (see references/method.md).
    max_branches: int = 32
    max_refinements: int = 20

    # Seeds the per-attempt seeds, so a rollout against a deterministic agent is
    # reproducible (AGENTS.md rule 5).
    seed: int = 0

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1 (§3: W ≥ 1), got {self.workers}")
        if self.max_branches < 1:
            # §B.2 validates ``1 <= W``, so a cap below one is a runner no plan
            # can satisfy rather than a runner that plans nothing.
            raise ValueError(f"max_branches must be at least 1, got {self.max_branches}")
        if self.max_refinements < 0:
            raise ValueError(f"max_refinements must not be negative, got {self.max_refinements}")
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
    """A completed online rollout: the tree it recorded and how it got there.

    ``cost`` is what it spent getting there (issue #18), which is not read off
    the tree: every attempt is a node and a discovery-agent call, but an attempt
    whose agent raised never reached the evaluator, so the calls and the
    measurements are two counts and only the rollout saw both.
    """

    tree: DiscoveryTree
    rounds: tuple[RoundRecord, ...]
    stop_reason: str
    cost: OnlineCost = field(default_factory=OnlineCost)

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
    rollout. The rest of the :class:`~dream_rsi.policy.OptimalPolicy` surface the
    policy-development agent writes against — ``solve`` and the observation
    signals — is issue #10, and the online loop calls none of it.

    One more method it does call where a policy defines it:
    ``plan_grid(context: GridPlanningContext) -> GridPlan``, §B.2's grid plan
    (issue #21), asked for once before the rollout opens anything. It is not
    declared here because it is optional — a protocol that required it would be
    one the baselines do not satisfy — so :func:`_plan` looks for it instead.
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
    # Before the grid exists, and once: §B.2's plan_grid "runs **before** a new
    # live grid is created" and "must never inspect a current episode's
    # outcomes", which a hook called per round could.
    plan = _plan(policy, config)
    # The grid is ``branch_count`` branches wide, so a round offered more
    # parallelism than that is offered workers the plan has nothing to spend them
    # on. §B.2: "controller thresholds may use less, but can never create
    # branches or attempts beyond the effective plan."
    width = config.workers if plan is None else min(config.workers, plan.branch_count)
    tree = DiscoveryTree.with_root(
        snapshot_ref=None if snapshots is None else snapshots.capture(workspace)
    )
    rounds: list[RoundRecord] = []
    deadline = None if config.max_seconds is None else time.monotonic() + config.max_seconds
    attempts = 0
    evaluations = 0
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

            eligible = _eligible(tree, plan)
            batch = tuple(policy.select(tree, eligible, width))
            _check_batch(batch, eligible)
            if not batch:
                stop_reason = STOP_EMPTY_BATCH
                break

            # Truncated rather than dropped whole: the budget is a ceiling on
            # attempts, and stopping a round short spends exactly what is left
            # instead of leaving it unused. The round still records the width the
            # policy asked for.
            admitted = _admitted(tree, batch, plan)
            scheduled = admitted if remaining is None else admitted[:remaining]
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
            for parent_id, (artifact, result, snapshot_ref, evaluated) in zip(
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
                evaluations += int(evaluated)
            attempts += len(produced)
            rounds.append(RoundRecord(index, len(batch), tuple(scheduled), tuple(produced)))

    return Rollout(
        tree=tree,
        rounds=tuple(rounds),
        stop_reason=stop_reason,
        # One discovery-agent call per attempt (§4), counted where the attempts
        # were scheduled rather than off the finished tree, which cannot say
        # which of its nodes reached an evaluator.
        cost=OnlineCost(agent_calls=attempts, evaluations=evaluations),
    )


def _plan(policy: ExplorationPolicy, config: RolloutConfig) -> GridPlan | None:
    """The grid this policy planned, or ``None`` where it plans none (§B.2).

    The hook is optional here (issue #21) — §B.2 requires every policy to
    implement it, and the three baselines in :mod:`dream_rsi.policy` do not — so
    a policy that does not define it is run ungridded, exactly as before. What it
    does define is validated here rather than trusted: §B.2 puts the check on the
    runner ("the runner validates ``1 <= W <= context.hard_max_branch_count`` and
    ``0 <= R <= context.hard_max_refine_count``"), and this is the runner.
    """
    plan_grid = getattr(policy, "plan_grid", None)
    if not callable(plan_grid):
        return None
    context = GridPlanningContext(
        hard_max_branch_count=config.max_branches,
        hard_max_refine_count=config.max_refinements,
        max_workers=config.workers,
    )
    plan = plan_grid(context)
    if not isinstance(plan, GridPlan):
        # §B.2: "It must always return a non-``None`` ``GridPlan``: do not
        # inherit the template stub and do not delegate grid choice to the
        # runner's fallback." A policy that defined the hook and answered with
        # something else planned nothing, and the rollout it would get is not the
        # one it asked for.
        raise ValueError(
            f"plan_grid must return a GridPlan, got {plan!r}; a policy that plans "
            f"no grid does not define plan_grid at all"
        )
    _count("branch_count", plan.branch_count, 1, config.max_branches)
    _count("refine_count", plan.refine_count, 0, config.max_refinements)
    return plan


def _count(name: str, value: int, low: int, high: int) -> None:
    """Reject a grid dimension the runner cannot open (§B.2's validation)."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(
            f"a grid plan's {name} must be a whole number in [{low}, {high}], got {value!r}"
        )


def _eligible(tree: DiscoveryTree, plan: GridPlan | None) -> tuple[str, ...]:
    """``A(T)``, narrowed to what the planned grid still has room for (§B.2).

    Narrowed rather than policed afterwards, because ``A(T)`` is the whole of
    what a policy is told it may do: a root the grid has no branch left for, or a
    leaf at the plan's full depth, is not an action this rollout can take, and
    offering it would have the policy spend rounds on selections the runner then
    dropped. With no plan the set is the paper's own, unchanged.

    PAPER-GAP: §B.2's grid is branches × attempts, in which the only thing a
    branch can do is get one attempt deeper, while a tree here lets one leaf be
    refined twice in a round and fork. We read ``refine_count`` as the depth
    bound it is described as — "the number of refinements allowed after each
    root" — so every frontier of a branch is capped at ``R`` refinements and a
    fork costs attempts rather than depth; bounding a branch's *nodes* instead
    would make the width of a fork the thing that ends a branch. Revisit if the
    authors' implementation lands (see references/method.md).
    """
    eligible = eligible_nodes(tree)
    if plan is None:
        return eligible
    opened = len(tree.children(tree.root_id))
    return tuple(
        node_id
        for node_id in eligible
        if (
            opened < plan.branch_count
            if node_id == tree.root_id
            else _depth(tree, node_id) <= plan.refine_count
        )
    )


def _admitted(
    tree: DiscoveryTree, batch: Sequence[str], plan: GridPlan | None
) -> tuple[str, ...]:
    """The selections of ``batch`` the grid has room to open this round (§B.2).

    Only the root needs this, and only for the repeats: :func:`_eligible` already
    took it out of ``A(T)`` once the branches were all opened, but a batch may
    name it several times — that is how one round opens several branches — and
    the last of those repeats is where a two-branch grid would become a
    three-branch one. Dropped rather than rejected, for the reason the node
    budget above truncates rather than failing: asking is legal, and the grid is
    a bound on what gets created.
    """
    if plan is None:
        return tuple(batch)
    room = plan.branch_count - len(tree.children(tree.root_id))
    admitted: list[str] = []
    for node_id in batch:
        if node_id == tree.root_id:
            if room <= 0:
                continue
            room -= 1
        admitted.append(node_id)
    return tuple(admitted)


def _depth(tree: DiscoveryTree, node_id: str) -> int:
    """How many attempts deep ``node_id`` sits: the root is 0, a branch's first 1."""
    depth = 0
    parent = tree.node(node_id).parent_id
    while parent is not None:
        depth += 1
        parent = tree.node(parent).parent_id
    return depth


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
) -> tuple[str | None, EvalResult, str | None, bool]:
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
        artifact, result, evaluated = _attempt(agent, evaluator, context)
        return artifact, result, None, evaluated
    with snapshots.checkout(base_ref, name) as workspace:
        artifact, result, evaluated = _attempt(
            agent, evaluator, replace(context, workspace=workspace)
        )
        return artifact, result, snapshots.capture(workspace), evaluated


def _attempt(
    agent: CodingAgent, evaluator: TaskEvaluator, context: AgentContext
) -> tuple[str | None, EvalResult, bool]:
    """One generation–evaluation attempt, which never raises.

    A worker that dies takes one node down, not the rollout: the attempt is
    recorded with no score and the tree keeps the failure, which is itself a
    replayable outcome — the paper's policies classify failed branches rather
    than never seeing them (§B.2). Evaluator crashes are already handled by
    :func:`safe_evaluate`; this adds the generation half.

    The third element is whether the evaluator was reached, which is what the
    rollout counts its evaluations from (issue #18). It is reported rather than
    inferred from the node: an evaluation that ran and failed also leaves a node
    with no score.
    """
    try:
        artifact = agent.propose(context)
    except Exception as exc:  # noqa: BLE001 - the whole point is to not propagate
        return None, EvalResult.failed(f"agent raised {type(exc).__name__}: {exc}"), False
    return (
        artifact.content,
        safe_evaluate(evaluator, artifact.content, context.workspace),
        True,
    )


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
