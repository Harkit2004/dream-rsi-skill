<!--
The prompt the policy-development agent is given (§3 step 4; §B.2, Listing 2).

It is one file, and one file only, so that it can be read, diffed and compared
against the authors' own prompt when their code lands — see the PAPER-GAP note
in dream_rsi/develop.py, which also lists the placeholders substituted below
(policy_name, version, score, source, replays, history, rejected) and is the
only code that renders this file.
-->

You are improving one **prefix-only exploration policy**. Reply with one Python
module and nothing else: no explanation, no commentary, no markdown fences. The
module is the whole of your output and the only thing that changes — the coding
agent, the evaluator and the execution interfaces stay fixed.

## The environment

A frozen replay world: a discovery tree an earlier online rollout recorded. Your
policy opens the root, which starts a further branch, or refines a revealed leaf,
which continues one. Each opened node reveals one recorded node and costs one
attempt. You see only what you have revealed; unrevealed scores do not exist for
you, and the world generates nothing new.

## Objective: quality, work, and parallelism

A replay is scored by Equation 1:

    V = max_v(s_v) - B1 * N + B2 * (N / max(1, k))

`max_v(s_v)` is the best score among the nodes you revealed, `N` is how many you
revealed, and `k` is how many decision rounds you took. So: reveal few nodes, and
when several are worth revealing, reveal them in the *same* round — a batch costs
one round however wide it is, which is the only way `N / k` rises. Choose only
promising attempts, but batch independent promising attempts whenever you can.

A local implementation failure does not by itself prove that its parent direction
is poor. Weigh recovery against new branches and ordinary refinements while
keeping batches parallel.

## The interface

Subclass the base policy and implement `choose`; `select`, the per-rollout state
and the `solve` loop are inherited:

    from dream_rsi.policy import OptimalPolicy as BasePolicy


    class $policy_name(BasePolicy):
        def __init__(self, config=None):
            super().__init__(config)

        def choose(self, tree, live, width):
            ...

A version that inherits nothing must instead define
`select(self, tree, eligible, width)` itself, and take its configuration from one
positional argument in `__init__`.

* `tree` is a `dream_rsi.tree.DiscoveryTree` holding the revealed nodes only.
  Each `dream_rsi.tree.Node` carries `.id`, `.parent_id`, `.score` — `None` where
  that attempt failed to evaluate — and `.observations`.
* `live` is the eligible set: the root first, then the revealed leaves in id
  order, with the nodes already shown to have no continuation removed.
* `width` is `W`, the most attempts one round may run at once.
* Return up to `width` node ids from `live`, or `()` to stop. The root may appear
  several times in one batch: each occurrence opens a further branch.
* `dream_rsi.policy` also exports the signals §B.2 names — `branch_promising`,
  `branch_failed_hard`, `probe_improved_vs_parent`, `probe_improved_vs_baseline`.

### Planning the grid (optional)

You may also plan the grid the *online* rollout that deploys you will run on:

    from dream_rsi.policy import GridPlan


    class $policy_name(BasePolicy):
        def plan_grid(self, context):
            return GridPlan(branch_count=..., refine_count=..., reason="...")

It is asked for once, before that rollout opens anything, and never during one —
it makes no within-episode decision and is not called while you are replaying a
frozen world, so it cannot read a current episode's outcomes. `branch_count` is
how many branches may be opened off the root and `refine_count` how many
refinements are allowed after each of them, so a branch is at most
`refine_count + 1` attempts deep. Both are whole numbers the runner validates
against `context`: `1 <= branch_count <= context.hard_max_branch_count` and
`0 <= refine_count <= context.hard_max_refine_count`. `context.max_workers` is
the `W` that grid will be explored under. A plan outside those bounds, or a
`plan_grid` that returns anything but a `GridPlan`, stops the rollout, so return
one on every path and give it a short factual `reason`.

The grid is a hard bound and not a target: it can never create branches or
attempts beyond the plan, and `choose` still decides which of the nodes on offer
to open, refine or stop at. Omit the method entirely if you do not want to plan
one — a policy without it runs unbounded by any grid, which is what the version
you are revising does unless its source says otherwise.

## Hard constraints

* **Prefix-only.** Never use unrevealed scores, a true optimum, hardcoded winning
  node ids, absolute score targets, or anything but the revealed tree. Every
  prune, widen, deepen, batch and stop decision must be explainable from the
  prefix you were shown.
* Read exactly one scalar knob in `__init__` —
  `beta = float(self.config.get("beta", <sensible default>))` — and route every
  behavioural threshold through it. High beta means more width, deeper patience
  and weaker pruning; low beta means fewer attempts, earlier stopping and
  stronger pruning. Never change beta from what you observe inside a rollout.
* A shallow weak score is not enough to discard a branch: deeper attempts
  recover, and a later success reopens one. A repairable failure must not erase
  the branch's earlier successful anchor.
* Never sample randomly, and do not fall back to a batch of one merely because
  its top candidate is clear.
* Always terminate. Stop when you select nothing; do not assume a budget cap.
* A batch holds ids that were legal before the call, at most `width` of them, and
  no duplicates other than repeated selections of the root.
* Import nothing but the standard library and `dream_rsi`. Your module runs in a
  sandbox that refuses the network, subprocesses, and writes outside its own
  scratch directory, and that caps its CPU, memory and wall clock: a version that
  oversteps scores nothing at all.

## The version you are revising

Version `$version` scored `$score` on average across the replay worlds.

Its source:

```python
$source
```

What it did on each world:

$replays

Earlier versions this round:

$history

Output of yours that was refused, and why — do not repeat it:

$rejected

## Now

Write the next version: the complete module, revising the source above in the
light of what it actually did. Change the decisions the trajectories show going
wrong; keep what the scores show working.
