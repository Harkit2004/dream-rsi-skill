# dream-rsi-skill

An agent skill that implements the **Dream-RSI** exploration loop — recursive self-improvement of *exploration policies* via replay simulators built from accumulated discovery history.

Based on [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858) (Zheng et al., 2026). The paper's reference implementation is not yet released; this repo is an independent implementation built from the paper.

## The idea in one paragraph

A discovery run records a tree: each node is one attempt by a coding agent, with its workspace, its artifact, its evaluation diagnostics, and a score. Once frozen, that tree is a **replay world**. A candidate exploration policy can be re-run against it for free — revealing pre-recorded branches in whatever order and batch shape it likes — and scored on what it attained versus how many nodes it had to open. That gives cheap off-policy feedback, which a policy-development agent uses to rewrite the policy's code. The improved policy goes back online, produces a new tree, and the simulator pool grows.

The coding agent underneath is never modified. All self-improvement happens in the orchestration layer.

## Quickstart

Three cycles of the loop on a toy search task: pick the `(width, depth)` plan with the
best throughput inside a cost budget. No API key, no provider and no network — the
discovery agent is scripted and the policy-development agent answers from a list, so
what runs is the orchestration layer and nothing else. From a clean checkout, after
`pip install -e .`:

```bash
python -m dream_rsi.run --cycles 3 runs/quickstart
```

```text
run: 3 cycle(s), best score 19.0000
cost split: 16 online agent call(s) against 21 policy-development call(s) and 48 replay cell(s) revealing 244 node(s)
  15.25 replayed node(s) per online agent call
pool: 3 tree(s), 19 node(s), 9319 byte(s) on disk

cycle     best  online.calls  online.total  online.evals  online.s  dream.calls  dream.cells  dream.reveals  dream.s  selected
    0  19.0000             8             8             8      0.06            7            8             52     0.42        v1
    1  19.0000             4            12             4      0.05            7           16             80     0.80        v0
    2  19.0000             4            16             4      0.06            7           24            112     1.17        v0

selection:
  cycle 0: selected v1 at V^m=18.9650 in place of the incumbent v0: beats V^0=18.9300 by +0.0350; 8 version(s) over 1 world(s)
  cycle 1: retained the incumbent v0 at V^0=18.9658: no version beat it, best other version v1 scored 18.9658 (+0.0000); 8 version(s) over 2 world(s)
  cycle 2: retained the incumbent v0 at V^0=18.9661: no version beat it, best other version v1 scored 18.9661 (+0.0000); 8 version(s) over 3 world(s)

policy:
  cycle 0: v0 -> v1
    --- cycle_000/policy.py
    +++ cycle_000/next_policy.py
    @@ -1,5 +1,6 @@
    -from dream_rsi.policy import BreadthFirstPolicy
    +from dream_rsi.policy import BudgetAwarePolicy
     
     
    -class OptimalPolicy(BreadthFirstPolicy):
    -    pass
    +class OptimalPolicy(BudgetAwarePolicy):
    +    def __init__(self, config=None):
    +        super().__init__({'beta': 0.5})
  cycle 1: unchanged
  cycle 2: unchanged
```

Cycle 0 deploys a breadth-first baseline that opens every branch it is offered: eight
discovery-agent calls, and Equation 1 charges it for all eight. Its tree is then frozen
and replayed — eight candidate versions over it, for no agent calls at all — and one of
them reaches the same best score while declining to spend once the score stops moving.
So cycle 1 deploys **different code**, and costs four calls instead of eight. The diff
under `policy:` is the loop working; the `cost split` line is what it cost.

Running the same command again resumes: cycles with a record are read back, not redone.

## Status

Early — nothing is implemented yet. The [issues](../../issues) hold the plan, phased and ordered, and each one names what blocks it. Issues labelled `blocked` have open blockers: don't start them, because the interfaces they depend on aren't settled.

| Phase | What lands | Issues |
|---|---|---|
| 0 — Foundations | Tree schema, evaluator and agent protocols | [#1](../../issues/1)–[#3](../../issues/3) |
| 1 — Record | Orchestrator, workspaces, first real trees | [#4](../../issues/4)–[#6](../../issues/6) |
| 2 — Replay | Frozen replay world, Equation 1, determinism | [#7](../../issues/7)–[#9](../../issues/9) |
| 3 — Dream | Policy interface, dreaming sweep, sandbox, refinement, selection | [#10](../../issues/10)–[#15](../../issues/15) |
| 4 — Loop | The full RSI driver, pool management, cost accounting | [#16](../../issues/16)–[#18](../../issues/18) |
| 5 — Surface | Skill description, quickstart, grid planning, validation | [#19](../../issues/19)–[#23](../../issues/23) |

Start at [#1](../../issues/1). It defines the data structure everything else reads and writes, so it genuinely has to come first.

## Where this differs from the paper

The reference implementation is unreleased, so wherever the paper is silent this repo makes a choice and marks it:

```bash
grep -rn "PAPER-GAP:" src/ tests/
```

[#23](../../issues/23) tracks those, and [#22](../../issues/22) is the standing task to revisit every one of them when the authors publish their code.

## Layout

| Path | What |
|---|---|
| `SKILL.md` | Skill entry point (triggers, workflow) |
| `references/method.md` | Paper → implementation mapping, loaded on demand |
| `references/paper/` | The paper itself — PDF (authoritative) plus a grep-able text extraction |
| `src/dream_rsi/` | The implementation |
| `tests/` | Fixtures and tests (written before implementation — see AGENTS.md) |

## License

MIT
