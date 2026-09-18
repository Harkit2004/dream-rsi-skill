# dream-rsi-skill

An agent skill that implements the **Dream-RSI** exploration loop — recursive self-improvement of *exploration policies* via replay simulators built from accumulated discovery history.

Based on [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858) (Zheng et al., 2026). The paper's reference implementation is not yet released; this repo is an independent implementation built from the paper.

## The idea in one paragraph

A discovery run records a tree: each node is one attempt by a coding agent, with its workspace, its artifact, its evaluation diagnostics, and a score. Once frozen, that tree is a **replay world**. A candidate exploration policy can be re-run against it for free — revealing pre-recorded branches in whatever order and batch shape it likes — and scored on what it attained versus how many nodes it had to open. That gives cheap off-policy feedback, which a policy-development agent uses to rewrite the policy's code. The improved policy goes back online, produces a new tree, and the simulator pool grows.

The coding agent underneath is never modified. All self-improvement happens in the orchestration layer.

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
| `references/` | Paper → implementation mapping, loaded on demand |
| `src/dream_rsi/` | The implementation |
| `tests/` | Fixtures and tests (written before implementation — see AGENTS.md) |

## License

MIT
