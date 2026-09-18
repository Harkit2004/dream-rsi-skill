# dream-rsi-skill

An agent skill that implements the **Dream-RSI** exploration loop — recursive self-improvement of *exploration policies* via replay simulators built from accumulated discovery history.

Based on [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858) (Zheng et al., 2026). The paper's reference implementation is not yet released; this repo is an independent implementation built from the paper.

## The idea in one paragraph

A discovery run records a tree: each node is one attempt by a coding agent, with its workspace, its artifact, its evaluation diagnostics, and a score. Once frozen, that tree is a **replay world**. A candidate exploration policy can be re-run against it for free — revealing pre-recorded branches in whatever order and batch shape it likes — and scored on what it attained versus how many nodes it had to open. That gives cheap off-policy feedback, which a policy-development agent uses to rewrite the policy's code. The improved policy goes back online, produces a new tree, and the simulator pool grows.

The coding agent underneath is never modified. All self-improvement happens in the orchestration layer.

## Status

Early. See the [issues](../../issues) for the implementation order — they are phased, and each one names what blocks it.

## Layout

| Path | What |
|---|---|
| `SKILL.md` | Skill entry point (triggers, workflow) |
| `references/` | Paper → implementation mapping, loaded on demand |
| `src/dream_rsi/` | The implementation |
| `tests/` | Fixtures and tests (written before implementation — see AGENTS.md) |

## License

MIT
