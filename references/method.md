# Paper → implementation map

Source: *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*, Zheng et al., [arXiv:2609.14858](https://arxiv.org/abs/2609.14858). The authors' code is **not released** as of this writing; their GitHub repo carries the paper, assets, and a "coming soon" release plan only. Everything below is read off the paper.

**The paper itself is in this repo**: [`paper/Dream-RSI.pdf`](paper/) (authoritative) alongside a grep-able text extraction. This file is a summary and a mapping onto modules — it is not a substitute. When an issue turns on a detail, open the PDF. See [`paper/README.md`](paper/README.md) for how to navigate between the two.

## Discovery tree node

Each non-root node `v` preserves:

| Field | Meaning |
|---|---|
| inherited history | saved workspace and accumulated observations from the parent |
| attempt outcome | the generated artifact plus evaluation diagnostics |
| filesystem snapshot | complete state after execution |
| `s_v` | score under the task-specific protocol |
| parent | exactly one primary parent (root, or the previous node) |

Single-parent means this is a tree, not a DAG. Do not add merge semantics.

→ `src/dream_rsi/tree.py`

## Replay simulator

A completed tree becomes a **frozen replay world**. A new policy navigating it reveals different subsets of the pre-recorded branches, in arbitrary order and with arbitrary parallelism.

Reveal is **strictly prefix-observable**: the policy's selectable set is the root (unopened branches) plus the leaves of branches it has already opened. The child lookup returns the pre-recorded children if they exist, otherwise the empty set. Nothing is generated during replay.

A batch repeats the root and nothing else. §3's action is a set `C ⊆ A(T)`, which cannot express the wide root fan-out §4 runs, so a batch here is a sequence — but only the root may appear in it twice, exactly as §B.2 tells the policy ("may contain several roots and/or one frontier from each opened branch", "no duplicate ids"). That is what keeps every recorded tree replayable: a non-root node never gains a second child, which is the one case §3 gives a reveal rule for ("for `v ≠ r`, `Child(v; T_i, T_i^{m,k})` is `v`'s unique recorded child"). A tree recorded elsewhere that does branch off a non-root node is still replayable, but only the round that first reveals such a node can reach the rest of its children, by naming it again in that same batch: eligibility is fixed when the round begins, and once the round ends the node is no longer a leaf and has left `A(T)` for good (issue #31).

→ `src/dream_rsi/replay.py`

## Scoring (Equation 1)

```
V_i^m = max_v(s_v) − β₁·N_i^m + β₂·(N_i^m / max{1, k_i^{m,★}})
```

- `max_v(s_v)` — attainment over revealed nodes
- `N_i^m` — count of revealed nodes, standing in for execution cost
- third term — batching efficiency reward, with `k_i^{m,★}` the number of decision rounds the replay completed at termination (§3), *not* the width of any one batch. It rewards the average number of attempts per round, so the same reveals grouped into fewer rounds score higher.

`PAPER-GAP:` the method text does not fix β₁ and β₂. Treat them as configuration with documented defaults, and make the sensitivity visible in any reported result.

→ `src/dream_rsi/scoring.py`

## Exploration policy

**Executable Python**, implementing a shared decision interface:

```python
class OptimalPolicy:
    def solve(self, question, budget=None): ...
```

Optional cross-cycle planning hook: `plan_grid(context) -> GridPlan`, deciding width and depth across cycles.

Observation helpers the paper names, from `see.policy.observation_signal`:
`branch_promising`, `branch_failed_hard`, `probe_improved_vs_parent`, `probe_improved_vs_baseline`.

→ `src/dream_rsi/policy.py`

## Dreaming loop

1. Deploy `π_t` online, collect `T_t`
2. Append `T_t` to history `H_t`
3. Evaluate `M` candidate versions over every tree in `H_t`, using recorded outcomes only
4. Policy-development agent revises the policy code from replay feedback
5. Select the best version — guaranteed no worse than the incumbent — as `π_{t+1}`

→ `src/dream_rsi/dream.py`, `src/dream_rsi/develop.py`, `src/dream_rsi/orchestrator.py`

## Components

- **Discovery agent** — the LLM generating candidates. The paper used Gemini-3.1 Pro and Gemini-3.7-Flash. Ours is adapter-shaped and provider-agnostic.
- **Evaluator** — task-specific scorer, deterministic correctness plus performance checks.
- **Orchestration layer** — branching, `W` parallel workers, stopping.
- **Policy-development agent** — the LLM that rewrites policy code.
- **SimResult / replay infrastructure** — curves, trajectories, per-round observations.

## Benchmarks in the paper

| Domain | Benchmarks |
|---|---|
| Algorithm engineering | Lasso regularization path — 17 synthetic instances, 6 held-out downstream tasks |
| Mathematical optimization | Sum–Difference; Circle Packing (n ∈ {26, 32}); Autocorrelation Inequalities (3 variants) |
| GPU kernel engineering | VGG16, LayerNorm, ConvDiv, ConvMax (KernelBench) |

These are not needed to build the skill. They matter for the validation issue, once the authors publish their discovered programs and reproduction scripts.

## Open gaps to revisit on code release

Grep the source for `PAPER-GAP:`. Known ones:

1. β₁ / β₂ values and their sensitivity.
2. Exact stopping criterion for an online rollout.
3. How the policy-development agent is prompted, and what `M` is in practice.
4. Whether filesystem snapshots are stored whole or as deltas at realistic tree sizes.
5. Tie-breaking rules in selection.
