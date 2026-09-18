---
name: dream-rsi
description: Run a Dream-RSI discovery loop — record exploration as a discovery tree, replay that tree as a frozen simulator to score candidate exploration policies off-policy, rewrite the policy from replay feedback, and redeploy. Use when a task needs repeated expensive search (algorithm engineering, kernel optimization, mathematical optimization) and the exploration strategy itself should improve across runs, not just the solutions.
---

# Dream-RSI

Self-improving exploration for expensive discovery tasks. The coding agent that writes candidate solutions is never modified — the thing that improves is the *policy deciding what to explore next*.

## When this applies

Use it when all of these hold:

- The task is search over programs or configurations, scored by an automatic evaluator.
- Evaluating a candidate is expensive (compilation, benchmarking, long runs).
- You expect to run discovery more than once, so history accumulates.

If the task is a one-shot question, or there is no automatic scorer, this is the wrong tool — the replay simulator has nothing to be built from.

## The loop

1. **Online rollout.** Deploy the current policy `π_t`. The orchestrator branches, runs `W` workers in parallel, and records every attempt as a node. Result: discovery tree `T_t`.
2. **Grow the pool.** Append `T_t` to history `H_t`.
3. **Dream.** Freeze each tree in `H_t` into a replay world. Run `M` candidate policy versions against them. No agent calls, no evaluator calls — outcomes are retrieved from the recording.
4. **Refine.** The policy-development agent rewrites the policy's Python from the replay trajectories and score comparisons.
5. **Select.** Take the best version, with the incumbent as a floor, and redeploy as `π_{t+1}`.

## Key invariants

- **Prefix-observable replay.** A policy in replay sees only what it has revealed. It may select the root or the leaves of branches it already opened — nothing else.
- **No generation during replay.** If a node has no recorded continuation, the child set is empty. That is a real signal, not an error to paper over.
- **Determinism.** Same tree, same policy, same seed produces an identical trajectory.
- **Policies are code.** An exploration policy is executable Python implementing `OptimalPolicy.solve(question, budget=None)` — not a prompt, not a parameter vector.

## Scoring a replayed policy

```
V = max_v(s_v) − β₁·N + β₂·(N / max(1, k))
```

`max_v(s_v)` is the best score attained among revealed nodes, `N` is the number of nodes revealed (the cost proxy), and `k` is the batch width — the last term rewards a policy for opening nodes in parallel rather than serially. See `references/method.md` for the parameter discussion.

## Working on this repo

Read [AGENTS.md](AGENTS.md) before changing anything. Test first, minimal diff, and mark every place the paper is silent with `PAPER-GAP:`.
