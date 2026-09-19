---
name: dream-rsi
description: Self-improving exploration for expensive search — record a discovery run as a tree of scored attempts, replay that frozen tree to score candidate exploration policies off-policy, rewrite the policy's Python from the replay feedback, and redeploy it (the Dream-RSI loop). Use when an automatic evaluator scores every candidate, one evaluation is expensive (compilation, benchmarking, long runs), and the same search will be run again — algorithm engineering, GPU kernel optimization, mathematical optimization, configuration or hyperparameter search — so that the search strategy itself improves across runs, not just the solutions it finds. Not for one-shot coding work: writing, debugging, reviewing or explaining a program, not for searching or grepping a codebase, and not for a search with no automatic scorer or no repeat runs, which leaves nothing for a replay simulator to be built from.
---

# Dream-RSI

Self-improving exploration for expensive discovery tasks. The coding agent that writes candidate solutions is never modified — the thing that improves is the *policy deciding what to explore next*.

## When this applies

Use it when all of these hold:

- The task is search over programs or configurations, scored by an automatic evaluator.
- Evaluating a candidate is expensive (compilation, benchmarking, long runs).
- You expect to run discovery more than once, so history accumulates.

If the task is a one-shot question, or there is no automatic scorer, this is the wrong tool — the replay simulator has nothing to be built from. Writing a function, fixing a bug, reviewing a diff, explaining a codebase and answering a question about a library are all one-shot: they get answered directly, not with a discovery loop. A search you will run exactly once is the same — there is no second deployment for an improved policy to be deployed into.

## The loop

One cycle `t`, as `dream_rsi.run` drives it:

1. **Online rollout.** Deploy the current policy `π_t`. The orchestrator branches, runs `W` workers in parallel over at most `K₁` decision rounds, and records every attempt as a node. Result: discovery tree `T_t`.
2. **Grow the pool.** Append `T_t` to the history `H_t`. The pool is a directory of recorded trees, so a run started tomorrow dreams over every tree the runs before it recorded.
3. **Dream and refine — one interleaved pass.** Version `v0` is the incumbent itself, replayed over every world in `H_t` and scored. The policy-development agent then reads `v0`'s replay trajectories and scores and rewrites the policy's Python into `v1`, which is scored on the same worlds under the same conditions, and so on up to `M` versions. Replay makes no agent calls and no evaluator calls — outcomes are retrieved from the recording.
4. **Select and redeploy.** Take the highest-scoring version. Where the incumbent has a `V⁰`, that score is a strict floor: a version merely tying it does not displace it. Where the incumbent scored nothing on this history there is no floor to clear, so a version that did score wins. The winner is `π_{t+1}`, and the next cycle deploys it.

Dreaming and refining are not two passes over the history, one after the other — each version is written *from* the previous one's replay feedback, which is why step 3 is a single call in the code.

## Key invariants

- **Prefix-observable replay.** A policy in replay sees only what it has revealed. It may select the root or the leaves of branches it already opened — nothing else.
- **No generation during replay.** If a node has no recorded continuation, the child set is empty. That is a real signal, not an error to paper over.
- **Determinism.** Same tree, same policy, same seed produces an identical trajectory, byte for byte.
- **Policies are code.** An exploration policy is executable Python — not a prompt, not a parameter vector. A module defines `class OptimalPolicy(...)`, or names another class by assigning `NAME`, and its instances answer `select(tree, eligible, width)` with the batch to open next or `()` to stop. Subclassing `dream_rsi.policy.OptimalPolicy` supplies `select` and the paper's `solve(question, budget=None)` loop; a subclass implements `choose`.
- **Policy code is sandboxed.** Both halves of the loop run model-written policy code, so both run it in a child process under wall-clock, CPU, memory and disk limits. There is no path by which policy source runs in the harness process — deployment included.

## Scoring a replayed policy

```
V_i = max_v(s_v) − β₁·N_i + β₂·(N_i / max(1, k_i))     for one world i
V   = mean of V_i over the worlds in H_t
```

`max_v(s_v)` is the best score attained among revealed nodes, `N_i` is the number of revealed non-root nodes (the cost proxy), and `k_i` is the number of decision rounds the replay completed — so the last term rewards a policy for opening nodes in batches rather than one per round. `s_v` is canonical larger-is-better; a lower-is-better task metric is mapped onto it by the evaluator's `ScoreDirection`, so do not assume the raw numbers a task reports run upward. β₁ and β₂ are configuration: see `references/method.md` for the parameter discussion.

## Running it

```bash
python -m dream_rsi.run --cycles 3 runs/loop   # the toy task, no model calls
```

A run writes one directory per cycle, each holding that cycle's tree, the policy it deployed, the policy it selected, and a record written last of all. The record is what says a cycle finished: running the same directory again resumes from the cycles it finds, so a crash in cycle 5 costs cycle 5 and not cycles 1–4. The report at the end prices each half of every cycle separately — discovery-agent calls online against model calls and revealed nodes offline — because the paper's claim is the ratio between them, not a single total.

## Working on this repo

Read [AGENTS.md](AGENTS.md) before changing anything. Test first, minimal diff, and mark every place the paper is silent with `PAPER-GAP:`.
