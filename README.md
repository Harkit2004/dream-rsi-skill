# dream-rsi-skill

[![ci](https://github.com/Harkit2004/dream-rsi-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/Harkit2004/dream-rsi-skill/actions/workflows/ci.yml)

An agent skill for expensive searches that will be run more than once. It records every
attempt of a search as a tree, replays that tree for free to try out better ways of
searching, and deploys the best one on the next run. The coding agent that writes the
candidates is never modified; what improves is the **policy that decides what to try next**.

It implements the loop from [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858)
(Zheng et al., 2026). The paper's code is not released; this is an independent implementation
built from the paper.

**Contents:** [What it is for](#what-it-is-for) · [Install](#install) · [Quickstart](#quickstart) ·
[Use it from your agent](#use-it-from-your-agent) · [Your own task](#your-own-task) ·
[A real run](#a-real-run) · [Checking on a run](#checking-on-a-run) · [How it works](#how-it-works) ·
[Status](#status)

## What it is for

**In one sentence:** when finding a good answer means trying many candidates, and each try is
slow or costly, this skill learns from the tries already made which ones are worth making, so
the next search wastes fewer.

**A picture.** You are perfecting a cake recipe, and every test bake takes an hour. After a few
weekends you have a notebook of everything you tried and how it turned out. Between weekends
you re-read the notebook and work out a smarter order to try things in: stop early when a line
of changes has gone flat, try several variations at once when it is promising. Reading the
notebook costs nothing; only baking costs. Dream-RSI is that habit for an AI coding agent: the
notebook is a recorded tree of attempts, and re-reading it is the "dreaming".

**Does my task fit?** Ask three questions. You need a yes to all three.

1. **Can a program grade an answer by itself, with a number?** A speed test, a count of tests
   passed, an error rate, a checker for a maths problem. If a person has to look and judge,
   the answer is no.
2. **Is each grading slow or costly?** It takes minutes, compute or money, or the attempt
   itself does (a model has to write it). If you could try a million answers in a second, you
   do not need this.
3. **Will you do this kind of search again?** What the skill improves is *how* it searches,
   and that only pays off the next time.

| You would say | Fits? | Why |
|---|---|---|
| "Make this function as fast as possible; here is the benchmark." | Yes | A number, a slow test, and you will keep optimizing |
| "Find the best settings for this compiler / model / pipeline." | Yes | Each setting is a candidate and each trial is costly |
| "Find the best arrangement": a packing, a schedule, a layout, with a checker | Yes | The checker is the grader |
| "Build me a login page." / "Fix this bug." | No | One job, done once: nothing to search |
| "Review this code." / "Which database should we use?" | No | No program can grade the answer |
| "Tune this once and never again." | No | There is no next search to do better in |
| "I have no grader, but I could make one up." | No | See below |

That last row is the common mistake. A grader invented just so the skill can be used is
usually easy to satisfy on the first attempt, and then there is nothing left to search. Start
from a measurement you already trust.

In the method's own words, the three questions are: an automatic evaluator exists, one
evaluation is expensive, and the search will be run again. The fields the paper works in are
algorithm engineering, GPU kernel optimization and mathematical optimization.

## Install

Two commands. The first installs the Python package the agent runs; the second writes the
skill directory your agent loads.

```bash
pip install git+https://github.com/Harkit2004/dream-rsi-skill
python -m dream_rsi.install --host claude    # or: opencode, cursor
```

The second command prints the path it wrote. By default that is the host's user-global skills
directory:

| Host | user-global (default) | `--project DIR` |
|---|---|---|
| Claude Code | `~/.claude/skills/dream-rsi/` | `DIR/.claude/skills/dream-rsi/` |
| OpenCode | `~/.config/opencode/skills/dream-rsi/` | `DIR/.opencode/skills/dream-rsi/` |
| Cursor | `~/.cursor/skills/dream-rsi/` | `DIR/.cursor/skills/dream-rsi/` |

Check that the host sees it: `opencode debug skill` lists `dream-rsi`; Claude Code and Cursor
list skills at the start of a new session.

**Requirements.** Python 3.10 or newer, on **Linux or macOS**. On Windows, use WSL: policy code
is model-written and runs in a POSIX sandbox with resource limits, and the loop refuses to
start on a platform without one. Under WSL, install the package inside WSL, and make sure the
coding-agent CLI you will name in `DREAM_RSI_AGENT_CMD` runs and is logged in *inside WSL*
too. A Windows-side CLI reached from WSL is not reliable enough to run a loop on.

<details>
<summary>What the installer writes, and why not to clone into a skills directory</summary>

It writes a thin directory: `SKILL.md`, `references/method.md` and
`references/task_template.py`, the files `SKILL.md` links, and nothing else.

User-global is the default on purpose. A project-local copy sits inside the host project, so
its own formatters, linters and protected-path rules apply to it. For the same reason, don't
clone this repo into a skills directory: that puts `src/`, `tests/`, the paper's PDF and CI
config inside the host project, and its gates then fail on them.

An existing skill directory that differs from what would be written (a local edit, or a
whole-repo clone) is refused unless you pass `--force`; one that is already identical is left
alone.

</details>

## Quickstart

One command runs three cycles of the loop on a built-in toy task, with no API key, no model
and no network. It works after the install above, or from a checkout after `pip install -e .`:

```bash
python -m dream_rsi.run --cycles 3 runs/quickstart
```

It prints a report. The part to look at is the cost: cycle 0 spends **8** agent calls, and
cycles 1 and 2 reach the same best score with **4**, because the loop rewrote its own policy
in between:

```
cycle     best  online.calls  selected
    0  19.0000             8        v1
    1  19.0000             4        v0
    2  19.0000             4        v0
```

<details>
<summary>The full report, and how to read it</summary>

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

The toy task is to pick the `(width, depth)` plan with the best throughput inside a cost
budget. The discovery agent is scripted and the policy-development agent answers from a list,
so what runs is the orchestration layer and nothing else.

Cycle 0 deploys a breadth-first baseline that opens every branch it is offered: eight
discovery-agent calls, and Equation 1 charges it for all eight. Its tree is then frozen
and replayed, eight candidate versions over it for no agent calls at all, and one of
them reaches the same best score while declining to spend once the score stops moving.
So cycle 1 deploys **different code**, and costs four calls instead of eight. The diff
under `policy:` is the loop working; the `cost split` line is what it cost.

</details>

Running the same command again resumes: cycles with a record are read back, not redone.

## Use it from your agent

Once installed, give your agent a task that fits and say to use the skill, for example:

> Use the dream-rsi skill to make `solve()` faster. `python bench.py` scores a candidate.

The skill then has the agent do five things, described in full in [SKILL.md](SKILL.md):

1. Check the task fits (the three conditions above).
2. Check `python -m dream_rsi.run --help` works.
3. Write a task file from `references/task_template.py`: the problem, your scorer's command,
   and which way the score runs.
4. Start a small run, detached, and poll it with `--report`.
5. Report the best score, what it cost, and whether the loop changed its policy.

## Your own task

`python -m dream_rsi.run --task PATH DIRECTORY` runs the same loop on a task of your own.
`PATH` is a Python file that defines `task()` and returns a `dream_rsi.task.Task`:

```python
from dream_rsi.task import Task


def task() -> Task:
    return Task(
        agent=...,      # writes candidates: anything with propose(context) -> Artifact
        evaluator=...,  # scores them: a TaskEvaluator
        developer=...,  # rewrites the exploration policy: anything with revise(context) -> str
        problem="what is being optimised, and which way the score runs",
        # policy=...    # optional: the starting policy's source (default: breadth-first)
    )
```

The file is a Python file, not a config format, because each role is a protocol and you may
need your own class for any of them. It is **your** code and runs in the command's own process;
the exploration *policies* the loop writes are model-written and run only in the sandbox.
A run directory is tied to its `problem` text and starting policy, so resuming it under a
different task is refused. Without `--task` the command runs the toy task above, unchanged.

## A real run

The quickstart proves the orchestration layer with scripted roles on both sides.
[`examples/packing`](examples/packing) is a task where a **model writes the candidates** and a
**scorer measures something real**: place ten points in the unit square so that the two closest
are as far apart as possible. A 4-by-3 grid scores `1/3`, a staggered arrangement `5/12`, and the
best known is about `0.4213`. The scorer (`score.py`) is deterministic and takes well under a
second, so a run's time and cost are the model's. It is in the spirit of §4's
mathematical-optimization domain, and its budgets are far smaller than §4's (see the `PAPER-GAP:` in
`task.py`), so its scores are not comparable to the paper's.

With any coding-agent CLI that has a non-interactive mode, it is two commands:

```bash
export DREAM_RSI_AGENT_CMD="claude -p --permission-mode acceptEdits"   # your CLI; there is no default
python -m dream_rsi.run --task examples/packing/task.py --cycles 2 --workers 3 --rounds 3 runs/packing
```

The variable is required: without it the example refuses to start and names it, because a default
would be a provider chosen for you. Let the CLI edit files without asking, since the loop cannot
answer a permission prompt. The run prints the report described above (`--report` checks on it),
and its directory holds the model-driven tree, each cycle's policy, and the revision that was
replayed, scored and selected or rejected.

**No run against a real model has been recorded here yet.** What is tested is everything up to the
CLI boundary: `tests/test_example_packing.py` drives this same `task.py` and `score.py` with
stand-in CLIs. When one real run has been done, its report belongs in this section, labelled as one
run and not a benchmark.

## Checking on a run

A real run outlasts a tool call, so launch it detached and check on it from a later one.
Running the command again would *resume* it, so ask for a report instead:

```bash
python -m dream_rsi.run --report runs/mine
```

It prints the run's report, headed by a `status:` line, and never starts a cycle or writes
to the directory. The exit status says which of four things is true:

| Exit | Meaning |
|---|---|
| `0` | finished: every cycle the last session asked for is on disk |
| `3` | running: a process holds the directory now (the status line names its cycle and pid) |
| `4` | stopped short: nothing is running it and fewer cycles are on disk than were asked for |
| `2` | not a run directory |

Only one process may write to a run directory at a time; a second `python -m dream_rsi.run`
on a directory another is running is refused with exit `2`.

## How it works

A discovery run records a tree: each node is one attempt by a coding agent, with its
workspace, its artifact, its evaluation diagnostics, and a score. Once frozen, that tree is a
**replay world**. A candidate exploration policy can be re-run against it for free, revealing
pre-recorded branches in whatever order and batch shape it likes, and scored on what it
attained versus how many nodes it had to open. That gives cheap off-policy feedback, which a
policy-development agent uses to rewrite the policy's code. The improved policy goes back
online, produces a new tree, and the pool of replay worlds grows.

[`references/method.md`](references/method.md) maps the paper's method onto the modules here.

## Status

Phases 0–4 and 6 have landed and the loop runs end to end, on the toy task or on your own — run the [Quickstart](#quickstart) above to see it. The [issues](../../issues) hold what remains. Issues labelled `blocked` have open blockers: don't start them, because the interfaces they depend on aren't settled.

| Phase | What lands | Issues | Status |
|---|---|---|---|
| 0 — Foundations | Tree schema, evaluator and agent protocols | [#1](../../issues/1)–[#3](../../issues/3) | landed |
| 1 — Record | Orchestrator, workspaces, first real trees | [#4](../../issues/4)–[#6](../../issues/6) | landed |
| 2 — Replay | Frozen replay world, Equation 1, determinism | [#7](../../issues/7)–[#9](../../issues/9) | landed |
| 3 — Dream | Policy interface, dreaming sweep, sandbox, refinement, selection | [#10](../../issues/10)–[#15](../../issues/15) | landed |
| 4 — Loop | The full RSI driver, pool management, cost accounting | [#16](../../issues/16)–[#18](../../issues/18) | landed |
| 5 — Surface | Skill description, quickstart, grid planning, validation | [#19](../../issues/19)–[#23](../../issues/23) | in progress |
| 6 — Harness | Usable as a skill inside an agent harness: run your own task, command-backed roles, install, workflow | [#66](../../issues/66)–[#77](../../issues/77) | landed; no real-model run recorded yet (see [A real run](#a-real-run)) |

Run the [Quickstart](#quickstart) to see the loop work, or read [SKILL.md](SKILL.md) for the skill interface. [Open issues](../../issues) are what's left.

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

## Contributing

Read [AGENTS.md](AGENTS.md) before changing anything: test first, minimal diff, and mark every
place the paper is silent with `PAPER-GAP:`. [CLAUDE.md](CLAUDE.md) adds the Claude-Code-specific
bits.

## License

MIT
