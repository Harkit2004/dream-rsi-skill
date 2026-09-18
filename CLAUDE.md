# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first — it holds the working rules for this repo, and they are not repeated here.** The short version: test before implementation, test behaviour rather than lines, minimal diff, one issue per PR, mark unspecified paper details with `PAPER-GAP:`, keep replay deterministic.

This file only adds the Claude-Code-specific bits.

## Orientation

Start with [`references/method.md`](references/method.md) — it maps the paper's method onto this codebase's modules. Don't infer the design from the code alone; the code is incomplete by design and the issues describe what's missing.

The paper is vendored at [`references/paper/`](references/paper/): the PDF is authoritative, and `Dream-RSI-fulltext.md` next to it is a text extraction for grepping. **Read the PDF whenever an issue turns on a specific detail** — an equation, a threshold, an experimental setup. `method.md` is a summary written by someone who could be wrong; the PDF is how you check. The text dump has no math in it at all, so never take an equation from there.

## Where things live

- `src/dream_rsi/tree.py` — node + tree schema, persistence. Everything depends on this; change it carefully.
- `src/dream_rsi/replay.py` — the frozen replay world, prefix-observable reveal.
- `src/dream_rsi/scoring.py` — Equation 1. Keep it a pure function.
- `src/dream_rsi/policy.py` — the `OptimalPolicy` interface candidate policies implement.
- `src/dream_rsi/dream.py` — offline dreaming and version selection.
- `src/dream_rsi/orchestrator.py` — the online rollout loop.
- `src/dream_rsi/adapters/` — wrappers for coding agents and task evaluators.

## Running things

```bash
pytest                      # all tests
pytest tests/test_replay.py # one file
ruff check src tests        # lint
```

## Working on issues

Issues are phased and each names its blockers. Do not start an issue whose blockers are open — the interfaces it depends on aren't settled yet, and you'll write code that has to be thrown away.

## Things that will bite you

- **Policy code is executed.** `dream.py` runs LLM-written Python. It must stay sandboxed and resource-capped. Never relax that to make a test pass.
- **Replay generates nothing.** During replay, outcomes are retrieved from the frozen tree only. If you find yourself calling an agent or an evaluator inside the replay path, you've made a mistake.
- **Scores are task-specific.** Don't assume higher-is-better everywhere without checking the evaluator's contract.
- **Reading the PDF needs setup.** `pdftotext` and `pdftoppm` are usually absent, and a system-wide `pip install pypdf` can collide with the system `cryptography`. A throwaway venv works: `python -m venv /tmp/pdfvenv && /tmp/pdfvenv/bin/pip install -q pypdf`, then read pages with `PdfReader`.

## Scope

This repo is an independent implementation from the paper. The authors' code is not released. Do not fabricate API compatibility with it, and do not cite line numbers or function names from a repo you cannot read.
