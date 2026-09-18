# Working rules

These apply to every agent and every human touching this repo. They are short on purpose. If a rule here conflicts with a habit, the rule wins.

## 1. Test before implementation

Write the failing test first. Every bug fix and every feature.

- **Bug:** reproduce it as a test that fails for the stated reason, *then* fix it. A bug with no regression test is not fixed.
- **Feature:** write the test that describes the behaviour, watch it fail, then make it pass.
- Never write the test after the code "to confirm it works". That test only confirms the code does what it does.

If a test is genuinely impractical to write first, say so explicitly in the PR and explain why — don't skip silently.

## 2. Minimal change

Solve the stated problem with the smallest diff that actually solves it.

- No drive-by refactors, renames, reformatting, or dependency additions in a change that is about something else. Open a separate issue.
- No speculative abstraction. Two call sites is not a pattern; three might be.
- No new dependency without a line in the PR saying what it replaces and why vendoring or stdlib won't do.
- Deleting code counts as a change too — don't remove things you merely find unfamiliar.

## 3. One issue, one PR

Work lands through a pull request, never by pushing to `main` directly. One branch, one PR, one issue, `Closes #N` in the description. If you found a second problem while in there, open an issue for it and move on.

**CodeRabbit reviews every PR, and its comments are not advisory.** Before merging, every one must be addressed — either fix it, or reply on the thread saying concretely why you are not (it conflicts with the issue's stated scope, it asks for the speculative abstraction rule 2 forbids, it is factually wrong about the code). "Noted" is not addressing it. Push the fixes to the same branch and let it re-review.

Merge only when **both** hold: CI is green on the head commit, and no review comment is outstanding. A red build or an unanswered comment means the PR is not ready, however small the remaining point looks.

## 4. Don't invent the paper's details

This repo implements a paper whose reference code is unreleased. Where the paper is explicit, follow it and cite the equation or section in a comment. Where it is silent, **make the choice explicit**:

```python
# PAPER-GAP: the paper does not state how ties are broken when two branches
# share max score. We take the lower node id for determinism. Revisit if the
# authors' implementation lands (see references/method.md).
```

Grep for `PAPER-GAP:` to find everything that needs re-checking when the official code is published. Never quietly guess.

## 5. Determinism

Replay must be reproducible. Anything that touches ordering, batching, or randomness takes an explicit seed. No reliance on dict ordering, set iteration order, or wall-clock time inside the replay path. A replay run twice on the same tree with the same policy and seed must produce byte-identical trajectories, and there is a test that asserts this.

## 6. The coding agent stays untouched

Dream-RSI's whole claim is that improvement lives in the orchestration layer. Anything that would require modifying, fine-tuning, or special-casing the underlying coding agent is out of scope for this repo. Adapters wrap agents; they do not change them.

## 7. Before you claim it works

Run the tests. Paste real output in the PR. "Should work" and "tests pass" without output are not acceptable. If something fails and you're landing anyway, say which test and why.

## 8. Commits and PRs

- Imperative subject line under 72 chars: `add prefix-observable reveal to ReplaySimulator`
- Body says *why*, not *what* — the diff already says what.
- Branch name: `issue-<N>-<short-slug>`.
- No force-push, on any branch someone else may have reviewed.

## 9. Keep CI and local identical

`pyproject.toml` pins exact versions of `pytest` and `ruff`. Leave them pinned. An unpinned linter means a run can report "all checks passed" locally and fail the identical command in CI, which is how the first green-locally/red-on-CI commit happened. Bump the pin deliberately, in its own change, with the new version's complaints fixed in the same PR.
