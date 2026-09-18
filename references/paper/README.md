# The paper

Vendored here so that anyone picking up an issue has the source in the repo rather than a link they have to go chase.

| File | Use it for |
|---|---|
| `Dream-RSI.pdf` | **Authoritative.** Equations, figures, tables, experimental detail. Read this for anything quantitative. |
| `Dream-RSI-fulltext.md` | Grepping and locating. Auto-extracted from the arXiv HTML; math is stripped and the layout is lossy. |

## How to use these

Locate first, then read. Grep the text dump to find *where* something is discussed, then open the PDF at that section:

```bash
grep -n -i "replay objective" references/paper/Dream-RSI-fulltext.md
```

The paper's own section order: motivation (discovery history as a replay simulator) → the method (discovery trees and the shared decision interface, online rollout, offline evaluation, the replay objective, policy improvement and selection) → experiments across the three domains → further analysis.

## Do not trust the text dump for

Equations — including the replay objective, which is the one piece of math this implementation depends on most directly. MathML was stripped during extraction, so symbols, subscripts and the whole of Equation 1 are missing or garbled. `references/method.md` transcribes Equation 1 by hand; the PDF is how you check that transcription.

## Provenance

- Paper: *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*, Zheng et al., 2026 — [arXiv:2609.14858](https://arxiv.org/abs/2609.14858)
- PDF copied verbatim from the authors' public repository, [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI), `papers/Dream-RSI.pdf`
- Text extracted from <https://arxiv.org/html/2609.14858v1>

Copyright in the paper belongs to its authors. It is included here unmodified for reference, and is not covered by this repository's MIT license — that license applies to the code in this repo only. If the authors would prefer it not be mirrored, open an issue and it will be removed and replaced with a link.
