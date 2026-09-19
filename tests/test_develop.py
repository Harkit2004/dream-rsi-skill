"""Revising policy code from replay feedback (issue #14).

Step 4 of the loop (§3): "the policy-development agent examines the replay
trajectories and scores of ``π_t^m``, together with feedback from earlier
revisions, to identify successful decisions and recurring failures. It then
revises the executable policy code to produce ``π_t^{m+1}``".

What matters about the pipeline is not what any particular model writes — no
test here calls one. It is that a round hands the agent the feedback it claims
to, that a revision is only ever run inside issue #13's sandbox, and that output
the harness cannot use comes back as a message the next attempt is given rather
than disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dream_rsi.develop import (
    DevelopmentReport,
    RevisionContext,
    develop,
    validate_source,
)
from dream_rsi.dream import DreamConfig, ReplayWorld
from dream_rsi.replay import ReplaySimulator
from dream_rsi.sandbox import SandboxLimits
from dream_rsi.tree import DiscoveryTree

TREES = Path(__file__).parent / "fixtures" / "trees"

# Generous next to what a policy here spends — a whole fixture replays through a
# freshly started child in tens of milliseconds — so nothing in this file is
# measuring the machine it runs on. The disk cap is small because no policy here
# writes at all.
TEST_LIMITS = SandboxLimits(
    wall_seconds=5.0,
    cpu_seconds=2,
    memory_bytes=512 * 1024 * 1024,
    disk_bytes=1024 * 1024,
)

# The incumbent π_t^0: a baseline reached by subclassing, which is the shape
# §B.2 asks a version for ("implement ``class OptimalPolicy(...)``").
BASELINE_SOURCE = """
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    pass
"""


def _revision(beta: float) -> str:
    """A legal revision: the baseline at a different baked-in beta (§B.2)."""
    return (
        "from dream_rsi.policy import GreedyBestFirstPolicy\n"
        "\n"
        "\n"
        "class OptimalPolicy(GreedyBestFirstPolicy):\n"
        "    def __init__(self, config=None):\n"
        f"        super().__init__({{'beta': {beta}}})\n"
    )


@dataclass
class ScriptedDeveloper:
    """A development agent that answers from a script and keeps what it was asked.

    Deterministic and inert: no model, no network, no clock. The contexts it
    collected are how a test sees what the pipeline actually handed the agent.
    """

    script: tuple[str, ...]
    contexts: list[RevisionContext] = field(default_factory=list)

    def revise(self, context: RevisionContext) -> str:
        self.contexts.append(context)
        # Indexing rather than cycling: a round that asks for more revisions
        # than the test scripted is a failure of the test's premise, not a pass.
        return self.script[len(self.contexts) - 1]


def _world(name: str) -> ReplayWorld:
    return ReplayWorld(
        name=name,
        simulator=ReplaySimulator(DiscoveryTree.load(TREES / name / "tree.json")),
    )


def _develop(
    source: str,
    developer: ScriptedDeveloper,
    scratch_root: Path,
    *,
    worlds: tuple[str, ...] = ("wide_shallow",),
    versions: int = 2,
    attempts: int = 1,
) -> DevelopmentReport:
    return develop(
        source,
        [_world(name) for name in worlds],
        developer,
        versions=versions,
        attempts=attempts,
        config=DreamConfig(width=2),
        limits=TEST_LIMITS,
        scratch_root=scratch_root,
    )


def test_a_stubbed_round_produces_m_sandboxed_scored_versions(tmp_path: Path) -> None:
    """Issue #14's "done when": M versions, every one of them replayed and scored.

    The incumbent is π_t^0 (§3) and the rest are what the agent wrote, so a round
    that quietly dropped a revision, or scored the incumbent M times, fails here.
    """
    developer = ScriptedDeveloper((_revision(0.5), _revision(2.0)))
    report = _develop(
        BASELINE_SOURCE,
        developer,
        tmp_path,
        worlds=("wide_shallow", "narrow_deep"),
        versions=3,
    )

    assert len(developer.contexts) == 2, "M versions need M-1 revisions"
    assert [version.source for version in report.versions] == [
        BASELINE_SOURCE,
        *developer.script,
    ]
    assert len({version.name for version in report.versions}) == 3
    assert not report.rejected
    for version in report.versions:
        assert version.report.score is not None, version.report.to_dict()
        assert [replay.world for replay in version.report.replays] == [
            "wide_shallow",
            "narrow_deep",
        ]
        for replay in version.report.replays:
            assert replay.result is not None and replay.result.rounds

    # The grid the whole round is compared on, which is what selection reads.
    assert report.comparison.worlds == ("wide_shallow", "narrow_deep")
    assert set(report.comparison.ranking) == {version.name for version in report.versions}


def test_a_revision_is_never_run_in_the_harness_process(tmp_path: Path) -> None:
    """"Every generated candidate goes through #13. No exceptions."

    The revision writes a file the moment its module body runs. Run in this
    process — to compile it, to check its interface, to score it — the file
    appears; run in the sandbox, the write is outside the scratch directory and
    is refused, which is a failed cell and a message.
    """
    escaped = tmp_path / "escaped.txt"
    source = (
        "import pathlib\n"
        "\n"
        "from dream_rsi.policy import GreedyBestFirstPolicy\n"
        "\n"
        f"pathlib.Path({str(escaped)!r}).write_text('ran in the harness')\n"
        "\n"
        "\n"
        "class OptimalPolicy(GreedyBestFirstPolicy):\n"
        "    pass\n"
    )
    report = _develop(BASELINE_SOURCE, ScriptedDeveloper((source,)), tmp_path)

    assert not escaped.exists(), "the revision ran outside the sandbox"
    version = report.versions[1]
    assert version.source == source
    assert version.report.score is None
    assert version.report.failures, "a revision the sandbox refused was scored anyway"
    assert "scratch" in version.report.failures[0].error


@pytest.mark.parametrize(
    ("revision", "expected"),
    [
        ("def broken(:\n", "does not parse"),
        ("class OptimalPolicy:\n    pass\n\x00", "does not parse"),
        ("class NotThePolicy:\n    pass\n", "OptimalPolicy"),
        ("", "OptimalPolicy"),
    ],
    ids=("syntax", "unscannable", "wrong-class", "empty"),
)
def test_an_unusable_revision_is_rejected_with_a_message_naming_the_fault(
    tmp_path: Path, revision: str, expected: str
) -> None:
    """Output that cannot be a policy is refused, and the refusal says why.

    Not scored, not run, and not silently dropped: the round comes back holding
    what was refused and a reason a reader — or the next attempt — can act on.
    """
    report = _develop(BASELINE_SOURCE, ScriptedDeveloper((revision,)), tmp_path)

    assert len(report.versions) == 1, "a revision the harness cannot use was scored anyway"
    assert len(report.rejected) == 1
    assert report.rejected[0].source == revision
    assert expected in report.rejected[0].reason


def test_a_policy_renamed_through_name_is_accepted() -> None:
    """§B.2 has ``NAME`` name the policy class, so the check follows it.

    Catches a check that hardcodes ``OptimalPolicy`` and refuses a legal version
    the sandbox would have loaded happily.
    """
    source = (
        'NAME = "MyPolicy"\n'
        "\n"
        "\n"
        "class MyPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        "        return ()\n"
    )
    assert validate_source(source) is None


def test_a_reassigned_name_is_read_the_way_the_sandbox_reads_it() -> None:
    """``NAME`` bound twice: the last binding is the class, as at runtime.

    ``_sandbox_child._load_policy`` does ``namespace.get("NAME", ...)`` *after*
    executing the module, so the final binding is the one it looks for. A check
    that stops at the first assignment refuses a version the sandbox would have
    loaded, and burns an attempt doing it. Both directions, because a check that
    gave up whenever ``NAME`` appears twice would pass the first assert alone.
    """
    body = (
        "class Actual:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        "        return ()\n"
    )
    accepted = 'NAME = "Stale"\nNAME = "Actual"\n\n\n' + body
    refused = 'NAME = "Actual"\nNAME = "Stale"\n\n\n' + body

    assert validate_source(accepted) is None
    assert "Stale" in (validate_source(refused) or "")


def test_a_rejected_revision_is_fed_back_to_the_next_attempt(tmp_path: Path) -> None:
    """A rejection is information (issue #14): the next attempt is shown it.

    Recording the refusal in the report but asking the agent again with the same
    context it already failed on is the failure this catches — the agent has no
    way to know it was refused, so it writes the same thing again.
    """
    developer = ScriptedDeveloper(("def broken(:\n", _revision(2.0)))
    report = _develop(BASELINE_SOURCE, developer, tmp_path, attempts=2)

    assert len(report.versions) == 2, "the accepted retry was not scored"
    assert report.versions[1].source == _revision(2.0)
    assert [rejection.source for rejection in report.rejected] == ["def broken(:\n"]

    retry = developer.contexts[1].prompt()
    assert "def broken(:" in retry
    assert report.rejected[0].reason in retry


def test_the_revision_context_carries_the_trajectories_and_scores_it_claims(
    tmp_path: Path,
) -> None:
    """§3: the agent examines "the replay trajectories and scores of π_t^m".

    So the context is the version's source, its per-world ``V_i^m``, and the
    trajectory each was computed from — and the prompt the agent is handed
    carries them rather than naming them.
    """
    developer = ScriptedDeveloper((_revision(2.0),))
    report = _develop(
        BASELINE_SOURCE,
        developer,
        tmp_path,
        worlds=("wide_shallow", "narrow_deep"),
    )

    context = developer.contexts[0]
    assert context.source == BASELINE_SOURCE
    assert context.report == report.versions[0].report
    assert context.report.score is not None
    assert not context.rejected and not context.history

    prompt = context.prompt()
    assert BASELINE_SOURCE.strip() in prompt
    for replay in context.report.replays:
        assert replay.score is not None
        assert replay.result is not None
        assert replay.world in prompt
        # The trajectory itself, not only the number it produced.
        assert replay.result.rounds[0].selected[0] in prompt


def test_a_version_that_crashed_hands_its_failure_text_to_the_next_revision(
    tmp_path: Path,
) -> None:
    """Issue #14's scope: "the failure text of anything that crashed".

    A version that cannot even be loaded is the common case for model-written
    code, and its error is the most useful thing the next revision can be told.
    """
    crashing = (
        "class OptimalPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        raise RuntimeError('boom')\n"
    )
    developer = ScriptedDeveloper((_revision(1.0),))
    _develop(crashing, developer, tmp_path)

    context = developer.contexts[0]
    assert context.report.score is None
    assert context.report.failures, "a version that crashed recorded no failure"
    assert "boom" in context.report.failures[0].error
    assert "boom" in context.prompt()
