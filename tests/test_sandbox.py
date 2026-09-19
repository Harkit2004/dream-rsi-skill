"""Running a model-written policy under limits (issue #13).

``dream.py`` executes Python an LLM wrote (issue #14). What matters about the
execution path is therefore not that it computes something — it is that the
things such code does wrong are contained: a policy that hangs, forks, eats
memory, reaches the network or writes outside its scratch directory is killed
and recorded as one failed cell, the sweep around it finishes, and the text it
produced on its way down survives as the feedback the development agent reads.

The other half is that none of that is paid for by a correct policy: a sandboxed
version of a baseline has to produce the trajectory the baseline produces
in-process, byte for byte, or the dreaming signal is measuring the sandbox.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dream_rsi.dream import DreamConfig, ReplayWorld, dream
from dream_rsi.policy import GreedyBestFirstPolicy
from dream_rsi.replay import ReplaySimulator
from dream_rsi.sandbox import (
    SandboxError,
    SandboxLimits,
    SandboxedPolicy,
    sandboxed_candidate,
)
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")

# Short enough that a hang test costs a second rather than ten, wide enough that
# a fresh interpreter importing ``dream_rsi`` is never the thing that trips it:
# starting a policy and replaying a whole fixture through it measures around 35ms
# of wall clock and 35ms of child CPU, so these leave a factor of tens in hand.
TEST_LIMITS = SandboxLimits(wall_seconds=1.5, cpu_seconds=1, memory_bytes=512 * 1024 * 1024)

# A legal candidate: the baseline, reached through the sandbox. Subclassing is
# what makes the comparison in ``test_a_sandboxed_baseline_replays_exactly_as_it
# _does_in_process`` a comparison of the *path* rather than of two strategies.
BASELINE_SOURCE = """
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    pass
"""


def _policy_source(body: str) -> str:
    """A policy whose decision runs ``body`` and then stops the replay."""
    return (
        "class OptimalPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        + textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
        + "\n        return ()\n"
    )


def _world(name: str) -> ReplayWorld:
    return ReplayWorld(
        name=name,
        simulator=ReplaySimulator(DiscoveryTree.load(TREES / name / "tree.json")),
    )


def _failure(source: str, *, scratch_root: Path | None = None) -> str:
    """Replay a policy that is going to fail, and return the recorded error text.

    Driven through :func:`~dream_rsi.dream.dream` rather than the sandbox alone,
    because "killed and recorded as a failed candidate" is a statement about
    what the harness ends up holding, not only about what the child process did.
    """
    candidate = sandboxed_candidate(
        "broken", source, limits=TEST_LIMITS, scratch_root=scratch_root
    )
    report = dream([candidate], [_world("wide_shallow")], config=DreamConfig(width=2))
    version = report.versions[0]
    assert version.score is None, f"a failing candidate scored {version.score}"
    assert len(version.failures) == 1, version.to_dict()
    error = version.failures[0].error
    assert error, "a failed cell recorded no error text"
    return error


@pytest.mark.parametrize("name", NAMES)
def test_a_sandboxed_baseline_replays_exactly_as_it_does_in_process(name: str) -> None:
    """The sandbox costs a correct policy nothing — not even a different tie-break.

    §3 compares versions by their replay scores, so a sandboxed candidate whose
    trajectory differs from the same strategy run in-process makes every number
    in a dreaming round a number about the execution path. Compared as the whole
    serialised trajectory rather than as the score: two runs can reach the same
    ``V_i^m`` down different branches, and it is the decisions that have to
    match. The baseline is the greedy one because it decides *from* the revealed
    scores: dropping them, or dropping ``W``, changes its trajectory on all three
    fixtures, where the breadth-first baseline would notice neither.
    """
    world = _world(name)
    config = DreamConfig(width=2)

    with SandboxedPolicy(BASELINE_SOURCE, config={"beta": 1.0}, limits=TEST_LIMITS) as policy:
        sandboxed = world.simulator.replay(
            policy, width=config.width, seed=config.seed
        ).result()
    in_process = world.simulator.replay(
        GreedyBestFirstPolicy(config={"beta": 1.0}), width=config.width, seed=config.seed
    ).result()

    assert sandboxed.to_json() == in_process.to_json()


def test_a_decision_is_shown_exactly_what_replay_passed_it() -> None:
    """The three arguments cross the boundary unchanged, in order.

    The trajectory comparison above cannot see this: the baselines sort what they
    are given, so a proxy that reversed the eligible set would still replay
    identically. A model-written policy that reads ``eligible[0]`` — the root, as
    ``eligible_nodes`` orders it — would not, and neither would one that batches
    up to ``width``.
    """
    tree = DiscoveryTree.load(TREES / "wide_shallow" / "tree.json")
    # The second decision, not the first: a one-node eligible set cannot show an
    # order, so the policy opens two branches off the root before reporting.
    error = _failure(
        "class OptimalPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        "        if len(tree) == 1:\n"
        "            return [eligible[0]] * 2\n"
        "        raise ValueError("
        "f'eligible={list(eligible)} width={width} nodes={len(tree)}')\n"
    )

    opened = sorted(node.id for node in tree.children(tree.root_id)[:2])
    expected = [tree.root_id, *opened]
    assert f"eligible={expected} width=2 nodes=3" in error, error


@pytest.mark.parametrize(
    ("what", "body"),
    [
        # No CPU burned at all, so only the wall clock can catch it.
        ("sleeping", "import time\ntime.sleep(60)"),
        # Burns CPU, so the CPU cap catches it first; either way it dies.
        ("spinning", "while True:\n    pass"),
    ],
)
def test_a_hanging_policy_is_killed_and_recorded_as_a_failure(what: str, body: str) -> None:
    """A version that never returns is one failed cell, not a stuck dreaming round.

    The bound on elapsed time is the test: a harness that simply waited for the
    child would pass every assertion about the failure record and hang a real
    round forever.
    """
    started = time.monotonic()
    error = _failure(_policy_source(body))
    elapsed = time.monotonic() - started

    assert elapsed < 30.0, f"a {what} policy took {elapsed:.1f}s to be killed"
    assert "SandboxError" in error, error


def test_a_memory_bomb_is_killed() -> None:
    """A version that asks for more memory than it may have fails, and alone.

    Without the address-space cap the allocation succeeds or the OOM killer
    picks a victim, and either way the parent's dreaming round is collateral.
    """
    error = _failure(_policy_source("self.hog = bytearray(8 * 1024 * 1024 * 1024)"))
    assert "SandboxError" in error, error


@pytest.mark.parametrize(
    ("what", "body"),
    [
        ("a socket", "import socket\nsocket.socket()"),
        ("a name lookup", "import socket\nsocket.getaddrinfo('example.com', 80)"),
        ("an HTTP request", "import urllib.request\nurllib.request.urlopen('http://example.com')"),
    ],
)
def test_network_access_is_refused(what: str, body: str) -> None:
    """Policy code decides from the revealed prefix; it has no business dialling out.

    Asserted on the refusal reaching the failure record rather than on the
    absence of traffic, because a sandbox that let the call through would fail
    here only when the network happened to be down.
    """
    error = _failure(_policy_source(body))
    assert "network" in error.lower(), error


@pytest.mark.parametrize(
    ("what", "body"),
    [
        ("forking", "import os\nos.fork()"),
        ("spawning a process", "import subprocess\nsubprocess.run(['/bin/true'])"),
        ("os.system", "import os\nos.system('true')"),
    ],
)
def test_starting_another_process_is_refused(what: str, body: str) -> None:
    """A child of the child outlives the kill and escapes every limit set on it."""
    error = _failure(_policy_source(body))
    assert "process" in error.lower(), error


def test_a_write_outside_the_scratch_directory_is_refused(tmp_path: Path) -> None:
    """Filesystem access is restricted to the scratch directory (issue #13, scope).

    A policy that can write anywhere can rewrite the recorded trees it is being
    scored against, or the source of the policy it is competing with.
    """
    escape = tmp_path / "escaped.txt"
    error = _failure(
        _policy_source(f"open({str(escape)!r}, 'w').write('x')"), scratch_root=tmp_path
    )

    assert "scratch" in error.lower(), error
    assert not escape.exists(), "the sandbox let a policy write outside its scratch directory"


def test_the_scratch_directory_is_writable_and_goes_away_afterwards(tmp_path: Path) -> None:
    """Restricted is not the same as read-only: the scratch directory is usable.

    A policy may keep notes between its rounds, and the paper's own skeleton
    carries per-rollout state. Catches a filesystem rule that refuses
    everything, which would pass every other test here.
    """
    source = _policy_source(
        """
        with open('notes.txt', 'a') as handle:
            handle.write('one round\\n')
        """
    )
    with SandboxedPolicy(source, limits=TEST_LIMITS, scratch_root=tmp_path) as policy:
        tree = DiscoveryTree.load(TREES / "wide_shallow" / "tree.json")
        assert policy.select(tree, (tree.root_id,), 1) == ()
        scratch = policy.scratch
        assert (scratch / "notes.txt").read_text() == "one round\n"

    assert not scratch.exists(), "the sandbox left its scratch directory behind"


def test_a_failure_record_carries_the_policys_output_and_its_exception() -> None:
    """The failure text is what issue #14 revises the next version from.

    A record that says only "it raised" tells the development agent nothing it
    can act on, so both what the policy printed and what it raised have to
    survive the process boundary.
    """
    error = _failure(
        _policy_source(
            """
            print('frontier looked empty')
            raise ZeroDivisionError('probe scores summed to zero')
            """
        )
    )
    assert "frontier looked empty" in error, error
    assert "ZeroDivisionError" in error, error
    assert "probe scores summed to zero" in error, error


@pytest.mark.parametrize(
    ("what", "source", "match"),
    [
        ("a syntax error", "class OptimalPolicy(:\n", "SyntaxError"),
        ("no policy at all", "answer = 42\n", "OptimalPolicy"),
        ("a policy that is not one", "OptimalPolicy = 42\n", "select"),
        ("one that raises on import", "raise RuntimeError('half-written')\n", "half-written"),
    ],
)
def test_source_that_cannot_be_used_is_reported_not_raised(
    what: str, source: str, match: str
) -> None:
    """Unusable source is a rejection with a message, which #14 feeds back.

    A model writes invalid code often enough that this is a normal event: it has
    to arrive as a failed candidate carrying the reason, never as an exception
    out of the harness or a candidate that quietly scores nothing.
    """
    assert match in _failure(source)


def test_a_failed_candidate_does_not_stop_the_sweep() -> None:
    """One cell's outcome, not the round's (issue #12).

    The dreaming round of a real loop always contains versions that fall over;
    if one of them took the round with it, the surviving versions would never be
    compared and the loop could not advance.
    """
    hanging = sandboxed_candidate("hangs", _policy_source("while True:\n    pass"), limits=TEST_LIMITS)
    good = sandboxed_candidate("baseline", BASELINE_SOURCE, limits=TEST_LIMITS)
    worlds = [_world(name) for name in NAMES]

    report = dream([hanging, good], worlds, config=DreamConfig(width=2))

    assert report.versions[0].score is None
    assert len(report.versions[0].failures) == len(worlds)
    assert report.versions[1].score is not None
    assert report.versions[1].failures == ()
    assert report.ranking == ("baseline", "hangs")


@pytest.mark.parametrize(
    ("field", "value"),
    [("wall_seconds", 0.0), ("cpu_seconds", 0), ("memory_bytes", 0)],
)
def test_limits_cannot_be_turned_off(field: str, value: float) -> None:
    """There is no bypass (issue #13, done-when; CLAUDE.md).

    The way a sandbox stops being one is a caller that asks for no limit at all,
    so "unlimited" is not expressible: every limit is a positive number.
    """
    with pytest.raises(ValueError, match=field):
        SandboxLimits(**{field: value})


def test_a_dead_sandbox_refuses_later_decisions() -> None:
    """Once a policy has been killed or closed, it does not answer again.

    A proxy that restarted its child would hand the next round a policy with no
    per-rollout state, and the trajectory would silently describe two different
    policies (§3: state is reset per policy-world pair, not per round). The
    second call has to fail the same way as the first, too: a caller holding a
    dead sandbox gets a ``SandboxError`` rather than an ``OSError`` about the
    descriptors it used to decide over.
    """
    tree = DiscoveryTree.load(TREES / "wide_shallow" / "tree.json")
    source = _policy_source("import time\ntime.sleep(60)")
    with SandboxedPolicy(source, limits=TEST_LIMITS) as policy:
        with pytest.raises(SandboxError):
            policy.select(tree, (tree.root_id,), 1)
        with pytest.raises(SandboxError):
            policy.select(tree, (tree.root_id,), 1)

    healthy = SandboxedPolicy(BASELINE_SOURCE, limits=TEST_LIMITS)
    healthy.close()
    with pytest.raises(SandboxError, match="closed"):
        healthy.select(tree, (tree.root_id,), 1)


def test_importing_the_sandbox_reaches_no_agent_and_no_evaluator() -> None:
    """The sandbox runs policies, and policies run in replay (§3).

    ``tests/test_dream.py`` asserts this of the harness and ``tests/test_replay
    .py`` of the simulator. If the execution path pulled an adapter in, a
    candidate could call a discovery agent and score itself on an outcome
    nothing ever executed.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dream_rsi.sandbox, dream_rsi._sandbox_child, sys; print('\\n'.join(sys.modules))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    adapters = sorted(
        name for name in completed.stdout.split() if name.startswith("dream_rsi.adapters")
    )
    assert not adapters, f"importing the sandbox pulled in {', '.join(adapters)}"
