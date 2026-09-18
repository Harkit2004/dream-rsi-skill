"""The committed recorded trees every later phase tests against (issue #6)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).parent / "fixtures" / "record_trees.py"
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")


def _load(name: str) -> DiscoveryTree:
    return DiscoveryTree.load(TREES / name / "tree.json")


def _depth(tree: DiscoveryTree, node_id: str) -> int:
    depth = 0
    current = tree.node(node_id).parent_id
    while current is not None:
        depth += 1
        current = tree.node(current).parent_id
    return depth


@pytest.mark.parametrize("name", NAMES)
def test_fixture_loads_under_the_current_schema(name: str) -> None:
    """The committed trees still load, which is what CI is here to verify.

    ``DiscoveryTree.load`` refuses an unrecognised ``schema_version`` and
    enforces the tree invariants — one root, one existing parent per node,
    unique ids, no cycles — so a schema bump that leaves the fixtures behind
    fails here rather than in whichever later test happened to use them.
    """
    tree = _load(name)

    assert len(tree) > 1
    assert tree.node(tree.root_id).parent_id is None

    rounds = json.loads((TREES / name / "rounds.json").read_text(encoding="utf-8"))
    produced = [node_id for round_ in rounds["rounds"] for node_id in round_["produced"]]
    # The round log is only usable alongside its tree: Equation 1 reads ``k`` off
    # a round and the scores off the nodes that round produced (issue #8).
    assert produced == [node.id for node in tree.iter_nodes() if node.id != tree.root_id]


def test_wide_shallow_fixture_is_wide_and_shallow() -> None:
    """A single round of branching off the root, and no deeper."""
    tree = _load("wide_shallow")
    children = tree.children(tree.root_id)

    assert len(children) >= 4
    assert len(children) == len(tree) - 1
    assert {_depth(tree, node.id) for node in children} == {1}


def test_narrow_deep_fixture_is_narrow_and_deep() -> None:
    """One unbranched chain, deep enough for a prefix reveal to have a prefix."""
    tree = _load("narrow_deep")

    assert max(_depth(tree, node.id) for node in tree.iter_nodes()) >= 5
    assert all(len(tree.children(node.id)) <= 1 for node in tree.iter_nodes())


def test_failing_branch_fixture_holds_a_dead_branch_and_a_live_one() -> None:
    """The mixed tree: a hard failure, an inadmissible plan, and real scores.

    Without all three a replay test cannot tell a branch that failed hard from
    one that merely scored badly — the distinction the paper's policies turn on
    (§B.2) and the one issue #11's observation signals are built from.
    """
    tree = _load("failing_branch")
    attempts = [node for node in tree.iter_nodes() if node.id != tree.root_id]

    failed = [node for node in attempts if node.score is None]
    assert failed, "no hard-failing node: nothing distinguishes a dead branch"
    assert all(node.diagnostics["fail_class"] != "ok" for node in failed)

    scored = [node for node in attempts if node.score is not None]
    assert [node for node in scored if node.diagnostics["correct"]], "no admissible plan recorded"
    assert [node for node in scored if not node.diagnostics["correct"]], (
        "no evaluated-but-incorrect node: the successful-evaluation distinction is untested"
    )
    assert len({node.score for node in scored}) > 1, "every scored node scored the same"
    assert max(_depth(tree, node.id) for node in attempts) >= 2


def test_recorder_reproduces_the_committed_fixtures_byte_for_byte(tmp_path: Path) -> None:
    """Re-recording from the fixed seed gives back exactly what is committed.

    This is the fixtures' determinism guarantee (AGENTS.md rule 5) and their
    staleness check in one: it fails if anything in the recording path depends
    on iteration order, the clock, the path it ran in or the worker that got
    there first, and it fails if the task changed without the fixtures being
    regenerated.
    """
    completed = subprocess.run(
        [sys.executable, str(RECORDER), str(tmp_path)],
        capture_output=True,
        text=True,
        # src on the path so this passes whether or not the package is
        # installed; the rest of the environment is inherited, because a hosted
        # runner's interpreter needs its own LD_LIBRARY_PATH to start at all.
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    committed = sorted(path.relative_to(TREES) for path in TREES.rglob("*.json"))
    regenerated = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json"))
    assert committed == regenerated

    for relative in committed:
        assert (tmp_path / relative).read_bytes() == (TREES / relative).read_bytes(), (
            f"{relative} differs from the committed fixture; "
            f"re-record with: python {RECORDER.relative_to(REPO)}"
        )
