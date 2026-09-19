"""The simulator pool: the history kept on disk between sessions (issue #17).

The pool is what §3's "continuously expanding simulator pool" is made of, and
the dreaming phase replays every tree in it. What matters here is therefore not
arithmetic but custody: that a tree added in one session is there and loadable
in the next, that an add which dies half-way leaves a pool that still reads,
that measuring the pool agrees with counting it by hand, and that a run which
cannot afford to dream over all of it drops the same trees every time it is
given the same seed.

Nothing in this file calls a model or an evaluator: the trees are built here,
node by node, and replayed from disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dream_rsi.pool import PoolConfig, PoolError, PoolStats, SimulatorPool, subsample
from dream_rsi.tree import DiscoveryTree


def _tree(attempts: int) -> DiscoveryTree:
    """A root and ``attempts`` children, so the node count is known by hand."""
    tree = DiscoveryTree.with_root()
    for index in range(attempts):
        tree.add_child(tree.root_id, score=float(index))
    return tree


def _filled(directory: Path, sizes: dict[str, int]) -> SimulatorPool:
    pool = SimulatorPool(directory)
    for name, attempts in sizes.items():
        pool.add(name, _tree(attempts))
    return pool


def test_a_pool_survives_a_restart_with_every_tree_loadable(tmp_path: Path) -> None:
    """Issue #17's first "tests first": a later session inherits the whole pool.

    The reader is a second :class:`SimulatorPool` built over the same directory
    and sharing nothing with the writer, which is what a run started tomorrow
    has. So a pool that kept its trees in memory and its directory as a
    write-behind cache — added names tracked in a list, an index flushed only on
    a clean shutdown — passes nothing here: the fresh pool has to name all three
    trees and hand back worlds holding the nodes that were recorded, not empty
    ones.
    """
    _filled(tmp_path / "pool", {"cycle_000": 4, "cycle_001": 2, "cycle_002": 7})

    reopened = SimulatorPool(tmp_path / "pool")

    assert reopened.names() == ("cycle_000", "cycle_001", "cycle_002")
    assert [(world.name, len(world.simulator)) for world in reopened.worlds()] == [
        ("cycle_000", 5),
        ("cycle_001", 3),
        ("cycle_002", 8),
    ]


def test_a_pool_reads_back_only_the_trees_asked_for(tmp_path: Path) -> None:
    """A dreaming round replays a named subset, in the order it named it.

    That is what subsampling is spent on, and what a cycle dreaming over ``ℋ_t``
    needs: the history it was handed, not everything the store happens to hold.
    A pool that ignored the argument and loaded its whole directory — or
    returned the trees in directory order instead of the caller's — fails here.
    """
    pool = _filled(tmp_path / "pool", {"a": 1, "b": 2, "c": 3})

    assert [world.name for world in pool.worlds(("c", "a"))] == ["c", "a"]
    with pytest.raises(PoolError, match="missing"):
        pool.worlds(("a", "nowhere"))


def test_an_add_that_dies_before_it_publishes_leaves_the_pool_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #17's second "tests first": an interrupted add is not a broken pool.

    The failure is injected at the moment of publication, which is where a crash
    costs the most: the bytes are written and the name is about to appear. A
    pool that wrote the tree into its final path directly would have published a
    truncated file by then, so the next session reads a tree that will not load
    — and for a store whose whole job is to be read back later, an unloadable
    tree is worse than a missing one. Here the pool afterwards holds exactly
    what it held before the attempt, still reads, and still takes new trees.
    """
    pool = _filled(tmp_path / "pool", {"cycle_000": 3})

    def die(source: object, target: object) -> None:
        raise OSError("interrupted")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", die)
        with pytest.raises(OSError):
            pool.add("cycle_001", _tree(5))

    reopened = SimulatorPool(tmp_path / "pool")
    assert reopened.names() == ("cycle_000",)
    assert [len(world.simulator) for world in reopened.worlds()] == [4]

    # And the leftovers of the failed attempt do not block the retry.
    reopened.add("cycle_001", _tree(5))
    assert SimulatorPool(tmp_path / "pool").names() == ("cycle_000", "cycle_001")


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "nested/tree"])
def test_a_tree_name_that_is_not_a_plain_filename_is_refused(tmp_path: Path, name: str) -> None:
    """A name is a name, not a path: nothing is written outside the pool.

    The names come from the driver today, but the pool is the thing that owns
    the directory, and a pool that pasted the name into a path would let
    ``../..`` write a tree anywhere the process can reach and then list a pool
    it cannot load back.
    """
    pool = SimulatorPool(tmp_path / "pool")

    with pytest.raises(PoolError):
        pool.add(name, _tree(1))
    assert pool.names() == ()


def test_stats_match_a_manual_count(tmp_path: Path) -> None:
    """Issue #17's fourth "tests first": the measurement agrees with counting.

    The numbers are what tells a run its dreaming bill is growing — the pool
    costs one replay per tree per version — so they have to be the trees and
    nodes actually on disk. A pool that counted the trees it added this session,
    or reported a tree's node count as its attempts, disagrees with the count
    made here from the directory and the trees themselves.
    """
    sizes = {"cycle_000": 4, "cycle_001": 2, "cycle_002": 7}
    _filled(tmp_path / "pool", sizes)

    stats = SimulatorPool(tmp_path / "pool").stats()

    assert stats.trees == 3
    assert stats.nodes == sum(attempts + 1 for attempts in sizes.values())
    assert stats.bytes == sum(
        path.stat().st_size for path in (tmp_path / "pool").iterdir() if path.is_file()
    )


def test_a_file_the_pool_did_not_write_is_not_a_world(tmp_path: Path) -> None:
    """Whatever the pool lists, it can load — so it lists only its own trees.

    A run directory is a place people and other tools write into: a note, an
    editor's leavings, a dotfile the filesystem put there. A pool that listed
    everything it found would name worlds it then refuses to load, and counting
    the pool — the one thing a run does with it every cycle — would raise
    instead of reporting a size.
    """
    pool = _filled(tmp_path / "pool", {"cycle_000": 3})
    (tmp_path / "pool" / "notes.txt").write_text("not a tree", encoding="utf-8")
    (tmp_path / "pool" / ".hidden.json").write_text("not a tree either", encoding="utf-8")

    assert pool.names() == ("cycle_000",)
    assert pool.stats().trees == 1
    assert [world.name for world in pool.worlds()] == ["cycle_000"]


def test_a_pool_with_nothing_in_it_reports_nothing(tmp_path: Path) -> None:
    """A run's first cycle dreams over a pool that was empty a moment ago.

    ``ℋ_0 = ()`` in §3, and the directory does not exist yet. A pool that
    raised on a missing directory, or reported one phantom tree, would break the
    cycle that creates the pool rather than the ones that inherit it.
    """
    pool = SimulatorPool(tmp_path / "pool")

    assert pool.names() == ()
    assert pool.worlds() == ()
    assert pool.stats() == PoolStats(trees=0, nodes=0, bytes=0)


def test_by_default_a_pool_is_dreamed_over_whole() -> None:
    """The honest default the issue asks for: no tree is dropped unasked.

    Subsampling is a cost control, and one that silently discarded history would
    make every ``V^m`` in the paper's average an average over something else.
    Nothing is dropped until a limit is set, and a limit no smaller than the
    pool drops nothing either.
    """
    names = ("cycle_000", "cycle_001", "cycle_002")

    assert subsample(names) == names
    assert subsample(names, PoolConfig()) == names
    assert subsample(names, PoolConfig(limit=3, seed=1)) == names
    assert subsample(names, PoolConfig(limit=9, seed=1)) == names


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_subsampling_is_deterministic_under_a_seed(seed: int) -> None:
    """Issue #17's third "tests first": the same seed keeps the same trees.

    A dreaming round that sampled the pool off ``set`` iteration or the clock
    would score two candidate versions over different histories and make
    selection's incumbent floor meaningless (working rule 5). The sample is also
    a subsample: a subset of what it was given, in the order it was given, and
    holding the newest tree — the one the policy just deployed recorded, which
    is the one a version has not already been tuned against.
    """
    names = tuple(f"cycle_{index:03d}" for index in range(8))
    config = PoolConfig(limit=3, seed=seed)

    chosen = subsample(names, config)

    assert chosen == subsample(names, config)
    assert len(chosen) == 3
    assert set(chosen) <= set(names)
    assert chosen == tuple(name for name in names if name in set(chosen))
    assert names[-1] in chosen


def test_a_different_seed_samples_a_different_history() -> None:
    """The seed is doing something: two of them do not fix one answer.

    Without this, a "deterministic" sampler that always took the last ``limit``
    trees — or the first — would pass every determinism check above while
    quietly making the seed a decoration and the older half of the pool dead
    weight.
    """
    names = tuple(f"cycle_{index:03d}" for index in range(12))

    samples = {subsample(names, PoolConfig(limit=4, seed=seed)) for seed in range(8)}

    assert len(samples) > 1


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_that_keeps_no_trees_is_refused(limit: int) -> None:
    """Dreaming over an empty history scores nothing, so it is a config error.

    §3's ``V^m`` is an average over ``t`` trees; a limit of zero makes it an
    average over none, which is a caller's mistake and should be refused where
    it is written rather than dividing by zero a cycle later.
    """
    with pytest.raises(ValueError, match="at least one"):
        PoolConfig(limit=limit)
