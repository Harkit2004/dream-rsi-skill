"""The simulator pool: the recorded trees, kept between sessions (issue #17).

§3's history ``ℋ_t`` is a collection of completed discovery trees, and the
abstract's "continuously expanding simulator pool" is what it becomes once a run
keeps adding to it. :mod:`dream_rsi.run` grows it by one tree per outer
iteration and :mod:`dream_rsi.dream` replays every tree in it, once per
candidate version. This module is the store between the two: a directory of
trees, added one at a time, read back by name.

Three things it owes its callers.

**A tree that is in is in.** :meth:`SimulatorPool.add` writes the tree beside
the pool and renames it into place, so the name appears only once the bytes are
there. A run interrupted mid-add leaves a pool that still lists and still loads
— it is simply one tree short, and the cycle that was recording that tree is
redone from the top anyway (see :mod:`dream_rsi.run`).

**The pool is measurable.** :meth:`SimulatorPool.stats` counts the trees, their
nodes and their bytes, because the dreaming bill is that count times ``M``:
this is the number that says when a pool has outgrown replaying in full.

**Dreaming over less than all of it is a decision, and a recorded one.**
:func:`subsample` picks the worlds one offline phase runs over. It drops
nothing by default, it never removes a tree from the store, and what it chose is
what the cycle's record carries.

Nothing here reaches an adapter: the pool is on the replay path, so it loads
trees and builds simulators, and never a discovery agent or an evaluator.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dream_rsi.dream import ReplayWorld
from dream_rsi.replay import DEFAULT_SEED, ReplaySimulator
from dream_rsi.tree import DiscoveryTree

__all__ = [
    "TREE_SUFFIX",
    "PoolConfig",
    "PoolError",
    "PoolStats",
    "SimulatorPool",
    "subsample",
]

# One tree per file, named for the world it records. The staging suffix is not a
# tree and is skipped when the pool is listed, so the leavings of an interrupted
# add are inert rather than half a world.
TREE_SUFFIX = ".json"
STAGING_SUFFIX = ".tmp"


class PoolError(ValueError):
    """The pool was asked for something it does not hold, or cannot name."""


@dataclass(frozen=True)
class PoolStats:
    """What the pool costs to dream over: its trees, their nodes, their bytes.

    ``nodes`` is the total across every tree, root nodes included — the same
    count :func:`len` gives a :class:`~dream_rsi.replay.ReplaySimulator` — and
    ``bytes`` is what the trees occupy on disk.
    """

    trees: int
    nodes: int
    bytes: int


@dataclass(frozen=True)
class PoolConfig:
    """How much of the pool one offline phase replays.

    ``limit`` is the number of worlds a dreaming round is allowed; ``None``, the
    default, is all of them. ``seed`` fixes which ones a limit keeps.
    """

    limit: int | None = None
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            # §3's V^m is an average over the trees replayed. A round of none of
            # them has no average to take, so this is refused where it is
            # written rather than a cycle later.
            raise ValueError(f"a dreaming round needs at least one world, got {self.limit}")


class SimulatorPool:
    """The trees under one directory, added atomically and read back by name.

    The directory is the whole of the pool's state: a pool built over an
    existing one inherits it, which is what makes a run started tomorrow dream
    over everything the runs before it recorded. Nothing is cached, because a
    cached listing is a listing that disagrees with the directory the moment
    another session adds to it.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def names(self) -> tuple[str, ...]:
        """Every tree in the pool, in name order.

        Sorted rather than in directory order so two sessions listing the same
        pool list it the same way (working rule 5); the driver's own names sort
        into the order the cycles ran. Only files this pool could have written
        are listed: the staging file of an interrupted add is not a tree, and
        neither is anything this pool would refuse to name.
        """
        if not self._directory.is_dir():
            return ()
        return tuple(
            sorted(
                name
                for name in (
                    path.name[: -len(TREE_SUFFIX)]
                    for path in self._directory.iterdir()
                    if path.is_file() and path.name.endswith(TREE_SUFFIX)
                )
                if _nameable(name)
            )
        )

    def add(self, name: str, tree: DiscoveryTree) -> None:
        """Add ``tree`` under ``name``, whole or not at all.

        Written beside its final path and renamed onto it, so a reader sees the
        tree it asked for or no tree at all — never a truncated one, which for a
        store whose point is being read back in a later session is the worse
        outcome. Adding a name the pool already holds replaces it: the driver
        re-adds a cycle's tree when that cycle is redone.
        """
        path = self._path(name)
        self._directory.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(path.name + STAGING_SUFFIX)
        tree.save(staging)
        try:
            os.replace(staging, path)
        except OSError:
            staging.unlink(missing_ok=True)
            raise

    def worlds(self, names: Iterable[str] | None = None) -> tuple[ReplayWorld, ...]:
        """The named trees as replay worlds, in the order asked for.

        ``None`` is the whole pool, in :meth:`names` order. A name the pool does
        not hold is an error rather than a gap: a dreaming round that quietly
        replayed fewer worlds than it was given would report a ``V^m`` averaged
        over a history nobody chose.
        """
        wanted = self.names() if names is None else tuple(names)
        return tuple(ReplayWorld(name=name, simulator=self._simulator(name)) for name in wanted)

    def stats(self) -> PoolStats:
        """Count the pool: trees, nodes, bytes. Every tree is read to count it."""
        trees = 0
        nodes = 0
        total = 0
        for name in self.names():
            path = self._path(name)
            trees += 1
            nodes += len(self._load(path))
            total += path.stat().st_size
        return PoolStats(trees=trees, nodes=nodes, bytes=total)

    def _simulator(self, name: str) -> ReplaySimulator:
        path = self._path(name)
        if not path.is_file():
            raise PoolError(f"the pool is missing a tree named {name!r}: {path}")
        return ReplaySimulator(self._load(path))

    def _load(self, path: Path) -> DiscoveryTree:
        try:
            return DiscoveryTree.load(path)
        except (OSError, ValueError) as exc:
            raise PoolError(f"{path} is not a readable tree: {exc}") from exc

    def _path(self, name: str) -> Path:
        """The file a tree of this name lives in, with the name checked first.

        A name is a name: the pool owns its directory, and a name that is a path
        would write a tree outside it and list a pool that cannot be read back.
        """
        if not _nameable(name):
            raise PoolError(f"a tree name must be a plain filename, got {name!r}")
        return self._directory / f"{name}{TREE_SUFFIX}"


def _nameable(name: str) -> bool:
    """Whether ``name`` names a tree in a pool: a plain filename, and not hidden."""
    return bool(name) and name == Path(name).name and not name.startswith(".")


def subsample(names: Sequence[str], config: PoolConfig | None = None) -> tuple[str, ...]:
    """The worlds one dreaming round replays, out of the history it was given.

    Returns a subsequence of ``names`` — the same names, in the same order — of
    at most ``config.limit`` entries. Nothing is removed from the pool itself:
    this bounds what one offline phase costs, and the next one is free to draw
    a different sample from the same store.
    """
    config = PoolConfig() if config is None else config
    ordered = tuple(names)
    if config.limit is None or len(ordered) <= config.limit:
        return ordered

    # PAPER-GAP: §3 expands the simulator pool every outer iteration — "expanding
    # the history available for the next offline improvement phase" — and never
    # states a retention or subsampling policy for a pool too large to replay in
    # full; §4 reports no ablation over history size either. The honest default
    # is therefore to use all of it, and a limit is opt-in. When one is set we
    # keep the newest tree — §3 appends 𝒯_t before dreaming, so it is the one
    # the policy in hand has not already been tuned against — and draw the rest
    # uniformly at random under the run's seed, rather than taking the newest
    # few: a window would make every older tree dead weight the moment the pool
    # outgrew the limit, and the paper's V^m is an average over the history, not
    # over its tail. Seeded because a dreaming round whose history depended on
    # iteration order would score two versions against different worlds
    # (working rule 5). Revisit if the authors' implementation lands (see
    # references/method.md).
    keep = {ordered[-1]}
    keep.update(random.Random(config.seed).sample(sorted(ordered[:-1]), config.limit - 1))
    return tuple(name for name in ordered if name in keep)
