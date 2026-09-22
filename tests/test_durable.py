"""Crash durability: the flushes that let a published name outlive the machine (issue #47).

A power loss is the one failure a test cannot stage — an in-process failure
cannot lose a page cache — so what is pinned here is the contract at the syscall
boundary: the bytes and the names are flushed, in the order the decision in
:mod:`dream_rsi.durable` states.

Only the primitives live here. Every writer — the pool's ``add``, a tree's
``save``, a rollout's ``save``, a snapshot's ``capture`` and the driver's own
files — is a call into them, and is tested where it is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dream_rsi import durable


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot open a directory to sync it")
def test_creating_a_directory_flushes_each_parent_that_names_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that was just created is named only in its parent until that is flushed.

    Flushing the files inside a directory does not persist the entry naming the
    directory, so the tree of directories a run creates — ``cycles/``,
    ``cycles/cycle_000``, a cycle's store — could come back from a power loss
    missing its top, taking finished cycles with it. Every level this creates
    therefore flushes the directory holding it, and nothing more: an existing
    directory is asked for on every attempt, so paying a device round trip for
    one would put the cost on the hot path.
    """
    root = tmp_path / "run"
    root.mkdir()
    synced: list[int] = []

    def fsync(descriptor: int) -> None:
        synced.append(descriptor)

    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", fsync)

        store = root / "cycles" / "cycle_000" / "store"
        durable.mkdir(store)

        assert store.is_dir()
        assert len(synced) == 3, "one flush per level that was created"

        synced.clear()
        durable.mkdir(store)

        assert synced == [], "an existing directory costs no device round trip"
