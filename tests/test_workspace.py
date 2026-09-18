"""Per-attempt workspaces and filesystem snapshots (issue #5)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dream_rsi.workspace import SnapshotError, SnapshotStore


def build_workspace(directory: Path) -> Path:
    """A workspace with nested directories, an executable, and an empty directory."""
    (directory / "src" / "deep").mkdir(parents=True)
    (directory / "src" / "deep" / "candidate.py").write_text("def solve(v):\n    return 0\n")
    (directory / "empty").mkdir()
    script = directory / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    return directory


def listing(directory: Path) -> list[tuple[str, int]]:
    """Every path under ``directory``, relative, with its permission bits."""
    return sorted(
        (path.relative_to(directory).as_posix(), path.lstat().st_mode & 0o777)
        for path in directory.rglob("*")
    )


def test_a_materialized_workspace_is_the_captured_one_again(tmp_path):
    # A child starts from its parent's snapshot, so what comes back has to be the
    # whole recorded state: nested directories, empty ones, and the mode bits an
    # agent's own build script depends on.
    store = SnapshotStore(tmp_path / "store")
    source = build_workspace(tmp_path / "source")

    ref = store.capture(source)
    restored = store.materialize(ref, tmp_path / "restored")

    assert listing(restored) == listing(source)
    candidate = restored / "src" / "deep" / "candidate.py"
    assert candidate.read_text() == "def solve(v):\n    return 0\n"
    assert os.access(restored / "run.sh", os.X_OK)


def test_a_symlink_is_restored_as_a_symlink(tmp_path):
    # Resolving it instead would copy the target's bytes into the snapshot — and
    # a link to somewhere that no longer exists would fail the capture outright.
    store = SnapshotStore(tmp_path / "store")
    source = build_workspace(tmp_path / "source")
    (source / "entry.py").symlink_to("src/deep/candidate.py")

    restored = store.materialize(store.capture(source), tmp_path / "restored")

    assert (restored / "entry.py").is_symlink()
    assert os.readlink(restored / "entry.py") == "src/deep/candidate.py"


def test_snapshots_of_equal_states_are_the_same_snapshot(tmp_path):
    # Refs are content addresses: a rollout records the same tree twice over
    # (AGENTS.md rule 5), and an attempt that changed nothing costs no storage.
    store = SnapshotStore(tmp_path / "store")
    source = build_workspace(tmp_path / "source")
    copy = tmp_path / "copy"
    shutil.copytree(source, copy, symlinks=True)

    ref = store.capture(source)
    assert store.capture(copy) == ref

    (copy / "src" / "deep" / "candidate.py").write_text("def solve(v):\n    return 1\n")
    assert store.capture(copy) != ref

    (source / "run.sh").chmod(0o644)
    assert store.capture(source) != ref


def test_siblings_resumed_from_one_snapshot_cannot_see_each_other(tmp_path):
    # Two workers expanding sibling nodes run against the same parent snapshot at
    # the same time; neither may observe the other's writes, and neither may
    # write back into the state their children will inherit.
    store = SnapshotStore(tmp_path / "store")
    source = build_workspace(tmp_path / "source")
    base = store.capture(source)

    with (
        store.checkout(base, "attempt_000000") as first,
        store.checkout(base, "attempt_000001") as second,
    ):
        # Each attempt starts from the parent's saved workspace (§3).
        assert listing(first) == listing(source)
        assert listing(second) == listing(source)

        (first / "notes.txt").write_text("first")
        (second / "notes.txt").write_text("second")

        assert first != second
        assert (first / "notes.txt").read_text() == "first"
        assert (second / "notes.txt").read_text() == "second"
        first_ref = store.capture(first)

    assert not (store.materialize(base, tmp_path / "base") / "notes.txt").exists()
    assert (store.materialize(first_ref, tmp_path / "first") / "notes.txt").read_text() == "first"


def test_the_workspace_is_removed_even_when_the_attempt_raises(tmp_path):
    # A worker that dies must not leave its scratch state behind for the next
    # attempt to inherit, nor leak the disk a long rollout needs.
    store = SnapshotStore(tmp_path / "store")
    base = store.capture(build_workspace(tmp_path / "source"))

    with (
        pytest.raises(RuntimeError, match="worker exploded"),
        store.checkout(base, "attempt_000000") as workspace,
    ):
        abandoned = workspace
        (workspace / "half-written.txt").write_text("...")
        raise RuntimeError("worker exploded")

    assert not abandoned.exists()
    with store.checkout(base, "attempt_000000") as reused:
        assert not (reused / "half-written.txt").exists()


def test_a_name_already_checked_out_is_refused(tmp_path):
    # Handing two live attempts the same directory is exactly the isolation
    # failure above, so it fails loudly instead of silently sharing.
    store = SnapshotStore(tmp_path / "store")
    base = store.capture(build_workspace(tmp_path / "source"))

    with store.checkout(base, "attempt_000000") as live:
        with (
            pytest.raises(SnapshotError, match="already in use"),
            store.checkout(base, "attempt_000000"),
        ):
            pass
        assert live.is_dir(), "the refused checkout must not disturb the live one"


@pytest.mark.parametrize("name", ["..", "nested/name", ""])
def test_a_workspace_name_that_is_not_one_directory_is_refused(tmp_path, name):
    # The name is joined onto the store and the directory is deleted afterwards:
    # a name that escapes would put the workspace, and the delete, anywhere.
    store = SnapshotStore(tmp_path / "store")
    base = store.capture(build_workspace(tmp_path / "source"))

    with pytest.raises(SnapshotError, match="workspace name"), store.checkout(base, name):
        pass


def test_an_unknown_snapshot_is_refused(tmp_path):
    # Starting from an empty directory instead would look like an agent that
    # deleted the workspace rather than a tree pointing at a snapshot that is gone.
    store = SnapshotStore(tmp_path / "store")

    with pytest.raises(SnapshotError, match="unknown snapshot"):
        store.materialize("0" * 64, tmp_path / "restored")
