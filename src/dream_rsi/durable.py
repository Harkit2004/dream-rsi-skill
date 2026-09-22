"""Publishing a run's files so that a power loss cannot undo them (issue #47).

A run publishes its files two ways: staged beside the final name and renamed
into place, or written where a later session reads it. Both are atomic against
an interrupted *process* — a reader never sees half a file — and neither is, by
itself, durable against the machine going away: ``os.replace`` orders the rename
against the file's data only once that data has reached the device, and a rename
is itself a change to the directory holding the name.

**The decision, stated once.** A published file is durable. :func:`write` flushes
a file's bytes and then the directory holding it; :func:`publish` renames a
file whose bytes were flushed and then flushes the directory the name appeared
in; :func:`mkdir` does the same for the directories a file is put in;
:func:`copy_tree` is :func:`write` for a whole copied state. So when a name
appears, its bytes are on the device, and — where the platform can sync a
directory at all — so is the rename. A cycle publishes in the order its record
needs: the policy it deployed, the rollout's tree and round log, the tree in the
pool, then the policy it selected and the timing, and the record last of all.
That is what lets :func:`~dream_rsi.run._resume` trust a record: a cycle whose
record exists is a cycle whose tree and policies reached the device before it
did. What a power loss can cost is the cycle in flight, never one that finished.

**The writers agree.** :meth:`~dream_rsi.tree.DiscoveryTree.save`,
:meth:`~dream_rsi.pool.SimulatorPool.add`,
:meth:`~dream_rsi.orchestrator.Rollout.save`,
:meth:`~dream_rsi.workspace.SnapshotStore.capture` and the driver's own files
all go through this module, so there is one answer to "what does a run
directory promise after the machine dies" rather than one per file.

**What it is not.** This is not a repair tool and not a file-integrity check:
what it guarantees is the order in which bytes and names reach the device.
Windows cannot open a directory to sync it (see :func:`sync_directory`), so
there a rename is only as durable as the filesystem's own journal made it; the
flushed files are flushed on both platforms.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["copy_tree", "mkdir", "publish", "sync_directory", "write"]


def write(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` with its bytes and its name on the device.

    ``Path.write_text`` plus the two flushes that make the result outlive the
    machine: the file's own, and its directory's. The second is not decoration —
    a file that was just created is not there for a later session until the
    directory holding its name is, whichever way the bytes got written.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(target.parent)


def publish(staging: str | Path, target: str | Path) -> None:
    """Rename ``staging`` onto ``target`` and flush the name into its directory.

    ``staging`` must have been written through :func:`write` — that call is what
    put its bytes on the device, so by the time the name is visible it is visible
    over a whole file. The sync after the rename is the other half: a rename is
    a change to the directory, and until that directory is flushed the old name
    is what a later session may find.
    """
    destination = Path(target)
    os.replace(Path(staging), destination)
    sync_directory(destination.parent)


def mkdir(directory: str | Path) -> None:
    """Create ``directory``, flushing every entry the creation added.

    ``Path.mkdir(parents=True, exist_ok=True)`` and the syncs a file's flush does
    not do for it: a directory that was just created is named only in its parent
    until that parent reaches the device, so every level this creates gets its
    parent flushed, deepest last. Without it a power loss can take a whole
    finished cycle's directory out of ``cycles/`` while the files inside it were
    flushed — the cycle would be redone, which is exactly what the record's
    durability exists to prevent.

    An existing directory costs nothing, which matters because the hot path —
    a snapshot per attempt, a workspace per cycle — asks for these over and over.
    """
    target = Path(directory)
    if target.is_dir():
        return
    created: list[Path] = []
    current = target
    while not current.is_dir():
        created.append(current)
        if current.parent == current:
            break
        current = current.parent
    target.mkdir(parents=True, exist_ok=True)
    for level in reversed(created):
        sync_directory(level.parent)


def copy_tree(source: str | Path, target: str | Path) -> None:
    """Copy a directory tree, flushing every file and directory entry it writes.

    ``shutil.copytree`` with the durability folded into the copy: each file is
    flushed while the copy still has it open for writing, and the directories'
    entries — ``target``'s own included — are flushed afterwards, children
    first. Metadata is applied before the flush, so what is flushed is the state
    the digest will name rather than a copy of it that still needs its mode set.
    """
    destination = Path(target)
    shutil.copytree(source, destination, symlinks=True, copy_function=_flushed_copy)
    _sync_directories(destination)
    sync_directory(destination.parent)


def sync_directory(directory: str | Path) -> None:
    """Flush a directory's own entries to the device, where the platform can.

    A rename's durability lives here: only a sync of the directory holding the
    name makes the name survive a power loss. Windows cannot open a directory to
    sync it, so there this is a no-op and a rename is left to the filesystem's
    own ordering; the files themselves are flushed on both platforms.
    """
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
        return
    descriptor = os.open(Path(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flushed_copy(source: str, target: str, *, follow_symlinks: bool = True) -> str:
    """``shutil.copy2`` with the finished copy flushed before the handle closes.

    The handle is opened writable before ``copystat`` runs, and the flush comes
    after it: a flush covers the file's metadata as well as its bytes, so it has
    to see the mode and timestamps the copy is supposed to have — and a
    read-only copy (which ``copystat`` can make) could not be opened for a flush
    afterwards.
    """
    shutil.copyfile(source, target, follow_symlinks=follow_symlinks)
    with open(target, "ab") as handle:
        shutil.copystat(source, target, follow_symlinks=follow_symlinks)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _sync_directories(root: Path) -> None:
    """Flush every directory under ``root``, children before parents.

    Bottom-up because a parent's entry for a child only means anything once the
    child is on the device. Symlinks are not followed: a link is copied as a
    link, and whatever it names is not part of this tree.
    """
    if os.name == "nt":  # pragma: no cover - no directory fsync on Windows
        return
    pending = [root]
    directories: list[Path] = []
    while pending:
        current = pending.pop()
        directories.append(current)
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
    for current in reversed(directories):
        sync_directory(current)
