"""Per-attempt workspaces and the filesystem snapshots nodes point at.

A node records "the resulting filesystem snapshot" of its attempt, and an
attempt beginning at that node "resumes the parent's saved workspace"
(Dream-RSI §3). Those two sentences are this module: :meth:`SnapshotStore.capture`
records a directory, :meth:`SnapshotStore.checkout` puts a recorded one back as
a fresh working directory, and the ``snapshot_ref`` it mints is what
:class:`~dream_rsi.tree.Node` carries between them.

Each attempt gets its own directory. The paper's exploration prompt fills in a
``$node_dir`` — "your own attempt directory --- exclude it when scanning sibling
``attempt_*/`` dirs" (§B.1) — so siblings are separate directories there too,
and here they are separate directories that cannot see each other at all: a
worker writes only into its checkout, and what its siblings inherit is the
parent snapshot, which no attempt can reach.

PAPER-GAP: the paper says a node preserves the complete filesystem state after
execution and never says how — whole copies or deltas, what fidelity, how
addressed. We store whole copies: it is obviously correct, and the storage cost
of a real tree is unknown until something measures it (issue #17; open a
follow-up if size turns out to be the binding constraint rather than building a
delta store speculatively, AGENTS.md rule 2). A snapshot preserves regular
files, directories, permission bits, and symlinks as links — resolving a link
would copy its target's bytes into the snapshot, and a link to something outside
the workspace would drag that in with it. References are content addresses (a
digest over the whole state), so an attempt that changed nothing costs no
storage and a re-recorded rollout mints identical refs, which is what keeps a
recorded tree byte-identical across runs (AGENTS.md rule 5). Modification times
are deliberately outside that state: a copy carries them along, but two states
differing only in an mtime are one state here, because a content address built
over wall-clock stamps would mint a fresh ref every time the same rollout was
recorded again and cost exactly that guarantee. Revisit if the authors'
implementation lands (see references/method.md).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dream_rsi import durable

__all__ = ["SnapshotError", "SnapshotStore"]

_SNAPSHOTS_DIRNAME = "snapshots"
_WORKSPACES_DIRNAME = "workspaces"

_CHUNK = 1 << 20


class SnapshotError(ValueError):
    """A snapshot could not be recorded, or a workspace could not be resumed."""


class SnapshotStore:
    """Filesystem snapshots on disk, and the attempt workspaces made from them.

    The store owns one directory: recorded states under ``snapshots/``, live
    attempt workspaces under ``workspaces/``. Nothing outside the store is
    written to, and neither directory is created until something is captured or
    checked out, so constructing a store touches no disk.

    Snapshots are immutable once recorded — a ref names one state forever, and
    the nodes referring to it are only correct as long as that holds. Attempts
    therefore never write inside ``snapshots/``; they write in their own
    checkout and capture the result.

    Captures and checkouts may run concurrently from several workers, which is
    how a rollout expands a batch (§3).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The directory holding the snapshots and the live workspaces."""
        return self._root

    def capture(self, directory: str | Path) -> str:
        """Record the state of ``directory`` and return its ``snapshot_ref``.

        The ref describes what was stored: it is the digest of the copy this
        made, so it stays a content address even if the workspace was still
        changing while it was read. Capturing a state already recorded returns
        the existing ref and copies nothing. The copy and the ref reach the
        device together (:mod:`dream_rsi.durable`), so a node a later session
        reads points at a state that survived the power loss with its tree.
        """
        source = Path(directory)
        # The copy below runs through the store's own staging directory, so a
        # store inside the captured directory would copy its own output, over
        # and over. A workspace *below* the store is the normal case and stays
        # allowed: that is where ``checkout`` puts one.
        if _contains(source, self._root):
            raise SnapshotError(
                f"snapshot store must not be inside the captured directory: {self._root}"
            )
        snapshots = self._root / _SNAPSHOTS_DIRNAME
        ref = _digest(source)
        if (snapshots / ref).is_dir():
            return ref

        snapshots.mkdir(parents=True, exist_ok=True)
        # Copied aside and renamed into place, so a reader never sees a
        # half-written snapshot under a ref that promises the whole state — and
        # flushed on the way (`durable`), so the ref survives a power loss with
        # bytes that do too rather than naming a copy the machine lost.
        staging = Path(tempfile.mkdtemp(dir=snapshots, prefix=".staging-"))
        try:
            state = staging / "state"
            durable.copy_tree(source, state)
            # Digested after the copy, not before: the ref has to describe the
            # bytes this actually stores. The two differ only if the workspace
            # changed while it was being read — an agent that left something
            # running in it — and then the earlier digest describes a state that
            # was never recorded anywhere.
            ref = _digest(state)
            target = snapshots / ref
            if not target.is_dir():
                try:
                    durable.publish(state, target)
                except OSError:
                    # Another worker captured the same state first. Same digest,
                    # same bytes: theirs will do.
                    if not target.is_dir():
                        raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return ref

    def materialize(self, ref: str, destination: str | Path) -> Path:
        """Write the state ``ref`` records into ``destination``, which must not exist."""
        source = self._root / _SNAPSHOTS_DIRNAME / _checked_ref(ref)
        if not source.is_dir():
            raise SnapshotError(f"unknown snapshot: {ref!r}")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=True)
        return target

    @contextmanager
    def checkout(self, ref: str, name: str) -> Iterator[Path]:
        """A fresh workspace holding the state ``ref`` records, removed on exit.

        ``name`` identifies the attempt and must not be in use by another
        checkout; two attempts sharing a directory is the isolation failure this
        exists to prevent. The workspace is removed however the body ends, so an
        attempt that raises leaves nothing for the next one to inherit — capture
        inside the block whatever the node should keep.
        """
        workspace = self._root / _WORKSPACES_DIRNAME / _checked_name(name)
        if workspace.exists():
            raise SnapshotError(f"workspace name {name!r} is already in use: {workspace}")
        self.materialize(ref, workspace)
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def _contains(directory: Path, other: Path) -> bool:
    """Is ``other`` ``directory`` itself or something under it?"""
    resolved = other.resolve()
    root = directory.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _checked_name(name: str) -> str:
    """Reject a workspace name that is not a single directory under the store.

    The name is joined onto the store and the directory it names is deleted
    afterwards, so ``..`` or a nested path would put both somewhere else.
    """
    if not name or name in {os.curdir, os.pardir} or Path(name).name != name:
        raise SnapshotError(f"workspace name must be a single directory, got {name!r}")
    return name


def _checked_ref(ref: str) -> str:
    """Reject anything that is not a ref this store could have minted."""
    nameable = isinstance(ref, str) and bool(ref) and ref not in {os.curdir, os.pardir}
    if not nameable or Path(ref).name != ref:
        raise SnapshotError(f"unknown snapshot: {ref!r}")
    return ref


def _digest(directory: Path) -> str:
    """A content address for ``directory``: the same state digests the same way.

    Entries are visited in a fixed order and each field is length-prefixed, so
    no two states can hash alike by running their paths together.
    """
    digest = hashlib.sha256()
    for relative, kind, mode, payload in _entries(directory):
        # ``os.fsencode``, not ``str.encode``: a filename the filesystem accepts
        # need not be valid UTF-8, and a workspace has to digest either way.
        for part in (kind.encode(), f"{mode:04o}".encode(), os.fsencode(relative), payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _entries(directory: Path) -> list[tuple[str, str, int, bytes]]:
    """``directory`` and every path under it as ``(relative, kind, mode, payload)``, sorted.

    The workspace root is an entry of its own, under the empty relative path: a
    copy preserves its mode too, so two states that differ only there are two
    states.
    """
    if not directory.is_dir() or directory.is_symlink():
        raise SnapshotError(f"not a workspace directory: {directory}")
    entries = [("", "dir", stat.S_IMODE(directory.lstat().st_mode), b"")]
    for path in _walk(directory):
        status = path.lstat()
        mode = stat.S_IMODE(status.st_mode)
        relative = path.relative_to(directory).as_posix()
        if stat.S_ISLNK(status.st_mode):
            entries.append((relative, "link", mode, os.fsencode(os.readlink(path))))
        elif stat.S_ISDIR(status.st_mode):
            entries.append((relative, "dir", mode, b""))
        elif stat.S_ISREG(status.st_mode):
            entries.append((relative, "file", mode, _file_digest(path)))
        else:
            # A socket or a fifo has no state a copy could preserve, and copying
            # one blocks on open rather than failing.
            raise SnapshotError(
                f"{path} is not a regular file, directory, or symlink, so it cannot be snapshotted"
            )
    entries.sort()
    return entries


def _walk(directory: Path) -> Iterator[Path]:
    """Every path under ``directory``, never descending through a symlink."""
    pending = [directory]
    while pending:
        for child in pending.pop().iterdir():
            yield child
            if child.is_dir() and not child.is_symlink():
                pending.append(child)


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.digest()
