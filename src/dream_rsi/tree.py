"""Discovery tree schema and on-disk format.

A discovery tree is rooted at ``r``, the initial workspace state. Each non-root
node has exactly one *primary parent* — the node whose saved workspace and
accumulated observations the discovery agent resumed from — and records the
outcome of one generation-evaluation attempt: the generated artifact, the
evaluation diagnostics, the resulting filesystem snapshot, and the score
``s_v`` under the task-scoring protocol (Dream-RSI §3, "Discovery trees and the
shared decision interface").

This is a tree, not a DAG: there is no merge or multi-parent semantics, and the
loader rejects any on-disk node that tries to express one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "DiscoveryTree",
    "Node",
    "SchemaVersionError",
    "TreeError",
    "TreeInvariantError",
    "eligible_nodes",
]

# Bumped whenever the on-disk layout changes in a way older readers would
# misread. The loader refuses anything it does not recognise rather than
# guessing, so recorded trees and fixtures fail loudly instead of silently
# decoding into the wrong shape.
SCHEMA_VERSION = 1

# PAPER-GAP: the paper does not specify a node id format, only that each node is
# identifiable and has one primary parent. We mint zero-padded sequential ids
# ("n000000") so that lexicographic order equals creation order, which gives
# replay a deterministic iteration order without a separate sort key. Revisit if
# the authors' implementation lands (see references/method.md).
_ID_TEMPLATE = "n{:06d}"
_MINTED_ID = re.compile(r"^n(\d{6,})$")


class TreeError(ValueError):
    """Base class for discovery tree errors."""


class TreeInvariantError(TreeError):
    """A tree invariant (single root, one parent, unique ids, no cycles) is broken."""


class SchemaVersionError(TreeError):
    """An on-disk tree carries a schema version this code cannot read."""


@dataclass(frozen=True)
class Node:
    """One attempt in the discovery tree, or the root workspace state.

    ``parent_id`` is ``None`` for the root and a node id for every other node —
    a single field, because a node has exactly one primary parent.

    ``snapshot_ref`` is an opaque handle to the filesystem state after
    execution. How snapshots are actually stored (whole copies versus deltas) is
    deliberately not this module's business; see issue #5.
    """

    id: str
    parent_id: str | None = None
    score: float | None = None
    artifact: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    observations: tuple[str, ...] = ()
    snapshot_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observations"] = list(self.observations)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> Node:
        if not isinstance(payload, dict):
            raise TreeInvariantError(f"node must be an object, got {type(payload).__name__}")

        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise TreeInvariantError(f"node has unknown field(s): {', '.join(unknown)}")

        if "id" not in payload:
            raise TreeInvariantError("node is missing required field: id")
        node_id = payload["id"]
        if not isinstance(node_id, str) or not node_id:
            raise TreeInvariantError(f"node id must be a non-empty string, got {node_id!r}")

        parent_id = payload.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            # A list here is the DAG that this schema refuses to be.
            raise TreeInvariantError(
                f"node {node_id!r}: parent_id must be a single node id or null, "
                f"got {parent_id!r}"
            )

        score = payload.get("score")
        if score is not None and not isinstance(score, (int, float)):
            raise TreeInvariantError(f"node {node_id!r}: score must be a number or null")

        diagnostics = payload.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise TreeInvariantError(f"node {node_id!r}: diagnostics must be an object")

        observations = payload.get("observations", ())
        if not isinstance(observations, (list, tuple)):
            raise TreeInvariantError(f"node {node_id!r}: observations must be a list")

        return cls(
            id=node_id,
            parent_id=parent_id,
            score=score,
            artifact=payload.get("artifact"),
            diagnostics=dict(diagnostics),
            observations=tuple(observations),
            snapshot_ref=payload.get("snapshot_ref"),
        )


class DiscoveryTree:
    """An immutable-by-convention collection of nodes with the tree invariants enforced.

    Invariants, checked on construction and after every mutation: ids are
    unique, exactly one node is the root, every other node names a parent that
    exists, and every node is reachable from the root (so there are no cycles).
    """

    def __init__(self, nodes: Iterable[Node]) -> None:
        by_id: dict[str, Node] = {}
        for node in nodes:
            if node.id in by_id:
                raise TreeInvariantError(f"duplicate node id: {node.id!r}")
            by_id[node.id] = node
        self._nodes = by_id
        self._root_id = _validate(by_id)
        self._next_index = _next_index(by_id)

    @classmethod
    def with_root(cls, **fields: Any) -> DiscoveryTree:
        """Start a tree containing only the root workspace state."""
        return cls([Node(id=_ID_TEMPLATE.format(0), parent_id=None, **fields)])

    @property
    def root_id(self) -> str:
        return self._root_id

    def __len__(self) -> int:
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscoveryTree):
            return NotImplemented
        return self._nodes == other._nodes

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def iter_nodes(self) -> Iterator[Node]:
        """Every node, in ascending id order. Stable across runs and reloads."""
        for node_id in sorted(self._nodes):
            yield self._nodes[node_id]

    def children(self, node_id: str) -> tuple[Node, ...]:
        """The recorded children of ``node_id``, in ascending id order."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        return tuple(n for n in self.iter_nodes() if n.parent_id == node_id)

    def add_child(self, parent_id: str, **fields: Any) -> Node:
        """Append a node under ``parent_id`` with a freshly minted id."""
        if parent_id not in self._nodes:
            raise TreeInvariantError(f"unknown parent id: {parent_id!r}")
        node_id = _ID_TEMPLATE.format(self._next_index)
        while node_id in self._nodes:
            self._next_index += 1
            node_id = _ID_TEMPLATE.format(self._next_index)
        node = Node(id=node_id, parent_id=parent_id, **fields)
        self._nodes[node.id] = node
        self._next_index += 1
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "root_id": self._root_id,
            "nodes": [node.to_dict() for node in self.iter_nodes()],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DiscoveryTree:
        if not isinstance(payload, dict):
            raise TreeInvariantError(f"tree must be an object, got {type(payload).__name__}")

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unreadable tree schema version {version!r}; this code reads {SCHEMA_VERSION}"
            )

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise TreeInvariantError("tree is missing required field: nodes")

        tree = cls(Node.from_dict(raw) for raw in raw_nodes)

        declared_root = payload.get("root_id")
        if declared_root is not None and declared_root != tree.root_id:
            raise TreeInvariantError(
                f"declared root_id {declared_root!r} is not the tree's root {tree.root_id!r}"
            )
        return tree

    def save(self, path: str | Path) -> None:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DiscoveryTree:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def eligible_nodes(tree: DiscoveryTree) -> tuple[str, ...]:
    """``A(T) = {r} ∪ {v ∈ T : v is a leaf}``, root first, then leaves by id (§3).

    The root stays selectable whether or not it has children — that is how a
    rollout opens further branches — so it appears exactly once even when it is
    itself a leaf.

    This lives beside the tree rather than beside either caller because the
    online rollout and offline replay "use this same decision interface" (§3):
    ``A(T)`` is read off whichever tree the policy is currently observing, the
    one being built online or the revealed subtree of a frozen world.
    """
    root_id = tree.root_id
    parents = {node.parent_id for node in tree.iter_nodes()}
    leaves = tuple(
        node.id for node in tree.iter_nodes() if node.id != root_id and node.id not in parents
    )
    return (root_id, *leaves)


def _validate(nodes: Mapping[str, Node]) -> str:
    roots = sorted(node.id for node in nodes.values() if node.parent_id is None)
    if not roots:
        raise TreeInvariantError("tree has no root node (no node with parent_id null)")
    if len(roots) > 1:
        raise TreeInvariantError(f"tree has more than one root node: {', '.join(roots)}")
    root_id = roots[0]

    for node in nodes.values():
        if node.parent_id is not None and node.parent_id not in nodes:
            raise TreeInvariantError(f"node {node.id!r} names an unknown parent {node.parent_id!r}")

    # Every node has exactly one existing parent and there is exactly one root,
    # so anything the root cannot reach sits on a parent cycle.
    reachable = {root_id}
    frontier = [root_id]
    children: dict[str, list[str]] = {}
    for node in nodes.values():
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.id)
    while frontier:
        for child in children.get(frontier.pop(), ()):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)

    orphaned = sorted(set(nodes) - reachable)
    if orphaned:
        raise TreeInvariantError(f"parent cycle among node(s): {', '.join(orphaned)}")
    return root_id


def _next_index(nodes: Mapping[str, Node]) -> int:
    indices = [int(m.group(1)) for m in map(_MINTED_ID.match, nodes) if m]
    return max(indices) + 1 if indices else 0
