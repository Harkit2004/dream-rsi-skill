"""Schema, invariants and on-disk round-trip for the discovery tree (issue #1)."""

import json

import pytest

from dream_rsi.tree import (
    SCHEMA_VERSION,
    DiscoveryTree,
    Node,
    SchemaVersionError,
    TreeInvariantError,
)


def _sample_tree() -> DiscoveryTree:
    tree = DiscoveryTree.with_root(snapshot_ref="snap/root")
    a = tree.add_child(
        tree.root_id,
        score=0.5,
        artifact="solve_a.py",
        diagnostics={"tests_passed": 3, "runtime_s": 1.25},
        observations=("compiles", "3/4 cases pass"),
        snapshot_ref="snap/a",
    )
    tree.add_child(tree.root_id, score=-2.0, artifact="solve_b.py", snapshot_ref="snap/b")
    tree.add_child(a.id, score=0.9, artifact="solve_c.py", snapshot_ref="snap/c")
    return tree


def test_round_trip_preserves_every_field(tmp_path):
    tree = _sample_tree()
    path = tmp_path / "tree.json"
    tree.save(path)
    loaded = DiscoveryTree.load(path)

    assert loaded == tree
    assert loaded.root_id == tree.root_id
    assert [n.id for n in loaded.iter_nodes()] == [n.id for n in tree.iter_nodes()]
    for original in tree.iter_nodes():
        assert loaded.node(original.id) == original


def test_round_trip_is_byte_identical_when_resaved(tmp_path):
    tree = _sample_tree()
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    tree.save(first)
    DiscoveryTree.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_iteration_order_is_stable_and_sorted():
    tree = _sample_tree()
    assert [n.id for n in tree.iter_nodes()] == [n.id for n in tree.iter_nodes()]
    assert [n.id for n in tree.iter_nodes()] == sorted(n.id for n in tree.iter_nodes())


def test_minted_ids_sort_in_creation_order():
    tree = _sample_tree()
    ids = [n.id for n in tree.iter_nodes()]
    assert ids == sorted(ids)
    extra = tree.add_child(tree.root_id)
    assert extra.id > max(ids)


def test_children_are_returned_in_stable_order():
    tree = _sample_tree()
    root_children = tree.children(tree.root_id)
    assert [n.id for n in root_children] == sorted(n.id for n in root_children)
    assert tree.children(tree.root_id) == root_children


def test_duplicate_id_raises():
    with pytest.raises(TreeInvariantError, match="duplicate"):
        DiscoveryTree([Node(id="n000000"), Node(id="n000000", parent_id="n000000")])


def test_missing_root_raises():
    with pytest.raises(TreeInvariantError, match="root"):
        DiscoveryTree([Node(id="a", parent_id="b"), Node(id="b", parent_id="a")])


def test_multiple_roots_raises():
    with pytest.raises(TreeInvariantError, match="root"):
        DiscoveryTree([Node(id="a"), Node(id="b")])


def test_unknown_parent_raises():
    with pytest.raises(TreeInvariantError, match="parent"):
        DiscoveryTree([Node(id="a"), Node(id="b", parent_id="ghost")])


def test_cycle_raises():
    nodes = [
        Node(id="root"),
        Node(id="a", parent_id="b"),
        Node(id="b", parent_id="a"),
    ]
    with pytest.raises(TreeInvariantError, match="cycle"):
        DiscoveryTree(nodes)


def test_add_child_rejects_unknown_parent():
    tree = DiscoveryTree.with_root()
    with pytest.raises(TreeInvariantError, match="parent"):
        tree.add_child("ghost")


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_future_schema_version_fails_loudly(tmp_path):
    tree = _sample_tree()
    path = tmp_path / "tree.json"
    tree.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION + 1
    _write(path, payload)
    with pytest.raises(SchemaVersionError):
        DiscoveryTree.load(path)


def test_missing_schema_version_fails_loudly(tmp_path):
    path = _write(tmp_path / "tree.json", {"nodes": [{"id": "a", "parent_id": None}]})
    with pytest.raises(SchemaVersionError):
        DiscoveryTree.load(path)


def test_two_parents_on_disk_fails_loudly(tmp_path):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            {"id": "root", "parent_id": None},
            {"id": "a", "parent_id": ["root", "other"]},
        ],
    }
    path = _write(tmp_path / "tree.json", payload)
    with pytest.raises(TreeInvariantError, match="parent_id"):
        DiscoveryTree.load(path)


def test_unknown_node_field_fails_loudly(tmp_path):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "nodes": [{"id": "root", "parent_id": None, "parents": ["x"]}],
    }
    path = _write(tmp_path / "tree.json", payload)
    with pytest.raises(TreeInvariantError, match="unknown"):
        DiscoveryTree.load(path)


def test_missing_node_field_fails_loudly(tmp_path):
    payload = {"schema_version": SCHEMA_VERSION, "nodes": [{"parent_id": None}]}
    path = _write(tmp_path / "tree.json", payload)
    with pytest.raises(TreeInvariantError, match="id"):
        DiscoveryTree.load(path)


def test_node_lookup_of_unknown_id_raises():
    tree = DiscoveryTree.with_root()
    with pytest.raises(KeyError):
        tree.node("nope")
