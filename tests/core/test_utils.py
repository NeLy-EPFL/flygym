"""Unit tests for flygym.utils (math, exceptions)."""

import pytest
import numpy as np

from flygym.utils.math import Tree, orderedset, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError


# ==============================================================================
# orderedset
# ==============================================================================


class TestOrderedset:
    def test_deduplication(self):
        result = orderedset([1, 2, 2, 3, 1])
        assert result == [1, 2, 3]

    def test_preserves_order(self):
        result = orderedset([3, 1, 2, 1, 3])
        assert result == [3, 1, 2]

    def test_empty(self):
        assert orderedset([]) == []

    def test_strings(self):
        result = orderedset(["b", "a", "b", "c", "a"])
        assert result == ["b", "a", "c"]

    def test_single_element(self):
        assert orderedset([42]) == [42]


# ==============================================================================
# Tree
# ==============================================================================


class TestTree:
    def test_valid_tree(self):
        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (1, 3), (3, 4)]
        tree = Tree(nodes, edges)
        assert tree is not None

    def test_dfs_edges_visits_all_nodes(self):
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("a", "c"), ("c", "d")]
        tree = Tree(nodes, edges)
        visited = set()
        for parent, child in tree.dfs_edges("a"):
            visited.add(parent)
            visited.add(child)
        assert visited == set(nodes)

    def test_dfs_edges_order(self):
        # Simple chain: root -> a -> b
        nodes = ["root", "a", "b"]
        edges = [("root", "a"), ("a", "b")]
        tree = Tree(nodes, edges)
        dfs = list(tree.dfs_edges("root"))
        # First edge must start from root
        assert dfs[0][0] == "root"

    def test_single_node_tree(self):
        tree = Tree([1], [])
        assert list(tree.dfs_edges(1)) == []

    def test_cycle_raises(self):
        nodes = [1, 2, 3]
        edges = [(1, 2), (2, 3), (3, 1)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_disconnected_raises(self):
        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (3, 4)]  # Two components
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_parallel_edges_raises(self):
        nodes = [1, 2]
        edges = [(1, 2), (1, 2)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_self_loop_raises(self):
        nodes = [1, 2]
        edges = [(1, 2), (1, 1)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_duplicate_nodes_raises(self):
        nodes = [1, 1, 2]
        edges = [(1, 2)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_edge_with_nonexistent_node_raises(self):
        nodes = [1, 2]
        edges = [(1, 99)]  # 99 not in nodes
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_invalid_root_raises(self):
        tree = Tree([1, 2], [(1, 2)])
        with pytest.raises(ValueError):
            list(tree.dfs_edges(99))


# ==============================================================================
# Rotation3D
# ==============================================================================


class TestRotation3D:
    def test_quat_valid(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        assert rot.format == "quat"
        assert tuple(rot.values) == (1, 0, 0, 0)

    def test_euler_valid(self):
        rot = Rotation3D("euler", (0.0, 0.0, 0.0))
        assert rot.format == "euler"

    def test_axisangle_valid(self):
        rot = Rotation3D("axisangle", (0, 0, 1))
        assert rot.format == "axisangle"

    def test_xyaxes_valid(self):
        rot = Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0))
        assert rot.format == "xyaxes"

    def test_zaxis_valid(self):
        rot = Rotation3D("zaxis", (0, 0, 1))
        assert rot.format == "zaxis"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            Rotation3D("badformat", (1, 0, 0, 0))

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError):
            Rotation3D("quat", (1, 0, 0))  # quat needs 4, got 3

    def test_as_kwargs_quat(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        kwargs = rot.as_kwargs()
        assert kwargs == {"quat": (1, 0, 0, 0)}

    def test_as_kwargs_euler(self):
        rot = Rotation3D("euler", (0.1, 0.2, 0.3))
        kwargs = rot.as_kwargs()
        assert kwargs == {"euler": (0.1, 0.2, 0.3)}

    def test_frozen_dataclass(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        with pytest.raises((AttributeError, TypeError)):
            rot.format = "euler"

    def test_non_number_values_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Rotation3D("euler", ("a", "b", "c"))


# ==============================================================================
# FlyGymInternalError
# ==============================================================================


class TestFlyGymInternalError:
    def test_is_exception(self):
        err = FlyGymInternalError("test")
        assert isinstance(err, Exception)

    def test_message(self):
        msg = "something went wrong internally"
        err = FlyGymInternalError(msg)
        assert str(err) == msg

    def test_can_be_raised(self):
        with pytest.raises(FlyGymInternalError):
            raise FlyGymInternalError("boom")
