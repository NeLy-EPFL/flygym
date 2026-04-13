from numbers import Number
from typing import Iterator, Hashable, TypeVar, Generic, Sequence, Literal
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

__all__ = [
    "Vec2",
    "Vec3",
    "Vec4",
    "Vec5",
    "Vec6",
    "Vec7",
    "Tree",
    "orderedset",
    "Rotation3D",
]

Vec2 = Float[np.ndarray, "2"]
Vec3 = Float[np.ndarray, "3"]
Vec4 = Float[np.ndarray, "4"]
Vec5 = Float[np.ndarray, "5"]
Vec6 = Float[np.ndarray, "6"]
Vec7 = Float[np.ndarray, "7"]

T = TypeVar("T", bound=Hashable)


class Tree(Generic[T]):
    """Minimal implementation of a tree data structure, made to parse and verify
    body skeletons without requiring extra dependency.

    Args:
        nodes:
            List of unique body segment identifiers.
        edges:
            List of directed (parent, child) tuples defining connections.

    Raises:
        ValueError:
            If graph is not a valid tree (has cycles, disconnected, duplicate nodes,
            self-loops, or parallel edges).
    """

    def __init__(self, nodes: list[T], edges: list[tuple[T, T]]) -> None:
        # Check for edges involving nonexistent nodes and self-loops
        nodes_set = set(nodes)
        if len(nodes_set) != len(nodes):
            raise ValueError("Tree contains duplicate nodes")
        for u, v in edges:
            if u not in nodes_set or v not in nodes_set:
                raise ValueError(f"Edge ({u}, {v}) not in tree")
            if u == v:
                raise ValueError(f"Edge ({u}, {v}) is a self-loop")

        # Check for parallel edges
        unique_edges = set(edges)
        if len(unique_edges) != len(edges):
            raise ValueError("Tree contains parallel edges")

        # Construct graph using adjacency list representation
        self.graph = {node: [] for node in nodes}
        self.in_degree = {node: 0 for node in nodes}
        for u, v in edges:
            self.graph[u].append(v)
            self.in_degree[v] += 1

        if not self._is_valid():
            raise ValueError("Tree is invalid")

    def _is_valid(self) -> bool:
        if len(self.graph) == 0:
            return True

        # Directed tree must have n-1 edges.
        n_edges = sum(len(neighbors) for neighbors in self.graph.values())
        if n_edges != len(self.graph) - 1:
            return False

        # Directed tree must have exactly one root and every other node exactly one parent.
        roots = [node for node, degree in self.in_degree.items() if degree == 0]
        if len(roots) != 1:
            return False
        if any(degree > 1 for degree in self.in_degree.values()):
            return False

        # All nodes must be reachable from the unique root via directed edges.
        visited = set()
        stack = [roots[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.graph[node])
        return len(visited) == len(self.graph)

    def dfs_edges(self, root: T) -> Iterator[tuple[T, T]]:
        """Yield edges in depth-first search order from root."""
        if root not in self.graph:
            raise ValueError(f"Root '{root}' not in tree")

        visited = set()
        stack = [(None, root)]  # (parent, child)
        while stack:
            parent, child = stack.pop()
            if child in visited:
                continue
            visited.add(child)
            if parent is not None:
                yield parent, child
            stack.extend((child, neighbor) for neighbor in reversed(self.graph[child]))


def orderedset(li: list) -> list:
    """Like set, but ordered (similar to dict keys in newer Python versions)."""
    return list(dict.fromkeys(li))


@dataclass(frozen=True)
class Rotation3D:
    """3D rotation representation in quaternion, axis-angle, xy-axes, z-axis, or Euler
    angles as allowed by MuJoCo. For details, see
    [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations).
    """

    format: Literal["quat", "axisangle", "xyaxes", "zaxis", "euler"]
    values: Sequence[Number]

    def __post_init__(self):
        expected_dims = {
            "quat": 4,
            "axisangle": 3,
            "xyaxes": 6,
            "zaxis": 3,
            "euler": 3,
        }
        if not (
            self.format in expected_dims
            and isinstance(self.values, Sequence)
            and all(isinstance(v, Number) for v in self.values)
        ):
            raise ValueError(
                f"Invalid rotation spec: format={self.format}, values={self.values}. "
                f"Format must be one of {list(expected_dims.keys())} and values must "
                "be a sequence of numbers."
            )
        if (dim := len(self.values)) != (exp_dim := expected_dims[self.format]):
            raise ValueError(
                f"Invalid rotation spec: format={self.format}, values={self.values}. "
                f"Format {self.format} should be {exp_dim}-dimensional, got {dim}."
            )

    def as_kwargs(self):
        """Convert to keyword arguments for MuJoCo MJCF elements as a dict.

        One should typically use `**` to expand the returned dict when passing to an
        MJCF element constructor. For example::

            rotation = Rotation3D("quat", (1, 0, 0, 0))
            camera = self.mjcf_root.worldbody.add(
                "camera", pos=pos_offset, **rotation.as_kwargs()
            )

        which expands to::

            camera = self.mjcf_root.worldbody.add(
                "camera", pos=pos_offset, quat=(1, 0, 0, 0)
            )
        """
        return {self.format: self.values}
