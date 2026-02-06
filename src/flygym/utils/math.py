from numbers import Number
from typing import Iterator, Hashable, TypeVar, Generic, Sequence, Literal, Annotated
from dataclasses import dataclass
from collections.abc import Collection

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=Hashable)
Vec2 = Annotated[npt.NDArray[np.float64], Literal[2]]
Vec3 = Annotated[npt.NDArray[np.float64], Literal[3]]
Vec4 = Annotated[npt.NDArray[np.float64], Literal[4]]
Vec5 = Annotated[npt.NDArray[np.float64], Literal[5]]
Vec6 = Annotated[npt.NDArray[np.float64], Literal[6]]
Vec7 = Annotated[npt.NDArray[np.float64], Literal[7]]


class Tree(Generic[T]):
    """Minimal implementation of a tree data structure, made to parse and verify
    body skeletons without requiring extra dependency."""

    def __init__(self, nodes: Collection[T], edges: Collection[tuple[T, T]]) -> None:
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
        unique_edges = set(frozenset(edge) for edge in edges)
        if len(unique_edges) != len(edges):
            raise ValueError("Tree contains parallel edges")

        # Construct graph using adjacency list representation
        self.graph = {node: [] for node in nodes}
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)

        if not self._is_valid():
            raise ValueError("Tree is invalid")

    def _is_valid(self) -> bool:
        if len(self.graph) == 0:
            return True

        # Check if the graph has the right number of edges
        n_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        if n_edges != len(self.graph) - 1:
            return False

        # DFS from an arbitrary node to check connectivity
        visited = set()
        stack = [next(iter(self.graph))]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.graph[node])
        return len(visited) == len(self.graph)

    def dfs_edges(self, root: T) -> Iterator[tuple[T, T]]:
        """Yield edges in DFS order starting from the root."""
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
        return {self.format: self.values}
