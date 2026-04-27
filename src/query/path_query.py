"""Reachability and shortest-path queries over the knowledge graph.

Provides BFS-based path finding between entities, optionally
constrained to specific relation types.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.store.triple_store import TripleStore, Triple


@dataclass
class PathResult:
    """Result of a path query between two entities.

    Attributes:
        source: Starting entity.
        target: Target entity.
        path: List of (entity, relation, entity) hops, empty if unreachable.
        length: Number of hops (-1 if unreachable).
    """

    source: str
    target: str
    path: list[tuple[str, str, str]]
    length: int

    @property
    def reachable(self) -> bool:
        """Whether the target is reachable from the source."""
        return self.length >= 0


class PathQueryEngine:
    """BFS-based path query engine over a triple store.

    Supports reachability checks, shortest-path finding, and
    neighborhood exploration with optional relation filters.

    Args:
        store: The triple store to query.
    """

    def __init__(self, store: TripleStore) -> None:
        self.store = store

    def shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
        relations: Optional[set[str]] = None,
        directed: bool = True,
    ) -> PathResult:
        """Find the shortest path between two entities using BFS.

        Args:
            source: Starting entity.
            target: Target entity.
            max_depth: Maximum number of hops to explore.
            relations: If provided, only traverse these relation types.
            directed: If False, also traverse edges in reverse.

        Returns:
            A PathResult with the shortest path or empty if unreachable.
        """
        if source == target:
            return PathResult(source, target, [], 0)

        visited: set[str] = {source}
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
        queue.append((source, []))

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            neighbors = self._get_neighbors(current, relations, directed)
            for relation, neighbor in neighbors:
                if neighbor == target:
                    full_path = path + [(current, relation, neighbor)]
                    return PathResult(source, target, full_path, len(full_path))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(current, relation, neighbor)]))

        return PathResult(source, target, [], -1)

    def reachable(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
        relations: Optional[set[str]] = None,
    ) -> bool:
        """Check if target is reachable from source.

        Args:
            source: Starting entity.
            target: Target entity.
            max_depth: Maximum search depth.
            relations: Optional relation filter.

        Returns:
            True if a path exists.
        """
        return self.shortest_path(source, target, max_depth, relations).reachable

    def neighbors(
        self,
        entity: str,
        depth: int = 1,
        relations: Optional[set[str]] = None,
    ) -> set[str]:
        """Find all entities within a given hop distance.

        Args:
            entity: The starting entity.
            depth: Maximum number of hops.
            relations: Optional relation filter.

        Returns:
            Set of reachable entity identifiers.
        """
        visited: set[str] = {entity}
        frontier: set[str] = {entity}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, neighbor in self._get_neighbors(node, relations, directed=True):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(entity)
        return visited

    def all_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
        relations: Optional[set[str]] = None,
    ) -> list[list[tuple[str, str, str]]]:
        """Find all simple paths between two entities up to max_depth.

        Args:
            source: Starting entity.
            target: Target entity.
            max_depth: Maximum path length.
            relations: Optional relation filter.

        Returns:
            List of paths, each a list of (entity, relation, entity) hops.
        """
        results: list[list[tuple[str, str, str]]] = []
        self._dfs_paths(source, target, max_depth, relations, set(), [], results)
        return results

    def _dfs_paths(
        self,
        current: str,
        target: str,
        max_depth: int,
        relations: Optional[set[str]],
        visited: set[str],
        path: list[tuple[str, str, str]],
        results: list[list[tuple[str, str, str]]],
    ) -> None:
        """DFS helper for finding all simple paths."""
        if len(path) > max_depth:
            return
        if current == target and path:
            results.append(list(path))
            return
        visited.add(current)
        for relation, neighbor in self._get_neighbors(current, relations, directed=True):
            if neighbor not in visited:
                path.append((current, relation, neighbor))
                self._dfs_paths(neighbor, target, max_depth, relations, visited, path, results)
                path.pop()
        visited.discard(current)

    def _get_neighbors(
        self,
        entity: str,
        relations: Optional[set[str]],
        directed: bool,
    ) -> list[tuple[str, str]]:
        """Get neighboring entities with their connecting relations.

        Args:
            entity: The entity to find neighbors for.
            relations: Optional filter on relation types.
            directed: If False, include reverse edges.

        Returns:
            List of (relation, neighbor) tuples.
        """
        neighbors: list[tuple[str, str]] = []
        for triple in self.store.lookup(subject=entity):
            if relations is None or triple.predicate in relations:
                neighbors.append((triple.predicate, triple.object))
        if not directed:
            for triple in self.store.lookup(obj=entity):
                if relations is None or triple.predicate in relations:
                    neighbors.append((triple.predicate, triple.subject))
        return neighbors


if __name__ == "__main__":
    store = TripleStore()
    edges = [
        Triple("A", "knows", "B"),
        Triple("B", "knows", "C"),
        Triple("C", "knows", "D"),
        Triple("A", "worksAt", "X"),
        Triple("B", "worksAt", "X"),
        Triple("D", "knows", "E"),
    ]
    for e in edges:
        store.add(e)

    engine = PathQueryEngine(store)

    result = engine.shortest_path("A", "D")
    print(f"Shortest path A → D (length={result.length}):")
    for hop in result.path:
        print(f"  {hop[0]} --{hop[1]}--> {hop[2]}")

    print(f"\nReachable from A (depth=2): {engine.neighbors('A', depth=2)}")
    print(f"A reachable to E? {engine.reachable('A', 'E')}")

    paths = engine.all_paths("A", "D", max_depth=5)
    print(f"\nAll paths A → D ({len(paths)} found):")
    for p in paths:
        print(f"  {' → '.join(h[0] for h in p)} → {p[-1][2]}")
