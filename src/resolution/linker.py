"""Transitive closure clustering for entity resolution.

Groups matched entity pairs into clusters using union-find,
then assigns canonical identifiers to each cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.resolution.blocker import EntityRecord, MinHashLSH, SortedNeighborhoodBlocker
from src.resolution.matcher import SimilarityMatcher, MatchResult, FieldConfig


class UnionFind:
    """Union-Find (disjoint set) data structure for clustering.

    Supports union and find operations with path compression
    and union by rank for near-constant amortized time.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative of x's cluster.

        Args:
            x: Element to find.

        Returns:
            Root representative of x's cluster.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the clusters containing x and y.

        Args:
            x: First element.
            y: Second element.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> dict[str, list[str]]:
        """Return all clusters as a mapping from root to members.

        Returns:
            Dictionary mapping cluster root to list of member elements.
        """
        groups: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


@dataclass
class LinkedCluster:
    """A cluster of resolved entity records.

    Attributes:
        canonical_id: The canonical identifier for this cluster.
        member_ids: List of record IDs in this cluster.
        matches: List of match results that formed this cluster.
    """

    canonical_id: str
    member_ids: list[str]
    matches: list[MatchResult]


class EntityLinker:
    """End-to-end entity resolution pipeline.

    Combines blocking, matching, and clustering into a single
    pipeline that takes raw records and produces linked clusters.

    Args:
        matcher: Similarity matcher for comparing records.
        blocking_method: 'lsh' or 'sorted' for blocking strategy.
        lsh_hashes: Number of MinHash functions (for LSH blocking).
        lsh_bands: Number of LSH bands.
        window_size: Window size (for sorted-neighborhood blocking).
    """

    def __init__(
        self,
        matcher: SimilarityMatcher | None = None,
        blocking_method: str = "lsh",
        lsh_hashes: int = 100,
        lsh_bands: int = 10,
        window_size: int = 5,
    ) -> None:
        self.matcher = matcher or SimilarityMatcher()
        self.blocking_method = blocking_method
        self.lsh_hashes = lsh_hashes
        self.lsh_bands = lsh_bands
        self.window_size = window_size

    def link(self, records: list[EntityRecord]) -> list[LinkedCluster]:
        """Run the full entity resolution pipeline.

        Steps:
        1. Generate candidate pairs via blocking.
        2. Compare candidates with the similarity matcher.
        3. Cluster matches using transitive closure (union-find).

        Args:
            records: List of entity records to resolve.

        Returns:
            List of LinkedCluster objects.
        """
        if self.blocking_method == "lsh":
            blocker = MinHashLSH(
                num_hashes=self.lsh_hashes,
                num_bands=self.lsh_bands,
            )
        else:
            blocker = SortedNeighborhoodBlocker(window_size=self.window_size)

        candidates = blocker.generate_candidates(records)
        records_map = {r.record_id: r for r in records}
        matches = self.matcher.match_candidates(records_map, candidates)

        uf = UnionFind()
        for r in records:
            uf.find(r.record_id)

        match_map: dict[tuple[str, str], MatchResult] = {}
        for m in matches:
            uf.union(m.record_id_1, m.record_id_2)
            key = tuple(sorted([m.record_id_1, m.record_id_2]))
            match_map[key] = m  # type: ignore[index]

        clusters_dict = uf.clusters()
        linked: list[LinkedCluster] = []
        for root, members in clusters_dict.items():
            cluster_matches: list[MatchResult] = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = tuple(sorted([members[i], members[j]]))
                    if key in match_map:
                        cluster_matches.append(match_map[key])  # type: ignore[index]
            linked.append(
                LinkedCluster(
                    canonical_id=root,
                    member_ids=sorted(members),
                    matches=cluster_matches,
                )
            )
        return sorted(linked, key=lambda c: c.canonical_id)

    def link_stats(self, clusters: list[LinkedCluster]) -> dict[str, int]:
        """Compute summary statistics for linked clusters.

        Args:
            clusters: List of linked clusters.

        Returns:
            Dictionary with cluster count, singleton count, and max size.
        """
        sizes = [len(c.member_ids) for c in clusters]
        return {
            "total_clusters": len(clusters),
            "singletons": sum(1 for s in sizes if s == 1),
            "max_cluster_size": max(sizes) if sizes else 0,
            "total_records": sum(sizes),
        }


if __name__ == "__main__":
    records = [
        EntityRecord("r1", {"name": "Albert Einstein", "city": "Ulm"}),
        EntityRecord("r2", {"name": "A. Einstein", "city": "Ulm"}),
        EntityRecord("r3", {"name": "Marie Curie", "city": "Warsaw"}),
        EntityRecord("r4", {"name": "M. Curie", "city": "Warsaw"}),
        EntityRecord("r5", {"name": "Isaac Newton", "city": "Woolsthorpe"}),
        EntityRecord("r6", {"name": "I. Newton", "city": "Woolsthorpe"}),
    ]

    linker = EntityLinker(
        matcher=SimilarityMatcher(threshold=0.5),
        blocking_method="sorted",
        window_size=4,
    )
    clusters = linker.link(records)

    print(f"Entity Resolution Results:")
    print(f"{'=' * 40}")
    for cluster in clusters:
        print(f"\nCluster '{cluster.canonical_id}':")
        print(f"  Members: {cluster.member_ids}")
        for m in cluster.matches:
            print(f"  Match: {m.record_id_1} <-> {m.record_id_2} (sim={m.similarity:.3f})")

    stats = linker.link_stats(clusters)
    print(f"\nStats: {stats}")
