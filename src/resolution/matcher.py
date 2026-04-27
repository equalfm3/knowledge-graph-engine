"""Field-level similarity matching for entity resolution.

Computes similarity between entity records using Jaccard similarity
for token sets, normalized Levenshtein distance for strings, and
weighted combination across fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.resolution.blocker import EntityRecord, CandidatePair


@dataclass
class MatchResult:
    """Result of comparing two entity records.

    Attributes:
        record_id_1: First record identifier.
        record_id_2: Second record identifier.
        similarity: Overall weighted similarity score.
        field_scores: Per-field similarity scores.
        is_match: Whether the pair exceeds the match threshold.
    """

    record_id_1: str
    record_id_2: str
    similarity: float
    field_scores: dict[str, float]
    is_match: bool


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Args:
        a: First token set.
        b: Second token set.

    Returns:
        Jaccard coefficient in [0, 1].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum number of single-character edits.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity in [0, 1] where 1 means identical.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def token_jaccard(s1: str, s2: str) -> float:
    """Compute Jaccard similarity on whitespace-tokenized strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Jaccard similarity of token sets.
    """
    return jaccard_similarity(set(s1.lower().split()), set(s2.lower().split()))


@dataclass
class FieldConfig:
    """Configuration for a single field comparison.

    Attributes:
        name: Field name.
        weight: Weight in the overall similarity score.
        similarity_fn: Similarity function to use.
    """

    name: str
    weight: float = 1.0
    similarity_fn: Callable[[str, str], float] = token_jaccard


class SimilarityMatcher:
    """Weighted multi-field similarity matcher.

    Computes per-field similarity scores and combines them into
    an overall weighted score. Pairs exceeding the threshold are
    declared matches.

    Args:
        field_configs: List of field comparison configurations.
        threshold: Minimum similarity for a match.
    """

    def __init__(
        self,
        field_configs: list[FieldConfig] | None = None,
        threshold: float = 0.7,
    ) -> None:
        self.field_configs = field_configs or [
            FieldConfig("name", weight=0.6, similarity_fn=normalized_levenshtein),
            FieldConfig("city", weight=0.4, similarity_fn=token_jaccard),
        ]
        self.threshold = threshold
        total_weight = sum(fc.weight for fc in self.field_configs)
        if total_weight > 0:
            for fc in self.field_configs:
                fc.weight /= total_weight

    def compare(self, record_a: EntityRecord, record_b: EntityRecord) -> MatchResult:
        """Compare two entity records across all configured fields.

        Args:
            record_a: First record.
            record_b: Second record.

        Returns:
            MatchResult with per-field and overall similarity.
        """
        field_scores: dict[str, float] = {}
        overall = 0.0

        for fc in self.field_configs:
            val_a = record_a.fields.get(fc.name, "")
            val_b = record_b.fields.get(fc.name, "")
            score = fc.similarity_fn(val_a, val_b)
            field_scores[fc.name] = score
            overall += fc.weight * score

        return MatchResult(
            record_id_1=record_a.record_id,
            record_id_2=record_b.record_id,
            similarity=overall,
            field_scores=field_scores,
            is_match=overall >= self.threshold,
        )

    def match_candidates(
        self,
        records: dict[str, EntityRecord],
        candidates: set[CandidatePair],
    ) -> list[MatchResult]:
        """Compare all candidate pairs and return matches.

        Args:
            records: Mapping from record_id to EntityRecord.
            candidates: Set of candidate pairs to compare.

        Returns:
            List of MatchResult objects for pairs that are matches.
        """
        matches: list[MatchResult] = []
        for id_a, id_b in candidates:
            if id_a in records and id_b in records:
                result = self.compare(records[id_a], records[id_b])
                if result.is_match:
                    matches.append(result)
        return matches


if __name__ == "__main__":
    records = {
        "r1": EntityRecord("r1", {"name": "Albert Einstein", "city": "Ulm"}),
        "r2": EntityRecord("r2", {"name": "A. Einstein", "city": "Ulm"}),
        "r3": EntityRecord("r3", {"name": "Marie Curie", "city": "Warsaw"}),
        "r4": EntityRecord("r4", {"name": "M. Curie", "city": "Warsaw"}),
        "r5": EntityRecord("r5", {"name": "Isaac Newton", "city": "Woolsthorpe"}),
    }

    matcher = SimilarityMatcher(threshold=0.5)

    result = matcher.compare(records["r1"], records["r2"])
    print(f"r1 vs r2: sim={result.similarity:.3f}, match={result.is_match}")
    print(f"  fields: {result.field_scores}")

    result2 = matcher.compare(records["r1"], records["r3"])
    print(f"\nr1 vs r3: sim={result2.similarity:.3f}, match={result2.is_match}")
    print(f"  fields: {result2.field_scores}")

    candidates = {("r1", "r2"), ("r3", "r4"), ("r1", "r5")}
    matches = matcher.match_candidates(records, candidates)
    print(f"\nMatches from {len(candidates)} candidates: {len(matches)}")
    for m in matches:
        print(f"  {m.record_id_1} <-> {m.record_id_2}: {m.similarity:.3f}")
