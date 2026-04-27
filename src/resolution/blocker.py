"""LSH and sorted-neighborhood blocking for entity resolution.

Generates candidate pairs for comparison without exhaustive O(n²)
pairwise matching. LSH uses MinHash on token sets; sorted-neighborhood
uses a sliding window over sorted keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class EntityRecord:
    """A record representing an entity with named fields.

    Attributes:
        record_id: Unique identifier for this record.
        fields: Dictionary of field name → field value.
    """

    record_id: str
    fields: dict[str, str]

    def tokens(self, field_name: str | None = None) -> set[str]:
        """Extract tokens from one or all fields.

        Args:
            field_name: Specific field to tokenize, or None for all fields.

        Returns:
            Set of lowercase tokens.
        """
        if field_name and field_name in self.fields:
            return set(self.fields[field_name].lower().split())
        tokens: set[str] = set()
        for v in self.fields.values():
            tokens.update(v.lower().split())
        return tokens


CandidatePair = tuple[str, str]


class MinHashLSH:
    """Locality-Sensitive Hashing blocker using MinHash signatures.

    Hashes token sets into compact signatures and groups records
    into buckets. Records sharing a bucket become candidate pairs.

    Args:
        num_hashes: Number of hash functions for the MinHash signature.
        num_bands: Number of bands for LSH banding (num_hashes must be
            divisible by num_bands).
        field_name: Which field to tokenize for blocking (None = all fields).
    """

    def __init__(
        self,
        num_hashes: int = 100,
        num_bands: int = 10,
        field_name: str | None = None,
    ) -> None:
        if num_hashes % num_bands != 0:
            raise ValueError("num_hashes must be divisible by num_bands")
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.field_name = field_name
        self._hash_params = self._generate_hash_params(num_hashes)

    def _generate_hash_params(self, n: int) -> list[tuple[int, int]]:
        """Generate random hash function parameters (a, b) for MinHash."""
        rng = np.random.RandomState(42)
        large_prime = 2**31 - 1
        a = rng.randint(1, large_prime, size=n)
        b = rng.randint(0, large_prime, size=n)
        return list(zip(a.tolist(), b.tolist()))

    def _minhash_signature(self, tokens: set[str]) -> list[int]:
        """Compute the MinHash signature for a token set.

        Args:
            tokens: Set of string tokens.

        Returns:
            List of MinHash values (one per hash function).
        """
        if not tokens:
            return [0] * self.num_hashes
        large_prime = 2**31 - 1
        hashes = []
        token_hashes = [
            int(hashlib.md5(t.encode()).hexdigest(), 16) % large_prime
            for t in tokens
        ]
        for a, b in self._hash_params:
            min_val = min((a * h + b) % large_prime for h in token_hashes)
            hashes.append(min_val)
        return hashes

    def generate_candidates(self, records: list[EntityRecord]) -> set[CandidatePair]:
        """Generate candidate pairs using LSH banding.

        Args:
            records: List of entity records.

        Returns:
            Set of (record_id_1, record_id_2) candidate pairs.
        """
        signatures: dict[str, list[int]] = {}
        for record in records:
            tokens = record.tokens(self.field_name)
            signatures[record.record_id] = self._minhash_signature(tokens)

        buckets: dict[tuple[int, ...], list[str]] = {}
        for record_id, sig in signatures.items():
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_hash = tuple(sig[start:end])
                key = (band_idx,) + band_hash
                buckets.setdefault(key, []).append(record_id)

        candidates: set[CandidatePair] = set()
        for bucket_records in buckets.values():
            if len(bucket_records) > 1:
                for i in range(len(bucket_records)):
                    for j in range(i + 1, len(bucket_records)):
                        pair = tuple(sorted([bucket_records[i], bucket_records[j]]))
                        candidates.add(pair)  # type: ignore[arg-type]
        return candidates


class SortedNeighborhoodBlocker:
    """Sorted-neighborhood blocking using a sliding window.

    Sorts records by a blocking key and compares records within
    a sliding window of fixed size.

    Args:
        window_size: Size of the sliding window.
        key_fn: Function to extract the sorting key from a record.
    """

    def __init__(
        self,
        window_size: int = 5,
        key_fn: Callable[[EntityRecord], str] | None = None,
    ) -> None:
        self.window_size = window_size
        self.key_fn = key_fn or self._default_key

    @staticmethod
    def _default_key(record: EntityRecord) -> str:
        """Default blocking key: concatenation of all field values."""
        return " ".join(record.fields.values()).lower()

    def generate_candidates(self, records: list[EntityRecord]) -> set[CandidatePair]:
        """Generate candidate pairs using sorted-neighborhood.

        Args:
            records: List of entity records.

        Returns:
            Set of (record_id_1, record_id_2) candidate pairs.
        """
        sorted_records = sorted(records, key=self.key_fn)
        candidates: set[CandidatePair] = set()

        for i in range(len(sorted_records)):
            for j in range(i + 1, min(i + self.window_size, len(sorted_records))):
                pair = tuple(sorted([sorted_records[i].record_id, sorted_records[j].record_id]))
                candidates.add(pair)  # type: ignore[arg-type]
        return candidates


if __name__ == "__main__":
    records = [
        EntityRecord("r1", {"name": "Albert Einstein", "city": "Ulm"}),
        EntityRecord("r2", {"name": "A. Einstein", "city": "Ulm Germany"}),
        EntityRecord("r3", {"name": "Marie Curie", "city": "Warsaw"}),
        EntityRecord("r4", {"name": "M. Curie", "city": "Warsaw Poland"}),
        EntityRecord("r5", {"name": "Isaac Newton", "city": "Woolsthorpe"}),
    ]

    lsh = MinHashLSH(num_hashes=50, num_bands=5, field_name="name")
    lsh_candidates = lsh.generate_candidates(records)
    print(f"LSH candidates ({len(lsh_candidates)}):")
    for pair in sorted(lsh_candidates):
        print(f"  {pair}")

    snb = SortedNeighborhoodBlocker(window_size=3)
    snb_candidates = snb.generate_candidates(records)
    print(f"\nSorted-neighborhood candidates ({len(snb_candidates)}):")
    for pair in sorted(snb_candidates):
        print(f"  {pair}")
