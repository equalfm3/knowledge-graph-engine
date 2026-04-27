"""In-memory triple store with SPO/POS/OSP indexes.

Stores RDF-style triples and maintains three hash-map indexes for
O(1) lookup on any combination of bound/unbound positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass(frozen=True)
class Triple:
    """An RDF-style triple (subject, predicate, object).

    Attributes:
        subject: The subject entity URI or literal.
        predicate: The predicate (relation) URI.
        object: The object entity URI or literal.
    """

    subject: str
    predicate: str
    object: str

    def __iter__(self) -> Iterator[str]:
        yield self.subject
        yield self.predicate
        yield self.object


class TripleStore:
    """In-memory triple store with three hash-map indexes.

    Maintains SPO, POS, and OSP indexes so that any triple pattern
    (with any combination of bound/unbound positions) resolves in O(1)
    per matching triple.

    Attributes:
        spo: Subject → Predicate → set of Objects.
        pos: Predicate → Object → set of Subjects.
        osp: Object → Subject → set of Predicates.
    """

    def __init__(self) -> None:
        self.spo: dict[str, dict[str, set[str]]] = {}
        self.pos: dict[str, dict[str, set[str]]] = {}
        self.osp: dict[str, dict[str, set[str]]] = {}
        self._size: int = 0

    def add(self, triple: Triple) -> bool:
        """Add a triple to the store.

        Args:
            triple: The triple to insert.

        Returns:
            True if the triple was new, False if it already existed.
        """
        s, p, o = triple.subject, triple.predicate, triple.object
        if self.contains(triple):
            return False
        self.spo.setdefault(s, {}).setdefault(p, set()).add(o)
        self.pos.setdefault(p, {}).setdefault(o, set()).add(s)
        self.osp.setdefault(o, {}).setdefault(s, set()).add(p)
        self._size += 1
        return True

    def remove(self, triple: Triple) -> bool:
        """Remove a triple from the store.

        Args:
            triple: The triple to remove.

        Returns:
            True if the triple was removed, False if it was not present.
        """
        s, p, o = triple.subject, triple.predicate, triple.object
        if not self.contains(triple):
            return False
        self.spo[s][p].discard(o)
        if not self.spo[s][p]:
            del self.spo[s][p]
        if not self.spo[s]:
            del self.spo[s]
        self.pos[p][o].discard(s)
        if not self.pos[p][o]:
            del self.pos[p][o]
        if not self.pos[p]:
            del self.pos[p]
        self.osp[o][s].discard(p)
        if not self.osp[o][s]:
            del self.osp[o][s]
        if not self.osp[o]:
            del self.osp[o]
        self._size -= 1
        return True

    def contains(self, triple: Triple) -> bool:
        """Check if a triple exists in the store.

        Args:
            triple: The triple to check.

        Returns:
            True if the triple is present.
        """
        s, p, o = triple.subject, triple.predicate, triple.object
        return o in self.spo.get(s, {}).get(p, set())

    def lookup(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> list[Triple]:
        """Look up triples matching a pattern with optional bound positions.

        Unbound positions (None) act as wildcards. Uses the most selective
        index based on which positions are bound.

        Args:
            subject: Bound subject or None for wildcard.
            predicate: Bound predicate or None for wildcard.
            obj: Bound object or None for wildcard.

        Returns:
            List of matching triples.
        """
        if subject and predicate and obj:
            t = Triple(subject, predicate, obj)
            return [t] if self.contains(t) else []
        if subject and predicate:
            return [
                Triple(subject, predicate, o)
                for o in self.spo.get(subject, {}).get(predicate, set())
            ]
        if predicate and obj:
            return [
                Triple(s, predicate, obj)
                for s in self.pos.get(predicate, {}).get(obj, set())
            ]
        if obj and subject:
            return [
                Triple(subject, p, obj)
                for p in self.osp.get(obj, {}).get(subject, set())
            ]
        if subject:
            results = []
            for p, objs in self.spo.get(subject, {}).items():
                for o in objs:
                    results.append(Triple(subject, p, o))
            return results
        if predicate:
            results = []
            for o, subjs in self.pos.get(predicate, {}).items():
                for s in subjs:
                    results.append(Triple(s, predicate, o))
            return results
        if obj:
            results = []
            for s, preds in self.osp.get(obj, {}).items():
                for p in preds:
                    results.append(Triple(s, p, obj))
            return results
        return list(self.all_triples())

    def all_triples(self) -> Iterator[Triple]:
        """Iterate over all triples in the store.

        Yields:
            Every triple in the store.
        """
        for s, po in self.spo.items():
            for p, objs in po.items():
                for o in objs:
                    yield Triple(s, p, o)

    def subjects(self) -> set[str]:
        """Return all unique subjects."""
        return set(self.spo.keys())

    def predicates(self) -> set[str]:
        """Return all unique predicates."""
        return set(self.pos.keys())

    def objects(self) -> set[str]:
        """Return all unique objects."""
        return set(self.osp.keys())

    def entities(self) -> set[str]:
        """Return all unique entities (subjects ∪ objects)."""
        return self.subjects() | self.objects()

    def cardinality(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        """Estimate the number of triples matching a pattern without materializing.

        Args:
            subject: Bound subject or None.
            predicate: Bound predicate or None.
            obj: Bound object or None.

        Returns:
            Estimated count of matching triples.
        """
        if subject and predicate:
            return len(self.spo.get(subject, {}).get(predicate, set()))
        if predicate and obj:
            return len(self.pos.get(predicate, {}).get(obj, set()))
        if obj and subject:
            return len(self.osp.get(obj, {}).get(subject, set()))
        if subject:
            return sum(len(os) for os in self.spo.get(subject, {}).values())
        if predicate:
            return sum(len(ss) for ss in self.pos.get(predicate, {}).values())
        if obj:
            return sum(len(ps) for ps in self.osp.get(obj, {}).values())
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"TripleStore(triples={self._size})"


if __name__ == "__main__":
    store = TripleStore()
    triples = [
        Triple("Einstein", "bornIn", "Ulm"),
        Triple("Einstein", "field", "Physics"),
        Triple("Ulm", "locatedIn", "Germany"),
        Triple("Curie", "bornIn", "Warsaw"),
        Triple("Warsaw", "locatedIn", "Poland"),
        Triple("Curie", "field", "Chemistry"),
        Triple("Newton", "bornIn", "Woolsthorpe"),
        Triple("Woolsthorpe", "locatedIn", "England"),
    ]
    for t in triples:
        store.add(t)
    print(f"Store: {store}")
    print(f"\nAll born-in triples:")
    for t in store.lookup(predicate="bornIn"):
        print(f"  {t.subject} bornIn {t.object}")
    print(f"\nEinstein's triples:")
    for t in store.lookup(subject="Einstein"):
        print(f"  {t.subject} {t.predicate} {t.object}")
    print(f"\nLocated in Germany:")
    for t in store.lookup(predicate="locatedIn", obj="Germany"):
        print(f"  {t.subject}")
    print(f"\nCardinality of (?x, bornIn, ?y): {store.cardinality(predicate='bornIn')}")
