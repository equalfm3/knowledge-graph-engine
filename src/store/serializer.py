"""Export triple store contents to N-Triples and JSON-LD formats.

Provides serialization of in-memory triples to standard RDF
interchange formats for persistence and interoperability.
"""

from __future__ import annotations

import json
from typing import Any

from src.store.triple_store import Triple, TripleStore


class NTriplesSerializer:
    """Serialize triples to N-Triples format.

    N-Triples is a line-based, plain-text format where each triple
    is written as ``<subject> <predicate> <object> .`` on its own line.
    """

    @staticmethod
    def serialize_triple(triple: Triple) -> str:
        """Serialize a single triple to N-Triples format.

        Args:
            triple: The triple to serialize.

        Returns:
            N-Triples formatted string.
        """
        s, p, o = triple.subject, triple.predicate, triple.object
        s_str = f"<{s}>" if not s.startswith("<") else s
        p_str = f"<{p}>" if not p.startswith("<") else p
        if o.startswith("http") or o.startswith("<"):
            o_str = f"<{o}>" if not o.startswith("<") else o
        else:
            o_str = f'"{o}"'
        return f"{s_str} {p_str} {o_str} ."

    @staticmethod
    def serialize(store: TripleStore) -> str:
        """Serialize all triples in a store to N-Triples format.

        Args:
            store: The triple store to serialize.

        Returns:
            Complete N-Triples document as a string.
        """
        lines = []
        for triple in store.all_triples():
            lines.append(NTriplesSerializer.serialize_triple(triple))
        return "\n".join(sorted(lines)) + "\n" if lines else ""


class JSONLDSerializer:
    """Serialize triples to a simplified JSON-LD format.

    Groups triples by subject into JSON-LD node objects with
    ``@id`` for the subject and predicate keys mapping to objects.
    """

    @staticmethod
    def serialize(store: TripleStore) -> str:
        """Serialize all triples to JSON-LD format.

        Args:
            store: The triple store to serialize.

        Returns:
            JSON-LD formatted string.
        """
        nodes: dict[str, dict[str, Any]] = {}
        for triple in store.all_triples():
            if triple.subject not in nodes:
                nodes[triple.subject] = {"@id": triple.subject}
            node = nodes[triple.subject]
            if triple.predicate in node:
                existing = node[triple.predicate]
                if isinstance(existing, list):
                    existing.append(triple.object)
                else:
                    node[triple.predicate] = [existing, triple.object]
            else:
                node[triple.predicate] = triple.object
        doc = {"@graph": list(nodes.values())}
        return json.dumps(doc, indent=2)

    @staticmethod
    def serialize_triple(triple: Triple) -> dict[str, Any]:
        """Serialize a single triple to a JSON-LD-like dict.

        Args:
            triple: The triple to serialize.

        Returns:
            Dictionary with @id, predicate, and object.
        """
        return {
            "@id": triple.subject,
            triple.predicate: triple.object,
        }


def export_store(store: TripleStore, fmt: str = "nt") -> str:
    """Export a triple store to the specified format.

    Args:
        store: The triple store to export.
        fmt: Output format — 'nt' for N-Triples, 'jsonld' for JSON-LD.

    Returns:
        Serialized string in the requested format.

    Raises:
        ValueError: If the format is not supported.
    """
    if fmt == "nt":
        return NTriplesSerializer.serialize(store)
    elif fmt == "jsonld":
        return JSONLDSerializer.serialize(store)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'nt' or 'jsonld'.")


if __name__ == "__main__":
    store = TripleStore()
    triples = [
        Triple("http://ex.org/Einstein", "http://ex.org/bornIn", "http://ex.org/Ulm"),
        Triple("http://ex.org/Einstein", "http://ex.org/field", "Physics"),
        Triple("http://ex.org/Ulm", "http://ex.org/locatedIn", "http://ex.org/Germany"),
        Triple("http://ex.org/Curie", "http://ex.org/bornIn", "http://ex.org/Warsaw"),
    ]
    for t in triples:
        store.add(t)

    print("=== N-Triples ===")
    print(export_store(store, "nt"))

    print("=== JSON-LD ===")
    print(export_store(store, "jsonld"))
