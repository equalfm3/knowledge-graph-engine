"""N-Triples and Turtle RDF parser.

Parses RDF serialization formats into Triple objects for loading
into the triple store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TextIO

from src.store.triple_store import Triple, TripleStore


@dataclass
class Namespace:
    """An RDF namespace prefix binding.

    Attributes:
        prefix: Short prefix (e.g., 'foaf').
        uri: Full URI (e.g., 'http://xmlns.com/foaf/0.1/').
    """

    prefix: str
    uri: str


class RDFParser:
    """Parser for N-Triples and simplified Turtle formats.

    Supports:
    - N-Triples (.nt): one triple per line, full URIs in angle brackets.
    - Simplified Turtle (.ttl): prefix declarations and prefixed names.

    Attributes:
        namespaces: Registered namespace prefix bindings.
    """

    # Pattern for N-Triples: <subject> <predicate> <object> .
    NT_PATTERN = re.compile(
        r'<([^>]+)>\s+<([^>]+)>\s+(?:<([^>]+)>|"([^"]*)")\s*\.'
    )
    # Pattern for prefix declarations: @prefix foaf: <http://...> .
    PREFIX_PATTERN = re.compile(r"@prefix\s+(\w*):\s*<([^>]+)>\s*\.")
    # Pattern for Turtle triples with prefixed names
    TURTLE_PATTERN = re.compile(
        r'(\S+)\s+(\S+)\s+(?:(\S+)|"([^"]*)")\s*[;.]'
    )

    def __init__(self) -> None:
        self.namespaces: dict[str, str] = {}

    def register_namespace(self, prefix: str, uri: str) -> None:
        """Register a namespace prefix.

        Args:
            prefix: Short prefix string.
            uri: Full namespace URI.
        """
        self.namespaces[prefix] = uri

    def expand_prefixed(self, term: str) -> str:
        """Expand a prefixed name to a full URI.

        Args:
            term: A prefixed name like 'foaf:name' or a full URI.

        Returns:
            The expanded URI string.
        """
        if ":" in term and not term.startswith("<") and not term.startswith("http"):
            prefix, local = term.split(":", 1)
            if prefix in self.namespaces:
                return self.namespaces[prefix] + local
        return term.strip("<>")

    def parse_nt_line(self, line: str) -> Triple | None:
        """Parse a single N-Triples line.

        Args:
            line: A line from an N-Triples file.

        Returns:
            A Triple if the line is valid, None otherwise.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        match = self.NT_PATTERN.match(line)
        if match:
            s, p = match.group(1), match.group(2)
            o = match.group(3) if match.group(3) else match.group(4)
            return Triple(s, p, o)
        return None

    def parse_nt(self, text: str) -> list[Triple]:
        """Parse N-Triples format text.

        Args:
            text: Full N-Triples content.

        Returns:
            List of parsed triples.
        """
        triples = []
        for line in text.strip().split("\n"):
            triple = self.parse_nt_line(line)
            if triple:
                triples.append(triple)
        return triples

    def parse_turtle(self, text: str) -> list[Triple]:
        """Parse simplified Turtle format text.

        Handles @prefix declarations and prefixed names. Does not support
        the full Turtle grammar (blank nodes, collections, etc.).

        Args:
            text: Turtle format content.

        Returns:
            List of parsed triples.
        """
        triples = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prefix_match = self.PREFIX_PATTERN.match(line)
            if prefix_match:
                self.namespaces[prefix_match.group(1)] = prefix_match.group(2)
                continue
            turtle_match = self.TURTLE_PATTERN.match(line)
            if turtle_match:
                s = self.expand_prefixed(turtle_match.group(1))
                p = self.expand_prefixed(turtle_match.group(2))
                o_uri = turtle_match.group(3)
                o_lit = turtle_match.group(4)
                o = self.expand_prefixed(o_uri) if o_uri else (o_lit or "")
                triples.append(Triple(s, p, o))
        return triples

    def load_into_store(self, text: str, store: TripleStore, fmt: str = "nt") -> int:
        """Parse text and load triples into a store.

        Args:
            text: RDF content string.
            store: Target triple store.
            fmt: Format — 'nt' for N-Triples, 'ttl' for Turtle.

        Returns:
            Number of triples added.
        """
        if fmt == "ttl":
            triples = self.parse_turtle(text)
        else:
            triples = self.parse_nt(text)
        count = 0
        for t in triples:
            if store.add(t):
                count += 1
        return count


if __name__ == "__main__":
    parser = RDFParser()

    nt_data = """
<http://example.org/Einstein> <http://example.org/bornIn> <http://example.org/Ulm> .
<http://example.org/Einstein> <http://example.org/field> "Physics" .
<http://example.org/Ulm> <http://example.org/locatedIn> <http://example.org/Germany> .
<http://example.org/Curie> <http://example.org/bornIn> <http://example.org/Warsaw> .
"""
    triples = parser.parse_nt(nt_data)
    print("N-Triples parsed:")
    for t in triples:
        print(f"  ({t.subject}, {t.predicate}, {t.object})")

    ttl_data = """
@prefix ex: <http://example.org/> .
ex:Einstein ex:bornIn ex:Ulm .
ex:Curie ex:bornIn ex:Warsaw .
ex:Ulm ex:locatedIn ex:Germany .
"""
    parser2 = RDFParser()
    triples2 = parser2.parse_turtle(ttl_data)
    print("\nTurtle parsed:")
    for t in triples2:
        print(f"  ({t.subject}, {t.predicate}, {t.object})")

    store = TripleStore()
    count = parser.load_into_store(nt_data, store, fmt="nt")
    print(f"\nLoaded {count} triples into store ({len(store)} total)")
