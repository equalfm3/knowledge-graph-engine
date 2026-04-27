"""SPARQL-like pattern parser.

Parses a simplified SPARQL SELECT query into a list of triple patterns
with variables (prefixed with '?') and bound terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TriplePattern:
    """A triple pattern with optional variable positions.

    Variables are strings starting with '?'. Bound terms are plain strings.

    Attributes:
        subject: Subject term or variable.
        predicate: Predicate term or variable.
        object: Object term or variable.
    """

    subject: str
    predicate: str
    object: str

    @property
    def variables(self) -> list[str]:
        """Return the list of variable names in this pattern."""
        return [t for t in (self.subject, self.predicate, self.object) if t.startswith("?")]

    @property
    def bound_count(self) -> int:
        """Return the number of bound (non-variable) positions."""
        return 3 - len(self.variables)

    def is_variable(self, position: str) -> bool:
        """Check if a position value is a variable.

        Args:
            position: The value at a triple position.

        Returns:
            True if the value is a variable (starts with '?').
        """
        return position.startswith("?")


@dataclass
class ParsedQuery:
    """A parsed SPARQL-like SELECT query.

    Attributes:
        select_vars: Variables to project in the result.
        patterns: List of triple patterns forming the basic graph pattern.
        limit: Optional result limit.
        offset: Optional result offset.
    """

    select_vars: list[str]
    patterns: list[TriplePattern]
    limit: Optional[int] = None
    offset: Optional[int] = None

    @property
    def all_variables(self) -> set[str]:
        """Return all variables across all patterns."""
        variables: set[str] = set()
        for p in self.patterns:
            variables.update(p.variables)
        return variables


class QueryParser:
    """Parser for simplified SPARQL SELECT queries.

    Supports queries of the form:
        SELECT ?x ?y WHERE { ?x predicate ?y . ?y predicate2 value }

    Does not support OPTIONAL, UNION, FILTER, or subqueries.
    """

    PATTERN_RE = re.compile(r"(\S+)\s+(\S+)\s+(\S+)")

    def parse(self, query: str) -> ParsedQuery:
        """Parse a SPARQL-like query string.

        Args:
            query: The query string.

        Returns:
            A ParsedQuery with select variables and triple patterns.

        Raises:
            ValueError: If the query syntax is invalid.
        """
        query = query.strip()
        select_match = re.match(
            r"SELECT\s+((?:\?\w+\s*)+)\s*WHERE\s*\{(.*)\}",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if not select_match:
            raise ValueError(f"Invalid query syntax: {query}")

        select_vars = re.findall(r"\?\w+", select_match.group(1))
        body = select_match.group(2).strip()

        patterns = self._parse_patterns(body)
        if not patterns:
            raise ValueError("No triple patterns found in WHERE clause")

        limit, offset = self._parse_modifiers(query)
        return ParsedQuery(
            select_vars=select_vars,
            patterns=patterns,
            limit=limit,
            offset=offset,
        )

    def _parse_patterns(self, body: str) -> list[TriplePattern]:
        """Parse the WHERE clause body into triple patterns.

        Args:
            body: The content between { and }.

        Returns:
            List of TriplePattern objects.
        """
        body = re.sub(r"\s+", " ", body).strip()
        parts = [p.strip() for p in body.split(".") if p.strip()]
        patterns = []
        for part in parts:
            match = self.PATTERN_RE.match(part.strip())
            if match:
                patterns.append(
                    TriplePattern(
                        subject=match.group(1),
                        predicate=match.group(2),
                        object=match.group(3),
                    )
                )
        return patterns

    def _parse_modifiers(self, query: str) -> tuple[Optional[int], Optional[int]]:
        """Extract LIMIT and OFFSET modifiers from the query.

        Args:
            query: Full query string.

        Returns:
            Tuple of (limit, offset), either may be None.
        """
        limit = None
        offset = None
        limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))
        offset_match = re.search(r"OFFSET\s+(\d+)", query, re.IGNORECASE)
        if offset_match:
            offset = int(offset_match.group(1))
        return limit, offset

    def parse_pattern(self, pattern_str: str) -> TriplePattern:
        """Parse a single triple pattern string.

        Args:
            pattern_str: A string like '?x bornIn ?city'.

        Returns:
            A TriplePattern object.

        Raises:
            ValueError: If the pattern is malformed.
        """
        match = self.PATTERN_RE.match(pattern_str.strip())
        if not match:
            raise ValueError(f"Invalid triple pattern: {pattern_str}")
        return TriplePattern(
            subject=match.group(1),
            predicate=match.group(2),
            object=match.group(3),
        )


if __name__ == "__main__":
    parser = QueryParser()

    query = "SELECT ?x ?city WHERE { ?x bornIn ?city . ?city locatedIn Germany }"
    parsed = parser.parse(query)
    print(f"Query: {query}")
    print(f"Select vars: {parsed.select_vars}")
    print(f"All vars: {parsed.all_variables}")
    print(f"Patterns:")
    for p in parsed.patterns:
        print(f"  ({p.subject}, {p.predicate}, {p.object})")
        print(f"    variables={p.variables}, bound={p.bound_count}")

    query2 = "SELECT ?x WHERE { ?x field Physics }"
    parsed2 = parser.parse(query2)
    print(f"\nQuery: {query2}")
    print(f"Patterns: {[(p.subject, p.predicate, p.object) for p in parsed2.patterns]}")
