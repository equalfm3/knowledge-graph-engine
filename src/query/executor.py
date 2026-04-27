"""BGP evaluation with index nested-loop join.

Executes SPARQL-like queries against the triple store by evaluating
triple patterns in the order determined by the query planner and
joining results on shared variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.query.parser import QueryParser, ParsedQuery, TriplePattern
from src.query.planner import QueryPlanner
from src.store.triple_store import TripleStore, Triple


Binding = dict[str, str]


class BGPExecutor:
    """Basic Graph Pattern executor using index nested-loop join.

    Evaluates triple patterns against the store in the order given
    by the planner, joining on shared variables.

    Args:
        store: The triple store to query.
    """

    def __init__(self, store: TripleStore) -> None:
        self.store = store
        self.planner = QueryPlanner(store)

    def execute(self, query: ParsedQuery) -> list[Binding]:
        """Execute a parsed query and return variable bindings.

        Args:
            query: A parsed SPARQL-like query.

        Returns:
            List of variable binding dictionaries.
        """
        plan = self.planner.plan(query)
        if not plan.ordered_patterns:
            return []

        bindings: list[Binding] = [{}]
        for pattern in plan.ordered_patterns:
            bindings = self._join_pattern(bindings, pattern)
            if not bindings:
                break

        projected = self._project(bindings, query.select_vars)
        if query.offset:
            projected = projected[query.offset:]
        if query.limit:
            projected = projected[: query.limit]
        return projected

    def _join_pattern(
        self, bindings: list[Binding], pattern: TriplePattern
    ) -> list[Binding]:
        """Join existing bindings with a new triple pattern.

        For each existing binding, substitute bound variables into the
        pattern and look up matching triples. Extend the binding with
        any new variable assignments.

        Args:
            bindings: Current list of partial bindings.
            pattern: The next triple pattern to join.

        Returns:
            Extended bindings after the join.
        """
        new_bindings: list[Binding] = []
        for binding in bindings:
            s = binding.get(pattern.subject, pattern.subject) if pattern.subject.startswith("?") else pattern.subject
            p = binding.get(pattern.predicate, pattern.predicate) if pattern.predicate.startswith("?") else pattern.predicate
            o = binding.get(pattern.object, pattern.object) if pattern.object.startswith("?") else pattern.object

            lookup_s = None if s.startswith("?") else s
            lookup_p = None if p.startswith("?") else p
            lookup_o = None if o.startswith("?") else o

            matches = self.store.lookup(subject=lookup_s, predicate=lookup_p, obj=lookup_o)
            for triple in matches:
                extended = dict(binding)
                valid = True
                for var, val in [
                    (pattern.subject, triple.subject),
                    (pattern.predicate, triple.predicate),
                    (pattern.object, triple.object),
                ]:
                    if var.startswith("?"):
                        if var in extended and extended[var] != val:
                            valid = False
                            break
                        extended[var] = val
                if valid:
                    new_bindings.append(extended)
        return new_bindings

    def _project(self, bindings: list[Binding], select_vars: list[str]) -> list[Binding]:
        """Project bindings to only the selected variables.

        Args:
            bindings: Full variable bindings.
            select_vars: Variables to keep.

        Returns:
            Projected bindings (deduplicated).
        """
        seen: set[tuple[tuple[str, str], ...]] = set()
        projected: list[Binding] = []
        for b in bindings:
            row = {v: b[v] for v in select_vars if v in b}
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                projected.append(row)
        return projected


def run_query(store: TripleStore, query_str: str) -> list[Binding]:
    """Parse and execute a SPARQL-like query against a store.

    Args:
        store: The triple store.
        query_str: The query string.

    Returns:
        List of result bindings.
    """
    parser = QueryParser()
    parsed = parser.parse(query_str)
    executor = BGPExecutor(store)
    return executor.execute(parsed)


if __name__ == "__main__":
    store = TripleStore()
    data = [
        Triple("Einstein", "bornIn", "Ulm"),
        Triple("Curie", "bornIn", "Warsaw"),
        Triple("Newton", "bornIn", "Woolsthorpe"),
        Triple("Ulm", "locatedIn", "Germany"),
        Triple("Warsaw", "locatedIn", "Poland"),
        Triple("Woolsthorpe", "locatedIn", "England"),
        Triple("Einstein", "field", "Physics"),
        Triple("Curie", "field", "Chemistry"),
        Triple("Newton", "field", "Physics"),
    ]
    for t in data:
        store.add(t)

    q1 = "SELECT ?x WHERE { ?x bornIn ?city . ?city locatedIn Germany }"
    print(f"Query: {q1}")
    for b in run_query(store, q1):
        print(f"  {b}")

    q2 = "SELECT ?x ?city WHERE { ?x bornIn ?city . ?x field Physics }"
    print(f"\nQuery: {q2}")
    for b in run_query(store, q2):
        print(f"  {b}")

    q3 = "SELECT ?x ?f WHERE { ?x field ?f }"
    print(f"\nQuery: {q3}")
    for b in run_query(store, q3):
        print(f"  {b}")
