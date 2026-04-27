"""Join ordering and selectivity estimation for query planning.

Uses index cardinalities to estimate pattern selectivity and orders
joins greedily to minimize intermediate result size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.query.parser import TriplePattern, ParsedQuery
from src.store.triple_store import TripleStore


@dataclass
class PatternEstimate:
    """Selectivity estimate for a triple pattern.

    Attributes:
        pattern: The triple pattern.
        cardinality: Estimated number of matching triples.
        selectivity: Normalized selectivity (0 = most selective, 1 = least).
    """

    pattern: TriplePattern
    cardinality: int
    selectivity: float


@dataclass
class JoinPlan:
    """An ordered sequence of patterns for join evaluation.

    Attributes:
        ordered_patterns: Patterns in evaluation order (most selective first).
        estimates: Selectivity estimates for each pattern.
        estimated_cost: Rough cost estimate for the full join.
    """

    ordered_patterns: list[TriplePattern]
    estimates: list[PatternEstimate]
    estimated_cost: float


class QueryPlanner:
    """Greedy selectivity-based query planner.

    Estimates pattern selectivity from index cardinalities and orders
    joins so the most selective pattern is evaluated first, reducing
    intermediate result sizes.

    Args:
        store: The triple store to estimate cardinalities from.
    """

    def __init__(self, store: TripleStore) -> None:
        self.store = store

    def estimate_cardinality(self, pattern: TriplePattern) -> int:
        """Estimate the number of triples matching a pattern.

        Uses the store's cardinality method with bound positions.

        Args:
            pattern: A triple pattern (variables treated as unbound).

        Returns:
            Estimated number of matching triples.
        """
        s = pattern.subject if not pattern.subject.startswith("?") else None
        p = pattern.predicate if not pattern.predicate.startswith("?") else None
        o = pattern.object if not pattern.object.startswith("?") else None
        return self.store.cardinality(subject=s, predicate=p, obj=o)

    def estimate_selectivity(self, pattern: TriplePattern) -> PatternEstimate:
        """Compute a selectivity estimate for a pattern.

        Args:
            pattern: The triple pattern.

        Returns:
            A PatternEstimate with cardinality and normalized selectivity.
        """
        card = self.estimate_cardinality(pattern)
        total = len(self.store) or 1
        selectivity = card / total
        return PatternEstimate(
            pattern=pattern,
            cardinality=card,
            selectivity=selectivity,
        )

    def plan(self, query: ParsedQuery) -> JoinPlan:
        """Create a join plan by ordering patterns by selectivity.

        Uses a greedy strategy: always pick the most selective unprocessed
        pattern that shares at least one variable with already-processed
        patterns (to enable index nested-loop join).

        Args:
            query: The parsed query with triple patterns.

        Returns:
            A JoinPlan with ordered patterns and cost estimate.
        """
        estimates = [self.estimate_selectivity(p) for p in query.patterns]
        estimates.sort(key=lambda e: e.cardinality)

        if not estimates:
            return JoinPlan([], [], 0.0)

        ordered: list[PatternEstimate] = [estimates[0]]
        remaining = estimates[1:]
        bound_vars: set[str] = set(estimates[0].pattern.variables)

        while remaining:
            best = None
            best_idx = -1
            for i, est in enumerate(remaining):
                pattern_vars = set(est.pattern.variables)
                shares_var = bool(pattern_vars & bound_vars)
                if best is None:
                    best = est
                    best_idx = i
                elif shares_var and est.cardinality < best.cardinality:
                    best = est
                    best_idx = i
                elif not (set(best.pattern.variables) & bound_vars) and shares_var:
                    best = est
                    best_idx = i
            if best is not None:
                ordered.append(best)
                bound_vars.update(best.pattern.variables)
                remaining.pop(best_idx)

        cost = sum(e.cardinality for e in ordered)
        return JoinPlan(
            ordered_patterns=[e.pattern for e in ordered],
            estimates=ordered,
            estimated_cost=float(cost),
        )


if __name__ == "__main__":
    from src.store.triple_store import Triple

    store = TripleStore()
    triples = [
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
    for t in triples:
        store.add(t)

    from src.query.parser import QueryParser

    parser = QueryParser()
    query = parser.parse("SELECT ?x WHERE { ?x bornIn ?city . ?city locatedIn Germany }")

    planner = QueryPlanner(store)
    plan = planner.plan(query)

    print("Join plan (most selective first):")
    for est in plan.estimates:
        p = est.pattern
        print(f"  ({p.subject}, {p.predicate}, {p.object})")
        print(f"    cardinality={est.cardinality}, selectivity={est.selectivity:.3f}")
    print(f"Estimated cost: {plan.estimated_cost}")
