"""Top-k link prediction using trained embedding models.

Given a partial triple (h, r, ?) or (?, r, t), ranks all candidate
entities by plausibility score and returns the top-k predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from src.embeddings.transe import TransE
from src.embeddings.rotate import RotatE


@dataclass
class Prediction:
    """A single link prediction result.

    Attributes:
        entity_id: Predicted entity index.
        entity_name: Entity name (if vocabulary provided).
        score: Plausibility score (higher = more plausible).
        rank: Rank among all candidates (1-indexed).
    """

    entity_id: int
    entity_name: str
    score: float
    rank: int


class LinkPredictor:
    """Top-k link prediction using a trained embedding model.

    Supports both tail prediction (h, r, ?) and head prediction (?, r, t).

    Args:
        model: A trained TransE or RotatE model.
        entity_vocab: Optional mapping from entity index to name.
        relation_vocab: Optional mapping from relation index to name.
    """

    def __init__(
        self,
        model: Union[TransE, RotatE],
        entity_vocab: dict[int, str] | None = None,
        relation_vocab: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.entity_vocab = entity_vocab or {}
        self.relation_vocab = relation_vocab or {}

    def predict_tail(
        self, head: int, relation: int, top_k: int = 10
    ) -> list[Prediction]:
        """Predict the most likely tail entities for (head, relation, ?).

        Args:
            head: Head entity index.
            relation: Relation index.
            top_k: Number of predictions to return.

        Returns:
            List of Prediction objects sorted by score (descending).
        """
        n = self.model.num_entities
        heads = np.full(n, head)
        relations = np.full(n, relation)
        tails = np.arange(n)
        scores = self.model.score_batch(heads, relations, tails)
        return self._rank_predictions(scores, top_k)

    def predict_head(
        self, relation: int, tail: int, top_k: int = 10
    ) -> list[Prediction]:
        """Predict the most likely head entities for (?, relation, tail).

        Args:
            relation: Relation index.
            tail: Tail entity index.
            top_k: Number of predictions to return.

        Returns:
            List of Prediction objects sorted by score (descending).
        """
        n = self.model.num_entities
        heads = np.arange(n)
        relations = np.full(n, relation)
        tails = np.full(n, tail)
        scores = self.model.score_batch(heads, relations, tails)
        return self._rank_predictions(scores, top_k)

    def predict_relation(
        self, head: int, tail: int, top_k: int = 5
    ) -> list[tuple[int, str, float]]:
        """Predict the most likely relations for (head, ?, tail).

        Args:
            head: Head entity index.
            tail: Tail entity index.
            top_k: Number of predictions to return.

        Returns:
            List of (relation_id, relation_name, score) tuples.
        """
        results: list[tuple[int, str, float]] = []
        for r in range(self.model.num_relations):
            score = self.model.score(head, r, tail)
            name = self.relation_vocab.get(r, f"rel_{r}")
            results.append((r, name, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _rank_predictions(
        self, scores: np.ndarray, top_k: int
    ) -> list[Prediction]:
        """Rank entities by score and return top-k.

        Args:
            scores: Array of scores for all entities.
            top_k: Number of results.

        Returns:
            Sorted list of Prediction objects.
        """
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            predictions.append(
                Prediction(
                    entity_id=int(idx),
                    entity_name=self.entity_vocab.get(int(idx), f"entity_{idx}"),
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return predictions

    def triple_score(self, head: int, relation: int, tail: int) -> float:
        """Get the plausibility score for a specific triple.

        Args:
            head: Head entity index.
            relation: Relation index.
            tail: Tail entity index.

        Returns:
            Plausibility score.
        """
        return self.model.score(head, relation, tail)


if __name__ == "__main__":
    from src.embeddings.transe import TransEConfig
    from src.embeddings.trainer import Trainer, TrainerConfig

    np.random.seed(42)

    entity_names = {0: "Einstein", 1: "Ulm", 2: "Germany", 3: "Curie", 4: "Warsaw"}
    relation_names = {0: "bornIn", 1: "locatedIn", 2: "field"}

    triples = np.array([
        [0, 0, 1], [3, 0, 4], [1, 1, 2],
        [4, 1, 2], [0, 2, 2], [3, 2, 2],
    ])

    model = TransE(num_entities=5, num_relations=3, config=TransEConfig(embedding_dim=32))
    trainer = Trainer(model, TrainerConfig(epochs=100, batch_size=6))
    trainer.train(triples, verbose=False)

    predictor = LinkPredictor(model, entity_names, relation_names)

    print("Predict tail: (Einstein, bornIn, ?)")
    for p in predictor.predict_tail(0, 0, top_k=3):
        print(f"  #{p.rank}: {p.entity_name} (score={p.score:.4f})")

    print("\nPredict head: (?, locatedIn, Germany)")
    for p in predictor.predict_head(1, 2, top_k=3):
        print(f"  #{p.rank}: {p.entity_name} (score={p.score:.4f})")

    print("\nPredict relation: (Einstein, ?, Ulm)")
    for r_id, r_name, score in predictor.predict_relation(0, 1, top_k=3):
        print(f"  {r_name} (score={score:.4f})")
