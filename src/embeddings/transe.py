"""TransE embedding model for knowledge graphs.

Implements the translational distance model where relations are
modeled as translations in embedding space: h + r ≈ t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransEConfig:
    """Configuration for the TransE model.

    Attributes:
        embedding_dim: Dimensionality of entity and relation embeddings.
        margin: Margin for the ranking loss.
        norm: L1 or L2 norm for distance computation.
        learning_rate: Learning rate for SGD updates.
    """

    embedding_dim: int = 128
    margin: float = 1.0
    norm: int = 2
    learning_rate: float = 0.01


class TransE:
    """TransE knowledge graph embedding model.

    Learns entity and relation embeddings such that for a true triple
    (h, r, t), the scoring function f(h,r,t) = -||h + r - t|| is
    maximized (distance is minimized).

    Args:
        num_entities: Number of unique entities.
        num_relations: Number of unique relations.
        config: Model configuration.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        config: Optional[TransEConfig] = None,
    ) -> None:
        self.config = config or TransEConfig()
        self.num_entities = num_entities
        self.num_relations = num_relations
        dim = self.config.embedding_dim

        bound = 6.0 / math.sqrt(dim)
        self.entity_embeddings = np.random.uniform(-bound, bound, (num_entities, dim))
        self.relation_embeddings = np.random.uniform(-bound, bound, (num_relations, dim))
        self._normalize_entities()

    def _normalize_entities(self) -> None:
        """Normalize entity embeddings to unit length."""
        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.entity_embeddings /= norms

    def score(self, head: int, relation: int, tail: int) -> float:
        """Compute the plausibility score for a triple.

        Lower distance means more plausible. Returns negative distance
        so higher score = more plausible.

        Args:
            head: Head entity index.
            relation: Relation index.
            tail: Tail entity index.

        Returns:
            Negative distance score.
        """
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]
        diff = h + r - t
        if self.config.norm == 1:
            return -float(np.sum(np.abs(diff)))
        return -float(np.sqrt(np.sum(diff ** 2)))

    def score_batch(
        self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray
    ) -> np.ndarray:
        """Score a batch of triples.

        Args:
            heads: Array of head entity indices.
            relations: Array of relation indices.
            tails: Array of tail entity indices.

        Returns:
            Array of negative distance scores.
        """
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        diff = h + r - t
        if self.config.norm == 1:
            return -np.sum(np.abs(diff), axis=1)
        return -np.sqrt(np.sum(diff ** 2, axis=1))

    def train_step(
        self,
        pos_heads: np.ndarray,
        pos_relations: np.ndarray,
        pos_tails: np.ndarray,
        neg_heads: np.ndarray,
        neg_tails: np.ndarray,
    ) -> float:
        """Perform one training step with margin-based ranking loss.

        Args:
            pos_heads: Positive triple head indices.
            pos_relations: Positive triple relation indices.
            pos_tails: Positive triple tail indices.
            neg_heads: Negative (corrupted) head indices.
            neg_tails: Negative (corrupted) tail indices.

        Returns:
            Average loss for this batch.
        """
        lr = self.config.learning_rate
        margin = self.config.margin

        pos_scores = self.score_batch(pos_heads, pos_relations, pos_tails)
        neg_scores = self.score_batch(neg_heads, pos_relations, neg_tails)

        losses = np.maximum(0, margin + (-pos_scores) - (-neg_scores))
        mask = losses > 0

        if not np.any(mask):
            return 0.0

        h_pos = self.entity_embeddings[pos_heads[mask]]
        r_pos = self.relation_embeddings[pos_relations[mask]]
        t_pos = self.entity_embeddings[pos_tails[mask]]
        h_neg = self.entity_embeddings[neg_heads[mask]]
        t_neg = self.entity_embeddings[neg_tails[mask]]

        pos_diff = h_pos + r_pos - t_pos
        neg_diff = h_neg + r_pos - t_neg

        pos_grad = 2 * pos_diff if self.config.norm == 2 else np.sign(pos_diff)
        neg_grad = 2 * neg_diff if self.config.norm == 2 else np.sign(neg_diff)

        np.add.at(self.entity_embeddings, pos_heads[mask], -lr * pos_grad)
        np.add.at(self.relation_embeddings, pos_relations[mask], -lr * pos_grad)
        np.add.at(self.entity_embeddings, pos_tails[mask], lr * pos_grad)
        np.add.at(self.entity_embeddings, neg_heads[mask], lr * neg_grad)
        np.add.at(self.entity_embeddings, neg_tails[mask], -lr * neg_grad)

        self._normalize_entities()
        return float(np.mean(losses))

    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """Get the embedding vector for an entity.

        Args:
            entity_id: Entity index.

        Returns:
            Embedding vector.
        """
        return self.entity_embeddings[entity_id].copy()

    def get_relation_embedding(self, relation_id: int) -> np.ndarray:
        """Get the embedding vector for a relation.

        Args:
            relation_id: Relation index.

        Returns:
            Embedding vector.
        """
        return self.relation_embeddings[relation_id].copy()


if __name__ == "__main__":
    np.random.seed(42)
    model = TransE(num_entities=10, num_relations=3, config=TransEConfig(embedding_dim=50))

    print(f"Entity embeddings shape: {model.entity_embeddings.shape}")
    print(f"Relation embeddings shape: {model.relation_embeddings.shape}")

    score = model.score(0, 1, 2)
    print(f"\nScore for triple (0, 1, 2): {score:.4f}")

    heads = np.array([0, 1, 2])
    rels = np.array([0, 1, 0])
    tails = np.array([3, 4, 5])
    scores = model.score_batch(heads, rels, tails)
    print(f"Batch scores: {scores}")

    neg_h = np.array([5, 6, 7])
    neg_t = np.array([8, 9, 0])
    loss = model.train_step(heads, rels, tails, neg_h, neg_t)
    print(f"Training loss: {loss:.4f}")
