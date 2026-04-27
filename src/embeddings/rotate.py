"""RotatE embedding model for knowledge graphs.

Implements the rotational model in complex vector space where each
relation is a rotation: t = h ∘ r, with |r_i| = 1 for all i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RotatEConfig:
    """Configuration for the RotatE model.

    Attributes:
        embedding_dim: Dimensionality of embeddings (in complex space,
            so real storage is 2 * embedding_dim).
        margin: Margin for the ranking loss.
        learning_rate: Learning rate for SGD updates.
        adversarial_temperature: Temperature for self-adversarial sampling.
    """

    embedding_dim: int = 128
    margin: float = 6.0
    learning_rate: float = 0.001
    adversarial_temperature: float = 1.0


class RotatE:
    """RotatE knowledge graph embedding model.

    Models relations as rotations in complex vector space. For a true
    triple (h, r, t), the tail is the element-wise product of head
    and relation: t = h ∘ r, where each r_i lies on the unit circle.

    Args:
        num_entities: Number of unique entities.
        num_relations: Number of unique relations.
        config: Model configuration.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        config: Optional[RotatEConfig] = None,
    ) -> None:
        self.config = config or RotatEConfig()
        self.num_entities = num_entities
        self.num_relations = num_relations
        dim = self.config.embedding_dim

        bound = 6.0 / math.sqrt(dim)
        self.entity_re = np.random.uniform(-bound, bound, (num_entities, dim))
        self.entity_im = np.random.uniform(-bound, bound, (num_entities, dim))

        phases = np.random.uniform(-math.pi, math.pi, (num_relations, dim))
        self.relation_re = np.cos(phases)
        self.relation_im = np.sin(phases)

    def score(self, head: int, relation: int, tail: int) -> float:
        """Compute the plausibility score for a triple.

        Score = -||h ∘ r - t|| where ∘ is complex multiplication.

        Args:
            head: Head entity index.
            relation: Relation index.
            tail: Tail entity index.

        Returns:
            Negative distance score (higher = more plausible).
        """
        h_re, h_im = self.entity_re[head], self.entity_im[head]
        r_re, r_im = self.relation_re[relation], self.relation_im[relation]
        t_re, t_im = self.entity_re[tail], self.entity_im[tail]

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        dist = np.sqrt(np.sum(diff_re ** 2 + diff_im ** 2))
        return -float(dist)

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
        h_re = self.entity_re[heads]
        h_im = self.entity_im[heads]
        r_re = self.relation_re[relations]
        r_im = self.relation_im[relations]
        t_re = self.entity_re[tails]
        t_im = self.entity_im[tails]

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        return -np.sqrt(np.sum(diff_re ** 2 + diff_im ** 2, axis=1))

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
            neg_heads: Negative head indices.
            neg_tails: Negative tail indices.

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

        self._update_embeddings(
            pos_heads[mask], pos_relations[mask], pos_tails[mask],
            neg_heads[mask], neg_tails[mask], lr,
        )
        self._project_relations()
        return float(np.mean(losses))

    def _update_embeddings(
        self,
        pos_h: np.ndarray, pos_r: np.ndarray, pos_t: np.ndarray,
        neg_h: np.ndarray, neg_t: np.ndarray,
        lr: float,
    ) -> None:
        """Apply gradient updates to embeddings."""
        h_re = self.entity_re[pos_h]
        h_im = self.entity_im[pos_h]
        r_re = self.relation_re[pos_r]
        r_im = self.relation_im[pos_r]
        t_re = self.entity_re[pos_t]
        t_im = self.entity_im[pos_t]

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re
        diff_re = rot_re - t_re
        diff_im = rot_im - t_im

        np.add.at(self.entity_re, pos_h, -lr * diff_re * r_re)
        np.add.at(self.entity_im, pos_h, -lr * diff_im * r_re)
        np.add.at(self.entity_re, pos_t, lr * diff_re)
        np.add.at(self.entity_im, pos_t, lr * diff_im)

    def _project_relations(self) -> None:
        """Project relation embeddings back onto the unit circle."""
        norms = np.sqrt(self.relation_re ** 2 + self.relation_im ** 2)
        norms = np.maximum(norms, 1e-8)
        self.relation_re /= norms
        self.relation_im /= norms

    def get_entity_embedding(self, entity_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the complex embedding for an entity.

        Args:
            entity_id: Entity index.

        Returns:
            Tuple of (real_part, imaginary_part) arrays.
        """
        return self.entity_re[entity_id].copy(), self.entity_im[entity_id].copy()

    def get_relation_phases(self, relation_id: int) -> np.ndarray:
        """Get the phase angles for a relation.

        Args:
            relation_id: Relation index.

        Returns:
            Array of phase angles in radians.
        """
        return np.arctan2(
            self.relation_im[relation_id], self.relation_re[relation_id]
        )


if __name__ == "__main__":
    np.random.seed(42)
    model = RotatE(num_entities=10, num_relations=3, config=RotatEConfig(embedding_dim=50))

    print(f"Entity embeddings: re={model.entity_re.shape}, im={model.entity_im.shape}")
    print(f"Relation embeddings: re={model.relation_re.shape}, im={model.relation_im.shape}")

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

    phases = model.get_relation_phases(0)
    print(f"\nRelation 0 phases (first 5): {phases[:5]}")
