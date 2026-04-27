"""Negative sampling trainer for knowledge graph embeddings.

Handles training loop, negative sample generation, and evaluation
metrics (MRR, Hits@K) for both TransE and RotatE models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union

import numpy as np

from src.embeddings.transe import TransE, TransEConfig
from src.embeddings.rotate import RotatE, RotatEConfig


class EmbeddingModel(Protocol):
    """Protocol for embedding models that support scoring and training."""

    num_entities: int
    num_relations: int

    def score(self, head: int, relation: int, tail: int) -> float: ...
    def score_batch(
        self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray
    ) -> np.ndarray: ...
    def train_step(
        self,
        pos_heads: np.ndarray, pos_relations: np.ndarray, pos_tails: np.ndarray,
        neg_heads: np.ndarray, neg_tails: np.ndarray,
    ) -> float: ...


@dataclass
class TrainerConfig:
    """Configuration for the embedding trainer.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Number of triples per batch.
        neg_samples: Number of negative samples per positive triple.
        seed: Random seed for reproducibility.
    """

    epochs: int = 100
    batch_size: int = 256
    neg_samples: int = 1
    seed: int = 42


@dataclass
class TrainResult:
    """Result of a training run.

    Attributes:
        epoch_losses: Average loss per epoch.
        final_loss: Loss at the last epoch.
    """

    epoch_losses: list[float]
    final_loss: float


class NegativeSampler:
    """Generate corrupted (negative) triples by replacing head or tail.

    Args:
        num_entities: Total number of entities.
        triples_set: Set of true triples for filtering.
        seed: Random seed.
    """

    def __init__(
        self,
        num_entities: int,
        triples_set: set[tuple[int, int, int]],
        seed: int = 42,
    ) -> None:
        self.num_entities = num_entities
        self.triples_set = triples_set
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        heads: np.ndarray,
        relations: np.ndarray,
        tails: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate negative samples by corrupting head or tail.

        For each positive triple, randomly replaces either the head
        or tail with a random entity.

        Args:
            heads: Positive head indices.
            relations: Positive relation indices.
            tails: Positive tail indices.

        Returns:
            Tuple of (corrupted_heads, corrupted_tails).
        """
        n = len(heads)
        corrupt_head = self.rng.random(n) < 0.5
        neg_heads = heads.copy()
        neg_tails = tails.copy()

        head_mask = corrupt_head
        tail_mask = ~corrupt_head

        neg_heads[head_mask] = self.rng.randint(0, self.num_entities, size=int(head_mask.sum()))
        neg_tails[tail_mask] = self.rng.randint(0, self.num_entities, size=int(tail_mask.sum()))

        return neg_heads, neg_tails


class Trainer:
    """Training loop for knowledge graph embedding models.

    Args:
        model: An embedding model (TransE or RotatE).
        config: Trainer configuration.
    """

    def __init__(
        self,
        model: Union[TransE, RotatE],
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()

    def train(
        self,
        triples: np.ndarray,
        verbose: bool = False,
    ) -> TrainResult:
        """Train the model on a set of triples.

        Args:
            triples: Array of shape (N, 3) with columns [head, relation, tail].
            verbose: If True, print loss every 10 epochs.

        Returns:
            TrainResult with per-epoch losses.
        """
        rng = np.random.RandomState(self.config.seed)
        triples_set = set(map(tuple, triples.tolist()))
        sampler = NegativeSampler(
            self.model.num_entities, triples_set, self.config.seed
        )

        epoch_losses: list[float] = []
        n = len(triples)

        for epoch in range(self.config.epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, n, self.config.batch_size):
                batch_idx = perm[start: start + self.config.batch_size]
                batch = triples[batch_idx]
                heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]

                neg_heads, neg_tails = sampler.generate(heads, relations, tails)
                loss = self.model.train_step(heads, relations, tails, neg_heads, neg_tails)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {avg_loss:.4f}")

        return TrainResult(epoch_losses=epoch_losses, final_loss=epoch_losses[-1])

    def evaluate(
        self,
        triples: np.ndarray,
        ks: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate the model with MRR and Hits@K metrics.

        Args:
            triples: Test triples of shape (N, 3).
            ks: List of K values for Hits@K (default [1, 3, 10]).

        Returns:
            Dictionary with 'mrr' and 'hits@k' metrics.
        """
        if ks is None:
            ks = [1, 3, 10]

        ranks: list[int] = []
        for i in range(len(triples)):
            h, r, t = int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2])
            all_tails = np.arange(self.model.num_entities)
            heads_arr = np.full(self.model.num_entities, h)
            rels_arr = np.full(self.model.num_entities, r)
            scores = self.model.score_batch(heads_arr, rels_arr, all_tails)
            rank = int((scores >= scores[t]).sum())
            ranks.append(max(rank, 1))

        ranks_arr = np.array(ranks, dtype=float)
        mrr = float(np.mean(1.0 / ranks_arr))
        metrics: dict[str, float] = {"mrr": mrr}
        for k in ks:
            metrics[f"hits@{k}"] = float(np.mean(ranks_arr <= k))
        return metrics


if __name__ == "__main__":
    np.random.seed(42)

    triples = np.array([
        [0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4],
        [0, 1, 2], [1, 1, 3], [2, 1, 4], [4, 0, 0],
        [0, 2, 3], [1, 2, 4], [3, 1, 0], [4, 2, 1],
    ])

    model = TransE(num_entities=5, num_relations=3, config=TransEConfig(embedding_dim=32))
    trainer = Trainer(model, TrainerConfig(epochs=50, batch_size=6))

    print("Training TransE...")
    result = trainer.train(triples, verbose=True)
    print(f"Final loss: {result.final_loss:.4f}")

    metrics = trainer.evaluate(triples[:4], ks=[1, 3])
    print(f"\nEvaluation: {metrics}")
