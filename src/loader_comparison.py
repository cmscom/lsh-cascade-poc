"""Multi-model embedding loader for comparison experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Model configurations
# All models use sentence-transformers for consistent API
MODELS = {
    "e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "trust_remote_code": False,
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dim": 1024,
        "passage_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
    },
    "jina-v3": {
        "name": "jinaai/jina-embeddings-v3",
        "dim": 1024,
        "passage_prefix": "",
        "query_prefix": "",
        "trust_remote_code": True,
    },
}

ModelName = Literal["e5-large", "bge-m3", "jina-v3"]


class MultiModelEmbedder:
    """Embedder supporting multiple models for comparison.

    All models are loaded via sentence-transformers for consistent API.
    """

    def __init__(
        self,
        model_name: ModelName,
        device: str = "cpu",
    ) -> None:
        """Initialize embedder with specified model.

        Args:
            model_name: One of "e5-large", "bge-m3", "jina-v3".
            device: Device for inference ("cpu" or "cuda").
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

        self.model_name = model_name
        self.config = MODELS[model_name]
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        kwargs = {"device": self.device}
        if self.config.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True

        self._model = SentenceTransformer(self.config["name"], **kwargs)

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> NDArray[np.floating]:
        """Embed passage texts.

        Args:
            texts: List of passage texts (without prefix).
            batch_size: Batch size for inference.
            show_progress: Show progress bar.

        Returns:
            Array of shape (n, dim) with normalized embeddings.
        """
        self._load_model()

        # Add prefix if required
        prefix = self.config["passage_prefix"]
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.floating]:
        """Embed a single query text.

        Args:
            query: Query text (without prefix).

        Returns:
            Vector of shape (dim,) with normalized embedding.
        """
        self._load_model()

        # Add prefix if required
        prefix = self.config["query_prefix"]
        if prefix:
            query = f"{prefix}{query}"

        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        return np.array(embedding, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        return self.config["dim"]

    @staticmethod
    def list_models() -> list[str]:
        """List available model names."""
        return list(MODELS.keys())


def compute_embedding_stats(
    embeddings: NDArray[np.floating],
    sample_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute statistics about embedding distribution.

    Args:
        embeddings: Array of shape (n, dim).
        sample_size: Number of pairs to sample for similarity computation.
        seed: Random seed.

    Returns:
        Dictionary with statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)

    # Sample random pairs for cosine similarity
    idx1 = rng.integers(0, n, size=sample_size)
    idx2 = rng.integers(0, n, size=sample_size)

    # Compute cosine similarities (embeddings are normalized)
    cos_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)

    # Compute statistics
    stats = {
        "n_vectors": n,
        "dim": embeddings.shape[1],
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
        "cos_sim_min": float(np.min(cos_sims)),
        "cos_sim_max": float(np.max(cos_sims)),
        "cos_sim_median": float(np.median(cos_sims)),
    }

    return stats
