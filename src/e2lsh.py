"""E2LSH (Euclidean LSH) implementation using p-stable distributions.

E2LSH uses random projections with bucketing for Euclidean distance-based
similarity search, as opposed to SimHash which is angle-based.

Hash function: h(v) = floor((a·v + b) / w)
- a: random vector from Gaussian N(0, 1)
- b: random offset from Uniform[0, w)
- w: bucket width parameter
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class E2LSHParams:
    """Parameters for E2LSH."""

    dim: int  # Vector dimension
    w: float = 4.0  # Bucket width
    k: int = 8  # Number of hash functions per table
    num_tables: int = 1  # Number of hash tables (L)
    seed: int = 42


class E2LSHHasher:
    """E2LSH hasher using random projections.

    Each hash function: h(v) = floor((a·v + b) / w)

    For similarity search, vectors that are close in Euclidean distance
    are more likely to have the same hash value.
    """

    def __init__(
        self,
        dim: int,
        w: float = 4.0,
        k: int = 8,
        num_tables: int = 1,
        seed: int = 42,
    ) -> None:
        """Initialize E2LSH hasher.

        Args:
            dim: Dimension of input vectors.
            w: Bucket width (larger = more collisions, coarser buckets).
            k: Number of hash functions per table.
            num_tables: Number of hash tables (L).
            seed: Random seed for reproducibility.
        """
        self.dim = dim
        self.w = w
        self.k = k
        self.num_tables = num_tables
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Generate random projection vectors: shape (num_tables, k, dim)
        # Each element from Gaussian N(0, 1)
        self.projections = rng.standard_normal(
            (num_tables, k, dim)
        ).astype(np.float32)

        # Generate random offsets: shape (num_tables, k)
        # Each element from Uniform[0, w)
        self.offsets = rng.uniform(0, w, (num_tables, k)).astype(np.float32)

    def hash_single(self, vector: NDArray[np.floating], table_idx: int = 0) -> tuple:
        """Compute hash for a single vector in one table.

        Args:
            vector: Input vector of shape (dim,).
            table_idx: Which hash table to use.

        Returns:
            Tuple of k hash values (bucket indices).
        """
        # h(v) = floor((a·v + b) / w)
        projections = self.projections[table_idx]  # (k, dim)
        offsets = self.offsets[table_idx]  # (k,)

        # Compute projections: (k,)
        proj = projections @ vector  # (k,)
        hash_values = np.floor((proj + offsets) / self.w).astype(np.int32)

        return tuple(hash_values)

    def hash_batch(
        self, vectors: NDArray[np.floating], table_idx: int = 0
    ) -> list[tuple]:
        """Compute hashes for multiple vectors in one table.

        Args:
            vectors: Input vectors of shape (n, dim).
            table_idx: Which hash table to use.

        Returns:
            List of n tuples, each containing k hash values.
        """
        projections = self.projections[table_idx]  # (k, dim)
        offsets = self.offsets[table_idx]  # (k,)

        # Batch projection: (n, dim) @ (dim, k) = (n, k)
        proj = vectors @ projections.T  # (n, k)
        hash_values = np.floor((proj + offsets) / self.w).astype(np.int32)

        return [tuple(row) for row in hash_values]

    def hash_all_tables(self, vector: NDArray[np.floating]) -> list[tuple]:
        """Compute hashes for a vector in all tables.

        Args:
            vector: Input vector of shape (dim,).

        Returns:
            List of num_tables tuples.
        """
        return [self.hash_single(vector, t) for t in range(self.num_tables)]

    def hash_batch_all_tables(
        self, vectors: NDArray[np.floating]
    ) -> list[list[tuple]]:
        """Compute hashes for multiple vectors in all tables.

        Args:
            vectors: Input vectors of shape (n, dim).

        Returns:
            List of n lists, each containing num_tables tuples.
        """
        n = len(vectors)
        # Initialize: n vectors, each with num_tables hashes
        result = [[] for _ in range(n)]

        for t in range(self.num_tables):
            table_hashes = self.hash_batch(vectors, t)
            for i, h in enumerate(table_hashes):
                result[i].append(h)

        return result


def hash_similarity(hash1: tuple, hash2: tuple) -> int:
    """Count number of matching hash components.

    Args:
        hash1: First hash tuple.
        hash2: Second hash tuple.

    Returns:
        Number of matching components (0 to k).
    """
    return sum(1 for a, b in zip(hash1, hash2) if a == b)


def hash_distance(hash1: tuple, hash2: tuple) -> int:
    """Count number of differing hash components.

    Args:
        hash1: First hash tuple.
        hash2: Second hash tuple.

    Returns:
        Number of differing components (0 to k).
    """
    return sum(1 for a, b in zip(hash1, hash2) if a != b)


class E2LSHIndex:
    """E2LSH index for approximate nearest neighbor search.

    Uses multiple hash tables to improve recall.
    """

    def __init__(self, hasher: E2LSHHasher) -> None:
        """Initialize index with hasher.

        Args:
            hasher: E2LSH hasher instance.
        """
        self.hasher = hasher
        self.tables: list[dict[tuple, list[int]]] = [
            {} for _ in range(hasher.num_tables)
        ]
        self.vectors: NDArray[np.floating] | None = None

    def build(self, vectors: NDArray[np.floating]) -> None:
        """Build index from vectors.

        Args:
            vectors: Input vectors of shape (n, dim).
        """
        self.vectors = vectors
        n = len(vectors)

        # Clear tables
        self.tables = [{} for _ in range(self.hasher.num_tables)]

        # Hash all vectors
        all_hashes = self.hasher.hash_batch_all_tables(vectors)

        # Insert into tables
        for i in range(n):
            for t in range(self.hasher.num_tables):
                h = all_hashes[i][t]
                if h not in self.tables[t]:
                    self.tables[t][h] = []
                self.tables[t][h].append(i)

    def query(self, vector: NDArray[np.floating], top_k: int = 10) -> list[int]:
        """Find approximate nearest neighbors.

        Args:
            vector: Query vector of shape (dim,).
            top_k: Number of neighbors to return.

        Returns:
            List of indices of nearest neighbors.
        """
        if self.vectors is None:
            raise ValueError("Index not built. Call build() first.")

        # Get hashes for query
        query_hashes = self.hasher.hash_all_tables(vector)

        # Collect candidates from all tables
        candidates = set()
        for t in range(self.hasher.num_tables):
            h = query_hashes[t]
            if h in self.tables[t]:
                candidates.update(self.tables[t][h])

        if not candidates:
            return []

        # Rank by actual distance
        distances = [
            (i, np.linalg.norm(self.vectors[i] - vector))
            for i in candidates
        ]
        distances.sort(key=lambda x: x[1])

        return [i for i, _ in distances[:top_k]]


def compute_e2lsh_collision_prob(
    distance: float, w: float, dim: int = 1024
) -> float:
    """Estimate collision probability for E2LSH.

    For Gaussian projections, the collision probability is approximately:
    P(h(p) = h(q)) ≈ 1 - 2*Φ(-w/(2*d)) - (2/sqrt(2π)) * (d/w) * (1 - exp(-w²/(2*d²)))

    where d is the Euclidean distance and Φ is the CDF of standard normal.

    Simplified approximation for small d:
    P ≈ 1 - d/w

    Args:
        distance: Euclidean distance between vectors.
        w: Bucket width.
        dim: Vector dimension (not used in simple approximation).

    Returns:
        Estimated collision probability.
    """
    # Simple linear approximation
    return max(0, 1 - distance / w)
