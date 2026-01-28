"""LSH (Locality Sensitive Hashing) core logic.

This module implements SimHash algorithm for vector fingerprinting
and provides utilities for hash manipulation and distance calculation.

Includes multiple hyperplane generation strategies:
- Random: Standard N(0,1) random hyperplanes
- Orthogonal: QR-decomposed orthogonal hyperplanes (BOLSH-style)
- DataSampled: Hyperplanes from data pair differences
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class HyperplaneStrategy(Enum):
    """Hyperplane generation strategy for SimHash."""

    RANDOM = "random"  # Standard N(0,1) random hyperplanes
    ORTHOGONAL = "orthogonal"  # QR-decomposed orthogonal hyperplanes
    DATA_SAMPLED = "data_sampled"  # From data pair differences


def generate_random_hyperplanes(
    dim: int, num_planes: int, seed: int = 42
) -> NDArray[np.floating]:
    """Generate standard random hyperplanes from N(0,1).

    Args:
        dim: Vector dimension.
        num_planes: Number of hyperplanes to generate.
        seed: Random seed.

    Returns:
        Array of shape (num_planes, dim).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_planes, dim)).astype(np.float32)


def generate_orthogonal_hyperplanes(
    dim: int, num_planes: int, seed: int = 42
) -> NDArray[np.floating]:
    """Generate orthogonalized hyperplanes using QR decomposition (BOLSH-style).

    Orthogonal projections reduce variance in angle estimation,
    potentially improving hash quality for narrow cone distributions.

    Args:
        dim: Vector dimension.
        num_planes: Number of hyperplanes (must be <= dim).
        seed: Random seed.

    Returns:
        Array of shape (num_planes, dim) with orthonormal rows.
    """
    if num_planes > dim:
        raise ValueError(f"num_planes ({num_planes}) must be <= dim ({dim})")

    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((dim, num_planes)).astype(np.float32)

    # QR decomposition: Q is orthonormal
    Q, R = np.linalg.qr(random_matrix)

    return Q.T[:num_planes]  # (num_planes, dim)


def generate_data_sampled_hyperplanes(
    embeddings: NDArray[np.floating], num_planes: int, seed: int = 42
) -> NDArray[np.floating]:
    """Generate hyperplanes from data pair differences.

    For narrow cone distributions, hyperplanes derived from actual data
    differences may better capture the structure within the cone.

    Args:
        embeddings: Data embeddings of shape (n, dim).
        num_planes: Number of hyperplanes to generate.
        seed: Random seed.

    Returns:
        Array of shape (num_planes, dim) with normalized difference vectors.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)

    if n < 2:
        raise ValueError("Need at least 2 embeddings to generate hyperplanes")

    hyperplanes = []
    for _ in range(num_planes):
        # Select random pair
        i, j = rng.choice(n, 2, replace=False)
        diff = embeddings[i] - embeddings[j]

        # Normalize
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diff = diff / norm
        else:
            # Fallback to random if difference is too small
            diff = rng.standard_normal(embeddings.shape[1])
            diff = diff / np.linalg.norm(diff)

        hyperplanes.append(diff)

    return np.array(hyperplanes, dtype=np.float32)


def generate_multiprobe_hashes(
    base_hash: int, num_bits: int = 128, max_flips: int = 1
) -> list[int]:
    """Generate multi-probe hashes by flipping bits.

    For SimHash, nearby hashes are those with small Hamming distance.
    This function generates candidate hashes by flipping 1 or more bits.

    Args:
        base_hash: Original hash value.
        num_bits: Number of bits in the hash.
        max_flips: Maximum number of bits to flip (1 or 2).

    Returns:
        List of probe hashes including the original.
    """
    probes = [base_hash]

    # Single bit flips
    for i in range(num_bits):
        flipped = base_hash ^ (1 << i)
        probes.append(flipped)

    # Double bit flips (if requested)
    if max_flips >= 2:
        for i in range(num_bits):
            for j in range(i + 1, num_bits):
                flipped = base_hash ^ (1 << i) ^ (1 << j)
                probes.append(flipped)

    return probes


class SimHashGenerator:
    """SimHash algorithm for converting vectors to binary fingerprints.

    Uses hyperplane projection to convert high-dimensional vectors
    to fixed-length binary representations. Vectors that are similar in
    the original space will have similar hash values (small Hamming distance).

    Supports multiple hyperplane generation strategies:
    - "random": Standard N(0,1) random hyperplanes
    - "orthogonal": QR-decomposed orthogonal hyperplanes (BOLSH-style)
    - "data_sampled": Hyperplanes from data pair differences (requires fit_data)
    """

    def __init__(
        self,
        dim: int = 1024,
        hash_bits: int = 128,
        seed: int = 42,
        strategy: Literal["random", "orthogonal", "data_sampled"] = "random",
    ) -> None:
        """Initialize SimHash generator.

        Args:
            dim: Input vector dimension (1024 for multilingual-e5-large).
            hash_bits: Number of output hash bits (default 128).
            seed: Random seed for reproducibility.
            strategy: Hyperplane generation strategy.
        """
        self.dim = dim
        self.hash_bits = hash_bits
        self.seed = seed
        self.strategy = strategy

        # Generate hyperplanes based on strategy
        if strategy == "random":
            self.hyperplanes = generate_random_hyperplanes(dim, hash_bits, seed)
        elif strategy == "orthogonal":
            self.hyperplanes = generate_orthogonal_hyperplanes(dim, hash_bits, seed)
        elif strategy == "data_sampled":
            # Will be set later via fit_data()
            self.hyperplanes = None
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def fit_data(self, embeddings: NDArray[np.floating]) -> None:
        """Fit hyperplanes from data (for data_sampled strategy).

        Args:
            embeddings: Training embeddings of shape (n, dim).
        """
        if self.strategy != "data_sampled":
            raise ValueError("fit_data() is only for data_sampled strategy")

        self.hyperplanes = generate_data_sampled_hyperplanes(
            embeddings, self.hash_bits, self.seed
        )

    def hash(self, vector: NDArray[np.floating]) -> int:
        """Convert a vector to a hash integer.

        Args:
            vector: 1D array of shape (dim,).

        Returns:
            Hash value as an integer (hash_bits bits).

        Raises:
            ValueError: If hyperplanes not initialized (data_sampled without fit_data).
        """
        if self.hyperplanes is None:
            raise ValueError("Hyperplanes not initialized. Call fit_data() first.")

        # Project vector onto hyperplanes
        projections = self.hyperplanes @ vector

        # Convert to binary: 1 if projection >= 0, else 0
        bits = (projections >= 0).astype(np.uint8)

        # Convert bit array to integer
        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)

        return hash_int

    def hash_batch(self, vectors: NDArray[np.floating]) -> list[int]:
        """Convert multiple vectors to hash integers.

        Args:
            vectors: 2D array of shape (n, dim).

        Returns:
            List of n hash integers.

        Raises:
            ValueError: If hyperplanes not initialized (data_sampled without fit_data).
        """
        if self.hyperplanes is None:
            raise ValueError("Hyperplanes not initialized. Call fit_data() first.")

        # Batch projection: (hash_bits, dim) @ (dim, n) -> (hash_bits, n)
        projections = self.hyperplanes @ vectors.T

        # Convert to binary
        bits = (projections >= 0).astype(np.uint8)  # (hash_bits, n)

        # Convert each column to integer
        result = []
        for i in range(bits.shape[1]):
            hash_int = 0
            for bit in bits[:, i]:
                hash_int = (hash_int << 1) | int(bit)
            result.append(hash_int)

        return result


def chunk_hash(simhash_int: int, num_chunks: int) -> list[str]:
    """Split a hash into chunks for indexing.

    Divides the 128-bit hash into num_chunks equal parts and returns
    them as prefixed hex strings for use in inverted index lookups.

    Args:
        simhash_int: Hash value as integer.
        num_chunks: Number of chunks (4, 8, or 16).

    Returns:
        List of prefixed hex strings like ["c0_A1B2C3D4", "c1_E5F6G7H8", ...].

    Raises:
        ValueError: If 128 is not divisible by num_chunks.
    """
    hash_bits = 128

    if hash_bits % num_chunks != 0:
        raise ValueError(f"128 must be divisible by num_chunks ({num_chunks})")

    bits_per_chunk = hash_bits // num_chunks
    mask = (1 << bits_per_chunk) - 1

    chunks = []
    for i in range(num_chunks):
        # Extract chunk from right to left
        chunk_idx = num_chunks - 1 - i
        chunk_value = (simhash_int >> (i * bits_per_chunk)) & mask

        # Format as hex with prefix
        hex_width = bits_per_chunk // 4
        hex_str = f"c{chunk_idx}_{chunk_value:0{hex_width}X}"
        chunks.append(hex_str)

    # Reverse to get c0, c1, c2, ... order
    chunks.reverse()

    return chunks


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Calculate Hamming distance between two hash values.

    Uses XOR and popcount to efficiently count differing bits.

    Args:
        hash_a: First hash value.
        hash_b: Second hash value.

    Returns:
        Number of differing bits (0 to hash_bits).
    """
    xor = hash_a ^ hash_b
    return xor.bit_count()
