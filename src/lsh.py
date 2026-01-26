"""LSH (Locality Sensitive Hashing) core logic.

This module implements SimHash algorithm for vector fingerprinting
and provides utilities for hash manipulation and distance calculation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SimHashGenerator:
    """SimHash algorithm for converting vectors to binary fingerprints.

    Uses random hyperplane projection to convert high-dimensional vectors
    to fixed-length binary representations. Vectors that are similar in
    the original space will have similar hash values (small Hamming distance).
    """

    def __init__(
        self,
        dim: int = 1024,
        hash_bits: int = 128,
        seed: int = 42,
    ) -> None:
        """Initialize SimHash generator with random hyperplanes.

        Args:
            dim: Input vector dimension (1024 for multilingual-e5-large).
            hash_bits: Number of output hash bits (default 128).
            seed: Random seed for reproducibility.
        """
        self.dim = dim
        self.hash_bits = hash_bits
        self.seed = seed

        # Generate random hyperplanes for projection
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.standard_normal((hash_bits, dim)).astype(np.float32)

    def hash(self, vector: NDArray[np.floating]) -> int:
        """Convert a vector to a hash integer.

        Args:
            vector: 1D array of shape (dim,).

        Returns:
            Hash value as an integer (hash_bits bits).
        """
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
        """
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
