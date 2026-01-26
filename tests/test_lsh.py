"""Tests for src/lsh.py."""

import numpy as np
import pytest

from src.lsh import SimHashGenerator, chunk_hash, hamming_distance


class TestSimHashGenerator:
    """Tests for SimHashGenerator class."""

    def test_deterministic_hash(self, simhash_generator, sample_vector):
        """Same seed and vector should produce same hash."""
        hash1 = simhash_generator.hash(sample_vector)
        hash2 = simhash_generator.hash(sample_vector)
        assert hash1 == hash2

    def test_different_seeds_different_hash(self, sample_vector):
        """Different seeds should produce different hashes."""
        gen1 = SimHashGenerator(seed=42)
        gen2 = SimHashGenerator(seed=123)

        hash1 = gen1.hash(sample_vector)
        hash2 = gen2.hash(sample_vector)

        assert hash1 != hash2

    def test_hash_is_128bit(self, simhash_generator, sample_vector):
        """Hash should be within 128-bit range."""
        hash_val = simhash_generator.hash(sample_vector)
        assert 0 <= hash_val < (1 << 128)

    def test_similar_vectors_small_distance(self, simhash_generator):
        """Similar vectors should have small Hamming distance."""
        rng = np.random.default_rng(42)
        v1 = rng.random(1024).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        # Create similar vector with small perturbation
        noise = rng.random(1024).astype(np.float32) * 0.01
        v2 = v1 + noise
        v2 = v2 / np.linalg.norm(v2)

        hash1 = simhash_generator.hash(v1)
        hash2 = simhash_generator.hash(v2)

        distance = hamming_distance(hash1, hash2)
        # Similar vectors should have distance < 32 (quarter of 128 bits)
        assert distance < 32

    def test_orthogonal_vectors_large_distance(self, simhash_generator):
        """Orthogonal vectors should have distance around 64."""
        # Create orthogonal vectors
        v1 = np.zeros(1024, dtype=np.float32)
        v2 = np.zeros(1024, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0

        hash1 = simhash_generator.hash(v1)
        hash2 = simhash_generator.hash(v2)

        distance = hamming_distance(hash1, hash2)
        # Orthogonal vectors should have distance around 64 (half of 128)
        assert 40 < distance < 88

    def test_hash_batch(self, simhash_generator, sample_vectors):
        """Batch hashing should produce same results as individual hashing."""
        batch_hashes = simhash_generator.hash_batch(sample_vectors)

        for i, vector in enumerate(sample_vectors):
            single_hash = simhash_generator.hash(vector)
            assert batch_hashes[i] == single_hash

    def test_batch_length(self, simhash_generator, sample_vectors):
        """Batch should return correct number of hashes."""
        hashes = simhash_generator.hash_batch(sample_vectors)
        assert len(hashes) == len(sample_vectors)


class TestChunkHash:
    """Tests for chunk_hash function."""

    def test_4_chunks(self):
        """128 bits should split into 4 x 32-bit chunks."""
        # Use a known value for predictable output
        test_hash = 0x123456789ABCDEF0FEDCBA9876543210
        chunks = chunk_hash(test_hash, 4)

        assert len(chunks) == 4
        # Each chunk should have c<n>_ prefix and 8 hex chars
        for i, chunk in enumerate(chunks):
            assert chunk.startswith(f"c{i}_")
            assert len(chunk) == 3 + 8  # "cN_" + 8 hex chars

    def test_8_chunks(self):
        """128 bits should split into 8 x 16-bit chunks."""
        test_hash = 0x123456789ABCDEF0FEDCBA9876543210
        chunks = chunk_hash(test_hash, 8)

        assert len(chunks) == 8
        for i, chunk in enumerate(chunks):
            assert chunk.startswith(f"c{i}_")
            assert len(chunk) == 3 + 4  # "cN_" + 4 hex chars

    def test_16_chunks(self):
        """128 bits should split into 16 x 8-bit chunks."""
        test_hash = 0x123456789ABCDEF0FEDCBA9876543210
        chunks = chunk_hash(test_hash, 16)

        assert len(chunks) == 16
        for i, chunk in enumerate(chunks):
            prefix = f"c{i}_" if i < 10 else f"c{i}_"
            assert chunk.startswith(prefix)
            # 2 hex chars for 8 bits
            assert chunk.endswith(chunk[-2:].upper())

    def test_invalid_chunks_raises(self):
        """Invalid chunk count should raise ValueError."""
        with pytest.raises(ValueError):
            chunk_hash(0, 7)  # 128 not divisible by 7

    def test_chunk_reconstruction(self):
        """Chunks should be able to reconstruct original hash."""
        test_hash = 0xABCDEF0123456789FEDCBA9876543210
        chunks = chunk_hash(test_hash, 4)

        # Extract values and reconstruct
        reconstructed = 0
        for i, chunk in enumerate(chunks):
            hex_value = chunk.split("_")[1]
            value = int(hex_value, 16)
            shift = (3 - i) * 32  # 4 chunks, 32 bits each
            reconstructed |= value << shift

        assert reconstructed == test_hash


class TestHammingDistance:
    """Tests for hamming_distance function."""

    def test_same_hash_zero_distance(self):
        """Same hash should have zero distance."""
        h = 0x123456789ABCDEF0
        assert hamming_distance(h, h) == 0

    def test_all_bits_different(self):
        """All bits different should have max distance."""
        h1 = 0
        h2 = (1 << 128) - 1  # All 1s
        assert hamming_distance(h1, h2) == 128

    def test_one_bit_different(self):
        """One bit difference should have distance 1."""
        h1 = 0
        h2 = 1
        assert hamming_distance(h1, h2) == 1

    def test_known_distance(self):
        """Test with known bit pattern."""
        h1 = 0b1010  # 2 bits set
        h2 = 0b0101  # 2 bits set, all different positions
        assert hamming_distance(h1, h2) == 4

    def test_symmetry(self):
        """Distance should be symmetric."""
        h1 = 0x123456789ABCDEF0
        h2 = 0xFEDCBA9876543210
        assert hamming_distance(h1, h2) == hamming_distance(h2, h1)
