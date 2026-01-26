"""Pytest fixtures for lsh-cascade-poc tests."""

import numpy as np
import pytest

from src.lsh import SimHashGenerator


@pytest.fixture
def simhash_generator():
    """Create a SimHashGenerator for tests."""
    return SimHashGenerator(dim=1024, hash_bits=128, seed=42)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    rng = np.random.default_rng(42)
    vectors = rng.random((100, 1024)).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def sample_vector():
    """Generate a single sample vector."""
    rng = np.random.default_rng(42)
    vector = rng.random(1024).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def in_memory_db():
    """Create an in-memory VectorDatabase."""
    from src.db import VectorDatabase

    db = VectorDatabase(in_memory=True)
    db.initialize()
    yield db
    db.close()
