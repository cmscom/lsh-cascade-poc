"""Tests for src/db.py."""

import numpy as np
import pandas as pd
import pytest

from src.db import VectorDatabase


class TestVectorDatabase:
    """Tests for VectorDatabase class."""

    def test_initialization(self, in_memory_db):
        """Database should initialize without errors."""
        assert in_memory_db.count() == 0

    def test_insert_and_count(self, in_memory_db):
        """Should insert documents and count correctly."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "text": ["doc1", "doc2", "doc3"],
                "vector": [
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                ],
                "simhash": ["A" * 32, "B" * 32, "C" * 32],
                "lsh_chunks": [
                    ["c0_1234", "c1_5678"],
                    ["c0_1234", "c1_ABCD"],
                    ["c0_FFFF", "c1_5678"],
                ],
            }
        )

        inserted = in_memory_db.insert_dataframe(df)
        assert inserted == 3
        assert in_memory_db.count() == 3

    def test_lsh_chunk_search(self, in_memory_db):
        """Should find documents with matching chunks."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "text": ["match1", "match2", "no_match"],
                "vector": [
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                ],
                "simhash": ["A" * 32, "B" * 32, "C" * 32],
                "lsh_chunks": [
                    ["c0_AAAA", "c1_BBBB"],
                    ["c0_AAAA", "c1_CCCC"],
                    ["c0_DDDD", "c1_EEEE"],
                ],
            }
        )

        in_memory_db.insert_dataframe(df)

        # Search for c0_AAAA - should match doc 0 and 1
        results = in_memory_db.search_lsh_chunks(["c0_AAAA"])
        assert len(results) == 2
        assert set(results["id"]) == {0, 1}

        # Search for c1_BBBB - should match only doc 0
        results = in_memory_db.search_lsh_chunks(["c1_BBBB"])
        assert len(results) == 1
        assert results.iloc[0]["id"] == 0

        # Search for non-existent chunk
        results = in_memory_db.search_lsh_chunks(["c0_ZZZZ"])
        assert len(results) == 0

    def test_lsh_chunk_or_search(self, in_memory_db):
        """Should find documents matching ANY of the query chunks."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "text": ["doc1", "doc2", "doc3"],
                "vector": [
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                    rng.random(1024).tolist(),
                ],
                "simhash": ["A" * 32, "B" * 32, "C" * 32],
                "lsh_chunks": [
                    ["c0_AAAA"],
                    ["c0_BBBB"],
                    ["c0_CCCC"],
                ],
            }
        )

        in_memory_db.insert_dataframe(df)

        # Search for multiple chunks - OR condition
        results = in_memory_db.search_lsh_chunks(["c0_AAAA", "c0_BBBB"])
        assert len(results) == 2
        assert set(results["id"]) == {0, 1}

    def test_get_by_ids(self, in_memory_db):
        """Should retrieve documents by ID list."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "text": ["a", "b", "c", "d", "e"],
                "vector": [rng.random(1024).tolist() for _ in range(5)],
                "simhash": [f"{i}" * 32 for i in range(5)],
                "lsh_chunks": [["c0_0000"] for _ in range(5)],
            }
        )

        in_memory_db.insert_dataframe(df)

        results = in_memory_db.get_by_ids([1, 3])
        assert len(results) == 2
        assert set(results["id"]) == {1, 3}

    def test_get_by_empty_ids(self, in_memory_db):
        """Empty ID list should return empty DataFrame."""
        results = in_memory_db.get_by_ids([])
        assert len(results) == 0

    def test_clear(self, in_memory_db):
        """Should clear all documents."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [0],
                "text": ["test"],
                "vector": [rng.random(1024).tolist()],
                "simhash": ["A" * 32],
                "lsh_chunks": [["c0_0000"]],
            }
        )

        in_memory_db.insert_dataframe(df)
        assert in_memory_db.count() == 1

        in_memory_db.clear()
        assert in_memory_db.count() == 0

    def test_hnsw_search(self, in_memory_db):
        """HNSW search should return results sorted by distance."""
        rng = np.random.default_rng(42)

        # Create a query vector
        query = rng.random(1024).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Create documents with known similarity to query
        vectors = []
        for i in range(10):
            if i == 0:
                # Most similar - same as query with small noise
                v = query + rng.random(1024).astype(np.float32) * 0.01
            else:
                v = rng.random(1024).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v.tolist())

        df = pd.DataFrame(
            {
                "id": list(range(10)),
                "text": [f"doc{i}" for i in range(10)],
                "vector": vectors,
                "simhash": [f"{i:032X}" for i in range(10)],
                "lsh_chunks": [["c0_0000"] for _ in range(10)],
            }
        )

        in_memory_db.insert_dataframe(df)
        in_memory_db.create_hnsw_index()

        results = in_memory_db.search_hnsw(query, top_k=5)
        assert len(results) == 5

        # First result should be the most similar (doc 0)
        assert results.iloc[0]["id"] == 0

        # Results should be sorted by distance (ascending)
        distances = results["distance"].tolist()
        assert distances == sorted(distances)

    def test_context_manager(self):
        """Should work as context manager."""
        with VectorDatabase(in_memory=True) as db:
            db.initialize()
            assert db.count() == 0
        # Connection should be closed after exiting context
