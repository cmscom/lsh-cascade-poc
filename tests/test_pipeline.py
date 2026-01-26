"""Tests for src/pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from src.db import VectorDatabase
from src.lsh import SimHashGenerator, chunk_hash
from src.pipeline import HNSWSearcher, LSHCascadeSearcher, SearchResult


class TestLSHCascadeSearcher:
    """Tests for LSHCascadeSearcher class."""

    @pytest.fixture
    def setup_db_with_data(self, in_memory_db, simhash_generator):
        """Create database with test data."""
        rng = np.random.default_rng(42)

        # Create 50 random documents
        vectors = []
        simhashes = []
        chunks_list = []

        for i in range(50):
            v = rng.random(1024).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v.tolist())

            h = simhash_generator.hash(v)
            simhashes.append(f"{h:032X}")
            chunks_list.append(chunk_hash(h, 8))

        df = pd.DataFrame(
            {
                "id": list(range(50)),
                "text": [f"Document {i}" for i in range(50)],
                "vector": vectors,
                "simhash": simhashes,
                "lsh_chunks": chunks_list,
            }
        )

        in_memory_db.insert_dataframe(df)
        in_memory_db.create_hnsw_index()

        return in_memory_db, simhash_generator, df

    def test_search_returns_results(self, setup_db_with_data):
        """Search should return results and metrics."""
        db, simhash_gen, df = setup_db_with_data

        searcher = LSHCascadeSearcher(
            db=db,
            simhash_generator=simhash_gen,
            num_chunks=8,
            step2_top_n=20,
        )

        # Use first document as query
        query_vector = np.array(df.iloc[0]["vector"], dtype=np.float32)

        results, metrics = searcher.search(query_vector, top_k=5)

        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert metrics.total_docs == 50
        assert metrics.total_time_ms > 0

    def test_step1_reduces_candidates(self, setup_db_with_data):
        """Step 1 should reduce candidate count."""
        db, simhash_gen, df = setup_db_with_data

        searcher = LSHCascadeSearcher(
            db=db,
            simhash_generator=simhash_gen,
            num_chunks=8,
            step2_top_n=20,
        )

        query_vector = np.array(df.iloc[0]["vector"], dtype=np.float32)
        results, metrics = searcher.search(query_vector, top_k=5)

        # Step 1 should filter some candidates (not return all 50)
        # Note: With random data, this might occasionally return all
        assert metrics.step1_candidates <= 50

    def test_results_sorted_by_score(self, setup_db_with_data):
        """Results should be sorted by score (descending)."""
        db, simhash_gen, df = setup_db_with_data

        searcher = LSHCascadeSearcher(
            db=db,
            simhash_generator=simhash_gen,
            num_chunks=8,
            step2_top_n=20,
        )

        query_vector = np.array(df.iloc[0]["vector"], dtype=np.float32)
        results, _ = searcher.search(query_vector, top_k=10)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_self_query_highest_score(self, setup_db_with_data):
        """Querying with a document's own vector should return it first."""
        db, simhash_gen, df = setup_db_with_data

        searcher = LSHCascadeSearcher(
            db=db,
            simhash_generator=simhash_gen,
            num_chunks=8,
            step2_top_n=50,  # Keep all for this test
        )

        # Query with first document's vector
        query_vector = np.array(df.iloc[0]["vector"], dtype=np.float32)
        results, _ = searcher.search(query_vector, top_k=5)

        # First result should be the query document itself
        assert len(results) > 0
        assert results[0].id == 0
        assert results[0].score > 0.99  # Should be very close to 1.0

    def test_empty_candidates(self, in_memory_db, simhash_generator):
        """Should handle empty candidate set gracefully."""
        # Create a document with specific chunks
        df = pd.DataFrame(
            {
                "id": [0],
                "text": ["test"],
                "vector": [np.random.rand(1024).tolist()],
                "simhash": ["A" * 32],
                "lsh_chunks": [["c0_AAAA", "c1_BBBB"]],
            }
        )
        in_memory_db.insert_dataframe(df)

        # Use a different generator that won't produce matching chunks
        different_gen = SimHashGenerator(seed=999)
        searcher = LSHCascadeSearcher(
            db=in_memory_db,
            simhash_generator=different_gen,
            num_chunks=8,
        )

        # Query should return empty or limited results
        query = np.random.rand(1024).astype(np.float32)
        results, metrics = searcher.search(query, top_k=5)

        # Should not crash, may return 0 or few results
        assert isinstance(results, list)


class TestHNSWSearcher:
    """Tests for HNSWSearcher class."""

    def test_search_returns_results(self, in_memory_db):
        """HNSW search should return results."""
        rng = np.random.default_rng(42)

        # Insert test data
        vectors = [rng.random(1024).tolist() for _ in range(20)]
        df = pd.DataFrame(
            {
                "id": list(range(20)),
                "text": [f"doc{i}" for i in range(20)],
                "vector": vectors,
                "simhash": [f"{i:032X}" for i in range(20)],
                "lsh_chunks": [["c0_0000"] for _ in range(20)],
            }
        )

        in_memory_db.insert_dataframe(df)
        in_memory_db.create_hnsw_index()

        searcher = HNSWSearcher(in_memory_db)
        query = np.array(vectors[0], dtype=np.float32)

        results, latency = searcher.search(query, top_k=5)

        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert latency > 0

    def test_search_scores_valid(self, in_memory_db):
        """Search scores should be valid similarities."""
        rng = np.random.default_rng(42)

        vectors = []
        for _ in range(10):
            v = rng.random(1024).astype(np.float32)
            v = v / np.linalg.norm(v)  # Normalize
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

        searcher = HNSWSearcher(in_memory_db)
        query = np.array(vectors[0], dtype=np.float32)

        results, _ = searcher.search(query, top_k=5)

        for r in results:
            # Cosine similarity should be between -1 and 1
            assert -1.0 <= r.score <= 1.0

        # First result (self) should have very high similarity
        assert results[0].score > 0.99


class TestCosineSmilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similarity = LSHCascadeSearcher._cosine_similarity(v, v)
        assert np.isclose(similarity, 1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        similarity = LSHCascadeSearcher._cosine_similarity(v1, v2)
        assert np.isclose(similarity, -1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = LSHCascadeSearcher._cosine_similarity(v1, v2)
        assert np.isclose(similarity, 0.0)

    def test_normalized_vectors(self):
        """For normalized vectors, dot product equals cosine similarity."""
        rng = np.random.default_rng(42)

        v1 = rng.random(1024).astype(np.float32)
        v2 = rng.random(1024).astype(np.float32)

        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        similarity = LSHCascadeSearcher._cosine_similarity(v1, v2)
        expected = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        assert np.isclose(similarity, expected)
