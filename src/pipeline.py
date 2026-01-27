"""Search pipeline with 3-stage LSH cascade filtering."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.db import VectorDatabase
from src.lsh import SimHashGenerator, chunk_hash, hamming_distance
from src.e2lsh import E2LSHHasher, E2LSHIndex


@dataclass
class SearchResult:
    """Single search result."""

    id: int
    text: str
    score: float  # Cosine similarity (0-1)


@dataclass
class SearchMetrics:
    """Performance metrics for a search operation."""

    total_docs: int
    step1_candidates: int
    step2_candidates: int
    final_results: int
    step1_time_ms: float
    step2_time_ms: float
    step3_time_ms: float
    total_time_ms: float


class LSHCascadeSearcher:
    """3-stage LSH cascade search.

    Stage 1: Coarse Filtering (SQL) - Filter by LSH chunk matching
    Stage 2: Binary Reranking (Python) - Rerank by Hamming distance
    Stage 3: Exact Reranking (Numpy) - Final scoring by cosine similarity
    """

    def __init__(
        self,
        db: VectorDatabase,
        simhash_generator: SimHashGenerator,
        num_chunks: int = 8,
        step2_top_n: int = 100,
    ) -> None:
        """Initialize the searcher.

        Args:
            db: VectorDatabase instance.
            simhash_generator: Must be the same instance used during indexing.
            num_chunks: Number of LSH chunks (4, 8, or 16).
            step2_top_n: Number of candidates to keep after Stage 2.
        """
        self.db = db
        self.simhash_generator = simhash_generator
        self.num_chunks = num_chunks
        self.step2_top_n = step2_top_n

    def search(
        self,
        query_vector: NDArray[np.floating],
        top_k: int = 10,
    ) -> tuple[list[SearchResult], SearchMetrics]:
        """Execute 3-stage cascade search.

        Args:
            query_vector: Query vector (1024 dimensions).
            top_k: Number of final results.

        Returns:
            Tuple of (results, metrics).
        """
        total_start = time.perf_counter()
        total_docs = self.db.count()

        # Generate query hash and chunks
        query_hash = self.simhash_generator.hash(query_vector)
        query_chunks = chunk_hash(query_hash, self.num_chunks)

        # Stage 1: Coarse Filtering
        step1_start = time.perf_counter()
        candidates = self._step1_coarse_filter(query_chunks)
        step1_time = (time.perf_counter() - step1_start) * 1000
        step1_count = len(candidates)

        # Stage 2: Binary Reranking
        step2_start = time.perf_counter()
        candidates = self._step2_binary_rerank(candidates, query_hash, self.step2_top_n)
        step2_time = (time.perf_counter() - step2_start) * 1000
        step2_count = len(candidates)

        # Stage 3: Exact Reranking
        step3_start = time.perf_counter()
        results = self._step3_exact_rerank(candidates, query_vector, top_k)
        step3_time = (time.perf_counter() - step3_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        metrics = SearchMetrics(
            total_docs=total_docs,
            step1_candidates=step1_count,
            step2_candidates=step2_count,
            final_results=len(results),
            step1_time_ms=step1_time,
            step2_time_ms=step2_time,
            step3_time_ms=step3_time,
            total_time_ms=total_time,
        )

        return results, metrics

    def _step1_coarse_filter(
        self,
        query_chunks: list[str],
    ) -> pd.DataFrame:
        """Stage 1: SQL-based chunk matching.

        Args:
            query_chunks: Query LSH chunks.

        Returns:
            DataFrame of candidate documents.
        """
        return self.db.search_lsh_chunks(query_chunks)

    def _step2_binary_rerank(
        self,
        candidates: pd.DataFrame,
        query_hash: int,
        top_n: int,
    ) -> pd.DataFrame:
        """Stage 2: Hamming distance reranking.

        Args:
            candidates: Candidate documents from Stage 1.
            query_hash: Query SimHash.
            top_n: Number of candidates to keep.

        Returns:
            Top N candidates sorted by Hamming distance.
        """
        if candidates.empty:
            return candidates

        # Calculate Hamming distance for each candidate
        distances = []
        for _, row in candidates.iterrows():
            # Parse hex string to int
            doc_hash = int(row["simhash"], 16)
            dist = hamming_distance(query_hash, doc_hash)
            distances.append(dist)

        candidates = candidates.copy()
        candidates["hamming_dist"] = distances

        # Sort by Hamming distance and take top N
        candidates = candidates.nsmallest(top_n, "hamming_dist")

        return candidates

    def _step3_exact_rerank(
        self,
        candidates: pd.DataFrame,
        query_vector: NDArray,
        top_k: int,
    ) -> list[SearchResult]:
        """Stage 3: Cosine similarity reranking.

        Args:
            candidates: Candidates from Stage 2.
            query_vector: Query vector.
            top_k: Number of final results.

        Returns:
            Top K results with cosine similarity scores.
        """
        if candidates.empty:
            return []

        results = []
        for _, row in candidates.iterrows():
            doc_vector = np.array(row["vector"], dtype=np.float32)
            similarity = self._cosine_similarity(query_vector, doc_vector)
            results.append(
                SearchResult(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    score=float(similarity),
                )
            )

        # Sort by similarity (descending) and take top K
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(v1: NDArray, v2: NDArray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            v1: First vector.
            v2: Second vector.

        Returns:
            Cosine similarity (0 to 1 for normalized vectors).
        """
        # For normalized vectors, dot product equals cosine similarity
        return float(np.dot(v1, v2))


class HNSWSearcher:
    """Baseline HNSW search wrapper."""

    def __init__(self, db: VectorDatabase) -> None:
        """Initialize the searcher.

        Args:
            db: VectorDatabase instance with HNSW index.
        """
        self.db = db

    def search(
        self,
        query_vector: NDArray[np.floating],
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """Execute HNSW search.

        Args:
            query_vector: Query vector (1024 dimensions).
            top_k: Number of results.

        Returns:
            Tuple of (results, search_time_ms).
        """
        start = time.perf_counter()

        df = self.db.search_hnsw(query_vector, top_k)

        elapsed = (time.perf_counter() - start) * 1000

        results = []
        for _, row in df.iterrows():
            # Convert distance to similarity
            # Cosine distance = 1 - cosine similarity
            similarity = 1.0 - float(row["distance"])
            results.append(
                SearchResult(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    score=similarity,
                )
            )

        return results, elapsed


class E2LSHCascadeSearcher:
    """E2LSH-based cascade search.

    Uses E2LSH (Euclidean LSH) with multiple hash tables for candidate
    retrieval, followed by exact cosine similarity reranking.

    Stage 1: E2LSH hash table lookup (candidates from all tables)
    Stage 2: Exact reranking by cosine similarity
    """

    def __init__(
        self,
        vectors: NDArray[np.floating],
        texts: list[str],
        ids: list[int],
        w: float = 8.0,
        k: int = 4,
        num_tables: int = 8,
        seed: int = 42,
    ) -> None:
        """Initialize the E2LSH searcher.

        Args:
            vectors: All document vectors of shape (n, dim).
            texts: List of document texts.
            ids: List of document IDs.
            w: E2LSH bucket width.
            k: Number of hash functions per table.
            num_tables: Number of hash tables (L).
            seed: Random seed for reproducibility.
        """
        self.vectors = vectors.astype(np.float32)
        self.texts = texts
        self.ids = ids
        self.dim = vectors.shape[1]

        # Initialize E2LSH
        self.hasher = E2LSHHasher(
            dim=self.dim,
            w=w,
            k=k,
            num_tables=num_tables,
            seed=seed,
        )
        self.index = E2LSHIndex(self.hasher)

        # Build index
        self.index.build(self.vectors)

    def search(
        self,
        query_vector: NDArray[np.floating],
        top_k: int = 10,
        step1_max_candidates: int = 100,
    ) -> tuple[list[SearchResult], SearchMetrics]:
        """Execute E2LSH cascade search.

        Args:
            query_vector: Query vector.
            top_k: Number of final results.
            step1_max_candidates: Max candidates from E2LSH lookup.

        Returns:
            Tuple of (results, metrics).
        """
        total_start = time.perf_counter()
        total_docs = len(self.vectors)

        # Stage 1: E2LSH hash table lookup
        step1_start = time.perf_counter()
        candidate_indices = self._step1_e2lsh_lookup(query_vector, step1_max_candidates)
        step1_time = (time.perf_counter() - step1_start) * 1000
        step1_count = len(candidate_indices)

        # Stage 2: Exact reranking
        step2_start = time.perf_counter()
        results = self._step2_exact_rerank(candidate_indices, query_vector, top_k)
        step2_time = (time.perf_counter() - step2_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        metrics = SearchMetrics(
            total_docs=total_docs,
            step1_candidates=step1_count,
            step2_candidates=step1_count,  # No intermediate step
            final_results=len(results),
            step1_time_ms=step1_time,
            step2_time_ms=0.0,
            step3_time_ms=step2_time,
            total_time_ms=total_time,
        )

        return results, metrics

    def _step1_e2lsh_lookup(
        self,
        query_vector: NDArray[np.floating],
        max_candidates: int,
    ) -> list[int]:
        """Stage 1: E2LSH hash table lookup.

        Args:
            query_vector: Query vector.
            max_candidates: Maximum candidates to return.

        Returns:
            List of candidate indices.
        """
        return self.index.query(query_vector, top_k=max_candidates)

    def _step2_exact_rerank(
        self,
        candidate_indices: list[int],
        query_vector: NDArray[np.floating],
        top_k: int,
    ) -> list[SearchResult]:
        """Stage 2: Exact cosine similarity reranking.

        Args:
            candidate_indices: Candidate vector indices.
            query_vector: Query vector.
            top_k: Number of final results.

        Returns:
            Top K results with cosine similarity scores.
        """
        if not candidate_indices:
            return []

        # Batch compute similarities
        candidate_vectors = self.vectors[candidate_indices]
        similarities = candidate_vectors @ query_vector

        # Create results with scores
        results = []
        for idx, sim in zip(candidate_indices, similarities):
            results.append(
                SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(sim),
                )
            )

        # Sort by similarity (descending) and take top K
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
