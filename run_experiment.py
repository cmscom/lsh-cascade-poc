#!/usr/bin/env python3
"""Experiment runner for LSH Cascade Search vs HNSW comparison."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from src.db import VectorDatabase
from src.loader import load_mixed_wikipedia, WikipediaLoader
from src.lsh import SimHashGenerator
from src.pipeline import HNSWSearcher, LSHCascadeSearcher, SearchResult

# Experiment parameters
NUM_QUERIES = 10
TOP_K = 10
CHUNK_CONFIGS = [4, 8, 16]


def run_experiment(
    ja_samples: int = 500,
    en_samples: int = 500,
    num_queries: int = NUM_QUERIES,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Run the complete experiment pipeline."""
    print("=" * 60)
    print("LSH Cascade Search Experiment")
    print("=" * 60)
    print(f"Settings: {ja_samples} ja + {en_samples} en samples, {num_queries} queries")
    print()

    # Initialize shared components
    simhash_generator = SimHashGenerator(dim=1024, hash_bits=128, seed=seed)

    # Prepare data
    print("[1/4] Loading and processing Wikipedia data...")
    df = load_mixed_wikipedia(
        ja_samples=ja_samples,
        en_samples=en_samples,
        num_chunks=max(CHUNK_CONFIGS),  # Use max chunks, we'll recalculate for others
        seed=seed,
        device=device,
    )
    print(f"      Loaded {len(df)} documents")

    # Initialize database
    print("[2/4] Inserting data into DuckDB...")
    db_path = Path("data/experiment.duckdb")
    with VectorDatabase(db_path=db_path) as db:
        db.initialize()
        db.clear()  # Clear existing data
        db.insert_dataframe(df)
        print(f"      Inserted {db.count()} documents")

        # Create HNSW index
        print("[3/4] Creating HNSW index...")
        db.create_hnsw_index()
        print("      HNSW index created")

        # Select random queries
        print("[4/4] Running experiments...")
        queries = select_random_queries(df, num_queries, seed)

        # Run experiments
        results = {
            "hnsw": [],
        }
        for num_chunks in CHUNK_CONFIGS:
            results[f"lsh_{num_chunks}"] = []

        # HNSW baseline
        hnsw_searcher = HNSWSearcher(db)
        hnsw_ground_truth = {}

        for query_id, query_vector in queries:
            search_results, latency = hnsw_searcher.search(query_vector, TOP_K)
            hnsw_ground_truth[query_id] = [r.id for r in search_results]
            results["hnsw"].append(
                {
                    "query_id": query_id,
                    "latency_ms": latency,
                    "result_ids": [r.id for r in search_results],
                }
            )

        # LSH cascade for each chunk configuration
        for num_chunks in CHUNK_CONFIGS:
            searcher = LSHCascadeSearcher(
                db=db,
                simhash_generator=simhash_generator,
                num_chunks=num_chunks,
                step2_top_n=100,
            )

            for query_id, query_vector in queries:
                search_results, metrics = searcher.search(query_vector, TOP_K)
                result_ids = [r.id for r in search_results]

                # Calculate recall against HNSW
                recall = evaluate_recall(hnsw_ground_truth[query_id], result_ids)

                results[f"lsh_{num_chunks}"].append(
                    {
                        "query_id": query_id,
                        "latency_ms": metrics.total_time_ms,
                        "result_ids": result_ids,
                        "recall": recall,
                        "step1_candidates": metrics.step1_candidates,
                        "total_docs": metrics.total_docs,
                    }
                )

    # Print results
    print()
    print_results(results, len(df))


def select_random_queries(
    df,
    num_queries: int,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    """Select random documents as queries.

    Returns:
        List of (id, vector) tuples.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(num_queries, len(df)), replace=False)

    queries = []
    for idx in indices:
        row = df.iloc[idx]
        vector = np.array(row["vector"], dtype=np.float32)
        queries.append((int(row["id"]), vector))

    return queries


def evaluate_recall(
    baseline_ids: list[int],
    result_ids: list[int],
) -> float:
    """Calculate Recall@K.

    Args:
        baseline_ids: Ground truth IDs (HNSW results).
        result_ids: Predicted IDs (LSH results).

    Returns:
        Recall score (0.0 to 1.0).
    """
    if not baseline_ids:
        return 0.0

    baseline_set = set(baseline_ids)
    result_set = set(result_ids)
    intersection = baseline_set & result_set

    return len(intersection) / len(baseline_set)


def print_results(results: dict, total_docs: int) -> None:
    """Print formatted experiment results."""
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total documents: {total_docs}")
    print()

    # HNSW results
    hnsw_latencies = [r["latency_ms"] for r in results["hnsw"]]
    print("--- Baseline: HNSW ---")
    print(f"  Avg Latency: {statistics.mean(hnsw_latencies):.2f} ms")
    print()

    # LSH results
    for num_chunks in CHUNK_CONFIGS:
        key = f"lsh_{num_chunks}"
        if key not in results:
            continue

        data = results[key]
        latencies = [r["latency_ms"] for r in data]
        recalls = [r["recall"] for r in data]
        candidates = [r["step1_candidates"] for r in data]

        avg_candidates = statistics.mean(candidates)
        reduction_rate = (1 - avg_candidates / total_docs) * 100

        bits_per_chunk = 128 // num_chunks

        print(f"--- LSH-{num_chunks} ({bits_per_chunk}-bit chunks) ---")
        print(f"  Recall@{TOP_K}: {statistics.mean(recalls):.2f}")
        print(f"  Avg Latency: {statistics.mean(latencies):.2f} ms")
        print(f"  Avg Step1 Candidates: {avg_candidates:.0f}")
        print(f"  Reduction Rate: {reduction_rate:.1f}%")
        print()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="LSH Cascade Search vs HNSW Experiment"
    )
    parser.add_argument(
        "--ja-samples",
        type=int,
        default=500,
        help="Number of Japanese Wikipedia samples (default: 500)",
    )
    parser.add_argument(
        "--en-samples",
        type=int,
        default=500,
        help="Number of English Wikipedia samples (default: 500)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=10,
        help="Number of random queries (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embedding model (default: cpu)",
    )
    args = parser.parse_args()

    run_experiment(
        ja_samples=args.ja_samples,
        en_samples=args.en_samples,
        num_queries=args.queries,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
