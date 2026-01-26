"""Database operations using DuckDB with vector search support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Constants
EMBEDDING_DIM = 1024
DEFAULT_DB_PATH = Path("data/vectors.duckdb")


class VectorDatabase:
    """DuckDB-based vector storage with HNSW and LSH support."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        in_memory: bool = False,
    ) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to database file (uses default if None).
            in_memory: If True, use in-memory database.
        """
        if in_memory:
            self.db_path = None
            self.conn = duckdb.connect(":memory:")
        else:
            self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))

        self._initialized = False

    def initialize(self) -> None:
        """Load vss extension and create schema."""
        if self._initialized:
            return

        # Install and load vss extension
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

        # Enable HNSW persistence
        self.conn.execute("SET hnsw_enable_experimental_persistence = true")

        self._create_schema()
        self._initialized = True

    def _create_schema(self) -> None:
        """Create documents table."""
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                text VARCHAR,
                vector FLOAT[{EMBEDDING_DIM}],
                simhash VARCHAR,
                lsh_chunks VARCHAR[]
            )
        """)

    def create_hnsw_index(self) -> None:
        """Create HNSW index on vector column."""
        # Check if index already exists
        result = self.conn.execute("""
            SELECT COUNT(*) FROM duckdb_indexes()
            WHERE index_name = 'hnsw_idx'
        """).fetchone()

        if result[0] == 0:
            self.conn.execute("""
                CREATE INDEX hnsw_idx
                ON documents USING HNSW (vector)
                WITH (metric = 'cosine')
            """)

    def insert_dataframe(self, df: pd.DataFrame) -> int:
        """Insert documents from DataFrame.

        Args:
            df: DataFrame with columns: id, text, vector, simhash, lsh_chunks.

        Returns:
            Number of inserted rows.
        """
        # Convert vectors to proper format
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "id": int(row["id"]),
                    "text": str(row["text"]),
                    "vector": list(row["vector"]),
                    "simhash": str(row["simhash"]),
                    "lsh_chunks": list(row["lsh_chunks"]),
                }
            )

        # Insert using executemany
        self.conn.executemany(
            """
            INSERT INTO documents (id, text, vector, simhash, lsh_chunks)
            VALUES ($id, $text, $vector, $simhash, $lsh_chunks)
            """,
            records,
        )

        return len(records)

    def search_hnsw(
        self,
        query_vector: NDArray | list[float],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Search using HNSW index.

        Args:
            query_vector: Query vector (EMBEDDING_DIM dimensions).
            top_k: Number of results to return.

        Returns:
            DataFrame with columns: id, text, vector, simhash, distance.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        result = self.conn.execute(
            """
            SELECT
                id, text, vector, simhash,
                array_cosine_distance(vector, $1::FLOAT[1024]) as distance
            FROM documents
            ORDER BY distance
            LIMIT $2
            """,
            [query_vector, top_k],
        ).fetchdf()

        return result

    def search_lsh_chunks(
        self,
        query_chunks: list[str],
    ) -> pd.DataFrame:
        """Search by LSH chunk matching (coarse filter).

        Args:
            query_chunks: List of LSH chunk strings to match.

        Returns:
            DataFrame with all documents that have at least one matching chunk.
        """
        result = self.conn.execute(
            """
            SELECT id, text, vector, simhash, lsh_chunks
            FROM documents
            WHERE list_has_any(lsh_chunks, $1)
            """,
            [query_chunks],
        ).fetchdf()

        return result

    def get_by_ids(self, ids: list[int]) -> pd.DataFrame:
        """Get documents by ID list.

        Args:
            ids: List of document IDs.

        Returns:
            DataFrame with matching documents.
        """
        if not ids:
            return pd.DataFrame(columns=["id", "text", "vector", "simhash", "lsh_chunks"])

        result = self.conn.execute(
            """
            SELECT id, text, vector, simhash, lsh_chunks
            FROM documents
            WHERE id IN (SELECT UNNEST($1::INTEGER[]))
            """,
            [ids],
        ).fetchdf()

        return result

    def get_all(self) -> pd.DataFrame:
        """Get all documents.

        Returns:
            DataFrame with all documents.
        """
        return self.conn.execute(
            "SELECT id, text, vector, simhash, lsh_chunks FROM documents"
        ).fetchdf()

    def count(self) -> int:
        """Get total document count.

        Returns:
            Number of documents in the database.
        """
        result = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0] if result else 0

    def clear(self) -> None:
        """Delete all documents."""
        self.conn.execute("DELETE FROM documents")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> VectorDatabase:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
