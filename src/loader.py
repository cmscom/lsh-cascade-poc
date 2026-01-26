"""Data loader for Wikipedia datasets with embedding and LSH processing."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from datasets import load_dataset
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.lsh import SimHashGenerator, chunk_hash

# Constants
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
MAX_TEXT_LENGTH = 2048  # Character limit for text


class WikipediaLoader:
    """Load and process Wikipedia data with embeddings and LSH hashes."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        simhash_generator: SimHashGenerator | None = None,
        num_chunks: int = 8,
        device: str = "cpu",
    ) -> None:
        """Initialize the loader.

        Args:
            model_name: Sentence Transformer model name.
            simhash_generator: SimHashGenerator instance (creates default if None).
            num_chunks: Number of LSH chunks for indexing.
            device: Inference device ("cpu" or "cuda").
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.simhash_generator = simhash_generator or SimHashGenerator(
            dim=EMBEDDING_DIM
        )
        self.num_chunks = num_chunks

    def load_wikipedia(
        self,
        lang: Literal["ja", "en"] = "ja",
        num_samples: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Load Wikipedia data and convert to vectors with LSH.

        Args:
            lang: Language ("ja" or "en").
            num_samples: Number of samples to load.
            seed: Random seed for shuffling.

        Returns:
            DataFrame with columns: id, text, vector, simhash, lsh_chunks.
        """
        # Load dataset
        config = "20220301.ja" if lang == "ja" else "20220301.en"
        dataset = load_dataset("wikipedia", config, split="train", streaming=True)

        # Shuffle and take samples
        dataset = dataset.shuffle(seed=seed)

        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = self._preprocess_text(item["text"])
            if text:  # Skip empty texts
                texts.append(text)

        # Embed texts
        vectors = self._embed_texts(texts)

        # Generate SimHash and chunks
        hashes = self.simhash_generator.hash_batch(vectors)
        chunks_list = [chunk_hash(h, self.num_chunks) for h in hashes]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "id": range(len(texts)),
                "text": texts,
                "vector": [v.tolist() for v in vectors],
                "simhash": [f"{h:032X}" for h in hashes],  # 128-bit as 32 hex chars
                "lsh_chunks": chunks_list,
            }
        )

        return df

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding.

        - Normalize whitespace
        - Add E5 model prefix
        - Truncate long texts

        Args:
            text: Raw text.

        Returns:
            Preprocessed text with "passage: " prefix.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        # Skip empty or very short texts
        if len(text) < 10:
            return ""

        # Add E5 prefix for passages
        return f"passage: {text}"

    def _embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> NDArray[np.floating]:
        """Embed texts using the sentence transformer model.

        Args:
            texts: List of preprocessed texts.
            batch_size: Batch size for inference.

        Returns:
            Array of shape (n, EMBEDDING_DIM).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.floating]:
        """Embed a single query text.

        Args:
            query: Query text.

        Returns:
            Vector of shape (EMBEDDING_DIM,).
        """
        # E5 requires "query: " prefix for queries
        query_text = f"query: {query}"
        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)


def load_mixed_wikipedia(
    ja_samples: int = 500,
    en_samples: int = 500,
    num_chunks: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """Load mixed Japanese and English Wikipedia data.

    Args:
        ja_samples: Number of Japanese samples.
        en_samples: Number of English samples.
        num_chunks: Number of LSH chunks.
        seed: Random seed.
        device: Inference device.

    Returns:
        Combined DataFrame with reassigned IDs.
    """
    loader = WikipediaLoader(num_chunks=num_chunks, device=device)

    dfs = []

    if ja_samples > 0:
        df_ja = loader.load_wikipedia(lang="ja", num_samples=ja_samples, seed=seed)
        dfs.append(df_ja)

    if en_samples > 0:
        df_en = loader.load_wikipedia(lang="en", num_samples=en_samples, seed=seed + 1)
        dfs.append(df_en)

    if not dfs:
        return pd.DataFrame(columns=["id", "text", "vector", "simhash", "lsh_chunks"])

    # Combine and reassign IDs
    combined = pd.concat(dfs, ignore_index=True)
    combined["id"] = range(len(combined))

    return combined
