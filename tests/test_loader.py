"""Tests for src/loader.py.

Note: These tests use mocking to avoid downloading Wikipedia data
and loading the embedding model during tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.lsh import SimHashGenerator


class TestWikipediaLoader:
    """Tests for WikipediaLoader class."""

    def test_preprocess_text_adds_prefix(self):
        """Preprocessed text should have 'passage: ' prefix."""
        from src.loader import WikipediaLoader

        # Create loader with mocked model
        with patch("src.loader.SentenceTransformer"):
            loader = WikipediaLoader()
            result = loader._preprocess_text("Hello world")
            assert result.startswith("passage: ")
            assert "Hello world" in result

    def test_preprocess_text_normalizes_whitespace(self):
        """Should normalize multiple whitespace to single space."""
        from src.loader import WikipediaLoader

        with patch("src.loader.SentenceTransformer"):
            loader = WikipediaLoader()
            result = loader._preprocess_text("Hello    world\n\ntest")
            assert "passage: Hello world test" == result

    def test_preprocess_text_truncates_long_text(self):
        """Should truncate text longer than MAX_TEXT_LENGTH."""
        from src.loader import WikipediaLoader, MAX_TEXT_LENGTH

        with patch("src.loader.SentenceTransformer"):
            loader = WikipediaLoader()
            long_text = "a" * (MAX_TEXT_LENGTH + 1000)
            result = loader._preprocess_text(long_text)
            # "passage: " is 9 chars
            assert len(result) <= MAX_TEXT_LENGTH + 9

    def test_preprocess_text_skips_short_text(self):
        """Should return empty string for very short text."""
        from src.loader import WikipediaLoader

        with patch("src.loader.SentenceTransformer"):
            loader = WikipediaLoader()
            result = loader._preprocess_text("Hi")
            assert result == ""

    def test_embed_texts_returns_correct_shape(self):
        """Embedding should return correct shape."""
        from src.loader import WikipediaLoader, EMBEDDING_DIM

        rng = np.random.default_rng(42)
        mock_model = MagicMock()
        mock_model.encode.return_value = rng.random((3, EMBEDDING_DIM))

        with patch("src.loader.SentenceTransformer", return_value=mock_model):
            loader = WikipediaLoader()
            embeddings = loader._embed_texts(["text1", "text2", "text3"])

            assert embeddings.shape == (3, EMBEDDING_DIM)
            mock_model.encode.assert_called_once()

    def test_embed_query_adds_query_prefix(self):
        """Query embedding should use 'query: ' prefix."""
        from src.loader import WikipediaLoader, EMBEDDING_DIM

        rng = np.random.default_rng(42)
        mock_model = MagicMock()
        mock_model.encode.return_value = rng.random((1, EMBEDDING_DIM))

        with patch("src.loader.SentenceTransformer", return_value=mock_model):
            loader = WikipediaLoader()
            loader.embed_query("test query")

            # Check that encode was called with query prefix
            call_args = mock_model.encode.call_args
            assert call_args[0][0] == ["query: test query"]


class TestLoadMixedWikipedia:
    """Tests for load_mixed_wikipedia function."""

    def test_returns_dataframe_with_correct_columns(self):
        """Should return DataFrame with required columns."""
        from src.loader import load_mixed_wikipedia, EMBEDDING_DIM

        rng = np.random.default_rng(42)
        mock_model = MagicMock()
        # Return embeddings for each text
        mock_model.encode.return_value = rng.random((5, EMBEDDING_DIM))

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.__iter__ = lambda self: iter(
            [{"text": f"Document {i} with enough text to pass filter"} for i in range(5)]
        )

        with patch("src.loader.SentenceTransformer", return_value=mock_model):
            with patch("src.loader.load_dataset", return_value=mock_dataset):
                df = load_mixed_wikipedia(ja_samples=5, en_samples=0)

                expected_columns = ["id", "text", "vector", "simhash", "lsh_chunks"]
                for col in expected_columns:
                    assert col in df.columns

    def test_simhash_format(self):
        """SimHash should be 32-char hex string."""
        from src.loader import load_mixed_wikipedia, EMBEDDING_DIM

        rng = np.random.default_rng(42)
        mock_model = MagicMock()
        mock_model.encode.return_value = rng.random((3, EMBEDDING_DIM))

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.__iter__ = lambda self: iter(
            [{"text": f"Document {i} with enough text content here"} for i in range(3)]
        )

        with patch("src.loader.SentenceTransformer", return_value=mock_model):
            with patch("src.loader.load_dataset", return_value=mock_dataset):
                df = load_mixed_wikipedia(ja_samples=3, en_samples=0)

                for simhash in df["simhash"]:
                    assert len(simhash) == 32
                    # Should be valid hex
                    int(simhash, 16)

    def test_lsh_chunks_format(self):
        """LSH chunks should be list of prefixed hex strings."""
        from src.loader import load_mixed_wikipedia, EMBEDDING_DIM

        rng = np.random.default_rng(42)
        mock_model = MagicMock()
        mock_model.encode.return_value = rng.random((2, EMBEDDING_DIM))

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.__iter__ = lambda self: iter(
            [{"text": f"Document {i} with sufficient text content"} for i in range(2)]
        )

        with patch("src.loader.SentenceTransformer", return_value=mock_model):
            with patch("src.loader.load_dataset", return_value=mock_dataset):
                df = load_mixed_wikipedia(ja_samples=2, en_samples=0, num_chunks=8)

                for chunks in df["lsh_chunks"]:
                    assert len(chunks) == 8
                    for i, chunk in enumerate(chunks):
                        assert chunk.startswith(f"c{i}_")
