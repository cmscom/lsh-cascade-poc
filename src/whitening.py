"""Pre-whitening transformation for embedding vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class EmbeddingWhitener:
    """Pre-whitening transformer for embedding vectors.

    Uses PCA-based whitening to transform anisotropic embeddings
    into a more isotropic distribution, improving LSH performance.

    The whitening transformation:
    1. Centers embeddings by subtracting the mean vector
    2. Applies whitening matrix (decorrelates and normalizes variance)
    3. Re-normalizes to unit length

    Attributes:
        mean_vector: Mean vector from training data (shape: dim,)
        whitening_matrix: Whitening transformation matrix (shape: dim x dim)
        dim: Embedding dimension
    """

    def __init__(self) -> None:
        """Initialize empty whitener."""
        self.mean_vector: NDArray[np.floating] | None = None
        self.whitening_matrix: NDArray[np.floating] | None = None
        self.dim: int | None = None

    def fit(self, embeddings: NDArray[np.floating]) -> None:
        """Compute whitening parameters from training embeddings.

        Args:
            embeddings: Training embeddings of shape (n_samples, dim).
                        Should be L2-normalized.
        """
        n_samples, dim = embeddings.shape
        self.dim = dim

        # 1. Compute mean vector
        self.mean_vector = np.mean(embeddings, axis=0).astype(np.float32)

        # 2. Center the embeddings
        centered = embeddings - self.mean_vector

        # 3. Compute covariance matrix
        # Use (n-1) for unbiased estimate
        cov = np.cov(centered.T)

        # 4. SVD decomposition of covariance matrix
        # cov = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(cov)

        # 5. Compute whitening matrix
        # Whitening: W = U @ diag(1/sqrt(S)) @ Vt
        # This makes cov(W @ centered) ≈ I
        epsilon = 1e-8  # Numerical stability
        self.whitening_matrix = (
            U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ Vt
        ).astype(np.float32)

    def transform(self, embeddings: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply whitening transformation to embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, dim) or (dim,).

        Returns:
            Whitened and L2-normalized embeddings of same shape.

        Raises:
            ValueError: If whitener has not been fitted.
        """
        if self.mean_vector is None or self.whitening_matrix is None:
            raise ValueError("Whitener has not been fitted. Call fit() first.")

        # Handle single vector
        single_vector = embeddings.ndim == 1
        if single_vector:
            embeddings = embeddings.reshape(1, -1)

        # 1. Center
        centered = embeddings - self.mean_vector

        # 2. Apply whitening transformation
        whitened = centered @ self.whitening_matrix.T

        # 3. L2 normalize
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = whitened / norms

        if single_vector:
            return normalized[0].astype(np.float32)
        return normalized.astype(np.float32)

    def fit_transform(
        self, embeddings: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Fit whitener and transform embeddings in one step.

        Args:
            embeddings: Training embeddings of shape (n_samples, dim).

        Returns:
            Whitened embeddings of same shape.
        """
        self.fit(embeddings)
        return self.transform(embeddings)

    def save(self, path: str | Path) -> None:
        """Save whitening parameters to file.

        Args:
            path: Path to save .npz file.
        """
        if self.mean_vector is None or self.whitening_matrix is None:
            raise ValueError("Whitener has not been fitted. Call fit() first.")

        np.savez(
            path,
            mean=self.mean_vector,
            matrix=self.whitening_matrix,
            dim=np.array([self.dim]),
        )

    def load(self, path: str | Path) -> None:
        """Load whitening parameters from file.

        Args:
            path: Path to .npz file.
        """
        data = np.load(path)
        self.mean_vector = data["mean"].astype(np.float32)
        self.whitening_matrix = data["matrix"].astype(np.float32)
        self.dim = int(data["dim"][0])

    @property
    def is_fitted(self) -> bool:
        """Check if whitener has been fitted."""
        return self.mean_vector is not None and self.whitening_matrix is not None


def compute_isotropy_score(embeddings: NDArray[np.floating]) -> dict:
    """Compute isotropy metrics for embeddings.

    Args:
        embeddings: Embeddings of shape (n_samples, dim).

    Returns:
        Dictionary with isotropy metrics.
    """
    # Compute covariance matrix
    centered = embeddings - np.mean(embeddings, axis=0)
    cov = np.cov(centered.T)

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Metrics
    total_var = np.sum(eigenvalues)
    top10_var = np.sum(eigenvalues[:10]) / total_var
    condition_number = eigenvalues[0] / (eigenvalues[-1] + 1e-10)

    return {
        "total_variance": float(total_var),
        "top10_variance_ratio": float(top10_var),
        "condition_number": float(condition_number),
        "max_eigenvalue": float(eigenvalues[0]),
        "min_eigenvalue": float(eigenvalues[-1]),
    }
