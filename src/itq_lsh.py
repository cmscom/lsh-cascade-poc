"""
ITQ (Iterative Quantization) LSH 実装

E5の異方性を解消し、ハミング距離とコサイン類似度の相関を最大化する。

参考: "Iterative Quantization: A Procrustean Approach to Learning Binary Codes"
      (Gong et al., CVPR 2011)
"""

import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple
import pickle


class ITQLSH:
    """
    ITQ (Iterative Quantization) に基づくLSH実装

    特徴:
    - Centering: 平均ベクトルを引いて分布を補正
    - PCA: 次元圧縮と主成分への投影
    - ITQ: 回転行列の最適化でハミング距離を改善
    """

    def __init__(
        self,
        n_bits: int = 128,
        n_iterations: int = 50,
        seed: int = 42
    ):
        """
        Args:
            n_bits: 出力ハッシュのビット数
            n_iterations: ITQの反復回数
            seed: 乱数シード
        """
        self.n_bits = n_bits
        self.n_iterations = n_iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 学習で得られるパラメータ
        self.mean_vector: Optional[np.ndarray] = None  # 平均ベクトル
        self.pca_matrix: Optional[np.ndarray] = None   # PCA変換行列 (dim, n_bits)
        self.rotation_matrix: Optional[np.ndarray] = None  # ITQ回転行列 (n_bits, n_bits)

        self._is_fitted = False

    def fit(self, X: np.ndarray) -> 'ITQLSH':
        """
        学習データからITQパラメータを学習

        Args:
            X: 学習データ (n_samples, dim)

        Returns:
            self
        """
        n_samples, dim = X.shape

        if n_samples < self.n_bits:
            raise ValueError(f"サンプル数({n_samples})がビット数({self.n_bits})より少ない")

        print(f"ITQ学習開始: samples={n_samples}, dim={dim}, bits={self.n_bits}")

        # Step 1: Centering (平均除去)
        self.mean_vector = X.mean(axis=0)
        X_centered = X - self.mean_vector
        print(f"  Centering完了: mean_norm={norm(self.mean_vector):.4f}")

        # Step 2: PCA
        # 共分散行列を計算
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # 固有値分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 大きい順にソート
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 上位n_bits個の主成分を選択
        self.pca_matrix = eigenvectors[:, :self.n_bits].astype(np.float32)

        # 説明される分散の割合
        explained_var = eigenvalues[:self.n_bits].sum() / eigenvalues.sum()
        print(f"  PCA完了: explained_variance={explained_var:.2%}")

        # Step 3: PCA空間への投影
        V = X_centered @ self.pca_matrix  # (n_samples, n_bits)

        # Step 4: ITQ (Iterative Quantization)
        # 初期回転行列（ランダム直交行列）
        R = self._random_orthogonal_matrix(self.n_bits)

        for iteration in range(self.n_iterations):
            # 回転後のデータ
            Z = V @ R

            # バイナリコード (符号で量子化)
            B = np.sign(Z)
            B[B == 0] = 1  # 0を1に置換

            # 回転行列の更新 (Procrustes問題)
            # min ||B - VR||^2 を解く
            # SVD(B^T V) = U S W^T
            # R = W U^T
            U, S, Vt = np.linalg.svd(B.T @ V)
            R = Vt.T @ U.T

            if (iteration + 1) % 10 == 0:
                # 量子化誤差を計算
                error = np.mean((B - Z) ** 2)
                print(f"  ITQ iteration {iteration + 1}: quantization_error={error:.4f}")

        self.rotation_matrix = R.astype(np.float32)
        self._is_fitted = True

        print(f"ITQ学習完了")
        return self

    def _random_orthogonal_matrix(self, n: int) -> np.ndarray:
        """ランダムな直交行列を生成"""
        H = self.rng.standard_normal((n, n))
        Q, R = np.linalg.qr(H)
        return Q

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        ベクトルをバイナリハッシュに変換

        Args:
            X: 入力ベクトル (n_samples, dim) or (dim,)

        Returns:
            バイナリハッシュ (n_samples, n_bits) or (n_bits,)
        """
        if not self._is_fitted:
            raise RuntimeError("fit()を先に呼び出してください")

        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        # Centering
        X_centered = X - self.mean_vector

        # PCA投影
        V = X_centered @ self.pca_matrix

        # ITQ回転
        Z = V @ self.rotation_matrix

        # 符号で量子化
        B = (Z > 0).astype(np.uint8)

        if single_input:
            return B[0]
        return B

    def transform_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ベクトルをバイナリハッシュに変換し、各ビットの確信度も返す

        確信度 = |Z| (ITQ回転後の射影値の絶対値)
        値が小さいほど決定境界に近く、ビットの確信度が低い

        Args:
            X: 入力ベクトル (n_samples, dim) or (dim,)

        Returns:
            binary_hash: バイナリハッシュ (n_samples, n_bits) or (n_bits,)
            projections: ITQ回転後の実数値射影 (n_samples, n_bits) or (n_bits,)
        """
        if not self._is_fitted:
            raise RuntimeError("fit()を先に呼び出してください")

        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        X_centered = X - self.mean_vector
        V = X_centered @ self.pca_matrix
        Z = V @ self.rotation_matrix
        B = (Z > 0).astype(np.uint8)

        if single_input:
            return B[0], Z[0].astype(np.float32)
        return B, Z.astype(np.float32)

    def hash_to_int(self, binary_hash: np.ndarray) -> int:
        """バイナリハッシュを整数に変換（128ビットまで対応）"""
        if binary_hash.ndim == 1:
            # 単一ハッシュ
            result = 0
            for bit in binary_hash:
                result = (result << 1) | int(bit)
            return result
        else:
            return [self.hash_to_int(h) for h in binary_hash]

    def save(self, path: str):
        """パラメータを保存"""
        if not self._is_fitted:
            raise RuntimeError("fit()を先に呼び出してください")

        params = {
            'n_bits': self.n_bits,
            'n_iterations': self.n_iterations,
            'seed': self.seed,
            'mean_vector': self.mean_vector,
            'pca_matrix': self.pca_matrix,
            'rotation_matrix': self.rotation_matrix,
        }

        with open(path, 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, path: str) -> 'ITQLSH':
        """パラメータをロード"""
        with open(path, 'rb') as f:
            params = pickle.load(f)

        instance = cls(
            n_bits=params['n_bits'],
            n_iterations=params['n_iterations'],
            seed=params['seed']
        )
        instance.mean_vector = params['mean_vector']
        instance.pca_matrix = params['pca_matrix']
        instance.rotation_matrix = params['rotation_matrix']
        instance._is_fitted = True

        return instance


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """2つのバイナリハッシュ間のハミング距離を計算"""
    return np.sum(h1 != h2)


def hamming_distance_batch(query_hash: np.ndarray, doc_hashes: np.ndarray) -> np.ndarray:
    """1つのクエリと複数のドキュメント間のハミング距離を一括計算"""
    return np.sum(query_hash != doc_hashes, axis=1)
