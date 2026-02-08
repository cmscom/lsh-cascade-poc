"""
DF-LSH (Double Filters LSH) 概念実装

DF-LSH論文 (Shan et al., 2025) の2つの中核概念を実装:
1. DAHBF (Data-Aware Hash Bloom Filter): データ分布を考慮したバンドインデックスによる高速候補フィルタリング
2. 幾何学的probe順序: 射影値の確信度に基づくmulti-probe最適化

参考: "DF-LSH: An efficient Double Filters Locality Sensitive Hashing
       for approximate nearest neighbor search" (Engineering Applications of AI, 2025)
"""

import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple, Dict, List, Set
from collections import defaultdict
import pickle


class DFLSH:
    """
    Data-Aware LSH with Band Indexing and Geometric Probe Ordering

    ITQとは独立に、PCA射影に基づくバンドインデックスを構築。
    幾何学的probe順序により、確信度の低いビットを優先的にフリップして候補を回復。
    """

    def __init__(
        self,
        n_projections: int = 128,
        band_width: int = 16,
        seed: int = 42
    ):
        """
        Args:
            n_projections: 射影数（バイナリコードのビット数）
            band_width: 1バンドあたりのビット数
            seed: 乱数シード
        """
        self.n_projections = n_projections
        self.band_width = band_width
        self.n_bands = n_projections // band_width
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.mean_vector: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None  # (dim, n_projections)
        self.band_index: Optional[Dict] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, method: str = 'pca') -> 'DFLSH':
        """
        データからdata-aware射影を学習

        Args:
            X: 学習データ (n_samples, dim)
            method: 'pca' (data-aware) or 'random'

        Returns:
            self
        """
        n_samples, dim = X.shape

        self.mean_vector = X.mean(axis=0)
        X_centered = X - self.mean_vector

        if method == 'pca':
            cov = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            self.projection_matrix = eigenvectors[:, idx[:self.n_projections]].astype(np.float32)

            explained_var = eigenvalues[idx[:self.n_projections]].sum() / eigenvalues.sum()
            print(f"DFLSH fit (PCA): dim={dim}, projections={self.n_projections}, "
                  f"explained_variance={explained_var:.2%}")
        elif method == 'random':
            H = self.rng.standard_normal((dim, self.n_projections))
            self.projection_matrix = (H / norm(H, axis=0, keepdims=True)).astype(np.float32)
            print(f"DFLSH fit (Random): dim={dim}, projections={self.n_projections}")
        else:
            raise ValueError(f"Unknown method: {method}")

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        ベクトルをバイナリコードに変換

        Args:
            X: (n_samples, dim) or (dim,)

        Returns:
            binary_codes: (n_samples, n_projections) or (n_projections,) uint8
        """
        if not self._is_fitted:
            raise RuntimeError("fit()を先に呼び出してください")

        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        X_centered = X - self.mean_vector
        Z = X_centered @ self.projection_matrix
        B = (Z > 0).astype(np.uint8)

        if single_input:
            return B[0]
        return B

    def transform_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        バイナリコードと確信度（射影値）を返す

        Args:
            X: (n_samples, dim) or (dim,)

        Returns:
            binary_codes: (n_samples, n_projections) or (n_projections,)
            projections: (n_samples, n_projections) or (n_projections,) float32
        """
        if not self._is_fitted:
            raise RuntimeError("fit()を先に呼び出してください")

        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        X_centered = X - self.mean_vector
        Z = X_centered @ self.projection_matrix
        B = (Z > 0).astype(np.uint8)

        if single_input:
            return B[0], Z[0].astype(np.float32)
        return B, Z.astype(np.float32)

    def build_index(self, binary_codes: np.ndarray):
        """
        バンドベース転置インデックスを構築

        Args:
            binary_codes: (n_docs, n_projections) uint8
        """
        self.band_index = {}
        for band_idx in range(self.n_bands):
            start = band_idx * self.band_width
            end = start + self.band_width
            band_bits = binary_codes[:, start:end]
            band_keys = _bits_to_int(band_bits)

            table = defaultdict(list)
            for doc_idx in range(len(binary_codes)):
                table[band_keys[doc_idx]].append(doc_idx)

            self.band_index[band_idx] = dict(table)

        print(f"Band index built: {self.n_bands} bands x {self.band_width} bits, "
              f"{len(binary_codes)} docs")

    def query(
        self,
        query_code: np.ndarray,
        min_band_matches: int = 1
    ) -> np.ndarray:
        """
        バンドインデックスで候補を検索

        Args:
            query_code: (n_projections,) uint8
            min_band_matches: 最低一致バンド数（1=OR, 2以上=AND的）

        Returns:
            candidate_indices: 候補ドキュメントのインデックス配列
        """
        if self.band_index is None:
            raise RuntimeError("build_index()を先に呼び出してください")

        if min_band_matches == 1:
            candidates = set()
            for band_idx in range(self.n_bands):
                start = band_idx * self.band_width
                end = start + self.band_width
                key = _bits_to_int_single(query_code[start:end])
                if key in self.band_index[band_idx]:
                    candidates.update(self.band_index[band_idx][key])
            return np.array(sorted(candidates), dtype=np.int64)
        else:
            match_counts = defaultdict(int)
            for band_idx in range(self.n_bands):
                start = band_idx * self.band_width
                end = start + self.band_width
                key = _bits_to_int_single(query_code[start:end])
                if key in self.band_index[band_idx]:
                    for doc_idx in self.band_index[band_idx][key]:
                        match_counts[doc_idx] += 1
            candidates = [idx for idx, count in match_counts.items()
                          if count >= min_band_matches]
            return np.array(sorted(candidates), dtype=np.int64)

    def query_with_multiprobe(
        self,
        query_code: np.ndarray,
        projections: np.ndarray,
        max_probes: int = 8,
        min_band_matches: int = 1
    ) -> np.ndarray:
        """
        確信度ベースのmulti-probeで候補を検索

        Args:
            query_code: (n_projections,) uint8
            projections: (n_projections,) float32 射影値（確信度計算用）
            max_probes: 追加probeの最大数
            min_band_matches: 最低一致バンド数

        Returns:
            candidate_indices: 候補ドキュメントのインデックス配列
        """
        if self.band_index is None:
            raise RuntimeError("build_index()を先に呼び出してください")

        # Phase 1: 通常の検索
        candidates = set()
        for band_idx in range(self.n_bands):
            start = band_idx * self.band_width
            end = start + self.band_width
            key = _bits_to_int_single(query_code[start:end])
            if key in self.band_index[band_idx]:
                candidates.update(self.band_index[band_idx][key])

        # Phase 2: 幾何学的probe順序による追加検索
        if max_probes > 0:
            # 各バンドの平均確信度を計算
            band_confidences = []
            for band_idx in range(self.n_bands):
                start = band_idx * self.band_width
                end = start + self.band_width
                avg_conf = np.mean(np.abs(projections[start:end]))
                band_confidences.append((band_idx, avg_conf))

            # 確信度の低い順にソート（最も不確実なバンドから）
            band_confidences.sort(key=lambda x: x[1])

            probes_used = 0
            for band_idx, _ in band_confidences:
                if probes_used >= max_probes:
                    break

                start = band_idx * self.band_width
                end = start + self.band_width
                band_proj = np.abs(projections[start:end])
                band_bits = query_code[start:end].copy()

                # 最も確信度の低いビットをフリップ
                least_conf_bit = np.argmin(band_proj)
                band_bits[least_conf_bit] ^= 1
                flipped_key = _bits_to_int_single(band_bits)

                if flipped_key in self.band_index[band_idx]:
                    candidates.update(self.band_index[band_idx][flipped_key])
                probes_used += 1

        return np.array(sorted(candidates), dtype=np.int64)

    def save(self, path: str):
        """パラメータを保存"""
        params = {
            'n_projections': self.n_projections,
            'band_width': self.band_width,
            'seed': self.seed,
            'mean_vector': self.mean_vector,
            'projection_matrix': self.projection_matrix,
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, path: str) -> 'DFLSH':
        """パラメータをロード"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        instance = cls(
            n_projections=params['n_projections'],
            band_width=params['band_width'],
            seed=params['seed']
        )
        instance.mean_vector = params['mean_vector']
        instance.projection_matrix = params['projection_matrix']
        instance._is_fitted = True
        return instance


# --- ユーティリティ関数 ---

def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    """ビット配列を整数配列に変換 (n_samples, width) -> (n_samples,)"""
    width = bits.shape[1]
    powers = 2 ** np.arange(width - 1, -1, -1, dtype=np.int64)
    return bits.astype(np.int64) @ powers


def _bits_to_int_single(bits: np.ndarray) -> int:
    """1つのビット配列を整数に変換 (width,) -> int"""
    result = 0
    for b in bits:
        result = (result << 1) | int(b)
    return result


def build_band_index(
    hashes: np.ndarray,
    band_width: int
) -> Dict[int, Dict[int, List[int]]]:
    """
    任意のバイナリハッシュからバンドインデックスを構築

    Args:
        hashes: (n_docs, n_bits) uint8
        band_width: バンド幅（ビット数）

    Returns:
        band_index: {band_idx: {band_key: [doc_ids]}}
    """
    n_bits = hashes.shape[1]
    n_bands = n_bits // band_width
    band_index = {}

    for band_idx in range(n_bands):
        start = band_idx * band_width
        end = start + band_width
        band_bits = hashes[:, start:end]
        band_keys = _bits_to_int(band_bits)

        table = defaultdict(list)
        for doc_idx in range(len(hashes)):
            table[band_keys[doc_idx]].append(doc_idx)

        band_index[band_idx] = dict(table)

    return band_index


def band_filter(
    query_hash: np.ndarray,
    band_index: Dict[int, Dict[int, List[int]]],
    band_width: int,
    min_matches: int = 1
) -> np.ndarray:
    """
    バンドインデックスで候補をフィルタリング

    Args:
        query_hash: (n_bits,) uint8
        band_index: build_band_index の出力
        band_width: バンド幅
        min_matches: 最低一致バンド数

    Returns:
        candidate_indices: 候補インデックス配列
    """
    n_bits = len(query_hash)
    n_bands = n_bits // band_width

    if min_matches == 1:
        candidates = set()
        for band_idx in range(n_bands):
            start = band_idx * band_width
            end = start + band_width
            key = _bits_to_int_single(query_hash[start:end])
            if key in band_index[band_idx]:
                candidates.update(band_index[band_idx][key])
        return np.array(sorted(candidates), dtype=np.int64)
    else:
        match_counts = defaultdict(int)
        for band_idx in range(n_bands):
            start = band_idx * band_width
            end = start + band_width
            key = _bits_to_int_single(query_hash[start:end])
            if key in band_index[band_idx]:
                for doc_idx in band_index[band_idx][key]:
                    match_counts[doc_idx] += 1
        candidates = [idx for idx, count in match_counts.items()
                      if count >= min_matches]
        return np.array(sorted(candidates), dtype=np.int64)


def confidence_multiprobe(
    query_hash: np.ndarray,
    projections: np.ndarray,
    band_index: Dict[int, Dict[int, List[int]]],
    band_width: int,
    max_probes: int = 8,
    order: str = 'confidence'
) -> np.ndarray:
    """
    確信度ベース（またはランダム）のmulti-probeで候補を検索

    Args:
        query_hash: (n_bits,) uint8
        projections: (n_bits,) float32 射影値
        band_index: build_band_index の出力
        band_width: バンド幅
        max_probes: 追加probe数
        order: 'confidence' (確信度順) or 'random' (ランダム順)

    Returns:
        candidate_indices: 候補インデックス配列
    """
    n_bits = len(query_hash)
    n_bands = n_bits // band_width

    # Phase 1: 通常の検索（全バンドの完全一致）
    candidates = set()
    for band_idx in range(n_bands):
        start = band_idx * band_width
        end = start + band_width
        key = _bits_to_int_single(query_hash[start:end])
        if key in band_index[band_idx]:
            candidates.update(band_index[band_idx][key])

    # Phase 2: multi-probe
    if max_probes > 0:
        if order == 'confidence':
            # 確信度の低い順にバンドをソート
            band_order = []
            for band_idx in range(n_bands):
                start = band_idx * band_width
                end = start + band_width
                avg_conf = np.mean(np.abs(projections[start:end]))
                band_order.append((band_idx, avg_conf))
            band_order.sort(key=lambda x: x[1])
            band_order = [b[0] for b in band_order]
        elif order == 'random':
            rng = np.random.default_rng(42)
            band_order = list(rng.permutation(n_bands))
        else:
            raise ValueError(f"Unknown order: {order}")

        probes_used = 0
        for band_idx in band_order:
            if probes_used >= max_probes:
                break

            start = band_idx * band_width
            end = start + band_width
            band_proj = np.abs(projections[start:end])
            band_bits = query_hash[start:end].copy()

            # 最も確信度の低いビットをフリップ
            least_conf_bit = np.argmin(band_proj)
            band_bits[least_conf_bit] ^= 1
            flipped_key = _bits_to_int_single(band_bits)

            if flipped_key in band_index[band_idx]:
                candidates.update(band_index[band_idx][flipped_key])
            probes_used += 1

    return np.array(sorted(candidates), dtype=np.int64)


def combined_band_pivot_filter(
    query_hash: np.ndarray,
    band_index: Dict[int, Dict[int, List[int]]],
    band_width: int,
    pivots: np.ndarray,
    pivot_distances: np.ndarray,
    pivot_threshold: int,
    min_band_matches: int = 1,
    projections: Optional[np.ndarray] = None,
    max_probes: int = 0
) -> np.ndarray:
    """
    バンドフィルタ + Pivotフィルタの2段フィルタリング

    Args:
        query_hash: (n_bits,) uint8
        band_index: バンドインデックス
        band_width: バンド幅
        pivots: (n_pivots, n_bits) uint8 ピボットハッシュ
        pivot_distances: (n_docs, n_pivots) uint8 全ドキュメントのピボット距離
        pivot_threshold: ピボット枝刈り閾値
        min_band_matches: バンド最低一致数
        projections: (n_bits,) float32 射影値（multi-probe用、Noneなら不使用）
        max_probes: 追加probe数

    Returns:
        candidate_indices: 2段フィルタ後の候補インデックス配列
    """
    from .itq_lsh import hamming_distance

    # Stage 1: バンドフィルタ
    if projections is not None and max_probes > 0:
        band_candidates = confidence_multiprobe(
            query_hash, projections, band_index, band_width,
            max_probes=max_probes, order='confidence'
        )
    else:
        band_candidates = band_filter(
            query_hash, band_index, band_width,
            min_matches=min_band_matches
        )

    if len(band_candidates) == 0:
        return band_candidates

    # Stage 2: Pivotフィルタ（バンド候補のみ対象）
    query_pivot_dists = np.array([
        hamming_distance(query_hash, p) for p in pivots
    ])

    mask = np.ones(len(band_candidates), dtype=bool)
    candidate_pivot_dists = pivot_distances[band_candidates]

    for i in range(len(pivots)):
        lower = query_pivot_dists[i] - pivot_threshold
        upper = query_pivot_dists[i] + pivot_threshold
        mask &= (candidate_pivot_dists[:, i] >= lower) & \
                (candidate_pivot_dists[:, i] <= upper)

    return band_candidates[mask]
