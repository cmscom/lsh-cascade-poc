"""
カスケード検索システム

ITQ LSHによる高速な候補選択とコサイン類似度によるリランキングを組み合わせた
2段階検索システム。

使用例:
    from src.cascade_search import CascadeSearcher
    from sentence_transformers import SentenceTransformer

    # 初期化
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    searcher = CascadeSearcher.from_itq_model('data/itq_model.pkl')

    # インデックス構築
    embeddings = model.encode([f'passage: {t}' for t in texts])
    searcher.build_index(embeddings, metadata=texts)

    # 検索
    query_emb = model.encode('passage: 検索クエリ')
    results = searcher.search(query_emb, candidates=1000, top_k=10)
"""

import numpy as np
from numpy.linalg import norm
from typing import Optional, List, Dict, Any, Tuple, Union
import pickle
import time

from .itq_lsh import ITQLSH, hamming_distance_batch


class CascadeSearcher:
    """
    ITQ LSH + コサイン類似度によるカスケード検索

    検索フロー:
    1. クエリをITQでハッシュ化
    2. ハミング距離で上位N件の候補を選択（高速）
    3. 候補に対してコサイン類似度を計算（精密）
    4. コサイン類似度でランキングして返却
    """

    def __init__(self, itq: ITQLSH):
        """
        Args:
            itq: 学習済みITQLSHインスタンス
        """
        self.itq = itq

        # インデックスデータ
        self._embeddings: Optional[np.ndarray] = None
        self._hashes: Optional[np.ndarray] = None
        self._metadata: Optional[List[Any]] = None
        self._is_indexed = False

        # 統計情報
        self._stats = {
            'n_documents': 0,
            'embedding_dim': 0,
            'hash_bits': itq.n_bits,
        }

    @classmethod
    def from_itq_model(cls, model_path: str) -> 'CascadeSearcher':
        """
        保存されたITQモデルからCascadeSearcherを作成

        Args:
            model_path: ITQモデルのパス

        Returns:
            CascadeSearcherインスタンス
        """
        itq = ITQLSH.load(model_path)
        return cls(itq)

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Any]] = None,
        show_progress: bool = True
    ) -> 'CascadeSearcher':
        """
        検索インデックスを構築

        Args:
            embeddings: ドキュメント埋め込み (n_docs, dim)
                        ※ passage: プレフィックスで生成されたものを想定
            metadata: 各ドキュメントのメタデータ（テキスト、IDなど）
            show_progress: 進捗を表示するか

        Returns:
            self
        """
        if show_progress:
            print(f"インデックス構築開始: {len(embeddings):,}件")

        start_time = time.time()

        # 埋め込みを保存
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

        # ITQハッシュを計算
        if show_progress:
            print("  ITQハッシュを計算中...")
        self._hashes = self.itq.transform(self._embeddings)

        # メタデータを保存
        self._metadata = metadata

        # 統計情報を更新
        self._stats['n_documents'] = len(embeddings)
        self._stats['embedding_dim'] = embeddings.shape[1]

        self._is_indexed = True

        elapsed = time.time() - start_time
        if show_progress:
            print(f"インデックス構築完了: {elapsed:.2f}秒")

        return self

    def search(
        self,
        query_embedding: np.ndarray,
        candidates: int = 1000,
        top_k: int = 10,
        return_scores: bool = True,
        return_distances: bool = False
    ) -> Dict[str, Any]:
        """
        カスケード検索を実行

        Args:
            query_embedding: クエリ埋め込み (dim,) または (1, dim)
                            ※ passage: プレフィックスで生成されたものを想定
            candidates: LSHで選択する候補数
            top_k: 最終的に返す結果数
            return_scores: コサイン類似度スコアを返すか
            return_distances: ハミング距離を返すか

        Returns:
            dict: {
                'indices': Top-kのドキュメントインデックス,
                'scores': コサイン類似度スコア（return_scores=Trueの場合）,
                'metadata': メタデータ（設定されている場合）,
                'hamming_distances': ハミング距離（return_distances=Trueの場合）,
                'timing': 各段階の処理時間
            }
        """
        if not self._is_indexed:
            raise RuntimeError("build_index()を先に呼び出してください")

        # クエリの形状を正規化
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        timing = {}

        # Stage 1: ITQ LSH候補選択
        t0 = time.time()
        query_hash = self.itq.transform(query_embedding)
        hamming_dists = hamming_distance_batch(query_hash, self._hashes)
        candidate_indices = np.argsort(hamming_dists)[:candidates]
        timing['lsh_ms'] = (time.time() - t0) * 1000

        # Stage 2: コサイン類似度リランキング
        t0 = time.time()
        candidate_embeddings = self._embeddings[candidate_indices]
        query_norm = norm(query_embedding)
        candidate_norms = norm(candidate_embeddings, axis=1)
        cosine_scores = (candidate_embeddings @ query_embedding) / (candidate_norms * query_norm + 1e-10)

        # Top-k選択
        top_k_in_candidates = np.argsort(cosine_scores)[-top_k:][::-1]
        top_k_indices = candidate_indices[top_k_in_candidates]
        top_k_scores = cosine_scores[top_k_in_candidates]
        timing['rerank_ms'] = (time.time() - t0) * 1000

        timing['total_ms'] = timing['lsh_ms'] + timing['rerank_ms']

        # 結果を構築
        result = {
            'indices': top_k_indices.tolist(),
            'timing': timing
        }

        if return_scores:
            result['scores'] = top_k_scores.tolist()

        if return_distances:
            result['hamming_distances'] = hamming_dists[top_k_indices].tolist()

        if self._metadata is not None:
            result['metadata'] = [self._metadata[i] for i in top_k_indices]

        return result

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        candidates: int = 1000,
        top_k: int = 10,
        return_scores: bool = True,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        複数クエリのバッチ検索

        Args:
            query_embeddings: クエリ埋め込み (n_queries, dim)
            candidates: LSHで選択する候補数
            top_k: 最終的に返す結果数
            return_scores: コサイン類似度スコアを返すか
            show_progress: 進捗を表示するか

        Returns:
            各クエリの検索結果のリスト
        """
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        n_queries = len(query_embeddings)

        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(n_queries), desc="検索中")
        else:
            iterator = range(n_queries)

        for i in iterator:
            result = self.search(
                query_embeddings[i],
                candidates=candidates,
                top_k=top_k,
                return_scores=return_scores
            )
            results.append(result)

        return results

    def evaluate_recall(
        self,
        query_embeddings: np.ndarray,
        candidates: int = 1000,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Recall@kを評価

        Ground Truthは全データに対するコサイン類似度で計算

        Args:
            query_embeddings: クエリ埋め込み (n_queries, dim)
            candidates: LSHで選択する候補数
            top_k: 評価するk

        Returns:
            dict: {'recall': Recall@k, 'queries': クエリ数}
        """
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        n_queries = len(query_embeddings)

        recall_sum = 0.0

        for i in range(n_queries):
            query_emb = query_embeddings[i]

            # Ground Truth（全データに対するコサイン類似度）
            all_cosines = self._embeddings @ query_emb / (
                norm(self._embeddings, axis=1) * norm(query_emb) + 1e-10
            )
            gt_indices = set(np.argsort(all_cosines)[-top_k:])

            # カスケード検索結果
            result = self.search(query_emb, candidates=candidates, top_k=top_k)
            pred_indices = set(result['indices'])

            # Recall計算
            recall = len(gt_indices & pred_indices) / top_k
            recall_sum += recall

        return {
            'recall': recall_sum / n_queries,
            'queries': n_queries,
            'candidates': candidates,
            'top_k': top_k
        }

    def get_stats(self) -> Dict[str, Any]:
        """インデックスの統計情報を取得"""
        stats = self._stats.copy()
        if self._is_indexed:
            stats['embeddings_memory_mb'] = self._embeddings.nbytes / (1024 * 1024)
            stats['hashes_memory_mb'] = self._hashes.nbytes / (1024 * 1024)
            stats['total_memory_mb'] = stats['embeddings_memory_mb'] + stats['hashes_memory_mb']
        return stats

    def save_index(self, path: str):
        """
        インデックスを保存

        Args:
            path: 保存先パス
        """
        if not self._is_indexed:
            raise RuntimeError("build_index()を先に呼び出してください")

        data = {
            'embeddings': self._embeddings,
            'hashes': self._hashes,
            'metadata': self._metadata,
            'stats': self._stats
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: str) -> 'CascadeSearcher':
        """
        インデックスをロード

        Args:
            path: インデックスファイルのパス

        Returns:
            self
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self._embeddings = data['embeddings']
        self._hashes = data['hashes']
        self._metadata = data['metadata']
        self._stats = data['stats']
        self._is_indexed = True

        return self


def benchmark_search(
    searcher: CascadeSearcher,
    query_embeddings: np.ndarray,
    candidate_sizes: List[int] = [500, 1000, 2000, 5000],
    top_k: int = 10,
    n_runs: int = 3
) -> Dict[str, Any]:
    """
    検索性能のベンチマーク

    Args:
        searcher: CascadeSearcherインスタンス
        query_embeddings: クエリ埋め込み
        candidate_sizes: テストする候補数リスト
        top_k: Top-k
        n_runs: 各設定の実行回数

    Returns:
        ベンチマーク結果
    """
    results = {
        'n_documents': searcher._stats['n_documents'],
        'n_queries': len(query_embeddings),
        'top_k': top_k,
        'benchmarks': []
    }

    for candidates in candidate_sizes:
        timings = {'lsh_ms': [], 'rerank_ms': [], 'total_ms': []}
        recalls = []

        for run in range(n_runs):
            # Recall評価
            if run == 0:
                eval_result = searcher.evaluate_recall(
                    query_embeddings, candidates=candidates, top_k=top_k
                )
                recalls.append(eval_result['recall'])

            # 時間計測
            for query_emb in query_embeddings:
                result = searcher.search(query_emb, candidates=candidates, top_k=top_k)
                for key in timings:
                    timings[key].append(result['timing'][key])

        results['benchmarks'].append({
            'candidates': candidates,
            'recall': recalls[0] if recalls else None,
            'avg_lsh_ms': np.mean(timings['lsh_ms']),
            'avg_rerank_ms': np.mean(timings['rerank_ms']),
            'avg_total_ms': np.mean(timings['total_ms']),
            'std_total_ms': np.std(timings['total_ms']),
            'queries_per_second': 1000 / np.mean(timings['total_ms'])
        })

    return results


def print_benchmark_results(results: Dict[str, Any]):
    """ベンチマーク結果を整形して表示"""
    print("=" * 80)
    print("カスケード検索ベンチマーク結果")
    print("=" * 80)
    print(f"ドキュメント数: {results['n_documents']:,}")
    print(f"クエリ数: {results['n_queries']}")
    print(f"Top-k: {results['top_k']}")
    print()

    print(f"{'候補数':>8} | {'Recall@k':>10} | {'LSH (ms)':>10} | {'Rerank (ms)':>12} | {'Total (ms)':>12} | {'QPS':>8}")
    print("-" * 80)

    for b in results['benchmarks']:
        recall_str = f"{b['recall']*100:.1f}%" if b['recall'] is not None else "N/A"
        print(f"{b['candidates']:>8} | {recall_str:>10} | {b['avg_lsh_ms']:>10.2f} | {b['avg_rerank_ms']:>12.2f} | {b['avg_total_ms']:>12.2f} | {b['queries_per_second']:>8.1f}")
