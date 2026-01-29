#!/usr/bin/env python3
"""
カスケード検索システムの使用例

このスクリプトは、ITQ LSH + コサイン類似度リランキングによる
2段階検索システムの基本的な使い方を示します。

実行:
    python examples/cascade_search_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer

from src.cascade_search import CascadeSearcher


def main():
    print("=" * 60)
    print("カスケード検索システム 使用例")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. モデルとデータの準備
    # -----------------------------------------------------------------
    print("\n[1] モデルとデータの準備...")

    # E5モデルのロード
    model = SentenceTransformer('intfloat/multilingual-e5-large')

    # サンプルデータ（実際のアプリケーションではDBや外部ソースから）
    # 重要: ドキュメントは必ず passage: プレフィックス付きで埋め込む
    documents = [
        "東京は日本の首都で、世界最大の都市圏を形成している。",
        "人工知能は機械学習を基盤として急速に発展している分野である。",
        "気候変動は地球規模の環境問題として注目されている。",
        "プログラミング言語Pythonはデータサイエンスで広く使用されている。",
        "日本の伝統文化には茶道、華道、書道などがある。",
    ]

    # -----------------------------------------------------------------
    # 2. CascadeSearcherの初期化
    # -----------------------------------------------------------------
    print("\n[2] CascadeSearcherの初期化...")

    # 学習済みITQモデルからSearcherを作成
    searcher = CascadeSearcher.from_itq_model('data/itq_model.pkl')
    print(f"  ハッシュビット数: {searcher.itq.n_bits}")

    # -----------------------------------------------------------------
    # 3. インデックス構築
    # -----------------------------------------------------------------
    print("\n[3] インデックス構築...")

    # ドキュメントを passage: プレフィックス付きで埋め込み
    doc_embeddings = model.encode(
        [f'passage: {doc}' for doc in documents],
        normalize_embeddings=False
    )

    # インデックスを構築（メタデータとしてテキストも保存）
    searcher.build_index(doc_embeddings, metadata=documents)

    stats = searcher.get_stats()
    print(f"  ドキュメント数: {stats['n_documents']}")

    # -----------------------------------------------------------------
    # 4. 検索実行
    # -----------------------------------------------------------------
    print("\n[4] 検索実行...")

    # 重要: 検索時も passage: プレフィックスを使用
    query = "日本の文化について"
    query_embedding = model.encode(f'passage: {query}', normalize_embeddings=False)

    # 検索実行
    results = searcher.search(
        query_embedding,
        candidates=3,  # 小さなデータなので候補数も少なく
        top_k=3,
        return_scores=True,
        return_distances=True
    )

    print(f"\n  クエリ: 「{query}」")
    print(f"  処理時間: {results['timing']['total_ms']:.2f} ms")
    print(f"    - LSH候補選択: {results['timing']['lsh_ms']:.2f} ms")
    print(f"    - リランキング: {results['timing']['rerank_ms']:.2f} ms")

    print("\n  検索結果:")
    for i, (idx, score, dist, text) in enumerate(zip(
        results['indices'],
        results['scores'],
        results['hamming_distances'],
        results['metadata']
    )):
        print(f"    {i+1}. [cos={score:.4f}, hamming={dist}] {text[:50]}...")

    # -----------------------------------------------------------------
    # 5. 大規模データでの使用例（DuckDBから）
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("大規模データでの使用例")
    print("=" * 60)

    if os.path.exists('data/experiment_400k.duckdb'):
        print("\n[5] 40万件データでの検索...")

        # データ読み込み
        conn = duckdb.connect('data/experiment_400k.duckdb', read_only=True)
        result = conn.execute("""
            SELECT text, embedding FROM documents
            ORDER BY id LIMIT 10000
        """).fetchall()
        conn.close()

        texts = [r[0] for r in result]
        embeddings = np.array([r[1] for r in result], dtype=np.float32)

        # 新しいSearcherでインデックス構築
        searcher_large = CascadeSearcher.from_itq_model('data/itq_model.pkl')
        searcher_large.build_index(embeddings, metadata=texts)

        # 検索
        query = "機械学習の基礎"
        query_emb = model.encode(f'passage: {query}', normalize_embeddings=False)

        results = searcher_large.search(
            query_emb,
            candidates=1000,
            top_k=5
        )

        print(f"\n  クエリ: 「{query}」")
        print(f"  処理時間: {results['timing']['total_ms']:.2f} ms")
        print("\n  Top-5 結果:")
        for i, (score, text) in enumerate(zip(results['scores'], results['metadata'])):
            print(f"    {i+1}. [cos={score:.4f}] {text[:60]}...")

        # Recall評価
        print("\n  Recall評価中...")
        # ランダムに50件のクエリを選択
        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(embeddings), 50, replace=False)
        query_embs = embeddings[query_indices]

        eval_result = searcher_large.evaluate_recall(
            query_embs, candidates=1000, top_k=10
        )
        print(f"  Recall@10 (候補1000件): {eval_result['recall']*100:.1f}%")

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
