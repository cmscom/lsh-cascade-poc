#!/usr/bin/env python3
"""
カスケード検索システムのベンチマーク

実行:
    python scripts/benchmark_cascade_search.py

出力:
    - 候補数別のRecall@10
    - 各段階の処理時間（LSH, リランキング）
    - メモリ使用量
    - QPS (Queries Per Second)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.linalg import norm
import duckdb
import time
import argparse
from sentence_transformers import SentenceTransformer

from src.cascade_search import CascadeSearcher, benchmark_search, print_benchmark_results


def main():
    parser = argparse.ArgumentParser(description='カスケード検索ベンチマーク')
    parser.add_argument('--db', default='data/experiment_400k.duckdb', help='DuckDBパス')
    parser.add_argument('--itq-model', default='data/itq_model.pkl', help='ITQモデルパス')
    parser.add_argument('--n-queries', type=int, default=100, help='ベンチマーク用クエリ数')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    args = parser.parse_args()

    print("=" * 80)
    print("カスケード検索システム ベンチマーク")
    print("=" * 80)

    # 1. データ読み込み
    print("\n[1/5] データ読み込み...")
    conn = duckdb.connect(args.db, read_only=True)

    result = conn.execute("""
        SELECT id, text, embedding
        FROM documents
        ORDER BY id
    """).fetchall()

    doc_ids = [r[0] for r in result]
    doc_texts = [r[1] for r in result]
    embeddings = np.array([r[2] for r in result], dtype=np.float32)

    print(f"  ドキュメント数: {len(doc_ids):,}")
    print(f"  埋め込み次元: {embeddings.shape[1]}")
    conn.close()

    # 2. CascadeSearcher初期化
    print("\n[2/5] CascadeSearcher初期化...")
    searcher = CascadeSearcher.from_itq_model(args.itq_model)
    print(f"  ITQモデル: {args.itq_model}")
    print(f"  ハッシュビット数: {searcher.itq.n_bits}")

    # 3. インデックス構築
    print("\n[3/5] インデックス構築...")
    t0 = time.time()
    searcher.build_index(embeddings, metadata=doc_texts, show_progress=True)
    build_time = time.time() - t0

    stats = searcher.get_stats()
    print(f"  構築時間: {build_time:.2f}秒")
    print(f"  埋め込みメモリ: {stats['embeddings_memory_mb']:.1f} MB")
    print(f"  ハッシュメモリ: {stats['hashes_memory_mb']:.1f} MB")
    print(f"  合計メモリ: {stats['total_memory_mb']:.1f} MB")

    # 4. ベンチマーク用クエリ生成
    print(f"\n[4/5] ベンチマーク用クエリ生成 ({args.n_queries}件)...")

    # ランダムにドキュメントを選択してクエリとして使用
    rng = np.random.default_rng(args.seed)
    query_indices = rng.choice(len(embeddings), args.n_queries, replace=False)
    query_embeddings = embeddings[query_indices]

    print(f"  クエリ数: {len(query_embeddings)}")

    # 5. ベンチマーク実行
    print("\n[5/5] ベンチマーク実行...")
    candidate_sizes = [500, 1000, 2000, 5000, 10000]

    results = benchmark_search(
        searcher,
        query_embeddings,
        candidate_sizes=candidate_sizes,
        top_k=10,
        n_runs=3
    )

    # 結果表示
    print()
    print_benchmark_results(results)

    # 追加分析
    print("\n" + "=" * 80)
    print("追加分析")
    print("=" * 80)

    # 90% Recall達成に必要な候補数
    print("\n■ Recall達成状況")
    for b in results['benchmarks']:
        recall = b['recall'] * 100
        status = "達成" if recall >= 90 else "未達"
        print(f"  候補{b['candidates']:>5}件: {recall:>5.1f}% ({status})")

    # ブルートフォースとの比較
    print("\n■ ブルートフォース比較")
    print("  ブルートフォースコサイン計算を実行中...")

    brute_times = []
    for query_emb in query_embeddings[:10]:  # 10件でテスト
        t0 = time.time()
        _ = embeddings @ query_emb / (norm(embeddings, axis=1) * norm(query_emb) + 1e-10)
        brute_times.append((time.time() - t0) * 1000)

    brute_avg = np.mean(brute_times)
    print(f"  ブルートフォース平均: {brute_avg:.2f} ms/query")

    for b in results['benchmarks']:
        speedup = brute_avg / b['avg_total_ms']
        print(f"  候補{b['candidates']:>5}件: {b['avg_total_ms']:.2f} ms ({speedup:.1f}x 高速)")

    # メモリ効率
    print("\n■ メモリ効率")
    embedding_per_doc = embeddings.shape[1] * 4 / 1024  # KB per doc (float32)
    hash_per_doc = searcher.itq.n_bits / 8 / 1024  # KB per doc (bit packed would be even smaller)
    print(f"  埋め込み: {embedding_per_doc:.2f} KB/doc")
    print(f"  ハッシュ: {hash_per_doc:.4f} KB/doc (現在uint8、ビットパック可能)")
    print(f"  ハッシュは埋め込みの {hash_per_doc/embedding_per_doc*100:.2f}% のサイズ")

    # 推奨設定
    print("\n" + "=" * 80)
    print("推奨設定")
    print("=" * 80)

    best_90 = None
    for b in results['benchmarks']:
        if b['recall'] >= 0.9:
            best_90 = b
            break

    if best_90:
        print(f"""
■ 90%+ Recall目標の場合:
  候補数: {best_90['candidates']}件
  Recall: {best_90['recall']*100:.1f}%
  処理時間: {best_90['avg_total_ms']:.2f} ms/query
  QPS: {best_90['queries_per_second']:.1f}

■ 使用方法:
  from src.cascade_search import CascadeSearcher

  searcher = CascadeSearcher.from_itq_model('data/itq_model.pkl')
  searcher.build_index(embeddings)

  # 検索時は passage: プレフィックスを使用
  query_emb = model.encode('passage: 検索クエリ')
  results = searcher.search(query_emb, candidates={best_90['candidates']}, top_k=10)
""")
    else:
        print("  90% Recall未達成。候補数を増やしてください。")


if __name__ == '__main__':
    main()
