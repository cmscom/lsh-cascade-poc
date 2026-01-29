#!/usr/bin/env python3
"""
ITQ LSH の評価スクリプト

従来のSimHash（ランダム超平面、DataSampled）と比較し、
ITQ + Centering がどの程度改善するかを検証する。

【重要検証項目】
- サンプリングによる学習の汎化性能
- 学習データと未知データの分離検証
- サンプリング率の影響

比較パターン:
1. SimHash (Random) - ベースライン
2. SimHash (DataSampled) - 従来手法
3. ITQ LSH - 新手法
4. ITQ LSH + passage/passage - プレフィックス統一版
"""

import sys
sys.path.insert(0, '/home/terapyon/dev/vibe-coding/lsh-cascade-poc')

import numpy as np
from numpy.linalg import norm
import pandas as pd
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.lsh import SimHashGenerator, hamming_distance as simhash_hamming
from src.itq_lsh import ITQLSH, hamming_distance_batch


def evaluate_lsh(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    doc_hashes: np.ndarray,
    hash_func,
    candidate_limits: list,
    top_k: int = 10
) -> list:
    """LSH手法を評価"""
    results = []

    for i in range(len(query_embeddings)):
        query_emb = query_embeddings[i]

        # Ground Truth（コサイン類似度Top-k）
        cos_sims = (doc_embeddings @ query_emb) / (norm(doc_embeddings, axis=1) * norm(query_emb) + 1e-9)
        gt_indices = set(np.argsort(cos_sims)[::-1][:top_k])

        # クエリのハッシュ
        query_hash = hash_func(query_emb)

        # ハミング距離
        if isinstance(query_hash, np.ndarray) and query_hash.dtype == np.uint8:
            # ITQ形式
            distances = hamming_distance_batch(query_hash, doc_hashes)
        else:
            # SimHash形式
            distances = np.array([simhash_hamming(h, query_hash) for h in doc_hashes])

        sorted_indices = np.argsort(distances)

        for limit in candidate_limits:
            candidates = set(sorted_indices[:limit])
            recall = len(gt_indices & candidates) / top_k
            results.append({
                'query_idx': i,
                'candidate_limit': limit,
                'recall': recall
            })

    return results


def main():
    print('=' * 80)
    print('ITQ LSH vs SimHash 比較評価')
    print('=' * 80)

    # DuckDBに接続
    print('\n1. データベースに接続...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb', read_only=True)

    # E5モデル読み込み
    print('\n2. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # データ読み込み（評価用にサブセットを使用）
    print('\n3. データ読み込み中...')
    datasets = ['body_en', 'body_ja', 'titles_en', 'titles_ja']

    # passage:プレフィックス埋め込み
    embeddings_passage = {}
    for dataset in tqdm(datasets, desc='   passage埋め込み'):
        df = con.execute(f"""
            SELECT embedding FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()
        embeddings_passage[dataset] = np.array(df['embedding'].tolist(), dtype=np.float32)

    # 全データを統合
    all_passage = np.vstack([embeddings_passage[d] for d in datasets])
    print(f'   完了: {all_passage.shape}')

    # 検索クエリ
    search_queries = [
        ('東京', 'ja', 'short'),
        ('人工知能', 'ja', 'short'),
        ('日本の歴史', 'ja', 'short'),
        ('プログラミング', 'ja', 'short'),
        ('音楽', 'ja', 'short'),
        ('環境問題', 'ja', 'short'),
        ('宇宙探査', 'ja', 'short'),
        ('経済学', 'ja', 'short'),
        ('医療技術', 'ja', 'short'),
        ('文学作品', 'ja', 'short'),
        ('最近話題になっている技術革新について知りたいのですが', 'ja', 'ambiguous'),
        ('日本の伝統的な文化や芸術に関する情報', 'ja', 'ambiguous'),
        ('環境に優しい持続可能な社会を実現するための取り組み', 'ja', 'ambiguous'),
        ('健康的な生活を送るために必要なこと', 'ja', 'ambiguous'),
        ('世界の政治情勢や国際関係についての最新動向', 'ja', 'ambiguous'),
        ('Tokyo', 'en', 'short'),
        ('Artificial intelligence', 'en', 'short'),
        ('World history', 'en', 'short'),
        ('Programming', 'en', 'short'),
        ('Climate change', 'en', 'short'),
        ('Recent technological innovations', 'en', 'ambiguous'),
        ('Traditional culture and arts', 'en', 'ambiguous'),
        ('Sustainable environmental protection', 'en', 'ambiguous'),
        ('Space exploration developments', 'en', 'ambiguous'),
        ('Business success factors', 'en', 'ambiguous'),
    ]

    query_texts = [q[0] for q in search_queries]

    # クエリ埋め込み生成
    print('\n4. クエリ埋め込みを生成中...')
    query_embs_query = model.encode(
        [f'query: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    query_embs_passage = model.encode(
        [f'passage: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)
    print('   完了')

    # ========================================
    # 各手法のセットアップ
    # ========================================
    print('\n5. 各手法をセットアップ中...')
    n_bits = 128
    candidate_limits = [500, 1000, 2000, 5000, 10000, 20000]
    all_results = []

    # --- 手法1: SimHash (Random) ---
    print('\n   [1/4] SimHash (Random)...')
    gen_random = SimHashGenerator(dim=1024, hash_bits=n_bits, seed=42, strategy='random')
    hashes_random = gen_random.hash_batch(all_passage)

    results = evaluate_lsh(
        all_passage, query_embs_query, hashes_random,
        lambda x: gen_random.hash_batch(x.reshape(1, -1))[0],
        candidate_limits
    )
    for r in results:
        r['method'] = 'SimHash (Random)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    # --- 手法2: SimHash (DataSampled) ---
    print('   [2/4] SimHash (DataSampled)...')
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(embeddings_passage['body_ja']), 300, replace=False)
    sample_embs = embeddings_passage['body_ja'][sample_indices]

    hyperplanes_ds = []
    for _ in range(n_bits):
        i, j = rng.choice(len(sample_embs), 2, replace=False)
        diff = sample_embs[i] - sample_embs[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes_ds.append(diff)
    hyperplanes_ds = np.array(hyperplanes_ds, dtype=np.float32)

    gen_ds = SimHashGenerator(dim=1024, hash_bits=n_bits, seed=0, strategy='random')
    gen_ds.hyperplanes = hyperplanes_ds
    hashes_ds = gen_ds.hash_batch(all_passage)

    results = evaluate_lsh(
        all_passage, query_embs_query, hashes_ds,
        lambda x: gen_ds.hash_batch(x.reshape(1, -1))[0],
        candidate_limits
    )
    for r in results:
        r['method'] = 'SimHash (DataSampled)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    # ========================================
    # サンプリング汎化性能の検証
    # ========================================
    print('\n' + '=' * 80)
    print('サンプリング汎化性能の検証')
    print('=' * 80)
    print('\n【重要】全データが事前にない場合を想定し、サンプリングで学習した')
    print('       ITQが未知データにも適用できるかを検証します。')

    # 学習/テスト分割（80%/20%）
    n_total = len(all_passage)
    n_test = n_total // 5  # 20%をテストデータに
    all_indices = rng.permutation(n_total)
    test_indices = all_indices[:n_test]
    pool_indices = all_indices[n_test:]  # 学習用プールから更にサンプリング

    test_embs = all_passage[test_indices]
    pool_embs = all_passage[pool_indices]

    print(f'   テストデータ: {len(test_embs):,}件（未知データとして扱う）')
    print(f'   学習用プール: {len(pool_embs):,}件')

    # サンプリング率別検証
    sampling_rates = [0.01, 0.05, 0.10, 0.25]  # 1%, 5%, 10%, 25%
    sampling_results = []

    for rate in sampling_rates:
        n_sample = max(500, int(len(pool_embs) * rate))
        sample_indices = rng.choice(len(pool_embs), n_sample, replace=False)
        sample_embs = pool_embs[sample_indices]

        print(f'\n   サンプリング率 {rate:.0%}: {n_sample:,}件で学習...')

        # ITQ学習（サンプリングデータのみ）
        itq_sample = ITQLSH(n_bits=n_bits, n_iterations=50, seed=42)
        itq_sample.fit(sample_embs)

        # テストデータに適用（未知データ）
        test_hashes = itq_sample.transform(test_embs)

        # 評価：テストデータ内での検索
        recalls = []
        for i in range(min(100, len(query_embs_query))):  # 最大100クエリで評価
            query_emb = query_embs_query[i % len(query_embs_query)]

            # Ground Truth（テストデータ内）
            cos_sims = (test_embs @ query_emb) / (norm(test_embs, axis=1) * norm(query_emb) + 1e-9)
            gt_indices = set(np.argsort(cos_sims)[::-1][:10])

            # ITQハッシュで検索
            query_hash = itq_sample.transform(query_emb)
            distances = hamming_distance_batch(query_hash, test_hashes)
            sorted_idx = np.argsort(distances)
            candidates = set(sorted_idx[:2000])  # 候補2000件

            recall = len(gt_indices & candidates) / 10
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        sampling_results.append({
            'sampling_rate': rate,
            'n_samples': n_sample,
            'recall': mean_recall
        })
        print(f'      → Recall@10 (候補2000件): {mean_recall:.1%}')

    print('\n   サンプリング率による影響まとめ:')
    for r in sampling_results:
        print(f"      {r['sampling_rate']:>5.0%} ({r['n_samples']:>6,}件) → Recall@10: {r['recall']:.1%}")

    # ========================================
    # 本評価（全データで学習）
    # ========================================
    print('\n' + '=' * 80)
    print('本評価（サンプリング10%で学習、全データで評価）')
    print('=' * 80)

    # --- 手法3: ITQ LSH ---
    print('\n   [3/4] ITQ LSH (query:プレフィックス)...')

    # 学習用サンプル（全データの10%）
    n_train = int(len(all_passage) * 0.10)
    train_indices = rng.choice(len(all_passage), n_train, replace=False)
    train_embs = all_passage[train_indices]

    print(f'      学習データ: {n_train:,}件（全体の10%）')

    itq = ITQLSH(n_bits=n_bits, n_iterations=50, seed=42)
    itq.fit(train_embs)

    # 全ドキュメントのハッシュ（学習に使っていないデータも含む）
    hashes_itq = itq.transform(all_passage)

    results = evaluate_lsh(
        all_passage, query_embs_query, hashes_itq,
        lambda x: itq.transform(x),
        candidate_limits
    )
    for r in results:
        r['method'] = 'ITQ LSH (query:)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    # --- 手法4: ITQ LSH + passage/passage ---
    print('   [4/4] ITQ LSH (passage:プレフィックス統一)...')

    results = evaluate_lsh(
        all_passage, query_embs_passage, hashes_itq,
        lambda x: itq.transform(x),
        candidate_limits
    )
    for r in results:
        r['method'] = 'ITQ LSH (passage:)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    df_results = pd.DataFrame(all_results)

    # ========================================
    # 結果表示
    # ========================================
    print('\n' + '=' * 100)
    print('結果: 手法別 Recall@10（25クエリ平均）')
    print('=' * 100)

    method_order = [
        'SimHash (Random)',
        'SimHash (DataSampled)',
        'ITQ LSH (query:)',
        'ITQ LSH (passage:)',
    ]

    print(f'\n{"手法":>30} | {"500":>8} | {"1000":>8} | {"2000":>8} | {"5000":>8} | {"10000":>8} | {"20000":>8}')
    print('-' * 105)

    for method in method_order:
        subset = df_results[df_results['method'] == method]
        pivot = subset.groupby('candidate_limit')['recall'].mean()
        row = [f'{pivot.get(l, 0):.1%}' for l in candidate_limits]
        print(f'{method:>30} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} | {row[5]:>8}')

    # クエリタイプ別
    print('\n' + '=' * 100)
    print('クエリタイプ別 Recall@10（候補2000件）')
    print('=' * 100)

    print(f'\n{"手法":>30} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 85)

    for method in method_order:
        subset = df_results[(df_results['method'] == method) & (df_results['candidate_limit'] == 2000)]

        ja_short = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'short')]['recall'].mean()
        ja_amb = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'ambiguous')]['recall'].mean()
        en_short = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'short')]['recall'].mean()
        en_amb = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'ambiguous')]['recall'].mean()

        print(f'{method:>30} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # ========================================
    # 改善度サマリー
    # ========================================
    print('\n' + '=' * 100)
    print('改善度サマリー（候補2000件）')
    print('=' * 100)

    baseline = df_results[(df_results['method'] == 'SimHash (Random)') &
                          (df_results['candidate_limit'] == 2000)]['recall'].mean()
    ds = df_results[(df_results['method'] == 'SimHash (DataSampled)') &
                    (df_results['candidate_limit'] == 2000)]['recall'].mean()
    itq_query = df_results[(df_results['method'] == 'ITQ LSH (query:)') &
                           (df_results['candidate_limit'] == 2000)]['recall'].mean()
    itq_passage = df_results[(df_results['method'] == 'ITQ LSH (passage:)') &
                             (df_results['candidate_limit'] == 2000)]['recall'].mean()

    print(f'''
   SimHash (Random):       {baseline:.1%} (ベースライン)
   SimHash (DataSampled):  {ds:.1%} ({ds - baseline:+.1%})
   ITQ LSH (query:):       {itq_query:.1%} ({itq_query - baseline:+.1%})
   ITQ LSH (passage:):     {itq_passage:.1%} ({itq_passage - baseline:+.1%})
    ''')

    if itq_passage > itq_query:
        print('   → ITQ + passage/passageが最も効果的')
    elif itq_query > baseline:
        print('   → ITQは改善に貢献している')
    else:
        print('   → ITQでは改善が見られない')

    # ITQパラメータ保存
    print('\n6. ITQパラメータを保存中...')
    itq.save('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/itq_model.pkl')
    print('   保存完了: data/itq_model.pkl')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
