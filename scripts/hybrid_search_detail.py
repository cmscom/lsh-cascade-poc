#!/usr/bin/env python3
"""
Hybrid検索の詳細検証スクリプト
LSH候補選択: passage:プレフィックス
最終コサイン計算: query:プレフィックス
"""

import sys
sys.path.insert(0, '/home/terapyon/dev/vibe-coding/lsh-cascade-poc')

import numpy as np
from numpy.linalg import norm
import pandas as pd
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.lsh import SimHashGenerator, hamming_distance


def hybrid_search(
    query_emb_passage: np.ndarray,
    query_emb_query: np.ndarray,
    all_embeddings: np.ndarray,
    all_hashes: list,
    gen: SimHashGenerator,
    candidate_limit: int,
    top_k: int = 10
) -> dict:
    """
    Hybrid検索: LSHにはpassage:、最終コサインにはquery:を使用
    """
    # Step 1: LSHで候補選択（passage:埋め込み使用）
    query_hash = gen.hash_batch(query_emb_passage.reshape(1, -1))[0]
    distances = [(j, hamming_distance(h, query_hash)) for j, h in enumerate(all_hashes)]
    distances.sort(key=lambda x: x[1])
    candidates = [idx for idx, _ in distances[:candidate_limit]]
    candidate_distances = [dist for _, dist in distances[:candidate_limit]]

    # Step 2: 候補に対してコサイン類似度を計算（query:埋め込み使用）
    candidate_embeddings = all_embeddings[candidates]
    cos_sims = (candidate_embeddings @ query_emb_query) / (
        norm(candidate_embeddings, axis=1) * norm(query_emb_query)
    )

    # Step 3: コサイン類似度でランキング
    sorted_indices = np.argsort(cos_sims)[::-1][:top_k]
    top_k_indices = [candidates[i] for i in sorted_indices]
    top_k_scores = [cos_sims[i] for i in sorted_indices]

    # Ground Truth（query:プレフィックスで全データから計算）
    all_cos_sims = (all_embeddings @ query_emb_query) / (
        norm(all_embeddings, axis=1) * norm(query_emb_query)
    )
    gt_query = set(np.argsort(all_cos_sims)[::-1][:top_k])

    # Recall計算
    recall = len(set(top_k_indices) & gt_query) / top_k

    return {
        'candidates': candidates,
        'candidate_distances': candidate_distances,
        'top_k_indices': top_k_indices,
        'top_k_scores': top_k_scores,
        'gt_query': gt_query,
        'recall': recall
    }


def main():
    print('=' * 80)
    print('Hybrid検索の詳細検証')
    print('LSH候補選択: passage:, 最終コサイン計算: query:')
    print('=' * 80)

    # DuckDBに接続
    print('\n1. データベースに接続...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb', read_only=True)

    # E5モデルを読み込み
    print('\n2. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # データ読み込み
    print('\n3. データ読み込み中...')
    datasets = ['body_en', 'body_ja', 'titles_en', 'titles_ja']
    all_embeddings = {}

    for dataset in tqdm(datasets, desc='   データセット'):
        df = con.execute(f"""
            SELECT embedding FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()
        all_embeddings[dataset] = np.array(df['embedding'].tolist(), dtype=np.float32)

    # 全データを統合
    all_embeddings_flat = np.vstack([all_embeddings[d] for d in datasets])
    all_datasets_flat = []
    for dataset in datasets:
        all_datasets_flat.extend([dataset] * len(all_embeddings[dataset]))
    print(f'   統合完了: {all_embeddings_flat.shape}')

    # 超平面を準備
    print('\n4. 超平面を準備中...')
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_embeddings['body_ja']), 300, replace=False)
    sample_embeddings = all_embeddings['body_ja'][sample_indices]

    hyperplanes = []
    for _ in range(128):
        i, j = rng.choice(len(sample_embeddings), 2, replace=False)
        diff = sample_embeddings[i] - sample_embeddings[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes.append(diff)
    hyperplanes = np.array(hyperplanes, dtype=np.float32)

    gen = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
    gen.hyperplanes = hyperplanes

    # 全ドキュメントのハッシュを計算
    print('\n5. SimHashを計算中...')
    all_hashes = gen.hash_batch(all_embeddings_flat)
    print(f'   完了: {len(all_hashes)}件')

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
        ('最近話題になっている技術革新について知りたいのですが、何かありますか', 'ja', 'ambiguous'),
        ('日本の伝統的な文化や芸術に関する情報を探しています', 'ja', 'ambiguous'),
        ('環境に優しい持続可能な社会を実現するための取り組みとは', 'ja', 'ambiguous'),
        ('健康的な生活を送るために必要なことは何でしょうか', 'ja', 'ambiguous'),
        ('世界の政治情勢や国際関係についての最新動向を教えて', 'ja', 'ambiguous'),
        ('子供の教育において大切にすべきポイントは何ですか', 'ja', 'ambiguous'),
        ('スポーツやフィットネスに関するトレンドを知りたい', 'ja', 'ambiguous'),
        ('美味しい料理のレシピや食文化についての情報', 'ja', 'ambiguous'),
        ('旅行や観光に関するおすすめの場所はありますか', 'ja', 'ambiguous'),
        ('ビジネスや起業に関する成功のヒントを教えてください', 'ja', 'ambiguous'),
        ('Tokyo', 'en', 'short'),
        ('Artificial intelligence', 'en', 'short'),
        ('World history', 'en', 'short'),
        ('Programming', 'en', 'short'),
        ('Climate change', 'en', 'short'),
        ('I want to learn about recent technological innovations', 'en', 'ambiguous'),
        ('Looking for information about traditional culture and arts', 'en', 'ambiguous'),
        ('What are sustainable approaches to environmental protection', 'en', 'ambiguous'),
        ('Tell me about the latest developments in space exploration', 'en', 'ambiguous'),
        ('What are the key factors for business success and entrepreneurship', 'en', 'ambiguous'),
    ]

    query_texts = [q[0] for q in search_queries]

    # クエリ埋め込みを生成
    print('\n6. クエリ埋め込みを生成中...')
    query_embs_passage = model.encode(
        [f'passage: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    query_embs_query = model.encode(
        [f'query: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)
    print('   完了')

    # ========================================
    # 評価実行
    # ========================================
    print('\n' + '=' * 80)
    print('7. Hybrid検索を評価中...')
    print('=' * 80)

    candidate_limits = [500, 1000, 2000, 5000, 10000, 20000]
    results = []

    for i, (query_text, lang, query_type) in enumerate(tqdm(search_queries, desc='   クエリ')):
        q_emb_p = query_embs_passage[i]
        q_emb_q = query_embs_query[i]

        for limit in candidate_limits:
            result = hybrid_search(
                q_emb_p, q_emb_q,
                all_embeddings_flat, all_hashes, gen,
                candidate_limit=limit
            )

            results.append({
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'candidate_limit': limit,
                'recall': result['recall']
            })

    df_results = pd.DataFrame(results)

    # ========================================
    # 結果表示
    # ========================================
    print('\n' + '=' * 80)
    print('結果: Hybrid検索（LSH=passage:, Cos=query:）')
    print('=' * 80)

    # 候補数別
    print('\n■ 候補数別 Recall@10（30クエリ平均）')
    pivot = df_results.groupby('candidate_limit')['recall'].agg(['mean', 'std', 'min', 'max'])
    print(f'{"候補数":>10} | {"平均":>10} | {"標準偏差":>10} | {"最小":>10} | {"最大":>10}')
    print('-' * 60)
    for limit in candidate_limits:
        row = pivot.loc[limit]
        print(f'{limit:>10} | {row["mean"]:>10.1%} | {row["std"]:>10.1%} | {row["min"]:>10.1%} | {row["max"]:>10.1%}')

    # クエリタイプ別（候補2000件）
    print('\n■ クエリタイプ別 Recall@10')
    print(f'{"候補数":>10} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 65)

    for limit in candidate_limits:
        subset = df_results[df_results['candidate_limit'] == limit]
        ja_short = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'short')]['recall'].mean()
        ja_amb = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'ambiguous')]['recall'].mean()
        en_short = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'short')]['recall'].mean()
        en_amb = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'ambiguous')]['recall'].mean()
        print(f'{limit:>10} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # ========================================
    # 詳細分析: 個別クエリ
    # ========================================
    print('\n' + '=' * 80)
    print('詳細分析: 代表的なクエリの結果')
    print('=' * 80)

    test_queries = [0, 10, 20, 25]  # 東京, 技術革新（曖昧）, Tokyo, innovations（曖昧）

    for test_idx in test_queries:
        query_text, lang, query_type = search_queries[test_idx]
        q_emb_p = query_embs_passage[test_idx]
        q_emb_q = query_embs_query[test_idx]

        print(f'\n--- クエリ: 「{query_text[:30]}...」 ({lang}, {query_type}) ---')

        for limit in [2000, 10000]:
            result = hybrid_search(
                q_emb_p, q_emb_q,
                all_embeddings_flat, all_hashes, gen,
                candidate_limit=limit
            )

            # GT内のドキュメントのハミング距離を確認
            gt_distances = []
            for gt_idx in result['gt_query']:
                query_hash = gen.hash_batch(q_emb_p.reshape(1, -1))[0]
                dist = hamming_distance(all_hashes[gt_idx], query_hash)
                gt_distances.append(dist)

            print(f'  候補{limit}件: Recall={result["recall"]:.1%}, ' +
                  f'GT内のハミング距離: 平均={np.mean(gt_distances):.1f}, 最大={np.max(gt_distances)}')

    # ========================================
    # 90% Recallに必要な候補数の推定
    # ========================================
    print('\n' + '=' * 80)
    print('90% Recall達成に必要な候補数の推定')
    print('=' * 80)

    for limit in candidate_limits:
        subset = df_results[df_results['candidate_limit'] == limit]
        recall_90_count = (subset['recall'] >= 0.9).sum()
        print(f'  候補{limit:>6}件: {recall_90_count:>2}/30 クエリが90%以上')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
