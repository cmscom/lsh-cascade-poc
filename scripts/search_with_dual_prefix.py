#!/usr/bin/env python3
"""
Phase 2: 2つのプレフィックスを使った検索手法の検証

検証パターン：
1. Baseline_Query: query:でLSH候補選択、query:でコサイン計算（従来手法）
2. Baseline_Passage: passage:でLSH候補選択、passage:でコサイン計算
3. Hybrid_PassageLSH_QueryCos: passage:でLSH候補選択、query:でコサイン計算
4. MeanDiff_Transform: query埋め込みを平均差分ベクトルで変換してLSH検索
5. Dual_Merge: 両方のプレフィックスで候補を取得してマージ
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


def main():
    print('=' * 80)
    print('Phase 2: 2つのプレフィックスを使った検索手法の検証')
    print('=' * 80)

    # 設定
    CANDIDATE_LIMITS = [1000, 2000, 5000]
    TOP_K = 10

    # DuckDBに接続
    print('\n1. データベースに接続...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb', read_only=True)

    # E5モデルを読み込み
    print('\n2. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # 平均差分ベクトルを読み込み
    print('\n3. 平均差分ベクトルを読み込み中...')
    mean_diff_vector = np.load('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/mean_diff_vector.npy')
    print(f'   形状: {mean_diff_vector.shape}')

    # データ読み込み
    print('\n4. データ読み込み中...')
    datasets = ['body_en', 'body_ja', 'titles_en', 'titles_ja']
    all_embeddings = {}
    all_texts = {}

    for dataset in tqdm(datasets, desc='   データセット'):
        df = con.execute(f"""
            SELECT text, embedding
            FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()

        embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
        all_embeddings[dataset] = embeddings
        all_texts[dataset] = df['text'].values

    # 全データを統合
    print('\n5. 全データを統合中...')
    all_embeddings_flat = np.vstack([all_embeddings[d] for d in datasets])
    all_datasets_flat = []
    for dataset in datasets:
        all_datasets_flat.extend([dataset] * len(all_embeddings[dataset]))
    print(f'   統合完了: {all_embeddings_flat.shape}')

    # 超平面を準備（DataSampled）
    print('\n6. 超平面を準備中...')
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_embeddings['body_ja']), 300, replace=False)
    sample_embeddings = all_embeddings['body_ja'][sample_indices]

    # DataSampled超平面を生成
    hyperplanes = []
    for _ in range(128):
        i, j = rng.choice(len(sample_embeddings), 2, replace=False)
        diff = sample_embeddings[i] - sample_embeddings[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes.append(diff)
    hyperplanes = np.array(hyperplanes, dtype=np.float32)
    print(f'   超平面形状: {hyperplanes.shape}')

    # SimHashGeneratorを設定
    gen = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
    gen.hyperplanes = hyperplanes

    # 全ドキュメントのハッシュを計算
    print('\n7. 全ドキュメントのSimHashを計算中...')
    all_hashes = gen.hash_batch(all_embeddings_flat)
    print(f'   完了: {len(all_hashes)}件')

    # 検索ワード
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
    print('\n8. クエリ埋め込みを生成中...')

    # query:プレフィックス
    query_embs_query = model.encode(
        [f'query: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    # passage:プレフィックス
    query_embs_passage = model.encode(
        [f'passage: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    # 平均差分ベクトルで変換（query - mean_diff → passage空間に近づける）
    query_embs_transformed = query_embs_query - mean_diff_vector

    print('   完了')

    # ========================================
    # 評価関数
    # ========================================

    def get_lsh_candidates(query_emb, all_hashes, gen, limit):
        """LSHで候補を取得"""
        query_hash = gen.hash_batch(query_emb.reshape(1, -1))[0]
        distances = [(j, hamming_distance(h, query_hash)) for j, h in enumerate(all_hashes)]
        distances.sort(key=lambda x: x[1])
        return set(idx for idx, _ in distances[:limit])

    def get_ground_truth(query_emb, all_embs, top_k):
        """コサイン類似度でGround Truthを取得"""
        cos_sims = (all_embs @ query_emb) / (norm(all_embs, axis=1) * norm(query_emb))
        return set(np.argsort(cos_sims)[::-1][:top_k])

    # ========================================
    # 評価実行
    # ========================================
    print('\n' + '=' * 100)
    print('9. 各手法の評価を実行中...')
    print('=' * 100)

    results = []

    for i, (query_text, lang, query_type) in enumerate(tqdm(search_queries, desc='   クエリ')):
        q_emb_query = query_embs_query[i]
        q_emb_passage = query_embs_passage[i]
        q_emb_transformed = query_embs_transformed[i]

        for limit in CANDIDATE_LIMITS:
            # ========================================
            # 手法1: Baseline_Query
            # LSH: query:, Ground Truth: query:
            # ========================================
            candidates_1 = get_lsh_candidates(q_emb_query, all_hashes, gen, limit)
            gt_1 = get_ground_truth(q_emb_query, all_embeddings_flat, TOP_K)
            recall_1 = len(candidates_1 & gt_1) / TOP_K

            results.append({
                'method': '1_Baseline_Query',
                'lsh_prefix': 'query:',
                'gt_prefix': 'query:',
                'candidate_limit': limit,
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'recall': recall_1
            })

            # ========================================
            # 手法2: Baseline_Passage
            # LSH: passage:, Ground Truth: passage:
            # ========================================
            candidates_2 = get_lsh_candidates(q_emb_passage, all_hashes, gen, limit)
            gt_2 = get_ground_truth(q_emb_passage, all_embeddings_flat, TOP_K)
            recall_2 = len(candidates_2 & gt_2) / TOP_K

            results.append({
                'method': '2_Baseline_Passage',
                'lsh_prefix': 'passage:',
                'gt_prefix': 'passage:',
                'candidate_limit': limit,
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'recall': recall_2
            })

            # ========================================
            # 手法3: Hybrid_PassageLSH_QueryCos
            # LSH: passage:, Ground Truth: query:
            # ========================================
            candidates_3 = get_lsh_candidates(q_emb_passage, all_hashes, gen, limit)
            gt_3 = get_ground_truth(q_emb_query, all_embeddings_flat, TOP_K)
            recall_3 = len(candidates_3 & gt_3) / TOP_K

            results.append({
                'method': '3_Hybrid_PassageLSH_QueryCos',
                'lsh_prefix': 'passage:',
                'gt_prefix': 'query:',
                'candidate_limit': limit,
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'recall': recall_3
            })

            # ========================================
            # 手法4: MeanDiff_Transform
            # LSH: query - mean_diff, Ground Truth: query:
            # ========================================
            candidates_4 = get_lsh_candidates(q_emb_transformed, all_hashes, gen, limit)
            gt_4 = get_ground_truth(q_emb_query, all_embeddings_flat, TOP_K)
            recall_4 = len(candidates_4 & gt_4) / TOP_K

            results.append({
                'method': '4_MeanDiff_Transform',
                'lsh_prefix': 'transformed',
                'gt_prefix': 'query:',
                'candidate_limit': limit,
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'recall': recall_4
            })

            # ========================================
            # 手法5: Dual_Merge
            # LSH: query: と passage: の両方から候補を取得してマージ
            # Ground Truth: query:
            # ========================================
            half_limit = limit // 2
            candidates_5a = get_lsh_candidates(q_emb_query, all_hashes, gen, half_limit)
            candidates_5b = get_lsh_candidates(q_emb_passage, all_hashes, gen, half_limit)
            candidates_5 = candidates_5a | candidates_5b
            gt_5 = get_ground_truth(q_emb_query, all_embeddings_flat, TOP_K)
            recall_5 = len(candidates_5 & gt_5) / TOP_K

            results.append({
                'method': '5_Dual_Merge',
                'lsh_prefix': 'query+passage',
                'gt_prefix': 'query:',
                'candidate_limit': limit,
                'query': query_text,
                'lang': lang,
                'query_type': query_type,
                'recall': recall_5
            })

    df_results = pd.DataFrame(results)

    # ========================================
    # 結果表示
    # ========================================
    print('\n' + '=' * 100)
    print('結果サマリー: 外部クエリ検索 Recall@10（30クエリ平均）')
    print('=' * 100)

    pivot = df_results.groupby(['method', 'candidate_limit'])['recall'].mean().unstack()

    print(f'\n{"手法":>35} | {"LSH":>12} | {"GT":>8} | {"1000件":>10} | {"2000件":>10} | {"5000件":>10}')
    print('-' * 100)

    method_order = [
        '1_Baseline_Query',
        '2_Baseline_Passage',
        '3_Hybrid_PassageLSH_QueryCos',
        '4_MeanDiff_Transform',
        '5_Dual_Merge'
    ]

    for method in method_order:
        if method in pivot.index:
            row = pivot.loc[method]
            method_info = df_results[df_results['method'] == method].iloc[0]
            lsh = method_info['lsh_prefix']
            gt = method_info['gt_prefix']
            print(f'{method:>35} | {lsh:>12} | {gt:>8} | {row[1000]:>10.1%} | {row[2000]:>10.1%} | {row[5000]:>10.1%}')

    # クエリタイプ別（候補2000件）
    print('\n' + '=' * 100)
    print('クエリタイプ別 Recall@10（候補2000件）')
    print('=' * 100)

    subset = df_results[df_results['candidate_limit'] == 2000]

    print(f'\n{"手法":>35} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 85)

    for method in method_order:
        if method in pivot.index:
            method_subset = subset[subset['method'] == method]

            ja_short = method_subset[(method_subset['lang'] == 'ja') & (method_subset['query_type'] == 'short')]['recall'].mean()
            ja_amb = method_subset[(method_subset['lang'] == 'ja') & (method_subset['query_type'] == 'ambiguous')]['recall'].mean()
            en_short = method_subset[(method_subset['lang'] == 'en') & (method_subset['query_type'] == 'short')]['recall'].mean()
            en_amb = method_subset[(method_subset['lang'] == 'en') & (method_subset['query_type'] == 'ambiguous')]['recall'].mean()

            print(f'{method:>35} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # ========================================
    # 詳細分析: Ground Truthの比較
    # ========================================
    print('\n' + '=' * 100)
    print('詳細分析: query: vs passage: のGround Truth比較')
    print('=' * 100)

    print('\n各クエリでのGround Truth（Top-10）の一致率:')

    gt_overlaps = []
    for i, (query_text, lang, query_type) in enumerate(search_queries):
        gt_query = get_ground_truth(query_embs_query[i], all_embeddings_flat, TOP_K)
        gt_passage = get_ground_truth(query_embs_passage[i], all_embeddings_flat, TOP_K)
        overlap = len(gt_query & gt_passage)
        gt_overlaps.append(overlap)

    print(f'   平均一致率: {np.mean(gt_overlaps):.1f}/10 件')
    print(f'   最小一致率: {np.min(gt_overlaps)}/10 件')
    print(f'   最大一致率: {np.max(gt_overlaps)}/10 件')

    # クエリタイプ別の一致率
    print('\n   クエリタイプ別:')
    for qtype in ['short', 'ambiguous']:
        overlaps = [gt_overlaps[i] for i, (_, _, qt) in enumerate(search_queries) if qt == qtype]
        print(f'     {qtype}: {np.mean(overlaps):.1f}/10 件')

    # ========================================
    # 「東京」クエリの詳細分析
    # ========================================
    print('\n' + '=' * 100)
    print('詳細分析: クエリ「東京」のGround TruthとLSH候補')
    print('=' * 100)

    test_idx = 0  # 「東京」
    q_emb_q = query_embs_query[test_idx]
    q_emb_p = query_embs_passage[test_idx]

    gt_q = get_ground_truth(q_emb_q, all_embeddings_flat, TOP_K)
    gt_p = get_ground_truth(q_emb_p, all_embeddings_flat, TOP_K)

    print(f'\n   Ground Truth (query:):   {sorted(gt_q)[:5]}...')
    print(f'   Ground Truth (passage:): {sorted(gt_p)[:5]}...')
    print(f'   共通件数: {len(gt_q & gt_p)}/10')

    # 各手法でのLSH候補
    print('\n   LSH候補（上位2000件）とGT(query:)の重複:')
    candidates_q = get_lsh_candidates(q_emb_q, all_hashes, gen, 2000)
    candidates_p = get_lsh_candidates(q_emb_p, all_hashes, gen, 2000)
    candidates_t = get_lsh_candidates(query_embs_transformed[test_idx], all_hashes, gen, 2000)

    print(f'     LSH(query:) ∩ GT(query:):      {len(candidates_q & gt_q)}/10')
    print(f'     LSH(passage:) ∩ GT(query:):    {len(candidates_p & gt_q)}/10')
    print(f'     LSH(transformed:) ∩ GT(query:): {len(candidates_t & gt_q)}/10')
    print(f'     LSH(passage:) ∩ GT(passage:):  {len(candidates_p & gt_p)}/10')

    # ========================================
    # 結論
    # ========================================
    print('\n' + '=' * 100)
    print('Phase 2 結論')
    print('=' * 100)

    best_method = pivot[2000].idxmax()
    best_recall = pivot[2000].max()

    print(f'''
   最良の手法（候補2000件）: {best_method}
   Recall@10: {best_recall:.1%}

   各手法の特徴:
   1. Baseline_Query:     従来手法。query:でLSH、query:でGT
   2. Baseline_Passage:   passage:でLSH、passage:でGT（LSHとGTが同じ空間）
   3. Hybrid:             passage:でLSH、query:でGT（LSH精度向上、GT維持）
   4. MeanDiff_Transform: クエリを変換してLSH（query→passage空間に近づける）
   5. Dual_Merge:         両方から候補を取得してマージ
    ''')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
