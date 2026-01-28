#!/usr/bin/env python3
"""
Query-Passage差分を考慮した超平面実験スクリプト
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from numpy.linalg import norm
import pandas as pd
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# src/lsh.pyをインポート
sys.path.insert(0, '/home/terapyon/dev/vibe-coding/lsh-cascade-poc')
from src.lsh import SimHashGenerator, hamming_distance


def generate_query_passage_diff_hyperplanes(
    texts: list,
    model: SentenceTransformer,
    num_hyperplanes: int,
    seed: int = 42
) -> np.ndarray:
    """
    同じテキストの query: と passage: 埋め込みの差分を超平面として生成
    """
    rng = np.random.default_rng(seed)

    # テキストをサンプリング
    selected_indices = rng.choice(len(texts), min(num_hyperplanes, len(texts)), replace=False)
    selected_texts = [texts[i] for i in selected_indices]

    # query と passage の埋め込みを生成
    query_texts = [f'query: {t}' for t in selected_texts]
    passage_texts = [f'passage: {t}' for t in selected_texts]

    print(f'  query/passage埋め込みを生成中 ({len(selected_texts)}件)...')
    query_embs = model.encode(query_texts, normalize_embeddings=False, show_progress_bar=False)
    passage_embs = model.encode(passage_texts, normalize_embeddings=False, show_progress_bar=False)

    # 差分ベクトルを計算（query - passage）
    diff_vectors = query_embs - passage_embs

    # 正規化
    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
    hyperplanes = diff_vectors / norms

    return hyperplanes.astype(np.float32)


def generate_data_sampled_hyperplanes(
    embeddings: np.ndarray,
    num_hyperplanes: int,
    seed: int = 42
) -> np.ndarray:
    """
    データの差分ベクトルから超平面を生成（従来手法）
    """
    rng = np.random.default_rng(seed)
    hyperplanes = []

    for _ in range(num_hyperplanes):
        i, j = rng.choice(len(embeddings), 2, replace=False)
        diff = embeddings[i] - embeddings[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes.append(diff)

    return np.array(hyperplanes, dtype=np.float32)


def evaluate_recall(
    query_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    hyperplanes: np.ndarray,
    candidate_limit: int = 2000,
    top_k: int = 10
) -> list:
    """
    クエリに対するRecall@kを評価
    """
    # SimHashGeneratorを作成して超平面を設定
    gen = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
    gen.hyperplanes = hyperplanes

    # 全ドキュメントのハッシュを計算
    all_hashes = gen.hash_batch(all_embeddings)

    # クエリのハッシュを計算
    query_hashes = gen.hash_batch(query_embeddings)

    recalls = []

    for i in range(len(query_embeddings)):
        query_emb = query_embeddings[i]
        query_hash = query_hashes[i]

        # Ground Truth（コサイン類似度Top-k）
        cos_sims = (all_embeddings @ query_emb) / (norm(all_embeddings, axis=1) * norm(query_emb))
        gt_indices = set(np.argsort(cos_sims)[::-1][:top_k])

        # LSH候補（ハミング距離Top-candidate_limit）
        distances = [(j, hamming_distance(h, query_hash)) for j, h in enumerate(all_hashes)]
        distances.sort(key=lambda x: x[1])
        candidates = set(idx for idx, _ in distances[:candidate_limit])

        recall = len(gt_indices & candidates) / top_k
        recalls.append(recall)

    return recalls


def main():
    print('=' * 80)
    print('Query-Passage差分を考慮した超平面実験')
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
    all_texts = {}
    all_ids = {}

    for dataset in tqdm(datasets, desc='   データセット'):
        df = con.execute(f"""
            SELECT id, text, embedding
            FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()

        embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
        all_embeddings[dataset] = embeddings
        all_texts[dataset] = df['text'].values
        all_ids[dataset] = df['id'].values

    # サンプルテキストを準備（body_jaから300件）
    print('\n4. サンプルデータを準備...')
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_texts['body_ja']), 300, replace=False)
    sample_texts = [all_texts['body_ja'][i] for i in sample_indices]
    sample_embeddings = all_embeddings['body_ja'][sample_indices]
    print(f'   サンプルテキスト数: {len(sample_texts)}')

    # 超平面を生成
    print('\n5. 超平面を生成中...')

    print('   5.1 DataSampled超平面...')
    hp_datasampled = generate_data_sampled_hyperplanes(sample_embeddings, 128, seed=42)

    print('   5.2 QueryPassage差分超平面...')
    hp_qp_diff = generate_query_passage_diff_hyperplanes(sample_texts, model, 128, seed=42)

    print('   5.3 ランダム超平面...')
    rng = np.random.default_rng(42)
    hp_random = rng.standard_normal((128, 1024)).astype(np.float32)
    hp_random = hp_random / np.linalg.norm(hp_random, axis=1, keepdims=True)

    # パターンを定義
    patterns = {
        'Baseline_DS': (128, 0, 0),
        'QP32': (96, 32, 0),
        'QP64': (64, 64, 0),
        'QP96': (32, 96, 0),
        'QP128': (0, 128, 0),
        'QP64_R32': (32, 64, 32),
    }

    hyperplanes_patterns = {}
    for name, (num_ds, num_qp, num_r) in patterns.items():
        parts = []
        if num_ds > 0:
            parts.append(hp_datasampled[:num_ds])
        if num_qp > 0:
            parts.append(hp_qp_diff[:num_qp])
        if num_r > 0:
            parts.append(hp_random[:num_r])
        hyperplanes_patterns[name] = np.vstack(parts)

    print('   完了')

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
    print('\n6. クエリ埋め込みを生成中...')

    query_texts_with_query_prefix = [f'query: {text}' for text in query_texts]
    query_embeddings_query = model.encode(query_texts_with_query_prefix, normalize_embeddings=False)
    query_embeddings_query = query_embeddings_query.astype(np.float32)

    query_texts_with_passage_prefix = [f'passage: {text}' for text in query_texts]
    query_embeddings_passage = model.encode(query_texts_with_passage_prefix, normalize_embeddings=False)
    query_embeddings_passage = query_embeddings_passage.astype(np.float32)

    print('   完了')

    # 全データを統合
    print('\n7. 全データを統合中...')
    all_embeddings_flat = np.vstack([all_embeddings[d] for d in datasets])
    all_datasets_flat = []
    for dataset in datasets:
        all_datasets_flat.extend([dataset] * len(all_embeddings[dataset]))
    print(f'   統合完了: {all_embeddings_flat.shape}')

    # 評価実行
    print('\n8. 評価を実行中...')
    print('=' * 100)

    candidate_limits = [1000, 2000, 5000]
    results = []

    for pattern_name, hyperplanes in tqdm(hyperplanes_patterns.items(), desc='   パターン'):
        for limit in candidate_limits:
            # アプローチ1: query:プレフィックス
            recalls_query = evaluate_recall(
                query_embeddings_query,
                all_embeddings_flat,
                hyperplanes,
                candidate_limit=limit
            )

            for i, (query_text, lang, query_type) in enumerate(search_queries):
                results.append({
                    'pattern': pattern_name,
                    'query_prefix': 'query:',
                    'candidate_limit': limit,
                    'query': query_text,
                    'lang': lang,
                    'query_type': query_type,
                    'recall': recalls_query[i]
                })

            # アプローチ2: passage:プレフィックス（Baseline_DSのみ）
            if pattern_name == 'Baseline_DS':
                recalls_passage = evaluate_recall(
                    query_embeddings_passage,
                    all_embeddings_flat,
                    hyperplanes,
                    candidate_limit=limit
                )

                for i, (query_text, lang, query_type) in enumerate(search_queries):
                    results.append({
                        'pattern': 'Baseline_DS_PassageQuery',
                        'query_prefix': 'passage:',
                        'candidate_limit': limit,
                        'query': query_text,
                        'lang': lang,
                        'query_type': query_type,
                        'recall': recalls_passage[i]
                    })

    df_results = pd.DataFrame(results)

    # 結果表示
    print('\n' + '=' * 100)
    print('外部クエリ検索 Recall@10（30クエリ平均）')
    print('=' * 100)

    pivot = df_results.groupby(['pattern', 'candidate_limit'])['recall'].mean().unstack()

    pattern_order = ['Baseline_DS', 'Baseline_DS_PassageQuery', 'QP32', 'QP64', 'QP96', 'QP128', 'QP64_R32']
    pivot = pivot.reindex([p for p in pattern_order if p in pivot.index])

    print(f'\n{"パターン":>30} | {"クエリ接頭辞":>12} | {"1000件":>10} | {"2000件":>10} | {"5000件":>10}')
    print('-' * 90)

    for pattern in pivot.index:
        row = pivot.loc[pattern]
        prefix = df_results[df_results['pattern'] == pattern]['query_prefix'].iloc[0]
        print(f'{pattern:>30} | {prefix:>12} | {row[1000]:>10.1%} | {row[2000]:>10.1%} | {row[5000]:>10.1%}')

    # クエリタイプ別
    print('\n' + '=' * 100)
    print('クエリタイプ別 Recall@10（候補2000件）')
    print('=' * 100)

    subset = df_results[df_results['candidate_limit'] == 2000]

    print(f'\n{"パターン":>30} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 85)

    for pattern in [p for p in pattern_order if p in pivot.index]:
        pattern_subset = subset[subset['pattern'] == pattern]

        ja_short = pattern_subset[(pattern_subset['lang'] == 'ja') & (pattern_subset['query_type'] == 'short')]['recall'].mean()
        ja_amb = pattern_subset[(pattern_subset['lang'] == 'ja') & (pattern_subset['query_type'] == 'ambiguous')]['recall'].mean()
        en_short = pattern_subset[(pattern_subset['lang'] == 'en') & (pattern_subset['query_type'] == 'short')]['recall'].mean()
        en_amb = pattern_subset[(pattern_subset['lang'] == 'en') & (pattern_subset['query_type'] == 'ambiguous')]['recall'].mean()

        print(f'{pattern:>30} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # 「東京」クエリの詳細分析
    print('\n' + '=' * 100)
    print('クエリ「東京」のGround Truth分析')
    print('=' * 100)

    test_idx = 0
    query_emb_q = query_embeddings_query[test_idx]
    cos_sims_q = (all_embeddings_flat @ query_emb_q) / (norm(all_embeddings_flat, axis=1) * norm(query_emb_q))
    gt_indices_q = np.argsort(cos_sims_q)[::-1][:10]

    query_emb_p = query_embeddings_passage[test_idx]
    cos_sims_p = (all_embeddings_flat @ query_emb_p) / (norm(all_embeddings_flat, axis=1) * norm(query_emb_p))
    gt_indices_p = np.argsort(cos_sims_p)[::-1][:10]

    print('\nGround Truth比較:')
    print(f'  query:プレフィックス時のTop-10: {list(gt_indices_q)}')
    print(f'  passage:プレフィックス時のTop-10: {list(gt_indices_p)}')
    print(f'  共通件数: {len(set(gt_indices_q) & set(gt_indices_p))}/10')

    # ハミング距離の詳細
    print('\n各パターンでのGT Top-10のハミング距離 (query:プレフィックス使用時):')
    print(f'{"GT#":>5} | {"cos_sim":>8} | {"dataset":>12} | {"Baseline":>10} | {"QP64":>10} | {"QP128":>10}')
    print('-' * 70)

    for rank, idx in enumerate(gt_indices_q):
        print(f'{rank+1:>5} | {cos_sims_q[idx]:>8.4f} | {all_datasets_flat[idx]:>12} | ', end='')

        for pattern in ['Baseline_DS', 'QP64', 'QP128']:
            gen = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
            gen.hyperplanes = hyperplanes_patterns[pattern]

            doc_hash = gen.hash_batch(all_embeddings_flat[idx:idx+1])[0]
            query_hash = gen.hash_batch(query_emb_q.reshape(1, -1))[0]
            ham_dist = hamming_distance(doc_hash, query_hash)
            print(f'{ham_dist:>10} | ', end='')
        print()

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
