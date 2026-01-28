#!/usr/bin/env python3
"""
プレフィックスなしインデックスでのLSH検索を評価するスクリプト

比較:
1. Doc=passage (元), Query=query (元) - ベースライン
2. Doc=none (新), Query=none (新) - プレフィックスなし
3. Doc=none (新), Query=query - ハイブリッド
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


def evaluate_pattern(
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    gen: SimHashGenerator,
    doc_hashes: list,
    candidate_limits: list,
    top_k: int = 10
) -> dict:
    """各クエリに対してRecall@kを計算"""
    results = []

    for i in range(len(query_embs)):
        query_emb = query_embs[i]

        # Ground Truth（コサイン類似度Top-k）
        cos_sims = (doc_embs @ query_emb) / (norm(doc_embs, axis=1) * norm(query_emb))
        gt_indices = set(np.argsort(cos_sims)[::-1][:top_k])

        # LSH候補
        query_hash = gen.hash_batch(query_emb.reshape(1, -1))[0]

        for limit in candidate_limits:
            distances = [(j, hamming_distance(h, query_hash)) for j, h in enumerate(doc_hashes)]
            distances.sort(key=lambda x: x[1])
            candidates = set(idx for idx, _ in distances[:limit])

            recall = len(gt_indices & candidates) / top_k

            results.append({
                'query_idx': i,
                'candidate_limit': limit,
                'recall': recall
            })

    return results


def main():
    print('=' * 80)
    print('プレフィックスなしインデックスのLSH検索評価')
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

    # 元のpassage:プレフィックス埋め込み
    embeddings_passage = {}
    for dataset in tqdm(datasets, desc='   passage:埋め込み'):
        df = con.execute(f"""
            SELECT embedding FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()
        embeddings_passage[dataset] = np.array(df['embedding'].tolist(), dtype=np.float32)

    # 新しいプレフィックスなし埋め込み
    embeddings_none = {}
    for dataset in tqdm(datasets, desc='   none埋め込み'):
        df = con.execute(f"""
            SELECT embedding FROM documents_no_prefix
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()
        embeddings_none[dataset] = np.array(df['embedding'].tolist(), dtype=np.float32)

    # 全データを統合
    all_passage = np.vstack([embeddings_passage[d] for d in datasets])
    all_none = np.vstack([embeddings_none[d] for d in datasets])
    print(f'   統合完了: passage={all_passage.shape}, none={all_none.shape}')

    # ========================================
    # 超平面を生成
    # ========================================
    print('\n4. 超平面を準備中...')

    rng = np.random.default_rng(42)

    # passage:埋め込みからの超平面（ベースライン）
    sample_indices = rng.choice(len(embeddings_passage['body_ja']), 300, replace=False)
    sample_passage = embeddings_passage['body_ja'][sample_indices]

    hyperplanes_passage = []
    for _ in range(128):
        i, j = rng.choice(len(sample_passage), 2, replace=False)
        diff = sample_passage[i] - sample_passage[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes_passage.append(diff)
    hyperplanes_passage = np.array(hyperplanes_passage, dtype=np.float32)

    # none埋め込みからの超平面（新規）
    sample_none = embeddings_none['body_ja'][sample_indices]

    hyperplanes_none = []
    for _ in range(128):
        i, j = rng.choice(len(sample_none), 2, replace=False)
        diff = sample_none[i] - sample_none[j]
        diff = diff / np.linalg.norm(diff)
        hyperplanes_none.append(diff)
    hyperplanes_none = np.array(hyperplanes_none, dtype=np.float32)

    # SimHashを計算
    print('\n5. SimHashを計算中...')
    gen_passage = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
    gen_passage.hyperplanes = hyperplanes_passage
    hashes_passage = gen_passage.hash_batch(all_passage)

    gen_none = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
    gen_none.hyperplanes = hyperplanes_none
    hashes_none = gen_none.hash_batch(all_none)

    print(f'   完了: passage={len(hashes_passage)}, none={len(hashes_none)}')

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
    query_embs_query = model.encode(
        [f'query: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    query_embs_passage = model.encode(
        [f'passage: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    query_embs_none = model.encode(
        query_texts,
        normalize_embeddings=False
    ).astype(np.float32)
    print('   完了')

    # ========================================
    # 評価パターン
    # ========================================
    print('\n' + '=' * 80)
    print('7. 各パターンで評価中...')
    print('=' * 80)

    candidate_limits = [500, 1000, 2000, 5000, 10000, 20000]
    all_results = []

    # パターン1: ベースライン（Doc=passage, Query=query, HP=passage）
    print('\n   [1/4] Doc=passage, Query=query, HP=passage（ベースライン）')
    results = evaluate_pattern(all_passage, query_embs_query, gen_passage, hashes_passage, candidate_limits)
    for r in results:
        r['pattern'] = 'Baseline (passage/query)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    # パターン2: プレフィックスなし（Doc=none, Query=none, HP=none）
    print('   [2/4] Doc=none, Query=none, HP=none（プレフィックスなし）')
    results = evaluate_pattern(all_none, query_embs_none, gen_none, hashes_none, candidate_limits)
    for r in results:
        r['pattern'] = 'No Prefix (none/none)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    # パターン3: Hybrid（Doc=none, Query=query, HP=none）
    print('   [3/4] Doc=none, Query=query, HP=none（Hybrid）')
    # LSHにはnone埋め込みを使い、コサイン計算時にquery埋め込みを使う
    results = []
    for i in range(len(query_embs_query)):
        q_emb_none = query_embs_none[i]
        q_emb_query = query_embs_query[i]

        # Ground Truth: query埋め込みで計算
        cos_sims_query = (all_none @ q_emb_query) / (norm(all_none, axis=1) * norm(q_emb_query))
        gt_indices = set(np.argsort(cos_sims_query)[::-1][:10])

        # LSH候補: none埋め込みで計算
        query_hash = gen_none.hash_batch(q_emb_none.reshape(1, -1))[0]

        for limit in candidate_limits:
            distances = [(j, hamming_distance(h, query_hash)) for j, h in enumerate(hashes_none)]
            distances.sort(key=lambda x: x[1])
            candidates = set(idx for idx, _ in distances[:limit])

            recall = len(gt_indices & candidates) / 10

            results.append({
                'pattern': 'Hybrid (none/query)',
                'query_idx': i,
                'query': query_texts[i],
                'lang': search_queries[i][1],
                'query_type': search_queries[i][2],
                'candidate_limit': limit,
                'recall': recall
            })
    all_results.extend(results)

    # パターン4: 参考 - 元のインデックスでpassage:クエリ（Phase1で最良だったパターン）
    print('   [4/4] Doc=passage, Query=passage, HP=passage（Phase1最良）')
    results = evaluate_pattern(all_passage, query_embs_passage, gen_passage, hashes_passage, candidate_limits)
    for r in results:
        r['pattern'] = 'Phase1 Best (passage/passage)'
        r['query'] = query_texts[r['query_idx']]
        r['lang'] = search_queries[r['query_idx']][1]
        r['query_type'] = search_queries[r['query_idx']][2]
    all_results.extend(results)

    df_results = pd.DataFrame(all_results)

    # ========================================
    # 結果表示
    # ========================================
    print('\n' + '=' * 100)
    print('結果: パターン別 Recall@10（30クエリ平均）')
    print('=' * 100)

    pattern_order = [
        'Baseline (passage/query)',
        'No Prefix (none/none)',
        'Hybrid (none/query)',
        'Phase1 Best (passage/passage)',
    ]

    print(f'\n{"パターン":>35} | {"500":>8} | {"1000":>8} | {"2000":>8} | {"5000":>8} | {"10000":>8} | {"20000":>8}')
    print('-' * 110)

    for pattern in pattern_order:
        subset = df_results[df_results['pattern'] == pattern]
        pivot = subset.groupby('candidate_limit')['recall'].mean()
        row = [f'{pivot.get(l, 0):.1%}' for l in candidate_limits]
        print(f'{pattern:>35} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} | {row[5]:>8}')

    # クエリタイプ別（候補2000件）
    print('\n' + '=' * 100)
    print('クエリタイプ別 Recall@10（候補2000件）')
    print('=' * 100)

    print(f'\n{"パターン":>35} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 85)

    for pattern in pattern_order:
        subset = df_results[(df_results['pattern'] == pattern) & (df_results['candidate_limit'] == 2000)]

        ja_short = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'short')]['recall'].mean()
        ja_amb = subset[(subset['lang'] == 'ja') & (subset['query_type'] == 'ambiguous')]['recall'].mean()
        en_short = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'short')]['recall'].mean()
        en_amb = subset[(subset['lang'] == 'en') & (subset['query_type'] == 'ambiguous')]['recall'].mean()

        print(f'{pattern:>35} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # ========================================
    # 詳細分析: 改善度
    # ========================================
    print('\n' + '=' * 100)
    print('改善度分析（候補2000件）')
    print('=' * 100)

    baseline_recalls = df_results[(df_results['pattern'] == 'Baseline (passage/query)') &
                                   (df_results['candidate_limit'] == 2000)].set_index('query_idx')['recall']
    noprefix_recalls = df_results[(df_results['pattern'] == 'No Prefix (none/none)') &
                                   (df_results['candidate_limit'] == 2000)].set_index('query_idx')['recall']

    print('\n■ 個別クエリの改善度（Top 5）')
    improvements = (noprefix_recalls - baseline_recalls).sort_values(ascending=False)
    for idx in improvements.head(5).index:
        query = query_texts[idx][:30]
        print(f'  「{query}...」: {baseline_recalls[idx]:.0%} → {noprefix_recalls[idx]:.0%} ({improvements[idx]:+.0%})')

    print('\n■ 個別クエリの悪化（Bottom 5）')
    for idx in improvements.tail(5).index:
        query = query_texts[idx][:30]
        print(f'  「{query}...」: {baseline_recalls[idx]:.0%} → {noprefix_recalls[idx]:.0%} ({improvements[idx]:+.0%})')

    # ========================================
    # 総括
    # ========================================
    print('\n' + '=' * 100)
    print('総括')
    print('=' * 100)

    baseline_mean = baseline_recalls.mean()
    noprefix_mean = noprefix_recalls.mean()
    hybrid_mean = df_results[(df_results['pattern'] == 'Hybrid (none/query)') &
                              (df_results['candidate_limit'] == 2000)]['recall'].mean()
    phase1_mean = df_results[(df_results['pattern'] == 'Phase1 Best (passage/passage)') &
                              (df_results['candidate_limit'] == 2000)]['recall'].mean()

    print(f'''
   候補2000件でのRecall@10平均:

   Baseline (passage/query):        {baseline_mean:.1%}
   No Prefix (none/none):           {noprefix_mean:.1%} ({noprefix_mean - baseline_mean:+.1%})
   Hybrid (none/query):             {hybrid_mean:.1%} ({hybrid_mean - baseline_mean:+.1%})
   Phase1 Best (passage/passage):   {phase1_mean:.1%} ({phase1_mean - baseline_mean:+.1%})
    ''')

    if noprefix_mean > baseline_mean:
        print('   → プレフィックスなしで改善が見られました！')
    elif noprefix_mean == baseline_mean:
        print('   → プレフィックスなしでも同等の結果')
    else:
        print('   → プレフィックスなしでは悪化しました')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
