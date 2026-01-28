#!/usr/bin/env python3
"""
プレフィックスなしの埋め込みの影響を検証するスクリプト（1000件の小規模実験）

検証パターン:
- ドキュメント埋め込み: passage: / なし
- クエリ埋め込み: query: / passage: / なし

組み合わせ:
1. Doc=passage:, Query=query: (現行)
2. Doc=passage:, Query=passage:
3. Doc=passage:, Query=なし
4. Doc=なし, Query=query:
5. Doc=なし, Query=passage:
6. Doc=なし, Query=なし
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
    print('プレフィックスなしの埋め込み実験（1000件）')
    print('=' * 80)

    # DuckDBに接続
    print('\n1. データベースに接続...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb', read_only=True)

    # E5モデルを読み込み
    print('\n2. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # 1000件のテキストを取得（各データセットから250件）
    print('\n3. テキストを取得中...')
    texts = []
    metadata = []

    for dataset in ['body_ja', 'body_en', 'titles_ja', 'titles_en']:
        df = con.execute(f"""
            SELECT text, lang FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY RANDOM()
            LIMIT 250
        """).fetchdf()

        for _, row in df.iterrows():
            texts.append(row['text'])
            metadata.append({
                'dataset': dataset,
                'lang': row['lang'],
                'type': 'body' if 'body' in dataset else 'title'
            })

    print(f'   テキスト数: {len(texts)}')

    # ========================================
    # 各パターンで埋め込みを生成
    # ========================================
    print('\n4. 埋め込みを生成中...')

    # ドキュメント埋め込み
    print('   4.1 ドキュメント埋め込み (passage:)...')
    doc_embs_passage = model.encode(
        [f'passage: {t}' for t in texts],
        normalize_embeddings=False,
        show_progress_bar=True
    ).astype(np.float32)

    print('   4.2 ドキュメント埋め込み (なし)...')
    doc_embs_none = model.encode(
        texts,
        normalize_embeddings=False,
        show_progress_bar=True
    ).astype(np.float32)

    # 検索クエリ（30件）
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
        ('最近話題になっている技術革新について', 'ja', 'ambiguous'),
        ('日本の伝統的な文化や芸術', 'ja', 'ambiguous'),
        ('環境に優しい持続可能な社会', 'ja', 'ambiguous'),
        ('健康的な生活を送るために', 'ja', 'ambiguous'),
        ('世界の政治情勢や国際関係', 'ja', 'ambiguous'),
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

    # クエリ埋め込み（3パターン）
    print('   4.3 クエリ埋め込み (query:)...')
    query_embs_query = model.encode(
        [f'query: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    print('   4.4 クエリ埋め込み (passage:)...')
    query_embs_passage = model.encode(
        [f'passage: {t}' for t in query_texts],
        normalize_embeddings=False
    ).astype(np.float32)

    print('   4.5 クエリ埋め込み (なし)...')
    query_embs_none = model.encode(
        query_texts,
        normalize_embeddings=False
    ).astype(np.float32)

    print('   完了')

    # ========================================
    # 埋め込みパターンの組み合わせを定義
    # ========================================
    patterns = {
        'Doc=passage, Query=query': (doc_embs_passage, query_embs_query),
        'Doc=passage, Query=passage': (doc_embs_passage, query_embs_passage),
        'Doc=passage, Query=none': (doc_embs_passage, query_embs_none),
        'Doc=none, Query=query': (doc_embs_none, query_embs_query),
        'Doc=none, Query=passage': (doc_embs_none, query_embs_passage),
        'Doc=none, Query=none': (doc_embs_none, query_embs_none),
    }

    # ========================================
    # LSH超平面を生成（各パターンごと）
    # ========================================
    print('\n5. 各パターンの評価を実行中...')
    print('=' * 100)

    candidate_limits = [50, 100, 200]  # 1000件中の候補数
    top_k = 10
    results = []

    for pattern_name, (doc_embs, query_embs) in tqdm(patterns.items(), desc='パターン'):
        # DataSampled超平面を生成（このドキュメント埋め込みから）
        rng = np.random.default_rng(42)
        hyperplanes = []
        for _ in range(128):
            i, j = rng.choice(len(doc_embs), 2, replace=False)
            diff = doc_embs[i] - doc_embs[j]
            diff = diff / np.linalg.norm(diff)
            hyperplanes.append(diff)
        hyperplanes = np.array(hyperplanes, dtype=np.float32)

        # SimHashを計算
        gen = SimHashGenerator(dim=1024, hash_bits=128, seed=0, strategy='random')
        gen.hyperplanes = hyperplanes
        doc_hashes = gen.hash_batch(doc_embs)

        # 各クエリで評価
        for i, (query_text, lang, query_type) in enumerate(search_queries):
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
                    'pattern': pattern_name,
                    'candidate_limit': limit,
                    'query': query_text,
                    'lang': lang,
                    'query_type': query_type,
                    'recall': recall
                })

    df_results = pd.DataFrame(results)

    # ========================================
    # 結果表示
    # ========================================
    print('\n' + '=' * 100)
    print('結果: プレフィックスパターン別 Recall@10（25クエリ平均）')
    print('=' * 100)

    pivot = df_results.groupby(['pattern', 'candidate_limit'])['recall'].mean().unstack()

    print(f'\n{"パターン":>35} | {"50件":>10} | {"100件":>10} | {"200件":>10}')
    print('-' * 75)

    pattern_order = [
        'Doc=passage, Query=query',
        'Doc=passage, Query=passage',
        'Doc=passage, Query=none',
        'Doc=none, Query=query',
        'Doc=none, Query=passage',
        'Doc=none, Query=none',
    ]

    for pattern in pattern_order:
        if pattern in pivot.index:
            row = pivot.loc[pattern]
            print(f'{pattern:>35} | {row[50]:>10.1%} | {row[100]:>10.1%} | {row[200]:>10.1%}')

    # クエリタイプ別（候補100件）
    print('\n' + '=' * 100)
    print('クエリタイプ別 Recall@10（候補100件）')
    print('=' * 100)

    subset = df_results[df_results['candidate_limit'] == 100]

    print(f'\n{"パターン":>35} | {"JA短文":>10} | {"JA曖昧":>10} | {"EN短文":>10} | {"EN曖昧":>10}')
    print('-' * 85)

    for pattern in pattern_order:
        if pattern in pivot.index:
            pattern_subset = subset[subset['pattern'] == pattern]

            ja_short = pattern_subset[(pattern_subset['lang'] == 'ja') & (pattern_subset['query_type'] == 'short')]['recall'].mean()
            ja_amb = pattern_subset[(pattern_subset['lang'] == 'ja') & (pattern_subset['query_type'] == 'ambiguous')]['recall'].mean()
            en_short = pattern_subset[(pattern_subset['lang'] == 'en') & (pattern_subset['query_type'] == 'short')]['recall'].mean()
            en_amb = pattern_subset[(pattern_subset['lang'] == 'en') & (pattern_subset['query_type'] == 'ambiguous')]['recall'].mean()

            print(f'{pattern:>35} | {ja_short:>10.1%} | {ja_amb:>10.1%} | {en_short:>10.1%} | {en_amb:>10.1%}')

    # ========================================
    # ベクトル空間の分析
    # ========================================
    print('\n' + '=' * 100)
    print('ベクトル空間の分析: ドキュメントとクエリの類似度')
    print('=' * 100)

    # 同じテキストのドキュメントとクエリの類似度
    print('\n■ 同じテキストのDoc-Query類似度（サンプル10件）:')
    print(f'{"テキスト":>30} | {"D=pass,Q=query":>15} | {"D=pass,Q=none":>15} | {"D=none,Q=query":>15} | {"D=none,Q=none":>15}')
    print('-' * 100)

    for i in range(10):
        text = texts[i][:25] + '...' if len(texts[i]) > 25 else texts[i]

        sim_pp_qq = np.dot(doc_embs_passage[i], query_embs_query[i]) / (norm(doc_embs_passage[i]) * norm(query_embs_query[i]))
        sim_pp_qn = np.dot(doc_embs_passage[i], query_embs_none[i]) / (norm(doc_embs_passage[i]) * norm(query_embs_none[i]))
        sim_pn_qq = np.dot(doc_embs_none[i], query_embs_query[i]) / (norm(doc_embs_none[i]) * norm(query_embs_query[i]))
        sim_pn_qn = np.dot(doc_embs_none[i], query_embs_none[i]) / (norm(doc_embs_none[i]) * norm(query_embs_none[i]))

        print(f'{text:>30} | {sim_pp_qq:>15.4f} | {sim_pp_qn:>15.4f} | {sim_pn_qq:>15.4f} | {sim_pn_qn:>15.4f}')

    # 平均類似度
    print('\n■ 平均類似度（全1000件）:')
    for doc_name, doc_embs in [('passage:', doc_embs_passage), ('none', doc_embs_none)]:
        for query_name, q_embs in [('query:', query_embs_query), ('passage:', query_embs_passage), ('none', query_embs_none)]:
            # 対応するテキストがないので、ランダムなペアの類似度を計算
            sims = []
            for i in range(min(100, len(texts))):
                for j in range(min(len(query_texts), 25)):
                    # クエリとドキュメントは異なるテキストなので、類似度は低くなる
                    pass
            # 同じテキストの場合の類似度（最初の25件がクエリと重複する場合を想定）
            # 実際にはクエリとドキュメントは異なるので、ここは省略

    # ========================================
    # ドキュメント間の類似度分析
    # ========================================
    print('\n■ ドキュメント間の類似度分布（ランダム100ペア）:')
    rng = np.random.default_rng(42)
    pairs = [(rng.integers(0, len(texts)), rng.integers(0, len(texts))) for _ in range(100)]

    for doc_name, doc_embs in [('Doc=passage:', doc_embs_passage), ('Doc=none', doc_embs_none)]:
        sims = []
        for i, j in pairs:
            if i != j:
                sim = np.dot(doc_embs[i], doc_embs[j]) / (norm(doc_embs[i]) * norm(doc_embs[j]))
                sims.append(sim)
        print(f'  {doc_name}: 平均={np.mean(sims):.4f}, 標準偏差={np.std(sims):.4f}')

    # ========================================
    # 結論
    # ========================================
    print('\n' + '=' * 100)
    print('結論')
    print('=' * 100)

    # 最良パターンを特定
    best_pattern = pivot[100].idxmax()
    best_recall = pivot[100].max()

    # 現行パターンとの比較
    current_pattern = 'Doc=passage, Query=query'
    current_recall = pivot.loc[current_pattern, 100] if current_pattern in pivot.index else 0

    none_pattern = 'Doc=none, Query=none'
    none_recall = pivot.loc[none_pattern, 100] if none_pattern in pivot.index else 0

    print(f'''
   最良パターン（候補100件）: {best_pattern}
   Recall@10: {best_recall:.1%}

   現行パターン（Doc=passage, Query=query）: {current_recall:.1%}
   プレフィックスなし（Doc=none, Query=none）: {none_recall:.1%}

   改善幅: {none_recall - current_recall:+.1%}
    ''')

    if none_recall > current_recall:
        print('   → プレフィックスなしの方が良い結果！全データでの再インデックスを検討すべき')
    else:
        print('   → プレフィックスありの方が良い（または同等）')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
