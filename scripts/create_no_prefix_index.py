#!/usr/bin/env python3
"""
40万件のデータに対してプレフィックスなしの埋め込みを生成し、
新しいDuckDBテーブルに保存するスクリプト
"""

import sys
sys.path.insert(0, '/home/terapyon/dev/vibe-coding/lsh-cascade-poc')

import numpy as np
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def main():
    print('=' * 80)
    print('プレフィックスなしの埋め込み生成（40万件）')
    print('=' * 80)

    # DuckDBに接続（読み書き可能）
    print('\n1. データベースに接続...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb')

    # E5モデルを読み込み
    print('\n2. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # 新しいテーブルを作成（既存の場合は削除）
    print('\n3. 新しいテーブルを作成...')
    con.execute("DROP TABLE IF EXISTS documents_no_prefix")
    con.execute("""
        CREATE TABLE documents_no_prefix (
            id INTEGER PRIMARY KEY,
            dataset VARCHAR,
            text VARCHAR,
            lang VARCHAR,
            embedding FLOAT[1024]
        )
    """)
    print('   テーブル documents_no_prefix を作成しました')

    # データセットごとに処理
    print('\n4. 埋め込みを生成中...')
    datasets = ['body_en', 'body_ja', 'titles_en', 'titles_ja']
    batch_size = 64
    total_processed = 0

    for dataset in datasets:
        print(f'\n   処理中: {dataset}')

        # テキストを取得
        df = con.execute(f"""
            SELECT id, text, lang FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY id
        """).fetchdf()

        texts = df['text'].tolist()
        ids = df['id'].tolist()
        langs = df['lang'].tolist()

        print(f'   テキスト数: {len(texts)}')

        # バッチ処理で埋め込み生成
        for i in tqdm(range(0, len(texts), batch_size), desc=f'   {dataset}'):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_langs = langs[i:i+batch_size]

            # プレフィックスなしで埋め込み生成
            embeddings = model.encode(
                batch_texts,  # プレフィックスなし
                normalize_embeddings=False,
                show_progress_bar=False
            ).astype(np.float32)

            # DuckDBに挿入
            for j, (doc_id, text, lang, emb) in enumerate(zip(batch_ids, batch_texts, batch_langs, embeddings)):
                con.execute("""
                    INSERT INTO documents_no_prefix (id, dataset, text, lang, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, [int(doc_id), dataset, text, lang, emb.tolist()])

            total_processed += len(batch_texts)

        # コミット
        con.commit()

    print(f'\n   合計 {total_processed:,} 件を処理しました')

    # 確認
    print('\n5. テーブルの確認...')
    result = con.execute("""
        SELECT dataset, COUNT(*) as cnt
        FROM documents_no_prefix
        GROUP BY dataset
        ORDER BY dataset
    """).fetchall()

    print('   データセット別件数:')
    for dataset, cnt in result:
        print(f'     {dataset}: {cnt:,}件')

    con.close()
    print('\n完了')


if __name__ == '__main__':
    main()
