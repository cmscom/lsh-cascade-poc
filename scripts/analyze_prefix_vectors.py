#!/usr/bin/env python3
"""
passage: と query: プレフィックスによるベクトル表現の変化を詳細に分析するスクリプト
"""

import sys
sys.path.insert(0, '/home/terapyon/dev/vibe-coding/lsh-cascade-poc')

import numpy as np
from numpy.linalg import norm
import pandas as pd
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用

from src.lsh import SimHashGenerator, hamming_distance


def main():
    print('=' * 80)
    print('Phase 1: passage: と query: プレフィックスによるベクトル変化の分析')
    print('=' * 80)

    # E5モデルを読み込み
    print('\n1. E5モデルを読み込み中...')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print('   完了')

    # DuckDBに接続してサンプルテキストを取得
    print('\n2. サンプルテキストを取得中...')
    con = duckdb.connect('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/experiment_400k.duckdb', read_only=True)

    # 各データセットから50件ずつサンプル（計200件）
    sample_texts = []
    sample_metadata = []

    for dataset in ['body_ja', 'body_en', 'titles_ja', 'titles_en']:
        df = con.execute(f"""
            SELECT text, lang FROM documents
            WHERE dataset = '{dataset}'
            ORDER BY RANDOM()
            LIMIT 50
        """).fetchdf()

        for _, row in df.iterrows():
            sample_texts.append(row['text'])
            sample_metadata.append({
                'dataset': dataset,
                'lang': row['lang'],
                'length': len(row['text']),
                'type': 'body' if 'body' in dataset else 'title'
            })

    print(f'   サンプルテキスト数: {len(sample_texts)}')

    # ========================================
    # 分析1: 基本的な統計
    # ========================================
    print('\n' + '=' * 80)
    print('分析1: 基本的な統計（passage vs query）')
    print('=' * 80)

    print('\n   埋め込み生成中...')
    passage_texts = [f'passage: {t}' for t in sample_texts]
    query_texts = [f'query: {t}' for t in sample_texts]

    passage_embs = model.encode(passage_texts, normalize_embeddings=False, show_progress_bar=True)
    query_embs = model.encode(query_texts, normalize_embeddings=False, show_progress_bar=True)

    # 各テキストごとのpassage vs queryの比較
    cos_sims = []
    l2_dists = []
    diff_norms = []
    diff_vectors = []

    for i in range(len(sample_texts)):
        p_emb = passage_embs[i]
        q_emb = query_embs[i]

        # コサイン類似度
        cos_sim = np.dot(p_emb, q_emb) / (norm(p_emb) * norm(q_emb))
        cos_sims.append(cos_sim)

        # L2距離
        l2_dist = norm(p_emb - q_emb)
        l2_dists.append(l2_dist)

        # 差分ベクトルのノルム
        diff = q_emb - p_emb  # query - passage
        diff_norms.append(norm(diff))
        diff_vectors.append(diff)

    diff_vectors = np.array(diff_vectors)

    print(f'\n   === 統計サマリー ===')
    print(f'   コサイン類似度（passage, query）:')
    print(f'     平均: {np.mean(cos_sims):.4f}')
    print(f'     標準偏差: {np.std(cos_sims):.4f}')
    print(f'     最小: {np.min(cos_sims):.4f}')
    print(f'     最大: {np.max(cos_sims):.4f}')

    print(f'\n   L2距離:')
    print(f'     平均: {np.mean(l2_dists):.4f}')
    print(f'     標準偏差: {np.std(l2_dists):.4f}')

    print(f'\n   差分ベクトル（query - passage）のノルム:')
    print(f'     平均: {np.mean(diff_norms):.4f}')
    print(f'     標準偏差: {np.std(diff_norms):.4f}')

    # データセット別の統計
    print(f'\n   === データセット別統計 ===')
    df_stats = pd.DataFrame({
        'dataset': [m['dataset'] for m in sample_metadata],
        'type': [m['type'] for m in sample_metadata],
        'lang': [m['lang'] for m in sample_metadata],
        'length': [m['length'] for m in sample_metadata],
        'cos_sim': cos_sims,
        'l2_dist': l2_dists,
        'diff_norm': diff_norms
    })

    print('\n   タイプ別（body vs title）:')
    for text_type in ['body', 'title']:
        subset = df_stats[df_stats['type'] == text_type]
        print(f'     {text_type}: cos_sim={subset["cos_sim"].mean():.4f}, l2_dist={subset["l2_dist"].mean():.4f}')

    print('\n   言語別:')
    for lang in ['ja', 'en']:
        subset = df_stats[df_stats['lang'] == lang]
        print(f'     {lang}: cos_sim={subset["cos_sim"].mean():.4f}, l2_dist={subset["l2_dist"].mean():.4f}')

    # ========================================
    # 分析2: 差分ベクトルの方向性の検証
    # ========================================
    print('\n' + '=' * 80)
    print('分析2: 差分ベクトルの方向性（共通の変換方向があるか）')
    print('=' * 80)

    # 差分ベクトルを正規化
    diff_vectors_normalized = diff_vectors / np.linalg.norm(diff_vectors, axis=1, keepdims=True)

    # 差分ベクトル同士のコサイン類似度（方向の一致度）
    print('\n   差分ベクトル同士のコサイン類似度を計算中...')
    diff_cos_sims = []
    for i in range(len(diff_vectors_normalized)):
        for j in range(i+1, len(diff_vectors_normalized)):
            sim = np.dot(diff_vectors_normalized[i], diff_vectors_normalized[j])
            diff_cos_sims.append(sim)

    print(f'\n   差分ベクトル間のコサイン類似度:')
    print(f'     平均: {np.mean(diff_cos_sims):.4f}')
    print(f'     標準偏差: {np.std(diff_cos_sims):.4f}')
    print(f'     最小: {np.min(diff_cos_sims):.4f}')
    print(f'     最大: {np.max(diff_cos_sims):.4f}')

    if np.mean(diff_cos_sims) > 0.5:
        print('\n   *** 重要発見: 差分ベクトルに共通の方向性がある！ ***')
        print('       → passage→query変換は一貫した方向への移動である可能性')
    elif np.mean(diff_cos_sims) > 0.2:
        print('\n   差分ベクトルにある程度の共通性がある')
    else:
        print('\n   差分ベクトルはランダムに近い（共通の方向性が弱い）')

    # 平均差分ベクトルの計算
    mean_diff_vector = np.mean(diff_vectors, axis=0)
    mean_diff_vector_normalized = mean_diff_vector / norm(mean_diff_vector)

    # 各差分ベクトルと平均差分ベクトルの類似度
    sims_to_mean = [np.dot(diff_vectors_normalized[i], mean_diff_vector_normalized)
                    for i in range(len(diff_vectors_normalized))]

    print(f'\n   各差分ベクトルと平均差分ベクトルの類似度:')
    print(f'     平均: {np.mean(sims_to_mean):.4f}')
    print(f'     標準偏差: {np.std(sims_to_mean):.4f}')

    # ========================================
    # 分析3: PCAによる差分ベクトルの構造分析
    # ========================================
    print('\n' + '=' * 80)
    print('分析3: 差分ベクトルのPCA分析')
    print('=' * 80)

    pca = PCA(n_components=10)
    diff_pca = pca.fit_transform(diff_vectors)

    print(f'\n   累積寄与率:')
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i in range(10):
        print(f'     PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% (累積: {cumsum[i]*100:.2f}%)')

    if pca.explained_variance_ratio_[0] > 0.5:
        print('\n   *** 重要発見: 第1主成分が50%以上を説明！ ***')
        print('       → 差分ベクトルは主に1つの方向に集中している')

    # ========================================
    # 分析4: 検索における影響
    # ========================================
    print('\n' + '=' * 80)
    print('分析4: 異なるテキスト間の類似度への影響')
    print('=' * 80)

    # ランダムに10ペアを選んで比較
    rng = np.random.default_rng(42)

    print('\n   異なるテキスト間の類似度比較（10ペア）:')
    print(f'   {"ペア":>5} | {"PP類似度":>10} | {"QQ類似度":>10} | {"QP類似度":>10} | {"差(QP-PP)":>10}')
    print('   ' + '-' * 60)

    pp_sims = []
    qq_sims = []
    qp_sims = []

    for _ in range(10):
        i, j = rng.choice(len(sample_texts), 2, replace=False)

        # passage同士
        pp_sim = np.dot(passage_embs[i], passage_embs[j]) / (norm(passage_embs[i]) * norm(passage_embs[j]))
        # query同士
        qq_sim = np.dot(query_embs[i], query_embs[j]) / (norm(query_embs[i]) * norm(query_embs[j]))
        # query i と passage j（検索シナリオ）
        qp_sim = np.dot(query_embs[i], passage_embs[j]) / (norm(query_embs[i]) * norm(passage_embs[j]))

        pp_sims.append(pp_sim)
        qq_sims.append(qq_sim)
        qp_sims.append(qp_sim)

        print(f'   {_+1:>5} | {pp_sim:>10.4f} | {qq_sim:>10.4f} | {qp_sim:>10.4f} | {qp_sim-pp_sim:>+10.4f}')

    print(f'\n   平均:')
    print(f'     passage-passage: {np.mean(pp_sims):.4f}')
    print(f'     query-query: {np.mean(qq_sims):.4f}')
    print(f'     query-passage: {np.mean(qp_sims):.4f}')

    # ========================================
    # 分析5: 平均差分ベクトルによる変換の検証
    # ========================================
    print('\n' + '=' * 80)
    print('分析5: 平均差分ベクトルによる変換の効果')
    print('=' * 80)

    print('\n   仮説: passage + mean_diff ≈ query')
    print('   検証: 変換後のベクトルと実際のqueryベクトルの類似度')

    # passage + mean_diff で疑似queryを作成
    pseudo_query_embs = passage_embs + mean_diff_vector

    # 実際のqueryとの類似度
    transform_sims = []
    for i in range(len(sample_texts)):
        sim = np.dot(pseudo_query_embs[i], query_embs[i]) / (norm(pseudo_query_embs[i]) * norm(query_embs[i]))
        transform_sims.append(sim)

    print(f'\n   変換後ベクトルと実際のqueryの類似度:')
    print(f'     平均: {np.mean(transform_sims):.4f}')
    print(f'     標準偏差: {np.std(transform_sims):.4f}')
    print(f'     最小: {np.min(transform_sims):.4f}')

    print(f'\n   比較（同じテキストのpassageとqueryの類似度）:')
    print(f'     平均: {np.mean(cos_sims):.4f}')

    if np.mean(transform_sims) > np.mean(cos_sims):
        print('\n   *** 平均差分ベクトルによる変換は有効！ ***')
        improvement = np.mean(transform_sims) - np.mean(cos_sims)
        print(f'       類似度が {improvement:.4f} 向上')

    # ========================================
    # 可視化
    # ========================================
    print('\n' + '=' * 80)
    print('可視化を生成中...')
    print('=' * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. コサイン類似度の分布
    ax1 = axes[0, 0]
    ax1.hist(cos_sims, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(cos_sims), color='red', linestyle='--', label=f'平均: {np.mean(cos_sims):.4f}')
    ax1.set_xlabel('Cosine Similarity (passage, query)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('同一テキストのpassageとqueryの類似度分布')
    ax1.legend()

    # 2. 差分ベクトル間の類似度分布
    ax2 = axes[0, 1]
    ax2.hist(diff_cos_sims, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(diff_cos_sims), color='red', linestyle='--', label=f'平均: {np.mean(diff_cos_sims):.4f}')
    ax2.set_xlabel('Cosine Similarity between diff vectors')
    ax2.set_ylabel('Frequency')
    ax2.set_title('差分ベクトル間の類似度分布（方向の一致度）')
    ax2.legend()

    # 3. PCA累積寄与率
    ax3 = axes[1, 0]
    ax3.bar(range(1, 11), pca.explained_variance_ratio_ * 100, alpha=0.7)
    ax3.plot(range(1, 11), cumsum * 100, 'ro-', label='累積')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance (%)')
    ax3.set_title('差分ベクトルのPCA寄与率')
    ax3.legend()
    ax3.set_xticks(range(1, 11))

    # 4. テキスト長と類似度の関係
    ax4 = axes[1, 1]
    lengths = [m['length'] for m in sample_metadata]
    colors = ['blue' if m['type'] == 'body' else 'orange' for m in sample_metadata]
    ax4.scatter(lengths, cos_sims, c=colors, alpha=0.6)
    ax4.set_xlabel('Text Length')
    ax4.set_ylabel('Cosine Similarity (passage, query)')
    ax4.set_title('テキスト長とpassage-query類似度の関係')
    ax4.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='body'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='title')
    ])

    plt.tight_layout()
    plt.savefig('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/19_prefix_analysis.png', dpi=150)
    print('   グラフを data/19_prefix_analysis.png に保存しました')

    # ========================================
    # 結論と次のステップ
    # ========================================
    print('\n' + '=' * 80)
    print('Phase 1 結論')
    print('=' * 80)

    print(f'''
   1. passageとqueryは同じテキストでも異なるベクトルになる
      - 平均コサイン類似度: {np.mean(cos_sims):.4f}
      - L2距離: {np.mean(l2_dists):.4f}

   2. 差分ベクトル（query - passage）の方向性:
      - 差分ベクトル間の平均類似度: {np.mean(diff_cos_sims):.4f}
      - PCA第1主成分の寄与率: {pca.explained_variance_ratio_[0]*100:.1f}%

   3. 変換の可能性:
      - 平均差分ベクトルによる変換後の類似度: {np.mean(transform_sims):.4f}
    ''')

    # 平均差分ベクトルを保存
    np.save('/home/terapyon/dev/vibe-coding/lsh-cascade-poc/data/mean_diff_vector.npy', mean_diff_vector)
    print('   平均差分ベクトルを data/mean_diff_vector.npy に保存しました')

    con.close()

    return {
        'mean_diff_vector': mean_diff_vector,
        'pca_ratio': pca.explained_variance_ratio_[0],
        'diff_cos_sim_mean': np.mean(diff_cos_sims),
        'transform_sim_mean': np.mean(transform_sims)
    }


if __name__ == '__main__':
    main()
