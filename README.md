# LSH Cascade PoC

LSH (Locality Sensitive Hashing) を用いた3段階フィルタリング手法による大規模ベクトル検索の概念実証 (PoC) プロジェクト。

DuckDBをストレージとして使用し、HNSWインデックスとLSHベースのカスケード検索のパフォーマンスを比較検証する。

## 特徴

- **SimHash**: ランダム超平面射影による128bitバイナリハッシュ生成
- **3段階フィルタリング**:
  1. Coarse Filtering: LSHチャンク一致によるSQL絞り込み
  2. Binary Reranking: ハミング距離による候補絞り込み
  3. Exact Reranking: コサイン類似度による最終スコアリング
- **DuckDB vss拡張**: HNSWインデックスによるベースライン比較

## 要件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャー)

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd lsh-cascade-poc

# 依存関係をインストール
uv sync
```

## 使い方

### 実験の実行

```bash
# デフォルト設定で実験を実行
uv run python run_experiment.py

# パラメータを指定して実行
uv run python run_experiment.py \
    --ja-samples 500 \
    --en-samples 500 \
    --queries 10 \
    --seed 42

# GPU使用 (CUDAが利用可能な場合)
uv run python run_experiment.py --device cuda
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--ja-samples` | 500 | 日本語Wikipediaのサンプル数 |
| `--en-samples` | 500 | 英語Wikipediaのサンプル数 |
| `--queries` | 10 | ランダムクエリ数 |
| `--seed` | 42 | 乱数シード |
| `--device` | cpu | 推論デバイス (cpu/cuda) |

### 出力例

```
============================================================
LSH Cascade Search Experiment
============================================================
Settings: 500 ja + 500 en samples, 10 queries

[1/4] Loading and processing Wikipedia data...
      Loaded 1000 documents
[2/4] Inserting data into DuckDB...
      Inserted 1000 documents
[3/4] Creating HNSW index...
      HNSW index created
[4/4] Running experiments...

============================================================
Results
============================================================
Total documents: 1000

--- Baseline: HNSW ---
  Avg Latency: 2.30 ms

--- LSH-4 (32-bit chunks) ---
  Recall@10: 0.85
  Avg Latency: 5.10 ms
  Avg Step1 Candidates: 234
  Reduction Rate: 76.6%

--- LSH-8 (16-bit chunks) ---
  Recall@10: 0.92
  Avg Latency: 8.20 ms
  Avg Step1 Candidates: 452
  Reduction Rate: 54.8%

--- LSH-16 (8-bit chunks) ---
  Recall@10: 0.98
  Avg Latency: 15.30 ms
  Avg Step1 Candidates: 723
  Reduction Rate: 27.7%
```

## プロジェクト構成

```
lsh-cascade-poc/
├── src/
│   ├── lsh.py         # SimHash生成・ハミング距離計算
│   ├── loader.py      # Wikipediaデータ取得・Embedding
│   ├── db.py          # DuckDB操作・スキーマ定義
│   └── pipeline.py    # 3段階検索パイプライン
├── tests/             # 単体テスト
├── data/              # データベースファイル (Git対象外)
├── run_experiment.py  # 実験実行スクリプト
└── SPECIFICATION.md   # 詳細仕様書
```

## テスト

```bash
# 全テスト実行
uv run pytest tests/ -v

# カバレッジ付き
uv run pytest tests/ --cov=src --cov-report=html
```

## 技術スタック

- **Database**: DuckDB (vss拡張)
- **Embedding**: sentence-transformers (`intfloat/multilingual-e5-large`, 1024次元)
- **Dataset**: Hugging Face Datasets (Wikipedia)

## ライセンス

MIT
