パッケージ開発の指針となる仕様書（`SPECIFICATION.md`）を作成しました。
このファイルをリポジトリのルートに置いておくと、実装時のブレを防ぎ、後から見返したときのマニュアル代わりにもなります。

---

# `lsh-cascade-poc` 仕様書

## 1. プロジェクト概要

本プロジェクトは、大規模ベクトル検索における **「LSH (Locality Sensitive Hashing) を用いた3段階フィルタリング手法」** の概念実証 (PoC) である。
DuckDBをストレージおよび検証環境として使用し、HNSW（近似近傍探索のデファクト）と、LSHを用いたNoSQL的アプローチの検索精度（再現率）とパフォーマンスを比較検証する。

## 2. システム構成

### 2.1 ディレクトリ構造

```text
lsh-cascade-poc/
├── data/                  # データセット・DBファイル (Git対象外)
├── src/
│   ├── lsh.py             # SimHash生成・ビット操作・ハミング距離計算
│   ├── db.py              # DuckDB接続・スキーマ定義・データ操作
│   ├── loader.py          # Wikipediaデータ取得・Embedding処理
│   └── pipeline.py        # 3段階検索ロジックの実装
├── run_experiment.py      # 実験実行スクリプト (Entry Point)
├── pyproject.toml         # プロジェクト設定・依存ライブラリ (uv)
└── uv.lock                # ロックファイル

```

### 2.2 技術スタック

* **Language:** Python 3.12+ (uv管理)
* **Database:** DuckDB (OLAP向け組込DB。ベクトル拡張 `vss` を使用)
* **Embedding:** `sentence-transformers` (Model: `intfloat/multilingual-e5-large`)
* **Dataset:** Hugging Face Datasets (`wikipedia`)

---

## 3. 実装詳細仕様

### 3.1 `src/lsh.py`: LSHコアロジック

SimHashアルゴリズムに基づき、ベクトルを「指紋（Fingerprint）」化する。

* **`SimHashGenerator` クラス**
* **初期化:** 次元数（例: 1024）とハッシュビット数（128bit）を指定。乱数シードを固定し、**ランダム射影用の超平面（Hyperplanes）** を生成・保持する。
* **`hash(vector)`:** 入力ベクトルと超平面の内積をとり、正負に応じて `0` / `1` を割り当て、128bit整数（またはビット列）を返す。


* **ユーティリティ関数**
* **`chunk_hash(simhash_int, num_chunks)`:** 128bitを `num_chunks` (4, 8, 16) に分割し、検索用トークンのリストを返す。
* 例: 128bit, 4分割 → 32bit × 4個のHEX文字列リスト `["c0_A1B2...", "c1_C3D4...", ...]`


* **`hamming_distance(int_a, int_b)`:** 2つのハッシュ値の排他的論理和 (XOR) をとり、立っているビット数 (`popcount`) を返す。



### 3.2 `src/db.py`: データベース操作

DuckDBを使用し、HNSWとLSHの両方をサポートするスキーマを構築する。

* **スキーマ定義 (`documents` テーブル)**
* `id`: INTEGER (Primary Key)
* `text`: VARCHAR (元のテキスト)
* `vector`: FLOAT[DIM] (Embeddingベクトル)
* `simhash`: UBIGINT または VARCHAR (128bitハッシュ値)
* `lsh_chunks`: VARCHAR[] (分割されたハッシュトークンの配列。NoSQLインデックス用)


* **インデックス**
* **HNSW Index:** `vector` カラムに対して作成（ベースライン比較用）。
* **Inverted Index:** `lsh_chunks` カラムの配列要素に対して作成（DuckDBのリストインデックス機能を活用）。



### 3.3 `src/loader.py`: データローダー

* **ソース:** Wikipedia (ja/en)
* **処理:**
1. 指定件数（例: 各1000〜10000件）を取得。
2. Sentence Transformers でベクトル化。
3. `LSH` モジュールを通して SimHash と Chunk を生成。
4. DataFrame または 辞書リストとして返す。



### 3.4 `src/pipeline.py`: 検索パイプライン

3段階のフィルタリングを実行するサーチャークラス。

* **`LSHCascadeSearcher` クラス**
* **Step 1: Coarse Filtering (SQL)**
* クエリベクトルをハッシュ化＆チャンク分割。
* SQLの `array_contains` や `unnest` を使い、**「チャンクのいずれかが一致する (OR検索)」** レコードをDBから取得。
* 目標: 数万件 → 数千件。


* **Step 2: Binary Reranking (Python)**
* Step 1の結果に対し、Python上でハミング距離を計算。
* 距離が近い順にソートし、上位 N件（例: 100件）に絞る。


* **Step 3: Exact Reranking (Python/Numpy)**
* Step 2の上位 N件に対し、元の `vector` を用いてコサイン類似度を計算。
* 最終的な Top-K（例: 10件）を返す。





---

## 4. 実験シナリオ (`run_experiment.py`)

以下の比較実験を行い、結果をコンソール（またはCSV）に出力する。

1. **準備フェーズ:**
* Wikipediaデータ (日英 混合) をロード。
* DBへインサート & インデックス構築。


2. **検索フェーズ:** ランダムなクエリ10件に対し、以下を実行。
* **Baseline (HNSW):** DuckDBのHNSWインデックス機能で Top-10 を取得。
* **Approach A (LSH-4):** 4分割 (32bit一致) での 3段階検索。
* **Approach B (LSH-8):** 8分割 (16bit一致) での 3段階検索。
* **Approach C (LSH-16):** 16分割 (8bit一致) での 3段階検索。


3. **評価フェーズ:**
* **再現率 (Recall@10):** HNSWのTop-10を正解とし、LSHの結果がどれだけ含んでいるか。
* **速度:** 各手法の平均レイテンシ(ms)。
* **候補削減率:** Step 1 でどれだけ絞り込めたか。



---

## 5. 開発マイルストーン

1. **Core:** `lsh.py` の実装と単体テスト（ハミング距離が正しく計算できるか）。
2. **Data:** `loader.py` でWikipediaからベクトル付きデータを作る。
3. **Storage:** `db.py` でデータをDuckDBに入れ、SQLで配列検索ができるか確認。
4. **Pipeline:** `pipeline.py` で3段階検索をつなぎこむ。
5. **Bench:** HNSWと比較し、パラメータ（分割数）を調整する。