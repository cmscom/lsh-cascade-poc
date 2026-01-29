# 3段階カスケード検索の設計

## 概要

ITQ LSHとOverlapセグメントインデックスを組み合わせた3段階カスケード検索方式の設計ドキュメント。

**推奨設定**: Overlap(8,4) + S1=10000 + S2=2000
- Recall@10 = 90.0%
- 処理時間 = 23.5ms（2段階検索より35%高速）

---

## 1. Overlap(8,4)の仕組み

### セグメント分割

128bit ITQハッシュを8bit幅、4bitストライドでスライディング分割する。

```
128bit ITQハッシュ
|<---------------------------- 128bit ---------------------------->|

セグメント0:  [bit 0-7]   (8bit = 0〜255の整数)
セグメント1:  [bit 4-11]  ← 4bitずらし
セグメント2:  [bit 8-15]
セグメント3:  [bit 12-19]
...
セグメント30: [bit 120-127]

合計31セグメント（オーバーラップあり）
```

### なぜオーバーラップが有効か

- 通常のセグメント分割（完全一致）では、1bitの違いでも別バケットになる
- オーバーラップにより、ビット列の「ずれ」を吸収
- 31セグメントのうちいずれかが一致すれば候補として収集（OR条件）

---

## 2. DBに持つデータ

### 2.1 メインテーブル（既存）

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    embedding FLOAT[1024],      -- 元ベクトル（4KB/件）
    hash BIT(128)               -- ITQハッシュ（16bytes/件）
);
```

**サイズ**: 40万件 × (4KB + 16bytes) ≈ **1.6GB**

### 2.2 セグメントインデックステーブル（新規）

```sql
CREATE TABLE segment_index (
    segment_id INTEGER,         -- 0〜30（31セグメント）
    segment_value INTEGER,      -- 0〜255（8bitの値）
    doc_id INTEGER              -- ドキュメントID
);

-- インデックス作成
CREATE INDEX idx_segment ON segment_index(segment_id, segment_value);
```

**サイズ見積もり**:
- 40万件 × 31セグメント = **1,240万行**
- 1行 ≈ 12bytes（segment_id: 4, segment_value: 4, doc_id: 4）
- 合計 ≈ **150MB**（インデックス込みで200MB程度）

### 2.3 データサイズまとめ

| テーブル | サイズ（40万件） | 用途 |
|---------|-----------------|------|
| documents.embedding | 1.6GB | コサイン計算用 |
| documents.hash | 6.4MB | ハミング距離計算用 |
| segment_index | **200MB** | 初段フィルタ用 |

**追加データ量**: 約200MB（元データの12%程度）

---

## 3. 検索フロー

### 3段階枝刈りフロー

```
全データ (400,000件)
    ↓ Step 1: Overlapセグメント一致 → 6.4万件（84%削減）
候補 (10,000件)
    ↓ Step 2: ハミング距離ソート
候補 (2,000件)
    ↓ Step 3: コサイン類似度
Top-K 結果
```

### Step 0: クエリ前処理（アプリ側）

```python
# クエリ文をembeddingに変換
query_text = "東京の観光スポット"
query_embedding = e5_model.encode(f"query: {query_text}")  # 1024次元

# ITQハッシュを計算
query_hash = itq.transform(query_embedding)  # 128bit

# 31個のセグメントに分割
query_segments = []
for i in range(31):
    start = i * 4
    segment_bits = query_hash[start:start+8]
    segment_value = int(bits_to_int(segment_bits))  # 0〜255
    query_segments.append(segment_value)

# 例: [142, 78, 203, 15, ..., 91]（31個の整数）
```

### Step 1: セグメント一致で候補取得（DB）

```sql
-- 31個のセグメントのいずれかが一致するdoc_idを取得
SELECT DISTINCT doc_id
FROM segment_index
WHERE (segment_id = 0 AND segment_value = 142)
   OR (segment_id = 1 AND segment_value = 78)
   OR (segment_id = 2 AND segment_value = 203)
   ...
   OR (segment_id = 30 AND segment_value = 91);

-- 結果: 約6.4万件のdoc_id
```

**この時点でのDB負荷**:
- インデックス引き31回（O(1) × 31）
- 結果は**6.4万件**（40万件の16%）

### Step 2: ハミング距離でソート（DB or アプリ）

```sql
-- 候補6.4万件のハッシュを取得してハミング距離計算
WITH candidates AS (
    SELECT DISTINCT doc_id FROM segment_index
    WHERE ...  -- Step1のクエリ
)
SELECT d.id, d.hash,
       bit_count(d.hash XOR ?::BIT(128)) as hamming_dist
FROM documents d
JOIN candidates c ON d.id = c.doc_id
ORDER BY hamming_dist
LIMIT 10000;  -- Step1 Limit

-- 結果: 1万件のdoc_id（ハミング距離順）
```

**この時点でのDB負荷**:
- ハッシュ読み込み: 6.4万件 × 16bytes = **1MB**
- ハミング距離計算: 6.4万回のXOR + popcount

### Step 3: さらにハミング距離で絞る（アプリ）

```python
# 1万件 → 2000件に絞る
top_2000 = sorted(candidates, key=lambda x: x.hamming_dist)[:2000]
```

### Step 4: コサイン類似度で最終ランキング（DB）

```sql
-- 2000件のembeddingを取得
SELECT id, embedding
FROM documents
WHERE id IN (?, ?, ..., ?);  -- 2000個のID

-- 結果: 2000件のembedding（各1024次元）
```

**この時点でのDB負荷**:
- embedding読み込み: **2000件 × 4KB = 8MB**
- コサイン計算: 2000回（アプリ側）

---

## 4. DB負荷の比較

| 処理 | 3段階カスケード | 2段階検索（従来） |
|------|----------------|------------------|
| **インデックス引き** | 31回 | なし |
| **ハッシュ読み込み** | 6.4万件(1MB) | **40万件(6.4MB)** |
| **ハミング距離計算** | 6.4万回 | **40万回** |
| **embedding読み込み** | 2000件(8MB) | 2000件(8MB) |
| **コサイン計算** | 2000回 | 2000回 |

**削減効果**:
- ハッシュ読み込み: **6分の1**
- ハミング距離計算: **6分の1**
- embedding読み込み: **同じ**

---

## 5. 実装のポイント

### 5.1 segment_indexにB-treeインデックス

```sql
CREATE INDEX idx_segment ON segment_index(segment_id, segment_value);
```

### 5.2 Step1のクエリ最適化

```sql
-- UNIONで並列実行
SELECT doc_id FROM segment_index WHERE segment_id=0 AND segment_value=142
UNION
SELECT doc_id FROM segment_index WHERE segment_id=1 AND segment_value=78
UNION
...
SELECT doc_id FROM segment_index WHERE segment_id=30 AND segment_value=91;
```

### 5.3 ハッシュの保持形式

```sql
-- DuckDBの場合
hash UHUGEINT  -- 128bit整数

-- PostgreSQLの場合
hash BIT(128)  -- または BYTEA
```

### 5.4 ハミング距離計算

```sql
-- DuckDBの場合
bit_count(hash1 # hash2)  -- XOR後のpopcount

-- PostgreSQLの場合（拡張必要）
SELECT length(replace(hash1::text # hash2::text, '0', ''))
```

---

## 6. シナリオ別推奨設定

| シナリオ | Overlap | S1 Limit | S2 Limit | R@10 | 処理時間 |
|---------|---------|----------|----------|------|----------|
| **精度重視** | (8,4) | 10000 | 2000 | 90.0% | 23.5ms |
| **バランス** | (8,4) | 5000 | 1000 | 84.5% | 21.3ms |
| **高速重視** | (8,4) | 5000 | 500 | 74.4% | 20.5ms |

---

## 7. 実験結果サマリー

### 実験33の結果（40万件データ）

| 設定 | R@10 | Step1候補 | 処理時間 |
|------|------|-----------|----------|
| Overlap(8,4) S1=10000 S2=2000 | **90.0%** | 6.4万件 | 23.5ms |
| Overlap(8,4) S1=5000 S2=1000 | 84.5% | 6.4万件 | 21.3ms |
| 2段階検索(2000) | 91.3% | 40万件 | 36.2ms |

### 重要な発見

1. **Step2のLimit（コサイン計算対象）が精度を決める**
   - S2=500件: R@10 ≈ 74-75%（不足）
   - S2=1000件: R@10 ≈ 84-85%（やや不足）
   - S2=2000件: R@10 ≈ 90%（十分）

2. **Overlap(16,x)は枝刈りしすぎて精度低下**
   - 削減率99%以上だが、R@10は40%台
   - 候補数が少なすぎて良い結果を逃す

---

## 8. 関連ファイル

- `src/itq_lsh.py` - ITQ LSHの実装
- `src/cascade_search.py` - カスケード検索の実装
- `notebooks/32_improved_segment_lsh.ipynb` - セグメント方式の比較実験
- `notebooks/33_overlap_cascade_filtering.ipynb` - 3段階カスケードの実験
- `data/itq_model.pkl` - 学習済みITQモデル（128bit）

---

## 9. 今後の課題

1. **DuckDBでの実装**
   - segment_indexテーブルの作成
   - 検索クエリの最適化

2. **外部クエリでの評価**
   - 学習データに含まれないクエリでの性能確認

3. **スケーラビリティ検証**
   - 数百万〜数千万件での性能評価
   - インデックスサイズと構築時間の計測
