# Experiment Notebooks

## 00s: 基礎検証・レポート

| # | Notebook | 概要 |
|---|----------|------|
| 00 | [final_report](00_final_report.ipynb) | 最終レポート |
| 01 | [basic_verification](01_basic_verification.ipynb), [v2](01_v2_basic_verification.ipynb) | 基本動作の検証 |
| 02 | [lsh_accuracy_analysis](02_lsh_accuracy_analysis.ipynb), [v2](02_v2_lsh_accuracy_analysis.ipynb) | LSH精度分析 |
| 03 | [embedding_model_comparison](03_embedding_model_comparison.ipynb) | Embeddingモデル比較 |
| 04 | [whitening_experiment](04_whitening_experiment.ipynb) | ホワイトニング実験 |
| 05 | [e2lsh_experiment](05_e2lsh_experiment.ipynb) | E2LSH実験 |
| 06 | [e2lsh_pipeline](06_e2lsh_pipeline.ipynb), [v2](06_v2_e2lsh_pipeline.ipynb) | E2LSHパイプライン |
| 07 | [e2lsh_accuracy_analysis v2](07_v2_e2lsh_accuracy_analysis.ipynb), [filtering_analysis v3](07_v3_e2lsh_filtering_analysis.ipynb) | E2LSH精度・フィルタリング分析 |
| 08 | [e2lsh_query_analysis](08_e2lsh_query_analysis.ipynb) | E2LSHクエリ分析 |
| 09 | [e2lsh_cascading_filter](09_e2lsh_cascading_filter.ipynb) | E2LSHカスケードフィルタ |

## 10s: E2LSH・SimHash改良

| # | Notebook | 概要 |
|---|----------|------|
| 10 | [e2lsh_multiprobe](10_e2lsh_multiprobe.ipynb) | E2LSHマルチプローブ |
| 11 | [e2lsh_multiprobe_query_analysis](11_e2lsh_multiprobe_query_analysis.ipynb) | マルチプローブクエリ分析 |
| 12 | [simhash_improvement](12_simhash_improvement.ipynb) | SimHash改良 |
| 13 | [presample_gridsearch](13_presample_gridsearch.ipynb) | プレサンプルグリッドサーチ |
| 14 | [create_experiment_db](14_create_experiment_db.ipynb) | 実験DB作成 |
| 15 | [large_scale_evaluation](15_large_scale_evaluation.ipynb) | 大規模評価 |
| 16 | [mixed_length_hyperplanes](16_mixed_length_hyperplanes.ipynb) | 混合長超平面 |
| 17 | [hybrid_hyperplanes](17_hybrid_hyperplanes.ipynb) | ハイブリッド超平面 |
| 18 | [query_passage_hyperplanes](18_query_passage_hyperplanes.ipynb) | クエリ・パッセージ超平面 |
| 19 | [dual_prefix_search](19_dual_prefix_search.ipynb) | デュアルプレフィックス検索 |

## 20s: ITQ-LSH

| # | Notebook | 概要 |
|---|----------|------|
| 20 | [no_prefix_experiment](20_no_prefix_experiment.ipynb) | プレフィックスなし実験 |
| 21 | [itq_lsh_experiment](21_itq_lsh_experiment.ipynb) | ITQ-LSH実験 |
| 22 | [itq_hybrid_search](22_itq_hybrid_search.ipynb) | ITQハイブリッド検索 |

## 30s: 多段フィルタリング

| # | Notebook | 概要 |
|---|----------|------|
| 30 | [multistage_lsh_filtering](30_multistage_lsh_filtering.ipynb) | 多段LSHフィルタリング |
| 31 | [multiprobe_lsh](31_multiprobe_lsh.ipynb) | マルチプローブLSH |
| 32 | [improved_segment_lsh](32_improved_segment_lsh.ipynb) | 改良セグメントLSH |
| 33 | [overlap_cascade_filtering](33_overlap_cascade_filtering.ipynb) | オーバーラップカスケードフィルタ |
| 34 | [alternative_lsh_filtering](34_alternative_lsh_filtering.ipynb) | 代替LSHフィルタリング |
| 35 | [segment_width_small_dataset](35_segment_width_small_dataset.ipynb) | セグメント幅（小規模データ） |
| 36 | [hnsw_topk_hamming_analysis](36_hnsw_topk_hamming_analysis.ipynb) | HNSW Top-K Hamming分析 |

## 40s: FastEmbed・E5モデル・大規模データ

| # | Notebook | 概要 |
|---|----------|------|
| 41 | [fastembed_investigation](41_fastembed_investigation.ipynb) | FastEmbed調査 |
| 42 | [itq_retraining_necessity](42_itq_retraining_necessity.ipynb) | ITQ再学習の必要性検証 |
| 43 | [e5_model_size_comparison](43_e5_model_size_comparison.ipynb) | E5モデルサイズ比較 |
| 44 | [e5_base_bits_evaluation](44_e5_base_bits_evaluation.ipynb) | E5-base ビット数評価 |
| 45 | [e5_base_fastembed_compatibility](45_e5_base_fastembed_compatibility.ipynb) | E5-base FastEmbed互換性 |
| 46 | [fastembed_practical_speed](46_fastembed_practical_speed.ipynb) | FastEmbed実用速度 |
| 47 | [fastembed_e5base_optimization](47_fastembed_e5base_optimization.ipynb) | FastEmbed E5-base最適化 |
| 48 | [wikipedia_400k_embedding](48_wikipedia_400k_embedding.ipynb) | Wikipedia 400K Embedding生成 |
| 49 | [overlap_evaluation_400k](49_overlap_evaluation_400k.ipynb) | オーバーラップ評価（400K） |

## 50s: Pivot・Gray Code・複合評価

| # | Notebook | 概要 |
|---|----------|------|
| 51 | [popcount_filtering](51_popcount_filtering.ipynb) | Popcountフィルタリング |
| 52 | [pivot_based_indexing](52_pivot_based_indexing.ipynb) | Pivotベースインデキシング |
| 53 | [gray_code_linearization](53_gray_code_linearization.ipynb) | Gray Code線形化 |
| 54 | [combined_evaluation](54_combined_evaluation.ipynb) | 複合評価 |
| 55 | [overlap_vs_pivot](55_overlap_vs_pivot.ipynb) | オーバーラップ vs Pivot |
| 56 | [small_data_pivot_evaluation](56_small_data_pivot_evaluation.ipynb) | 小規模データPivot評価 |

## 60s: Embeddingモデル比較

| # | Notebook | 概要 |
|---|----------|------|
| 61 | [gte_small_evaluation](61_gte_small_evaluation.ipynb) | GTE-small評価 |
| 62 | [e5_small_evaluation](62_e5_small_evaluation.ipynb) | E5-small評価 |
| 63 | [bge_m3_evaluation](63_bge_m3_evaluation.ipynb) | BGE-M3評価 |
| 64 | [model_characteristics_comparison](64_model_characteristics_comparison.ipynb) | モデル特性比較 |
| 65 | [cpu_performance_comparison](65_cpu_performance_comparison.ipynb) | CPU性能比較 |
| 66 | [minilm_english_evaluation](66_minilm_english_evaluation.ipynb) | MiniLM英語評価 |
| 67 | [bge_small_fastembed_evaluation](67_bge_small_fastembed_evaluation.ipynb) | BGE-small FastEmbed評価 |
| 68 | [hnsw_model_comparison](68_hnsw_model_comparison.ipynb) | HNSWモデル比較 |

## 70s: ANN Benchmark

| # | Notebook | 概要 |
|---|----------|------|
| 71 | [export_itq_pivot_data](71_export_itq_pivot_data.ipynb) | ITQ・Pivotデータエクスポート |
| 72 | [ann_benchmark_data_preparation](72_ann_benchmark_data_preparation.ipynb) | ANNベンチマークデータ準備 |
| 73 | [ann_benchmark_itq_accuracy](73_ann_benchmark_itq_accuracy.ipynb) | ANNベンチマークITQ精度 |
| 74 | [ann_benchmark_pivot_hnsw_comparison](74_ann_benchmark_pivot_hnsw_comparison.ipynb) | ANNベンチマークPivot・HNSW比較 |
| 75 | [ann_benchmark_summary](75_ann_benchmark_summary.ipynb) | ANNベンチマークまとめ |

## 80s: DF-LSH・複合手法

| # | Notebook | 概要 |
|---|----------|------|
| 80 | [dflsh_data_preparation](80_dflsh_data_preparation.ipynb) | DF-LSHデータ準備 |
| 81 | [dflsh_standalone](81_dflsh_standalone.ipynb) | DF-LSH単体評価 |
| 82 | [bloom_filter_itq](82_bloom_filter_itq.ipynb) | Bloomフィルタ + ITQ |
| 83 | [confidence_multiprobe](83_confidence_multiprobe.ipynb) | 信頼度マルチプローブ |
| 84 | [combined_evaluation](84_combined_evaluation.ipynb) | 複合評価 |
| 85 | [pivot_lb_fair_comparison](85_pivot_lb_fair_comparison.ipynb) | Pivot下界の公平比較 |
| 86 | [itq_whitening_experiment](86_itq_whitening_experiment.ipynb) | ITQホワイトニング実験 |

## 90s: 軽量モデル・バイナリEmbedding

| # | Notebook | 概要 |
|---|----------|------|
| 91 | [lightweight_embedding_comparison](91_lightweight_embedding_comparison.ipynb) | 軽量Embeddingモデル比較 |
| 92 | [sts_benchmark_evaluation](92_sts_benchmark_evaluation.ipynb) | STSベンチマーク評価 |
| 93 | [wikipedia_article_discrimination](93_wikipedia_article_discrimination.ipynb) | Wikipedia記事識別 |
| 94 | [binary_embedding_vs_itq_lsh](94_binary_embedding_vs_itq_lsh.ipynb) | バイナリEmbedding vs ITQ-LSH |
| 95 | [onnx_cpu_inference_comparison](95_onnx_cpu_inference_comparison.ipynb) | ONNX CPU推論比較 |

## 100s: チャンキング戦略

| # | Notebook | 概要 |
|---|----------|------|
| 101 | [chunking_data_preparation](101_chunking_data_preparation.ipynb) | チャンキング実験用データ準備（Wikipedia JA/EN 各1000記事） |
| 102 | [fixed_size_overlap_sweep](102_fixed_size_overlap_sweep.ipynb) | 固定長チャンク × オーバーラップのグリッドサーチ（E5-base, Qwen3） |
| 103 | [boundary_aware_chunking](103_boundary_aware_chunking.ipynb) | 文・段落・セクション境界チャンキングの評価 |
| 104 | [long_context_head_bias](104_long_context_head_bias.ipynb) | Qwen3 長文コンテキストの先頭偏重バイアス検証 |
| 105 | [chunking_strategy_comparison](105_chunking_strategy_comparison.ipynb) | 101-104の総合比較とベストプラクティス |

### 100s シリーズの結論
- **固定長 + 25% オーバーラップが最良**（境界認識は改善なし）
- 推奨: E5-base=256/64, Qwen3=512/128
- Qwen3 は強い先頭偏重あり → 大チャンク (2048) の優位性は競合緩和効果が主因
- 記事レベル検索には max-sim 集約が最適（R@10=0.956〜0.999）

## 110s: Voronoi分割

| # | Notebook | 概要 |
|---|----------|------|
| 111 | [voronoi_partition_evaluation](111_voronoi_partition_evaluation.ipynb) | Voronoi分割（k-meansパーティション）の基本評価。C×Pグリッドサーチ、ITQ系との比較、汎化テスト |
| 112 | [voronoi_generalization_private](112_voronoi_generalization_private.ipynb) | Wikipedia学習セントロイドのドメイン外（非公開データ）への汎化性能評価 |
| 113 | [voronoi_image_text_comparison](113_voronoi_image_text_comparison.ipynb) | 画像(顔認識) vs テキストの最適パラメータ比較。Embedding空間構造分析、top-10散布分析 |
| 114 | [voronoi_multi_assign](114_voronoi_multi_assign.ipynb) | マルチアサイン(assign=2,3)の効果。画像プロジェクトと同方式での評価 |
| 115 | [voronoi_centroid_export](115_voronoi_centroid_export.ipynb) | セントロイドのエクスポート（NumPy + JSON）。利用例コード付き |

### 110s シリーズの結論
- **Voronoi分割はITQ系パイプラインを全削減率帯で上回る**（Pareto最適が全てVoronoi）
- 画像(顔認識)ではC=256, assign=2, P=2で有効 → テキストでも同方式が有効
- テキストはembedding空間のランダム-近傍Gapが画像の1/5 → probeを多く取る必要あるが、assign=2で緩和
- **assign=2, C=256が推奨**: JA R@10≥85%はP=2(IN句2要素)、EN R@10≥90%はA=3,P=4(IN句4要素)で達成
- Wikipedia学習セントロイドはドメイン外データでも汎化する（再学習不要）
- Firestore: `pivot_ids` KeywordIndex/array-contains-any、Zope: KeywordIndex + `operator='or'`
