# Citation Pipeline Ablation (2026-04-19)

5 curated cases / 25 segments run under 4 configurations.

| config | top-1 | recall@3 | has_cit prec | miss_req | Δ top-1 | Δ recall@3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100.0% | 100.0% | 100.0% | 0 | — | — |
| no_dp | 92.0% | 96.0% | 92.0% | 0 | -8.0 pp | -4.0 pp |
| no_anchor | 80.0% | 92.0% | 80.0% | 0 | -20.0 pp | -8.0 pp |
| no_hybrid | 92.0% | 96.0% | 92.0% | 0 | -8.0 pp | -4.0 pp |

- **baseline**: 完全現行パイプライン (hybrid + hierarchical narrowing + Viterbi DP + margin gate)
- **no_dp**: `citation_section_alignment_enabled: false` — DP alignment 無効、pool-cluster fallback のみ
- **no_anchor**: `citation_section_anchor_enabled: false` — section anchoring 全停止、全 segment が full-file 検索
- **no_hybrid**: `citation_hybrid_enabled: false` — dense-only retrieval、BM25 rerank なし

## どこが崩れたか (segment 単位)

### `no_dp`

| case | section_path | top-1 | r@3 | top-3 chunks |
|---|---|:-:|:-:|---|
| `002_dq_section1_dp_anchor` | `詳細内容/4` | ❌ | ✅ | `transcript:41`, `transcript:2`, `transcript:14` |
| `005_recipe_section_transitions` | `詳細内容/29` | ❌ | ❌ | `transcript:92`, `transcript:88`, `transcript:91` |

### `no_anchor`

| case | section_path | top-1 | r@3 | top-3 chunks |
|---|---|:-:|:-:|---|
| `001_recipe_three_segment_types` | `詳細内容/4` | ❌ | ✅ | `transcript:27`, `transcript:12`, `transcript:11` |
| `002_dq_section1_dp_anchor` | `導入/0` | ❌ | ✅ | `transcript:40`, `transcript:4`, `transcript:15` |
| `002_dq_section1_dp_anchor` | `詳細内容/4` | ❌ | ❌ | `transcript:14`, `transcript:18`, `transcript:39` |
| `005_recipe_section_transitions` | `詳細内容/23` | ❌ | ✅ | `transcript:3`, `transcript:2`, `transcript:79` |
| `005_recipe_section_transitions` | `詳細内容/29` | ❌ | ❌ | `transcript:41`, `transcript:42`, `transcript:7` |

### `no_hybrid`

| case | section_path | top-1 | r@3 | top-3 chunks |
|---|---|:-:|:-:|---|
| `003_recipe_all_table_rows` | `重要ポイントまとめ/row/1` | ❌ | ✅ | `transcript:92`, `transcript:78`, `transcript:61` |
| `003_recipe_all_table_rows` | `重要ポイントまとめ/row/3` | ❌ | ❌ | `transcript:78`, `transcript:84`, `transcript:61` |

## 解釈

- **no_dp**: DQ 動画の `詳細内容/4` が chunk 41 (動画末尾の無関係な話題) に飛ぶ — hako `Qu5avq5Mdxig6U_9LhE-J` で記録された regression が再現。Viterbi DP が section 1 (chunks 0-5) に拘束していた効果が -8 pp として定量化された
- **no_anchor**: 最大の劣化 -20 pp。section 全体の pool vector による chunk 範囲絞り込みが外れ、各 segment が full-file 検索に退行して別 section の chunk に引っ張られる
- **no_hybrid**: table row の 2 行が別 recipe 帯に漏れる — cell embedding だけでは numeric 値 (「にんじん3本」「こんにゃく1枚」) の signal が弱く、BM25 の salient token ヒットが効いていたことが裏付けられる
