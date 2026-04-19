# detailed_summary Citation Eval — no_anchor

_Generated: 2026-04-19 12:41:31 UTC_

## Aggregate

- total segments scored: **25**
- top-1 accuracy: **80.0%**
- recall @ 3: **92.0%**
- has_citation precision: **80.0%**
- missing required citations: **0**

### By segment type

| type | top-1 | recall@3 | n |
|---|---:|---:|---:|
| bullet | 84.2% | 94.7% | 19 |
| paragraph | 66.7% | 83.3% | 6 |

## Cases

### `001_recipe_three_segment_types`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | top1 | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | ✅ | ✅ | ✅ | 0.94 | `transcript:0`, `transcript:2`, `transcript:3` |
| `詳細内容/0` | bullet | ✅ | ✅ | ✅ | 0.92 | `transcript:2`, `transcript:20`, `transcript:95` |
| `詳細内容/4` | bullet | ❌ | ✅ | ✅ | 0.92 | `transcript:27`, `transcript:12`, `transcript:11` |
| `詳細内容/7` | bullet | ✅ | ✅ | ✅ | 0.88 | `transcript:29`, `transcript:27`, `transcript:23` |
| `重要ポイントまとめ/row/0` | bullet | ✅ | ✅ | ✅ | 0.84 | `transcript:26`, `transcript:23`, `transcript:24` |
| `重要ポイントまとめ/row/4` | bullet | ✅ | ✅ | ✅ | 0.83 | `transcript:113`, `transcript:109`, `transcript:110` |

### `002_dq_section1_dp_anchor`

- file: `YouTube/fsdjh983jhrf.mp4`
- file_id: `KtVKUiry6S_d`
| section_path | type | top1 | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | ❌ | ✅ | ✅ | 0.96 | `transcript:40`, `transcript:4`, `transcript:15` |
| `詳細内容/0` | paragraph | ✅ | ✅ | ✅ | 0.90 | `transcript:0`, `transcript:53`, `transcript:8` |
| `詳細内容/1` | bullet | ✅ | ✅ | ✅ | 0.92 | `transcript:0`, `transcript:27`, `transcript:40` |
| `詳細内容/2` | bullet | ✅ | ✅ | ✅ | 0.93 | `transcript:0`, `transcript:53`, `transcript:40` |
| `詳細内容/3` | bullet | ✅ | ✅ | ✅ | 0.89 | `transcript:1`, `transcript:0`, `transcript:53` |
| `詳細内容/4` | paragraph | ❌ | ❌ | ✅ | 0.88 | `transcript:14`, `transcript:18`, `transcript:39` |

### `003_recipe_all_table_rows`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | top1 | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `重要ポイントまとめ/row/1` | bullet | ✅ | ✅ | ✅ | 0.80 | `transcript:43`, `transcript:92`, `transcript:72` |
| `重要ポイントまとめ/row/2` | bullet | ✅ | ✅ | ✅ | 0.83 | `transcript:93`, `transcript:43`, `transcript:26` |
| `重要ポイントまとめ/row/3` | bullet | ✅ | ✅ | ✅ | 0.81 | `transcript:109`, `transcript:78`, `transcript:84` |

### `004_dq_section_boundary`

- file: `YouTube/fsdjh983jhrf.mp4`
- file_id: `KtVKUiry6S_d`
| section_path | type | top1 | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/6` | paragraph | ✅ | ✅ | ✅ | 0.93 | `transcript:5`, `transcript:4`, `transcript:47` |
| `詳細内容/7` | bullet | ✅ | ✅ | ✅ | 0.94 | `transcript:5`, `transcript:8`, `transcript:36` |
| `詳細内容/9` | bullet | ✅ | ✅ | ✅ | 0.96 | `transcript:8`, `transcript:27`, `transcript:40` |
| `詳細内容/14` | bullet | ✅ | ✅ | ✅ | 0.90 | `transcript:15`, `transcript:18`, `transcript:40` |
| `詳細内容/15` | bullet | ✅ | ✅ | ✅ | 0.87 | `transcript:15`, `transcript:22`, `transcript:12` |
| `詳細内容/20` | paragraph | ✅ | ✅ | ✅ | 0.94 | `transcript:18`, `transcript:34`, `transcript:37` |

### `005_recipe_section_transitions`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | top1 | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/11` | bullet | ✅ | ✅ | ✅ | 0.89 | `transcript:61`, `transcript:92`, `transcript:79` |
| `詳細内容/23` | bullet | ❌ | ✅ | ✅ | 0.88 | `transcript:3`, `transcript:2`, `transcript:79` |
| `詳細内容/29` | bullet | ❌ | ❌ | ✅ | 0.91 | `transcript:41`, `transcript:42`, `transcript:7` |
| `詳細内容/37` | bullet | ✅ | ✅ | ✅ | 0.92 | `transcript:110`, `transcript:2`, `transcript:20` |
