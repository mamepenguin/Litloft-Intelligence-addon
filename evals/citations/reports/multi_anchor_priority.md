# detailed_summary Citation Eval — multi-anchor-priority

_Generated: 2026-04-19 13:40:34 UTC_

## Aggregate

- total segments scored: **25**
- has_citation precision: **80.0%**  _(when a citation was returned, it pointed at an exact-hit chunk)_
- missing required citations: **0**  _(must_have_citation=true segments flipped to ⚠)_

### Location offset (primary metric)

``offset_at_top1`` = chunk-index distance between the system's top-1 chunk and the nearest ground-truth chunk. 0 = exact hit, 1–2 = adjacent, 5+ = different part of the file. Computed only for segments with known GT (25 of 25).

- mean: **0.92**  median (p50): **0.0**  p95: **3.0**  max: **15**

| threshold | hit rate (offset ≤ N) |
|---|---:|
| offset ≤ 0 | 80.0% _(== strict top-1 accuracy)_ |
| offset ≤ 1 | 88.0% |
| offset ≤ 2 | 88.0% |
| offset ≤ 5 | 96.0% |

### Calibration by top_score band

Sanity-checks whether the system's own confidence signal predicts location correctness. If mean offset does NOT decrease as score increases, the 2-state ⚠/citation UI is discarding information.

| top_score | n | mean offset | median offset | hit@0 |
|---|---:|---:|---:|---:|
| <0.70 | 0 | — | — | — |
| [0.70-0.80) | 0 | — | — | — |
| [0.80-0.85) | 5 | 3.00 | 0.0 | 80.0% |
| [0.85-0.90) | 6 | 1.00 | 0.0 | 66.7% |
| ≥0.90 | 14 | 0.14 | 0.0 | 85.7% |

### By segment type (legacy binary)

| type | top-1 (offset==0) | recall@3 | n |
|---|---:|---:|---:|
| bullet | 78.9% | 84.2% | 19 |
| paragraph | 83.3% | 100.0% | 6 |

## Cases

### `001_recipe_three_segment_types`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:2`, `transcript:0`, `transcript:3` |
| `詳細内容/0` | bullet | 1 | ❌ | ✅ | 0.91 | `transcript:2`, `transcript:3`, `transcript:6` |
| `詳細内容/4` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:14`, `transcript:9`, `transcript:12` |
| `詳細内容/7` | bullet | 0 ✅ | ✅ | ✅ | 0.90 | `transcript:29`, `transcript:30`, `transcript:28` |
| `重要ポイントまとめ/row/0` | bullet | 0 ✅ | ✅ | ✅ | 0.84 | `transcript:26`, `transcript:23`, `transcript:24` |
| `重要ポイントまとめ/row/4` | bullet | 0 ✅ | ✅ | ✅ | 0.83 | `transcript:113`, `transcript:109`, `transcript:110` |

### `002_dq_section1_dp_anchor`

- file: `YouTube/fsdjh983jhrf.mp4`
- file_id: `KtVKUiry6S_d`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.96 | `transcript:4`, `transcript:3`, `transcript:0` |
| `詳細内容/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.90 | `transcript:0`, `transcript:1`, `transcript:3` |
| `詳細内容/1` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:0`, `transcript:1`, `transcript:3` |
| `詳細内容/2` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:0`, `transcript:1`, `transcript:3` |
| `詳細内容/3` | bullet | 0 ✅ | ✅ | ✅ | 0.89 | `transcript:1`, `transcript:0`, `transcript:3` |
| `詳細内容/4` | paragraph | 0 ✅ | ✅ | ✅ | 0.89 | `transcript:2`, `transcript:1`, `transcript:3` |

### `003_recipe_all_table_rows`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `重要ポイントまとめ/row/1` | bullet | 15 | ❌ | ✅ | 0.80 | `transcript:43`, `transcript:92`, `transcript:72` |
| `重要ポイントまとめ/row/2` | bullet | 0 ✅ | ✅ | ✅ | 0.83 | `transcript:93`, `transcript:43`, `transcript:26` |
| `重要ポイントまとめ/row/3` | bullet | 0 ✅ | ✅ | ✅ | 0.81 | `transcript:109`, `transcript:78`, `transcript:84` |

### `004_dq_section_boundary`

- file: `YouTube/fsdjh983jhrf.mp4`
- file_id: `KtVKUiry6S_d`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/6` | paragraph | 1 | ✅ | ✅ | 0.93 | `transcript:5`, `transcript:4`, `transcript:3` |
| `詳細内容/7` | bullet | 0 ✅ | ✅ | ✅ | 0.91 | `transcript:5`, `transcript:8`, `transcript:0` |
| `詳細内容/9` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:8`, `transcript:0`, `transcript:5` |
| `詳細内容/14` | bullet | 0 ✅ | ✅ | ✅ | 0.90 | `transcript:15`, `transcript:18`, `transcript:16` |
| `詳細内容/15` | bullet | 3 | ✅ | ✅ | 0.88 | `transcript:18`, `transcript:15`, `transcript:12` |
| `詳細内容/20` | paragraph | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:18`, `transcript:13`, `transcript:14` |

### `005_recipe_section_transitions`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/11` | bullet | 3 | ❌ | ✅ | 0.89 | `transcript:61`, `transcript:79`, `transcript:59` |
| `詳細内容/23` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:79`, `transcript:92`, `transcript:85` |
| `詳細内容/29` | bullet | 0 ✅ | ✅ | ✅ | 0.91 | `transcript:93`, `transcript:95`, `transcript:92` |
| `詳細内容/37` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:110`, `transcript:121`, `transcript:120` |

## Baseline comparison

| metric | baseline | current | delta |
|---|---:|---:|---:|
| top1_accuracy | 76.0% | 80.0% | +4.0% ✅ |
| recall_at_3 | 88.0% | 88.0% | +0.0% (tied) |
| has_citation_precision | 76.0% | 80.0% | +4.0% ✅ |
