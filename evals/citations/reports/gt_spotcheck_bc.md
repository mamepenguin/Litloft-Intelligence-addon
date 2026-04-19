# detailed_summary Citation Eval — gt-spotcheck-BC-expanded

_Generated: 2026-04-19 16:19:10 UTC_

## Aggregate

- total segments scored: **69**
- has_citation precision: **78.3%**  _(when a citation was returned, it pointed at an exact-hit chunk)_
- missing required citations: **0**  _(must_have_citation=true segments flipped to ⚠)_

### Location offset (primary metric)

``offset_at_top1`` = chunk-index distance between the system's top-1 chunk and the nearest ground-truth chunk. 0 = exact hit, 1–2 = adjacent, 5+ = different part of the file. Computed only for segments with known GT (69 of 69).

- mean: **0.94**  median (p50): **0.0**  p95: **7.0**  max: **15**

| threshold | hit rate (offset ≤ N) |
|---|---:|
| offset ≤ 0 | 78.3% _(== strict top-1 accuracy)_ |
| offset ≤ 1 | 88.4% |
| offset ≤ 2 | 88.4% |
| offset ≤ 5 | 94.2% |

### Calibration by top_score band

Sanity-checks whether the system's own confidence signal predicts location correctness. If mean offset does NOT decrease as score increases, the 2-state ⚠/citation UI is discarding information.

| top_score | n | mean offset | median offset | hit@0 |
|---|---:|---:|---:|---:|
| <0.70 | 0 | — | — | — |
| [0.70-0.80) | 1 | 9.00 | 9.0 | 0.0% |
| [0.80-0.85) | 9 | 2.22 | 0.0 | 66.7% |
| [0.85-0.90) | 16 | 1.06 | 0.0 | 68.8% |
| ≥0.90 | 43 | 0.44 | 0.0 | 86.0% |

### By segment type (legacy binary)

| type | top-1 (offset==0) | recall@3 | n |
|---|---:|---:|---:|
| bullet | 79.6% | 88.9% | 54 |
| paragraph | 73.3% | 86.7% | 15 |

## Cases

### `001_recipe_three_segment_types`

- file: `YouTube/【常備菜5選】一度覚えたら一生使える基本の副菜｜管理栄養士の作り置きレシピ.hvlink`
- file_id: `yFpvytc5zGpM`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:2`, `transcript:0`, `transcript:3` |
| `詳細内容/0` | bullet | 1 | ✅ | ✅ | 0.92 | `transcript:2`, `transcript:3`, `transcript:0` |
| `詳細内容/4` | bullet | 1 | ✅ | ✅ | 0.93 | `transcript:9`, `transcript:12`, `transcript:14` |
| `詳細内容/7` | bullet | 0 ✅ | ✅ | ✅ | 0.90 | `transcript:27`, `transcript:29`, `transcript:30` |
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
| `詳細内容/7` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:5`, `transcript:8`, `transcript:0` |
| `詳細内容/9` | bullet | 0 ✅ | ✅ | ✅ | 0.96 | `transcript:8`, `transcript:0`, `transcript:5` |
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
| `詳細内容/29` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:93`, `transcript:92`, `transcript:95` |
| `詳細内容/37` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:110`, `transcript:121`, `transcript:120` |

### `006_lecture_yamanaka_bracket_anchor`

- file: `YouTube/iPS細胞研究所 山中伸弥教授 卒業スピーチ「塞翁が馬…だから人生は楽しい」平成27年度近畿大学卒業式.mp4`
- file_id: `mQb3-7ABpBzo`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 1 | ✅ | ✅ | 0.84 | `transcript:1`, `transcript:0`, `transcript:2` |
| `詳細内容/2` | bullet | 0 ✅ | ✅ | ✅ | 0.91 | `transcript:15`, `transcript:14`, `transcript:11` |
| `詳細内容/17` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:59`, `transcript:58`, `transcript:61` |
| `重要ポイントまとめ/row/0` | bullet | 0 ✅ | ✅ | ✅ | 0.84 | `transcript:9`, `transcript:10`, `transcript:66` |

### `007_lecture_matayoshi_triple_bracket`

- file: `YouTube/ピース又吉卒業スピーチ  「バッドエンドはない、僕達は途中だ」  平成29年度近畿大学卒業式.hvlink`
- file_id: `4hdPYDUjlEw2`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 4 | ❌ | ✅ | 0.83 | `transcript:0`, `transcript:1`, `transcript:2` |
| `詳細内容/0` | bullet | 0 ✅ | ✅ | ✅ | 0.96 | `transcript:13`, `transcript:23`, `transcript:16` |
| `詳細内容/5` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:40`, `transcript:30`, `transcript:49` |
| `詳細内容/10` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:78`, `transcript:77`, `transcript:76` |

### `008_lecture_honda_claim_vs_example`

- file: `YouTube/プロサッカー選手 本田圭佑氏 卒業式スピーチ「欲望を解放しろ、環境にこだわれ」｜令和4年度近畿大学卒業式.hvlink`
- file_id: `LGVpwmWf-2vd`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入：スピーチの心構えと現状認識/0` | paragraph | 9 | ❌ | ✅ | 0.80 | `transcript:0`, `transcript:2`, `transcript:1` |
| `詳細内容：人生を切り開くための二つの指針/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.85 | `transcript:20`, `transcript:36`, `transcript:53` |
| `詳細内容：人生を切り開くための二つの指針/1` | bullet | 9 | ❌ | ✅ | 0.89 | `transcript:21`, `transcript:22`, `transcript:19` |
| `詳細内容：人生を切り開くための二つの指針/9` | bullet | 0 ✅ | ✅ | ✅ | 0.90 | `transcript:69`, `transcript:74`, `transcript:71` |

### `009_lecture_jobs_crosslang`

- file: `YouTube/【英語スピーチ】Apple創業者スティーブ・ジョブズのスタンフォード大卒業式スピーチ｜日英字幕.mp4`
- file_id: `7yaTstDyaD--`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `導入/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.87 | `transcript:0`, `transcript:2`, `transcript:1` |
| `詳細内容/11` | bullet | 0 ✅ | ✅ | ✅ | 0.88 | `transcript:20`, `transcript:21`, `transcript:4` |
| `詳細内容/21` | bullet | 0 ✅ | ✅ | ✅ | 0.88 | `transcript:33`, `transcript:24`, `transcript:37` |
| `結論/0` | paragraph | 0 ✅ | ✅ | ✅ | 0.87 | `transcript:58`, `transcript:59`, `transcript:57` |

### `010_press_weather_agency_numeric`

- file: `YouTube/【長野県北部で震度5強】気象庁 緊急記者会見 生中継（2026年4月18日）.hvlink`
- file_id: `JNgHRLLuc5uN`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `1. 地震の概要と発生状況/1` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:4`, `transcript:30`, `transcript:10` |
| `1. 地震の概要と発生状況/3` | bullet | 0 ✅ | ✅ | ✅ | 0.96 | `transcript:6`, `transcript:47`, `transcript:24` |
| `2. 防災上の留意事項と今後の見通し/3` | bullet | 11 | ❌ | ✅ | 0.90 | `transcript:2`, `transcript:1`, `transcript:3` |
| `2. 防災上の留意事項と今後の見通し/6` | bullet | 0 ✅ | ✅ | ✅ | 0.99 | `transcript:40`, `transcript:74`, `transcript:13` |

### `011_press_onoda_nested_bracket`

- file: `YouTube/【会見ノーカット】閣議後　小野田経済安保相 記者会見 宇宙船「オリオン」人類の最遠到達記録について──政治ニュース（日テレNEWS）.hvlink`
- file_id: `BGvpgKtSfILh`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `4. 宇宙開発・国際協力に関する質疑応答（経済安全保障担当大臣）/1` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:22`, `transcript:21`, `transcript:18` |
| `8. 政治的発言に関する質疑応答（AI担当大臣）/1` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:64`, `transcript:69`, `transcript:65` |
| `5. 海賊版対策に関する質疑応答（文化庁担当者）/3` | paragraph | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:36`, `transcript:37`, `transcript:35` |
| `6. 重要物資の安定確保に関する質疑応答（経済産業大臣）/3` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:46`, `transcript:47`, `transcript:49` |

### `012_commentary_snp_numeric_triple`

- file: `YouTube/S&P500は連続の急回復、でも何もするな！【S&P500, NASDAQ100】.hvlink`
- file_id: `Y-cjKrgtON1X`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `📈 市場の直近の動向（米国株・為替）/1` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:2`, `transcript:4`, `transcript:1` |
| `📊 主要指標と市場の構造分析/3` | bullet | 0 ✅ | ✅ | ✅ | 0.97 | `transcript:10`, `transcript:12`, `transcript:11` |
| `📊 主要指標と市場の構造分析/9` | bullet | 1 | ❌ | ✅ | 0.92 | `transcript:17`, `transcript:11`, `transcript:4` |
| `💡 投資戦略と市場の評価軸/4` | bullet | 0 ✅ | ✅ | ✅ | 0.95 | `transcript:47`, `transcript:48`, `transcript:44` |

### `013_commentary_ff14_propernoun`

- file: `YouTube/FF14　忙しい人のための第92回PLL要約.hvlink`
- file_id: `avSk8Nj4jrnV`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `1. パッチ7.5の全体概要と主要コンテンツ/1` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:4`, `transcript:3`, `transcript:8` |
| `2. 主要コンテンツの詳細解説/1` | bullet | 0 ✅ | ✅ | ✅ | 0.91 | `transcript:21`, `transcript:20`, `transcript:31` |
| `2. 主要コンテンツの詳細解説/10` | bullet | 0 ✅ | ✅ | ✅ | 0.94 | `transcript:29`, `transcript:31`, `transcript:30` |
| `2. 主要コンテンツの詳細解説/11` | paragraph | 0 ✅ | ✅ | ✅ | 0.97 | `transcript:33`, `transcript:35`, `transcript:34` |

### `014_interview_mihashi_quote_heavy`

- file: `YouTube/グラドル好き同人作家が人気グラドル三橋くん様に過激！？なインタビュー！同人も読んで頂いた！【塚本のべる_個人Vtuber】.hvlink`
- file_id: `u9zw0zYoZTOB`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/9` | bullet | 0 ✅ | ✅ | ✅ | 0.93 | `transcript:31`, `transcript:21`, `transcript:35` |
| `詳細内容/15` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:46`, `transcript:43`, `transcript:47` |
| `詳細内容/22` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `transcript:67`, `transcript:68`, `transcript:69` |
| `詳細内容/24` | bullet | 0 ✅ | ✅ | ✅ | 0.88 | `transcript:73`, `transcript:71`, `transcript:76` |

### `015_doc_workspace_chunk_granularity`

- file: `Knowledge/Google Workspaceを契約すべきかについて.md`
- file_id: `tK9ag6CUVeWL`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/4` | bullet | 0 ✅ | ✅ | ✅ | 0.86 | `document:0`, `document:2`, `document:1` |
| `詳細内容/6` | paragraph | 0 ✅ | ✅ | ✅ | 0.92 | `document:1`, `document:2`, `document:3` |
| `詳細内容/11` | bullet | 0 ✅ | ✅ | ✅ | 0.86 | `document:4`, `document:2`, `document:3` |
| `詳細内容/19` | bullet | 4 | ❌ | ✅ | 0.92 | `document:12`, `document:10`, `document:11` |

### `016_doc_claude_skills_code_anchor`

- file: `Knowledge/Claude Skillsで簡単にApple風デザインを自動生成！AIっぽいデザインから脱却する方法.md`
- file_id: `d8jk2Z5nxNYV`
| section_path | type | offset | r@3 | has_cit | score | chunks |
|---|---|:-:|:-:|:-:|---:|---|
| `詳細内容/5` | bullet | 0 ✅ | ✅ | ✅ | 0.92 | `document:7`, `document:6`, `document:4` |
| `詳細内容/17` | bullet | 1 | ✅ | ✅ | 0.87 | `document:17`, `document:16`, `document:18` |
| `詳細内容/18` | bullet | 0 ✅ | ✅ | ✅ | 0.90 | `document:17`, `document:16`, `document:18` |
| `詳細内容/22` | bullet | 1 | ✅ | ✅ | 0.89 | `document:34`, `document:35`, `document:33` |

## Baseline comparison

| metric | baseline | current | delta |
|---|---:|---:|---:|
| top1_accuracy | 75.4% | 78.3% | +2.9% ✅ |
| recall_at_3 | 88.4% | 88.4% | +0.0% (tied) |
| has_citation_precision | 75.4% | 78.3% | +2.9% ✅ |
