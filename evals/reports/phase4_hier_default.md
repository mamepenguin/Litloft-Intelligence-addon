# Eval Report: phase4 hier default (clue_count=3)

- date: 2026-04-28T08:23:05Z
- git_commit: unknown
- llm_model: gemma4:e4b
- llm_base_url: http://host.docker.internal:11434/v1
- llm_temperature: 0 (forced by runner)
- search_config:
    rag.top_k: 5
    rag.max_tokens: 2048
    search.mode: recall
- index_snapshot:
    file: /eval-data/test-drive/snapshot/search.db
    sha256: 09bff72c346c4adaeaf1ad627697435e7fd1ae69bec639d794e8770ed2c13632
    indexed_with:
      whisper: 'openai/whisper-small'
      clip: 'llm-jp/llm-jp-clip-vit-base-patch16'
      blip: 'Salesforce/blip-image-captioning-base'
- blocklists:
    question_words.txt: b2aa63dd5c23475e50ebe81eaaf4bc8585165d6fbcb96fae0e56d5d65b12be13
    file_type_words.txt: 1a54cff895ca51f0f9d9845b0d158b6408c34d182bfcf88fc806b64438a66661
- runs: stage1=1 stage2=1 stage3=3
- epsilon: 0.1
- drive: eval-drive
- total_cases: 14

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 1.00 |
| Stage 1: must_exclude_violations (sum) | 5 |
| Stage 2: file recall@5 (median) | 1.00 |
| Stage 2: file recall@10 (median) | 1.00 |
| Stage 2: segment recall@5 (median) | 1.00 |
| Stage 2: MRR (median) | 1.00 |
| Stage 3: must_mention_coverage (median) | 1.00 |
| Stage 3: citation_in_ground_truth (median) | 1.00 |
| Stage 3: citation_segment_match (median) | 1.00 |
| Stage 3: citation_in_retrieved (median) | 1.00 |

## Per-case summary

| id | recall@5 | seg-recall@5 | MRR | must_mention | citations | flags |
|---|---|---|---|---|---|---|
| 001_proper_noun_kyoto | 1.00 | 0.00 | 0.50 | 1.00 | 1.00 | seg-recall<1 |
| 002_question_word_noise | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | seg-recall<1 |
| 003_recipe_segment | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 004_segment_overlap | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 005_cross_modal_image_text | 0.50 | 1.00 | 1.00 | 1.00 | 0.33 | recall<1 |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - |
| 008_proper_noun_preservation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 009_long_video_olive | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 010_long_video_chiikawa_taste | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_long_video_atashinchi_game | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_vault_md_tech_paper | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | recall<1 |
| 012_vault_md_cat_diary_primary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 012_vocab_mismatch_vit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都 紅葉'
- top_10: ['✗mJLIgl6IaF3d', '✓xv2nmOPUTJvi', '✗z6H1Vnkm-4Z7', '✗E91YCBdvjlOT', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫'
- top_10: ['✓ZmxTegJvjxAy', '✓5z3uW83PgOb1', '✗j_UPRviAs1Mm', '✗z6H1Vnkm-4Z7']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 005_cross_modal_image_text

- query: '黒猫の写真と日記を見せて'
- keywords: '黒猫'
- top_10: ['✓ZmxTegJvjxAy', '✗5z3uW83PgOb1', '✗qf--MrL_g_Gt', '✗z6H1Vnkm-4Z7', '✗mJLIgl6IaF3d']
- ground_truth not retrieved: ['images/blackcat.jpg']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 011_vault_md_tech_paper

- query: 'Vision Transformer ってどういう仕組み？'
- keywords: 'Vision Transformer 仕組み'
- top_10: []
- ground_truth not retrieved: ['docs/tech_paper.md']
- Stage 3 must_mention values: [0.0, 0.0, 0.0]

<details>
<summary>Full appendix (raw runs)</summary>

```yaml
case: 001_proper_noun_kyoto
query: 京都の紅葉について何を言ってた？
stage1:
  keywords: 京都 紅葉
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - mJLIgl6IaF3d
  - xv2nmOPUTJvi
  - z6H1Vnkm-4Z7
  - E91YCBdvjlOT
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 0.5
  precision@5: 0.4
stage3_runs:
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜間はライトアップされます。また、哲学の道を歩きながらもみじ狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4904
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜間はライトアップされます。また、哲学の道を歩きながらもみじ狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5585
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜間はライトアップされます。また、哲学の道を歩きながらもみじ狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5862

---
case: 002_question_word_noise
query: 黒猫の共通点は？
stage1:
  keywords: 黒猫
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  - j_UPRviAs1Mm
  - z6H1Vnkm-4Z7
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 黒猫には、遺伝的に病気に強いという点や、日本では幸運の象徴とされている点、そして人懐こく賢い個体が多いという性格的特徴が共通点として挙げられます。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 3718
- answer: 黒猫には、遺伝的に病気に強いという点や、日本では幸運の象徴とされている点、そして人懐こく賢い個体が多いという性格的特徴が共通点として挙げられます。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 3870
- answer: 黒猫には、遺伝的に病気に強いという点や、日本では幸運の象徴とされている点、そして人懐こく賢い個体が多いという性格的特徴が共通点として挙げられます。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 3392

---
case: 003_recipe_segment
query: カレーの作り方は？
stage1:
  keywords: カレー
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  - z6H1Vnkm-4Z7
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 'カレーの作り方として、以下の手順が紹介されています。


    1. 玉ねぎをみじん切りにして、飴色になるまでよく炒めるのがコツです。

    2. 鶏肉を加えて表面を焼き、水とカレールーを入れて20分ほど煮込みます。

    3. 最後にガラムマサラを振ると本格的な味になります。


    また、ルーを入れる前に一度火を止めるとダマになりにくいという知恵も紹介されています。'
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5576
- answer: 'カレーの作り方として、以下の手順が紹介されています。


    1. 玉ねぎをみじん切りにして、飴色になるまでよく炒めるのがコツです。

    2. 鶏肉を加えて表面を焼き、水とカレールーを入れて20分ほど煮込みます。

    3. 最後にガラムマサラを振ると本格的な味になります。


    また、ルーを入れる前に一度火を止めるとダマになりにくいという知恵も紹介されています。'
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5592
- answer: 'カレーの作り方として、以下の手順が紹介されています。


    1. 玉ねぎをみじん切りにして、飴色になるまでよく炒めるのがコツです。

    2. 鶏肉を加えて表面を焼き、水とカレールーを入れて20分ほど煮込みます。

    3. 最後にガラムマサラを振ると本格的な味になります。


    また、ルーを入れる前に一度火を止めるとダマになりにくいという知恵も紹介されています。'
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5279

---
case: 004_segment_overlap
query: 永観堂のライトアップについて何て言ってた？
stage1:
  keywords: 永観堂 ライトアップ
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - qf--MrL_g_Gt
  - xv2nmOPUTJvi
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 3892
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4505
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4967

---
case: 005_cross_modal_image_text
query: 黒猫の写真と日記を見せて
stage1:
  keywords: 黒猫
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  - qf--MrL_g_Gt
  - z6H1Vnkm-4Z7
  - mJLIgl6IaF3d
  file_recall@5: 0.5
  file_recall@10: 0.5
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 黒猫の写真として、黒猫のミケが押し入れの奥で丸まって眠っていたという記述があります。また、日記には「黒猫のミケが押し入れで子猫のように丸まっていた」という記述もあり、これは黒猫に関する記録です。動画ファイル（black_cat_facts.mp4）は黒猫に関する情報を含んでいますが、具体的な写真や画像は提供されていません。
  citation_file_ids:
  - qf--MrL_g_Gt
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8179
- answer: 黒猫の写真として、黒猫のミケが押し入れの奥で丸まって眠っていたという記述があります。また、日記には「黒猫のミケが押し入れで子猫のように丸まっていた」という記述もあり、これは黒猫に関する記録です。動画ファイル（black_cat_facts.mp4）は黒猫に関する情報を含んでいますが、具体的な写真や画像は提供されていません。
  citation_file_ids:
  - qf--MrL_g_Gt
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7224
- answer: 黒猫の写真として、黒猫のミケが押し入れの奥で丸まって眠っていたという記述があります。また、日記には「黒猫のミケが押し入れで子猫のように丸まっていた」という記述もあり、これは黒猫に関する記録です。動画ファイル（black_cat_facts.mp4）は黒猫に関する情報を含んでいますが、具体的な写真や画像は提供されていません。
  citation_file_ids:
  - qf--MrL_g_Gt
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6898

---
case: 006_no_answer
query: 宇宙ステーションの食事メニューは何ですか？
stage1:
  keywords: 宇宙ステーション 食事メニュー
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - z6H1Vnkm-4Z7
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 0.0
  precision@5: 0.0
stage3_runs:
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4119
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4545
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4734

---
case: 007_blocklist_question_word
query: 機械学習と京都の紅葉動画の共通点は？
stage1:
  keywords: 機械学習 京都 紅葉
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  - QzLEo8LTOgGr
  - E91YCBdvjlOT
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.6
stage3_runs:
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について判断できる情報は見つかりませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 3861
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について判断できる情報は見つかりませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4214
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について判断できる情報は見つかりませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4009

---
case: 008_proper_noun_preservation
query: 東福寺の通天橋と哲学の道はどんな感じ？
stage1:
  keywords: 東福寺 通天橋 哲学の道
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - qf--MrL_g_Gt
  - xv2nmOPUTJvi
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 東福寺の通天橋からの紅葉の眺めは絶景で、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、良い思い出になると述べられています。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7489
- answer: 東福寺の通天橋からの紅葉の眺めは絶景で、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、良い思い出になると述べられています。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6909
- answer: 東福寺の通天橋からの紅葉の眺めは絶景で、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、良い思い出になると述べられています。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6907

---
case: 009_long_video_olive
query: 200年以上の古いオリーブの話、何て言ってた？
stage1:
  keywords: 200年 オリーブ
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - mJLIgl6IaF3d
  - E91YCBdvjlOT
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 小木のオリーブは、少なくとも200年から300年以上生きているとされています。200年という年数は、日本が江戸時代、徳川幕府の時代から生きていることに相当します。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6218
- answer: 小木のオリーブは、少なくとも200年から300年以上生きているとされています。200年という年数は、日本が江戸時代、徳川幕府の時代から生きていることに相当します。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6129
- answer: 小木のオリーブは、少なくとも200年から300年以上生きているとされています。200年という年数は、日本が江戸時代、徳川幕府の時代から生きていることに相当します。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6105

---
case: 010_long_video_chiikawa_taste
query: ちいかわアイスの味の特徴は？
stage1:
  keywords: ちいかわアイス 味 特徴
  must_include_coverage: 0.5
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - z6H1Vnkm-4Z7
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 5866
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 6019
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 6087

---
case: 011_long_video_atashinchi_game
query: 半年かけてゲームを手に入れた話、何て言ってた？
stage1:
  keywords: 半年 ゲーム
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - E91YCBdvjlOT
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 「半年かかって溜めた小豆がいと落としたまで ようやく手に入れたこのゲームと共に今日この日曜日だけは ゆいぎな一日にしたいんだあ」というセリフを言っていました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4216
- answer: 「半年かかって溜めた小豆がいと落としたまで ようやく手に入れたこのゲームと共に今日この日曜日だけは ゆいぎな一日にしたいんだあ」というセリフを言っていました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 3991
- answer: 「半年かかって溜めた小豆がいと落としたまで ようやく手に入れたこのゲームと共に今日この日曜日だけは ゆいぎな一日にしたいんだあ」というセリフを言っていました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4719

---
case: 011_vault_md_tech_paper
query: Vision Transformer ってどういう仕組み？
stage1:
  keywords: Vision Transformer 仕組み
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids: []
  file_recall@5: 0.0
  file_recall@10: 0.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 0.0
  precision@5: 0.0
stage3_runs:
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 570
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 579
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 632

---
case: 012_vault_md_cat_diary_primary
query: うちのミケってどんな性格？
stage1:
  keywords: ミケ 性格
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - E91YCBdvjlOT
  - 5z3uW83PgOb1
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: ミケは人懐っこく、とても賢い性格であると記載されています。また、来客に対しても警戒心が薄く、膝に乗ってくるほど人懐っこい様子が描かれています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4482
- answer: ミケは人懐っこく、とても賢い性格であると記載されています。また、来客に対しても警戒心が薄く、膝に乗ってくるほど人懐っこい様子が描かれています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4719
- answer: ミケは人懐っこく、とても賢い性格であると記載されています。また、来客に対しても警戒心が薄く、膝に乗ってくるほど人懐っこい様子が描かれています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4699

---
case: 012_vocab_mismatch_vit
query: 畳み込みを使わずに画像処理を実現した新しいアーキテクチャは何？
stage1:
  keywords: 畳み込みを使わずに画像処理を実現した新しいアーキテクチャは何？
  must_include_coverage: 1.0
  must_exclude_violations: 3
stage2:
  top_file_ids:
  - QzLEo8LTOgGr
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4321
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4286
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4197

---
```
</details>

## Pair comparison vs phase4 baseline (hierarchical off, HEAD=aa1dfb5)

- baseline: `/eval-data/reports/phase4_baseline_off.json`
- common cases: 14

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 1: must_include_coverage | 0 | 0 | 14 |
| Stage 1: must_exclude_violations | 0 | 0 | 14 |
| Stage 2: recall@5 (file) | 0 | 0 | 14 |
| Stage 2: recall@10 (file) | 0 | 0 | 14 |
| Stage 2: segment recall@5 | 0 | 0 | 14 |
| Stage 2: MRR | 0 | 0 | 14 |
| Stage 3: must_mention (median) | 0 | 0 | 14 |
| Stage 3: citation_in_ground_truth (median) | 0 | 1 | 13 |
| Stage 3: citation_segment_match (median) | 1 | 1 | 12 |
| Stage 3: citation_in_retrieved (median) | 0 | 1 | 13 |
