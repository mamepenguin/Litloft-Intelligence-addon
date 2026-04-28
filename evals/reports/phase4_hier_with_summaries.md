# Eval Report: phase4 hier with summaries (14 files, 11 with summary)

- date: 2026-04-28T08:48:01Z
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
    sha256: 5b6478efc3ac453dc87e5755772f85eab85e12a69541af680950492d1f40c92e
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
| 005_cross_modal_image_text | 0.50 | 1.00 | 1.00 | 1.00 | 0.75 | recall<1 |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - |
| 008_proper_noun_preservation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 009_long_video_olive | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 010_long_video_chiikawa_taste | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_long_video_atashinchi_game | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | unstable |
| 011_vault_md_tech_paper | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 012_vault_md_cat_diary_primary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 012_vocab_mismatch_vit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都 紅葉'
- top_10: ['✗mJLIgl6IaF3d', '✓xv2nmOPUTJvi', '✓qf--MrL_g_Gt', '✗z6H1Vnkm-4Z7', '✗E91YCBdvjlOT']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫'
- top_10: ['✓5z3uW83PgOb1', '✓ZmxTegJvjxAy', '✗j_UPRviAs1Mm', '✗z6H1Vnkm-4Z7']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 005_cross_modal_image_text

- query: '黒猫の写真と日記を見せて'
- keywords: '黒猫'
- top_10: ['✓ZmxTegJvjxAy', '✗5z3uW83PgOb1', '✗qf--MrL_g_Gt', '✗z6H1Vnkm-4Z7', '✗mJLIgl6IaF3d']
- ground_truth not retrieved: ['images/blackcat.jpg']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 011_long_video_atashinchi_game

- query: '半年かけてゲームを手に入れた話、何て言ってた？'
- keywords: '半年 ゲーム'
- top_10: ['✓E91YCBdvjlOT']
- Stage 3 must_mention values: [0.0, 0.0, 1.0]

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
  - qf--MrL_g_Gt
  - z6H1Vnkm-4Z7
  - E91YCBdvjlOT
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 0.5
  precision@5: 0.4
stage3_runs:
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、永観堂のライトアップも素晴らしく、昼とは違う表情を楽しめます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6415
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、永観堂のライトアップも素晴らしく、昼とは違う表情を楽しめます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7409
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、永観堂のライトアップも素晴らしく、昼とは違う表情を楽しめます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7140

---
case: 002_question_word_noise
query: 黒猫の共通点は？
stage1:
  keywords: 黒猫
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  - j_UPRviAs1Mm
  - z6H1Vnkm-4Z7
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 黒猫には、遺伝的に病気に強いとされる点、日本では幸運の象徴とされる一方で欧米では不吉とされる文化的な側面、そして賢く人懐こい個体が多いという性格的な特徴が共通点として挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5334
- answer: 黒猫には、遺伝的に病気に強いとされる点、日本では幸運の象徴とされる一方で欧米では不吉とされる文化的な側面、そして賢く人懐こい個体が多いという性格的な特徴が共通点として挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5921
- answer: 黒猫には、遺伝的に病気に強いとされる点、日本では幸運の象徴とされる一方で欧米では不吉とされる文化的な側面、そして賢く人懐こい個体が多いという性格的な特徴が共通点として挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5746

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
- answer: カレーの作り方としては、まず玉ねぎをみじん切りにして飴色になるまで炒める工程から始まります。次に鶏肉を加えて表面を焼き、水とカレールーを加えて二十分ほど煮込みます。最後にガラムマサラを振ると本格的な味になると説明されています。また、カレールーを入れる前には一度火を止めるとダマになりにくいという知恵も紹介されています。
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6804
- answer: カレーの作り方としては、まず玉ねぎをみじん切りにして飴色になるまで炒める工程から始まります。次に鶏肉を加えて表面を焼き、水とカレールーを加えて二十分ほど煮込みます。最後にガラムマサラを振ると本格的な味になると説明されています。また、カレールーを入れる前には一度火を止めるとダマになりにくいという知恵も紹介されています。
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7306
- answer: カレーの作り方としては、まず玉ねぎをみじん切りにして飴色になるまで炒める工程から始まります。次に鶏肉を加えて表面を焼き、水とカレールーを加えて二十分ほど煮込みます。最後にガラムマサラを振ると本格的な味になると説明されています。また、カレールーを入れる前には一度火を止めるとダマになりにくいという知恵も紹介されています。
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7036

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
- answer: 永観堂のライトアップについては、夜になると昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4912
- answer: 永観堂のライトアップについては、夜になると昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4856
- answer: 永観堂のライトアップについては、夜になると昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4854

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
- answer: '黒猫の写真や日記として、黒猫ミケに関する情報が「cat_diary.md」に記録されています。この日記には、ミケのプロフィール（真っ黒な毛色、人懐こく賢い性格）や、最近の様子（窓辺での日向ぼっこ、新しいおもちゃへの対応、来客への人懐っこさ）が記されています。また、「小さな事件」として、押し入れの奥で丸まって眠っていたエピソードも記載されています。


    また、別の「家族の思い出ノート」には、黒猫のミケが押し入れで子猫のように丸まっていたという記述があります。'
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.75
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8722
- answer: '黒猫の写真や日記として、黒猫ミケに関する情報が「cat_diary.md」に記録されています。この日記には、ミケのプロフィール（真っ黒な毛色、人懐こく賢い性格）や、最近の様子（窓辺での日向ぼっこ、新しいおもちゃへの対応、来客への人懐っこさ）が記されています。また、「小さな事件」として、押し入れの奥で丸まって眠っていたエピソードも記載されています。


    また、別の「家族の思い出ノート」には、黒猫のミケが押し入れで子猫のように丸まっていたという記述があります。'
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.75
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10899
- answer: '黒猫の写真や日記として、黒猫ミケに関する情報が「cat_diary.md」に記録されています。この日記には、ミケのプロフィール（真っ黒な毛色、人懐こく賢い性格）や、最近の様子（窓辺での日向ぼっこ、新しいおもちゃへの対応、来客への人懐っこさ）が記されています。また、「小さな事件」として、押し入れの奥で丸まって眠っていたエピソードも記載されています。


    また、別の「家族の思い出ノート」には、黒猫のミケが押し入れで子猫のように丸まっていたという記述があります。'
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.75
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11775

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
  took_ms: 4365
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 5067
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 5043

---
case: 007_blocklist_question_word
query: 機械学習と京都の紅葉動画の共通点は？
stage1:
  keywords: 機械学習 京都 紅葉
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - QzLEo8LTOgGr
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  - E91YCBdvjlOT
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.6
stage3_runs:
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について直接的な関連性を見つけることはできませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7854
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について直接的な関連性を見つけることはできませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7527
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の共通点について直接的な関連性を見つけることはできませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 9994

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
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、永観堂の紅葉も素晴らしく、夜間のライトアップでは昼とは異なる表情を見せてくれます。哲学の道では、散歩をしながら紅葉狩りを楽しむことができ、家族にとって良い思い出になった場所です。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 13488
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、永観堂の紅葉も素晴らしく、夜間のライトアップでは昼とは異なる表情を見せてくれます。哲学の道では、散歩をしながら紅葉狩りを楽しむことができ、家族にとって良い思い出になった場所です。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11213
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、永観堂の紅葉も素晴らしく、夜間のライトアップでは昼とは異なる表情を見せてくれます。哲学の道では、散歩をしながら紅葉狩りを楽しむことができ、家族にとって良い思い出になった場所です。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11561

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
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 小木のオリーブは、少なくとも200年から300年以上生きている非常に古い歴史を持つオリーブであると説明されています。このオリーブは、徳川幕府の時代から生きている可能性があり、その歴史が体に刻まれていると述べられています。
  citation_file_ids:
  - mJLIgl6IaF3d
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 14478
- answer: 小木のオリーブは、少なくとも200年から300年以上生きている非常に古い歴史を持つオリーブであると説明されています。このオリーブは、徳川幕府の時代から生きている可能性があり、その歴史が体に刻まれていると述べられています。
  citation_file_ids:
  - mJLIgl6IaF3d
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 13402
- answer: 小木のオリーブは、少なくとも200年から300年以上生きている非常に古い歴史を持つオリーブであると説明されています。このオリーブは、徳川幕府の時代から生きている可能性があり、その歴史が体に刻まれていると述べられています。
  citation_file_ids:
  - mJLIgl6IaF3d
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 14576

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
  took_ms: 14262
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 14933
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 13259

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
- answer: 半年前くらいに、インスタグラムを見ていたところから、イベントに参加する機会があり、そこで樹齢少なくとも200年から300年以上ある小木のオリーブを購入したという話が語られています。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 13691
- answer: 半年前くらいに、インスタグラムを見ていたところから、イベントに参加する機会があり、そこで樹齢少なくとも200年から300年以上ある小木のオリーブを購入したという話が語られています。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10706
- answer: 半年かけて手に入れたゲームについて、あるエピソードでは「ファイナルスタークエスト13」を手に入れたことが語られています。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 9191

---
case: 011_vault_md_tech_paper
query: Vision Transformer ってどういう仕組み？
stage1:
  keywords: Vision Transformer 仕組み
  must_include_coverage: 1.0
  must_exclude_violations: 1
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
- answer: Vision Transformer (ViT) は、画像をパッチ系列に分割し、Transformerを用いて直接処理する仕組みを持っています。このアプローチは、大規模な事前学習を前提としており、ResNet系のモデルを上回る精度を示すことが報告されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10036
- answer: Vision Transformer (ViT) は、画像をパッチ系列に分割し、Transformerを用いて直接処理する仕組みを持っています。このアプローチは、大規模な事前学習を前提としており、ResNet系のモデルを上回る精度を示すことが報告されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7339
- answer: Vision Transformer (ViT) は、画像をパッチ系列に分割し、Transformerを用いて直接処理する仕組みを持っています。このアプローチは、大規模な事前学習を前提としており、ResNet系のモデルを上回る精度を示すことが報告されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6909

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
- answer: ミケの性格は、人懐こく、とても賢いとされています。来客に対しても警戒心が薄く、人懐こい様子が記されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9899
- answer: ミケの性格は、人懐こく、とても賢いとされています。来客に対しても警戒心が薄く、人懐こい様子が記されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7992
- answer: ミケの性格は、人懐こく、とても賢いとされています。来客に対しても警戒心が薄く、人懐こい様子が記されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9546

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
- answer: 提供されたファイル情報の中には、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャについての具体的な記述は見当たりませんでした。ただし、画像分類の代表的なアーキテクチャとして、画像をパッチ系列として処理しTransformerで直接処理するVision
    Transformer (ViT)が紹介されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10355
- answer: 提供されたファイル情報の中には、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャについての具体的な記述は見当たりませんでした。ただし、画像分類の代表的なアーキテクチャとして、画像をパッチ系列として処理しTransformerで直接処理するVision
    Transformer (ViT)が紹介されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9048
- answer: 提供されたファイル情報の中には、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャについての具体的な記述は見当たりませんでした。ただし、画像分類の代表的なアーキテクチャとして、画像をパッチ系列として処理しTransformerで直接処理するVision
    Transformer (ViT)が紹介されています。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8525

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
| Stage 2: recall@5 (file) | 1 | 0 | 13 |
| Stage 2: recall@10 (file) | 1 | 0 | 13 |
| Stage 2: segment recall@5 | 0 | 0 | 14 |
| Stage 2: MRR | 1 | 0 | 13 |
| Stage 3: must_mention (median) | 1 | 1 | 12 |
| Stage 3: citation_in_ground_truth (median) | 2 | 2 | 10 |
| Stage 3: citation_segment_match (median) | 1 | 2 | 11 |
| Stage 3: citation_in_retrieved (median) | 1 | 1 | 12 |
