# Eval Report

- date: 2026-04-30T10:26:09Z
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
    sha256: 0eaf2240d392df4807f508a26f549a132b18ede7c0041107abe330bfab98ad27
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
- total_cases: 20

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 1.00 |
| Stage 1: must_exclude_violations (sum) | 8 |
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
| 004_segment_overlap | 0.50 | 0.00 | 1.00 | 1.00 | 1.00 | recall<1,seg-recall<1 |
| 005_cross_modal_image_text | 0.50 | 1.00 | 1.00 | 1.00 | 0.33 | recall<1 |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 008_proper_noun_preservation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 009_long_video_olive | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 010_long_video_chiikawa_taste | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_long_video_atashinchi_game | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_vault_md_tech_paper | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| 012_vault_md_cat_diary_primary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 012_vocab_mismatch_vit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 013_required_proper_noun_han | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 014_required_proper_noun_latin | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| 015_required_kana_chiikawa | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | - |
| 016_required_compound_old_olive | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | - |
| 017_required_kuroneko_curry | 0.00 | 1.00 | 0.17 | 1.00 | 0.00 | recall<1 |
| 018_required_machine_learning_method | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都 紅葉'
- top_10: ['✗mJLIgl6IaF3d', '✓xv2nmOPUTJvi', '✗z6H1Vnkm-4Z7', '✗E91YCBdvjlOT', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫 共通点'
- top_10: ['✓ZmxTegJvjxAy', '✓5z3uW83PgOb1', '✗j_UPRviAs1Mm', '✗z6H1Vnkm-4Z7']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 004_segment_overlap

- query: '永観堂のライトアップについて何て言ってた？'
- keywords: '永観堂 ライトアップ'
- top_10: ['✓qf--MrL_g_Gt']
- ground_truth not retrieved: ['videos/kyoto_autumn.mp4']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 005_cross_modal_image_text

- query: '黒猫の写真と日記を見せて'
- keywords: '黒猫 写真 日記'
- top_10: ['✓ZmxTegJvjxAy', '✗5z3uW83PgOb1', '✗qf--MrL_g_Gt', '✗z6H1Vnkm-4Z7', '✗mJLIgl6IaF3d']
- ground_truth not retrieved: ['images/blackcat.jpg']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 017_required_kuroneko_curry

- query: '黒猫が出てくるカレーの話を教えて'
- keywords: '黒猫 カレー 話'
- top_10: ['✗ZmxTegJvjxAy', '✗5z3uW83PgOb1', '✗j_UPRviAs1Mm', '✗6ZUevJjXSBVz', '✗z6H1Vnkm-4Z7', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

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
- answer: 去年の秋に家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めが絶景で、橋の下一面に広がる紅葉の海に皆で感動しました。また、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは異なる表情を見せてくれました。哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、家族にとって良い思い出になったそうです。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7283
- answer: 去年の秋に家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めが絶景で、橋の下一面に広がる紅葉の海に皆で感動しました。また、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは異なる表情を見せてくれました。哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、家族にとって良い思い出になったそうです。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9031
- answer: 去年の秋に家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めが絶景で、橋の下一面に広がる紅葉の海に皆で感動しました。また、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは異なる表情を見せてくれました。哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、家族にとって良い思い出になったそうです。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9557

---
case: 002_question_word_noise
query: 黒猫の共通点は？
stage1:
  keywords: 黒猫 共通点
  must_include_coverage: 1.0
  must_exclude_violations: 1
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
- answer: 黒猫には、遺伝的に病気に強いと言われている点や、日本では古くから幸運の象徴とされてきたという共通点があります。また、性格面では人懐こく賢い個体が多いという傾向が指摘されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4959
- answer: 黒猫には、遺伝的に病気に強いと言われている点や、日本では古くから幸運の象徴とされてきたという共通点があります。また、性格面では人懐こく賢い個体が多いという傾向が指摘されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 6754
- answer: 黒猫には、遺伝的に病気に強いと言われている点や、日本では古くから幸運の象徴とされてきたという共通点があります。また、性格面では人懐こく賢い個体が多いという傾向が指摘されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8062

---
case: 003_recipe_segment
query: カレーの作り方は？
stage1:
  keywords: カレー 作り方
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
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
  took_ms: 6329
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
  took_ms: 6663
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
  took_ms: 6673

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
  file_recall@5: 0.5
  file_recall@10: 0.5
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 3788
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 5170
- answer: 永観堂の紅葉は夜のライトアップでは、昼とは全く違う表情を見せてくれると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 5835

---
case: 005_cross_modal_image_text
query: 黒猫の写真と日記を見せて
stage1:
  keywords: 黒猫 写真 日記
  must_include_coverage: 1.0
  must_exclude_violations: 2
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
- answer: 黒猫に関する情報として、日記には「黒猫ミケ」という名前の黒猫のプロフィールや、最近の様子、小さな事件についての記述があります。また、家族の思い出ノートの「その他の日々」の項目には、「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。写真については、提供されたファイル情報の中には黒猫の写真はありませんが、動画ファイルには黒猫に関する情報が含まれているものもあります。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8958
- answer: 黒猫に関する情報として、日記には「黒猫ミケ」という名前の黒猫のプロフィールや、最近の様子、小さな事件についての記述があります。また、家族の思い出ノートの「その他の日々」の項目には、「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。写真については、提供されたファイル情報の中には黒猫の写真はありませんが、動画ファイルには黒猫に関する情報が含まれているものもあります。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8231
- answer: 黒猫に関する情報として、日記には「黒猫ミケ」という名前の黒猫のプロフィールや、最近の様子、小さな事件についての記述があります。また、家族の思い出ノートの「その他の日々」の項目には、「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。写真については、提供されたファイル情報の中には黒猫の写真はありませんが、動画ファイルには黒猫に関する情報が含まれているものもあります。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  - 5z3uW83PgOb1
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.3333333333333333
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7574

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
  took_ms: 3740
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 5038
- answer: 提供されたファイル情報には、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 5183

---
case: 007_blocklist_question_word
query: 機械学習と京都の紅葉動画の共通点は？
stage1:
  keywords: 京都 機械学習 紅葉 動画
  must_include_coverage: 1.0
  must_exclude_violations: 1
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
- answer: 提供されたファイル情報からは、機械学習に関する情報と京都の紅葉に関する情報がそれぞれ別々に存在することは確認できますが、これら二つのトピックに共通する点についての直接的な記述は見当たりませんでした。
  citation_file_ids:
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 5784
- answer: 提供されたファイル情報からは、機械学習に関する情報と京都の紅葉に関する情報がそれぞれ別々に存在することは確認できますが、これら二つのトピックに共通する点についての直接的な記述は見当たりませんでした。
  citation_file_ids:
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 9224
- answer: 提供されたファイル情報からは、機械学習に関する情報と京都の紅葉に関する情報がそれぞれ別々に存在することは確認できますが、これら二つのトピックに共通する点についての直接的な記述は見当たりませんでした。
  citation_file_ids:
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7258

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
  took_ms: 6415
- answer: 東福寺の通天橋からの紅葉の眺めは絶景で、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、良い思い出になると述べられています。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7765
- answer: 東福寺の通天橋からの紅葉の眺めは絶景で、橋の下一面に広がる紅葉の海は息をのむほど美しいです。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、良い思い出になると述べられています。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7831

---
case: 009_long_video_olive
query: 200年以上の古いオリーブの話、何て言ってた？
stage1:
  keywords: オリーブ 200年以上 古い 話
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - mJLIgl6IaF3d
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
  took_ms: 5998
- answer: 小木のオリーブは、少なくとも200年から300年以上生きているとされています。200年という年数は、日本が江戸時代、徳川幕府の時代から生きていることに相当します。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7485
- answer: 小木のオリーブは、少なくとも200年から300年以上生きているとされています。200年という年数は、日本が江戸時代、徳川幕府の時代から生きていることに相当します。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7665

---
case: 010_long_video_chiikawa_taste
query: ちいかわアイスの味の特徴は？
stage1:
  keywords: ちいかわ アイス 味 特徴
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
  took_ms: 5861
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 7519
- answer: ちいかわ一口ミルキーアイスは、ヨーグルトの酸味があり、甘さが控えめな味わいです。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 7756

---
case: 011_long_video_atashinchi_game
query: 半年かけてゲームを手に入れた話、何て言ってた？
stage1:
  keywords: ゲーム 話
  must_include_coverage: 0.5
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
  took_ms: 3180
- answer: 「半年かかって溜めた小豆がいと落としたまで ようやく手に入れたこのゲームと共に今日この日曜日だけは ゆいぎな一日にしたいんだあ」というセリフを言っていました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4885
- answer: 「半年かかって溜めた小豆がいと落としたまで ようやく手に入れたこのゲームと共に今日この日曜日だけは ゆいぎな一日にしたいんだあ」というセリフを言っていました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4994

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
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1869
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1900
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1903

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
  took_ms: 5955
- answer: ミケは人懐っこく、とても賢い性格であると記載されています。また、来客に対しても警戒心が薄く、膝に乗ってくるほど人懐っこい様子が描かれています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7994
- answer: ミケは人懐っこく、とても賢い性格であると記載されています。また、来客に対しても警戒心が薄く、膝に乗ってくるほど人懐っこい様子が描かれています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8330

---
case: 012_vocab_mismatch_vit
query: 畳み込みを使わずに画像処理を実現した新しいアーキテクチャは何？
stage1:
  keywords: 畳み込み 画像処理 アーキテクチャ
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
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 4913
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7016
- answer: 提供された技術ノートには、畳み込み（CNN）を使わずに画像処理を実現した新しいアーキテクチャとして、Vision Transformer (ViT)
    が挙げられています。ViTは画像をパッチ系列に分割し、Transformerで直接処理するモデルです。
  citation_file_ids:
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7158

---
case: 013_required_proper_noun_han
query: 東福寺の紅葉はどうだった？
stage1:
  keywords: 東福寺 紅葉
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、栄感道の紅葉も有名で、夜にはライトアップされるとのことです。また、鉄画区の道を歩きながら紅葉狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9315
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、栄感道の紅葉も有名で、夜にはライトアップされるとのことです。また、鉄画区の道を歩きながら紅葉狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11424
- answer: 東福寺の通天橋からの紅葉の眺めは絶景であり、栄感道の紅葉も有名で、夜にはライトアップされるとのことです。また、鉄画区の道を歩きながら紅葉狩りを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11883

---
case: 014_required_proper_noun_latin
query: ViTの強みは何ですか？
stage1:
  keywords: ViT 強み
  must_include_coverage: 1.0
  must_exclude_violations: 0
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
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 2410
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 2310
- answer: null
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 2399

---
case: 015_required_kana_chiikawa
query: ちいかわのアイスはどんな味？
stage1:
  keywords: ちいかわ アイス
  must_include_coverage: 1.0
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
- answer: 試食したところ、ヨーグルトの酸味があり甘さが控えめな味わいでした。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 0.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 13406
- answer: 試食したところ、ヨーグルトの酸味があり甘さが控えめな味わいでした。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 0.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 14977
- answer: 試食したところ、ヨーグルトの酸味があり甘さが控えめな味わいでした。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 0.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 14325

---
case: 016_required_compound_old_olive
query: 古木オリーブの値段はいくらだった？
stage1:
  keywords: 古木オリーブ 値段
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 動画内では、古木オリーブの具体的な購入価格についての言及はありませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 5309
- answer: 動画内では、古木オリーブの具体的な購入価格についての言及はありませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7089
- answer: 動画内では、古木オリーブの具体的な購入価格についての言及はありませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7410

---
case: 017_required_kuroneko_curry
query: 黒猫が出てくるカレーの話を教えて
stage1:
  keywords: 黒猫 カレー 話
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  - j_UPRviAs1Mm
  - 6ZUevJjXSBVz
  - z6H1Vnkm-4Z7
  - qf--MrL_g_Gt
  file_recall@5: 0.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 0.16666666666666666
  precision@5: 0.0
stage3_runs:
- answer: 提供されたファイル情報の中には、「黒猫が出てくるカレー」に関する具体的な情報は含まれていませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 4544
- answer: 提供されたファイル情報の中には、「黒猫が出てくるカレー」に関する具体的な情報は含まれていませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 6279
- answer: 提供されたファイル情報の中には、「黒猫が出てくるカレー」に関する具体的な情報は含まれていませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 6223

---
case: 018_required_machine_learning_method
query: 機械学習でよく使う手法には何がありますか？
stage1:
  keywords: 機械学習 手法
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids:
  - rG-RB9DbOblv
  - QzLEo8LTOgGr
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: '機械学習でよく使われる代表的なアルゴリズムや手法には、以下のようなものがあります。


    *   **アルゴリズムの類型として:** 線形回帰、ロジスティック回帰、決定木、ランダムフォレスト、勾配ブースティング、ニューラルネットワーク（MLP、CNN、Transformer）などが挙げられます。

    *   **代表的なモデルやアーキテクチャとして:** 選挙会期、決定木、ニューラルネットワークなどが挙げられています。

    *   **画像分類などの具体的なタスクでは:** ResNet（残差接続を導入したモデル）、Vision Transformer (ViT)（…'
  citation_file_ids:
  - rG-RB9DbOblv
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7659
- answer: '機械学習でよく使われる代表的なアルゴリズムや手法には、以下のようなものがあります。


    *   **アルゴリズムの類型として:** 線形回帰、ロジスティック回帰、決定木、ランダムフォレスト、勾配ブースティング、ニューラルネットワーク（MLP、CNN、Transformer）などが挙げられます。

    *   **代表的なモデルやアーキテクチャとして:** 選挙会期、決定木、ニューラルネットワークなどが挙げられています。

    *   **画像分類などの具体的なタスクでは:** ResNet（残差接続を導入したモデル）、Vision Transformer (ViT)（…'
  citation_file_ids:
  - rG-RB9DbOblv
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9319
- answer: '機械学習でよく使われる代表的なアルゴリズムや手法には、以下のようなものがあります。


    *   **アルゴリズムの類型として:** 線形回帰、ロジスティック回帰、決定木、ランダムフォレスト、勾配ブースティング、ニューラルネットワーク（MLP、CNN、Transformer）などが挙げられます。

    *   **代表的なモデルやアーキテクチャとして:** 選挙会期、決定木、ニューラルネットワークなどが挙げられています。

    *   **画像分類などの具体的なタスクでは:** ResNet（残差接続を導入したモデル）、Vision Transformer (ViT)（…'
  citation_file_ids:
  - rG-RB9DbOblv
  - QzLEo8LTOgGr
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9227

---
```
</details>

## Pair comparison vs 2026-04-30-required-keyword-phase0-baseline.json

- baseline: `/eval-data/reports/2026-04-30-required-keyword-phase0-baseline.json`
- common cases: 20

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 1: must_include_coverage | 0 | 1 | 19 |
| Stage 1: must_exclude_violations | 1 | 4 | 15 |
| Stage 2: recall@5 (file) | 3 | 1 | 16 |
| Stage 2: recall@10 (file) | 3 | 1 | 16 |
| Stage 2: segment recall@5 | 0 | 1 | 19 |
| Stage 2: MRR | 2 | 0 | 18 |
| Stage 3: must_mention (median) | 0 | 0 | 20 |
| Stage 3: citation_in_ground_truth (median) | 0 | 0 | 20 |
| Stage 3: citation_segment_match (median) | 0 | 0 | 20 |
| Stage 3: citation_in_retrieved (median) | 0 | 0 | 20 |
