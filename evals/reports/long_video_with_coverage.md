# Eval Report: long videos + coverage-based segment match

- date: 2026-04-14T13:46:41Z
- git_commit: unknown
- llm_model: gemma4:e2b
- llm_base_url: http://host.docker.internal:11434/v1
- llm_temperature: 0 (forced by runner)
- search_config:
    rag.top_k: 5
    rag.max_tokens: 1024
    search.mode: recall
- index_snapshot:
    file: /eval-data/test-drive/snapshot/search.db
    sha256: 68ca5275ea9f82f53a9175b38394109ffed7ecb774ca7b2a100692a6bb59517f
    indexed_with:
      whisper: 'openai/whisper-small'
      clip: 'llm-jp/llm-jp-clip-vit-base-patch16'
      blip: 'Salesforce/blip-image-captioning-base'
- blocklists:
    question_words.txt: b2aa63dd5c23475e50ebe81eaaf4bc8585165d6fbcb96fae0e56d5d65b12be13
    file_type_words.txt: 9c304d6f3ee5743511d3783b64f095b9507ed8233dcdf2aca665481873468907
- runs: stage1=1 stage2=1 stage3=3
- epsilon: 0.1
- drive: eval-drive
- total_cases: 11

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
| Stage 3: citation_segment_match (median) | 0.50 |
| Stage 3: citation_in_retrieved (median) | 1.00 |

## Per-case summary

| id | recall@5 | seg-recall@5 | MRR | must_mention | citations | flags |
|---|---|---|---|---|---|---|
| 001_proper_noun_kyoto | 1.00 | 0.00 | 0.50 | 1.00 | 1.00 | seg-recall<1 |
| 002_question_word_noise | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | seg-recall<1 |
| 003_recipe_segment | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 004_segment_overlap | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 005_cross_modal_image_text | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | recall<1 |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - |
| 008_proper_noun_preservation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 009_long_video_olive | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 010_long_video_chiikawa_taste | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 011_long_video_atashinchi_game | 1.00 | 0.00 | 0.50 | 0.00 | 0.00 | seg-recall<1 |

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

### 005_cross_modal_image_text

- query: '黒猫の写真と日記を見せて'
- keywords: '黒猫 写真 日記'
- top_10: ['✓ZmxTegJvjxAy', '✗5z3uW83PgOb1', '✗qf--MrL_g_Gt', '✗z6H1Vnkm-4Z7', '✗mJLIgl6IaF3d']
- ground_truth not retrieved: ['images/blackcat.jpg']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 011_long_video_atashinchi_game

- query: 'ユズが半年かけて手に入れたものは？'
- keywords: 'ユズが半年かけて手に入れたものは？'
- top_10: ['✗z6H1Vnkm-4Z7', '✓E91YCBdvjlOT', '✗mJLIgl6IaF3d']
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
- answer: 家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは絶景であり、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せるため、家族全員の良い思い出になったと記録されています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も印象に残っています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 11291
- answer: 家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは絶景であり、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せるため、家族全員の良い思い出になったと記録されています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も印象に残っています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10035
- answer: 家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは絶景であり、永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せるため、家族全員の良い思い出になったと記録されています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も印象に残っています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10298

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
- answer: 黒猫に関する共通点として、遺伝的に病気に強いと言われていること、そして性格的に人懐こく賢い個体が多いという点が挙げられます。また、日本では古くから幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7414
- answer: 黒猫に関する共通点として、遺伝的に病気に強いと言われていること、そして性格的に人懐こく賢い個体が多いという点が挙げられます。また、日本では古くから幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7436
- answer: 黒猫に関する共通点として、遺伝的に病気に強いと言われていること、そして性格的に人懐こく賢い個体が多いという点が挙げられます。また、日本では古くから幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7491

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
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めるとダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 10476
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めるとダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 10924
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めるとダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 10661

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
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6591
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6663
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6628

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
- answer: 黒猫ミケに関する日記の内容は、ミケが人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出す賢さを持っていること、また、来客に対しても人懐こい様子が記されています。また、押し入れで丸まって眠っていたという小さな事件も記録されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.5
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10380
- answer: 黒猫ミケに関する日記の内容は、ミケが人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出す賢さを持っていること、また、来客に対しても人懐こい様子が記されています。また、押し入れで丸まって眠っていたという小さな事件も記録されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.5
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10478
- answer: 黒猫ミケに関する日記の内容は、ミケが人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出す賢さを持っていること、また、来客に対しても人懐こい様子が記されています。また、押し入れで丸まって眠っていたという小さな事件も記録されています。
  citation_file_ids:
  - ZmxTegJvjxAy
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.5
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 10458

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
- answer: 提供されたファイルには、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 3451
- answer: 提供されたファイルには、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 3559
- answer: 提供されたファイルには、宇宙ステーションの食事メニューに関する情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 3545

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
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画との具体的な共通点についての直接的な情報は確認できませんでした。ファイルには機械学習の概要や、京都の紅葉に関する情報、その他の動画や文書が含まれていますが、両者の関連性についての比較情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 12066
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画との具体的な共通点についての直接的な情報は確認できませんでした。ファイルには機械学習の概要や、京都の紅葉に関する情報、その他の動画や文書が含まれていますが、両者の関連性についての比較情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 12187
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画との具体的な共通点についての直接的な情報は確認できませんでした。ファイルには機械学習の概要や、京都の紅葉に関する情報、その他の動画や文書が含まれていますが、両者の関連性についての比較情報は含まれていません。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 12451

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
- answer: 家族の思い出ノートによると、家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは「本当に絶景」であり、「橋の下一面に広がる紅葉の海」だと述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も、家族全員の良い思い出になったと記されています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8535
- answer: 家族の思い出ノートによると、家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは「本当に絶景」であり、「橋の下一面に広がる紅葉の海」だと述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も、家族全員の良い思い出になったと記されています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8975
- answer: 家族の思い出ノートによると、家族旅行で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは「本当に絶景」であり、「橋の下一面に広がる紅葉の海」だと述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しんだ時間も、家族全員の良い思い出になったと記されています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9250

---
case: 009_long_video_olive
query: 200年以上の古いオリーブの話、何て言ってた？
stage1:
  keywords: オリーブ 話
  must_include_coverage: 0.5
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
- answer: 小木オリーブは、少なくとも200年から300年以上あるとされる、非常に古い歴史を持つオリーブです。このオリーブの株は、かつてより大きなオリーブの一部が割れてできたかけらの一部であるとされています。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 9015
- answer: 小木オリーブは、少なくとも200年から300年以上あるとされる、非常に古い歴史を持つオリーブです。このオリーブの株は、かつてより大きなオリーブの一部が割れてできたかけらの一部であるとされています。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 9549
- answer: 小木オリーブは、少なくとも200年から300年以上あるとされる、非常に古い歴史を持つオリーブです。このオリーブの株は、かつてより大きなオリーブの一部が割れてできたかけらの一部であるとされています。
  citation_file_ids:
  - mJLIgl6IaF3d
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.0
  citation_in_retrieved: 1.0
  took_ms: 9999

---
case: 010_long_video_chiikawa_taste
query: ちいかわアイスの味の特徴は？
stage1:
  keywords: ちいかわアイス 味
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
- answer: ちいかわ一口ミルキーアイスは、無茶馬ヨーグルト味で、シャリシャリとした食感で、周りは硬く中はチョコレートの甘さ控えめなヨーグルトの酸味があるような感じがするとのことです。また、パッケージにはちいかわやアイスの見本、ラッコ先生、モモンガちゃんのシールがついており、カットするとスタンドにもなるという特徴があります。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.5
  citation_in_retrieved: 1.0
  took_ms: 11873
- answer: ちいかわ一口ミルキーアイスは、無茶馬ヨーグルト味で、シャリシャリとした食感で、周りは硬く中はチョコレートの甘さ控えめなヨーグルトの酸味があるような感じがするとのことです。また、パッケージにはちいかわやアイスの見本、ラッコ先生、モモンガちゃんのシールがついており、カットするとスタンドにもなるという特徴があります。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.5
  citation_in_retrieved: 1.0
  took_ms: 12547
- answer: ちいかわ一口ミルキーアイスは、無茶馬ヨーグルト味で、シャリシャリとした食感で、周りは硬く中はチョコレートの甘さ控えめなヨーグルトの酸味があるような感じがするとのことです。また、パッケージにはちいかわやアイスの見本、ラッコ先生、モモンガちゃんのシールがついており、カットするとスタンドにもなるという特徴があります。
  citation_file_ids:
  - z6H1Vnkm-4Z7
  - z6H1Vnkm-4Z7
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 0.5
  citation_in_retrieved: 1.0
  took_ms: 13304

---
case: 011_long_video_atashinchi_game
query: ユズが半年かけて手に入れたものは？
stage1:
  keywords: ユズが半年かけて手に入れたものは？
  must_include_coverage: 0.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - z6H1Vnkm-4Z7
  - E91YCBdvjlOT
  - mJLIgl6IaF3d
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 0.5
  precision@5: 0.2
stage3_runs:
- answer: 提供されたファイル情報からは、「ユズが半年かけて手に入れたもの」が具体的に何であるかについての直接的な情報は確認できませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 14783
- answer: 提供されたファイル情報からは、「ユズが半年かけて手に入れたもの」が具体的に何であるかについての直接的な情報は確認できませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 15108
- answer: 提供されたファイル情報からは、「ユズが半年かけて手に入れたもの」が具体的に何であるかについての直接的な情報は確認できませんでした。
  citation_file_ids: []
  must_mention_coverage: 0.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 15187

---
```
</details>

## Pair comparison vs snapshot+3 long videos, 3 new cases (009/010/011)

- baseline: `/eval-data/reports/long_video_cases.json`
- common cases: 11

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 1: must_include_coverage | 0 | 0 | 11 |
| Stage 1: must_exclude_violations | 0 | 0 | 11 |
| Stage 2: recall@5 (file) | 0 | 0 | 11 |
| Stage 2: recall@10 (file) | 0 | 0 | 11 |
| Stage 2: segment recall@5 | 2 | 0 | 9 |
| Stage 2: MRR | 0 | 0 | 11 |
| Stage 3: must_mention (median) | 0 | 0 | 11 |
| Stage 3: citation_in_ground_truth (median) | 0 | 0 | 11 |
| Stage 3: citation_segment_match (median) | 0 | 0 | 11 |
| Stage 3: citation_in_retrieved (median) | 0 | 0 | 11 |
