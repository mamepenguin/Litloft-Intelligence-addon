# Eval Report

- date: 2026-04-14T10:29:51Z
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
    sha256: 1263c433364e2e0a06598490dfdd3d668573dec44852ffa82e52ef392ca31016
    indexed_with:
      whisper: 'openai/whisper-small'
      clip: 'llm-jp/llm-jp-clip-vit-base-patch16'
      blip: 'Salesforce/blip-image-captioning-base'
- blocklists:
    question_words.txt: b2aa63dd5c23475e50ebe81eaaf4bc8585165d6fbcb96fae0e56d5d65b12be13
    file_type_words.txt: 9c304d6f3ee5743511d3783b64f095b9507ed8233dcdf2aca665481873468907
- runs: stage1=1 stage2=1 stage3=2
- epsilon: 0.1
- drive: eval-drive
- total_cases: 8

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 1.00 |
| Stage 1: must_exclude_violations (sum) | 17 |
| Stage 2: file recall@5 (median) | 1.00 |
| Stage 2: file recall@10 (median) | 1.00 |
| Stage 2: segment recall@5 (median) | 1.00 |
| Stage 2: MRR (median) | 1.00 |
| Stage 3: must_mention_coverage (median) | 1.00 |
| Stage 3: citation_in_ground_truth (median) | 1.00 |
| Stage 3: citation_segment_match (median) | 0.00 |
| Stage 3: citation_in_retrieved (median) | 1.00 |

## Per-case summary

| id | recall@5 | seg-recall@5 | MRR | must_mention | citations | flags |
|---|---|---|---|---|---|---|
| 001_proper_noun_kyoto | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | seg-recall<1 |
| 002_question_word_noise | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | seg-recall<1 |
| 003_recipe_segment | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 004_segment_overlap | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 005_cross_modal_image_text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - |
| 008_proper_noun_preservation | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | recall<1 |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都の紅葉について何を言ってた？'
- top_10: ['✓xv2nmOPUTJvi', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫の共通点は？'
- top_10: ['✓ZmxTegJvjxAy', '✓5z3uW83PgOb1', '✗j_UPRviAs1Mm']
- Stage 3 must_mention values: [1.0, 1.0]

### 008_proper_noun_preservation

- query: '東福寺の通天橋と哲学の道はどんな感じ？'
- keywords: '東福寺の通天橋と哲学の道はどんな感じ？'
- top_10: ['✓xv2nmOPUTJvi']
- ground_truth not retrieved: ['docs/family_notes.md']
- Stage 3 must_mention values: [1.0, 1.0]

<details>
<summary>Full appendix (raw runs)</summary>

```yaml
case: 001_proper_noun_kyoto
query: 京都の紅葉について何を言ってた？
stage1:
  keywords: 京都の紅葉について何を言ってた？
  must_include_coverage: 1.0
  must_exclude_violations: 5
stage2:
  top_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7420
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7429

---
case: 002_question_word_noise
query: 黒猫の共通点は？
stage1:
  keywords: 黒猫の共通点は？
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  - j_UPRviAs1Mm
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 提供されたファイルの情報に基づくと、黒猫の共通点として、遺伝的に病気に強いとされること、そして性格が人懐こく賢い個体が多いという点が挙げられます。[2]また、日本では幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。[1]（ただし、これは特定の個体に関する情報や一般的な傾向として言及されています。）
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 8602
- answer: 提供されたファイルの情報に基づくと、黒猫の共通点として、遺伝的に病気に強いとされること、そして性格が人懐こく賢い個体が多いという点が挙げられます。[2]また、日本では幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。[1]（ただし、これは特定の個体に関する情報や一般的な傾向として言及されています。）
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 8634

---
case: 003_recipe_segment
query: カレーの作り方は？
stage1:
  keywords: カレーの作り方は？
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 'カレーの作り方について、提供された情報から以下の手順が分かります。


    **音声情報からの手順:**


    1. 玉ねぎをみじん切りにしてよく炒める。

    2. 鶏肉を加えて表面を焼く。

    3. 水とカレールーを入れて20分煮込む。

    4. 最後にガラムマサラを振る。


    **家族の思い出ノートからの補足情報:**


    お姉ちゃんが妹に教えた方法として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツであること、また「ルーを入れる前に一度火を止めるとダマにならない」という知恵が伝えられていることが記載されています。'
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7980
- answer: 'カレーの作り方について、提供された情報から以下の手順が分かります。


    **音声情報からの手順:**


    1. 玉ねぎをみじん切りにしてよく炒める。

    2. 鶏肉を加えて表面を焼く。

    3. 水とカレールーを入れて20分煮込む。

    4. 最後にガラムマサラを振る。


    **家族の思い出ノートからの補足情報:**


    お姉ちゃんが妹に教えた方法として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツであること、また「ルーを入れる前に一度火を止めるとダマにならない」という知恵が伝えられていることが記載されています。'
  citation_file_ids:
  - 6ZUevJjXSBVz
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8017

---
case: 004_segment_overlap
query: 永観堂のライトアップについて何て言ってた？
stage1:
  keywords: 永観堂のライトアップについて何て言ってた？
  must_include_coverage: 1.0
  must_exclude_violations: 5
stage2:
  top_file_ids:
  - xv2nmOPUTJvi
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 栄感道のもみじは夜にライトアップされます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4045
- answer: 栄感道のもみじは夜にライトアップされます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 4116

---
case: 005_cross_modal_image_text
query: 黒猫の写真と日記を見せて
stage1:
  keywords: 黒猫の写真と日記を見せて
  must_include_coverage: 1.0
  must_exclude_violations: 2
stage2:
  top_file_ids:
  - ZmxTegJvjxAy
  - 5z3uW83PgOb1
  - qf--MrL_g_Gt
  - j_UPRviAs1Mm
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 提供されたファイル情報に基づくと、「黒猫の写真」と「日記」の両方を見せることはできません。ファイルには黒猫に関する日記（Cat Diary）と、黒猫に関する事実（Black
    Cat Facts）、そして猫に関する写真（blackcat.jpg）が含まれていますが、具体的な「黒猫の写真」が日記や他の情報と直接結びついているわけではありません。また、日記の内容は黒猫ミケに関するものであり、写真そのものではありません。
  citation_file_ids:
  - ZmxTegJvjxAy
  - j_UPRviAs1Mm
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9820
- answer: 提供されたファイル情報に基づくと、「黒猫の写真」と「日記」の両方を見せることはできません。ファイルには黒猫に関する日記（Cat Diary）と、黒猫に関する事実（Black
    Cat Facts）、そして猫に関する写真（blackcat.jpg）が含まれていますが、具体的な「黒猫の写真」が日記や他の情報と直接結びついているわけではありません。また、日記の内容は黒猫ミケに関するものであり、写真そのものではありません。
  citation_file_ids:
  - ZmxTegJvjxAy
  - j_UPRviAs1Mm
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 9948

---
case: 006_no_answer
query: 宇宙ステーションの食事メニューは何ですか？
stage1:
  keywords: 宇宙ステーション 食事メニュー
  must_include_coverage: 1.0
  must_exclude_violations: 0
stage2:
  top_file_ids: []
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 0.0
  precision@5: 0.0
stage3_runs:
- answer: null
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1189
- answer: null
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1156

---
case: 007_blocklist_question_word
query: 機械学習と京都の紅葉動画の共通点は？
stage1:
  keywords: 機械学習と京都の紅葉動画の共通点は？
  must_include_coverage: 1.0
  must_exclude_violations: 2
stage2:
  top_file_ids:
  - rG-RB9DbOblv
  - xv2nmOPUTJvi
  - QzLEo8LTOgGr
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.6
stage3_runs:
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の具体的な共通点についての直接的な情報は得られませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7525
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の具体的な共通点についての直接的な情報は得られませんでした。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 7646

---
case: 008_proper_noun_preservation
query: 東福寺の通天橋と哲学の道はどんな感じ？
stage1:
  keywords: 東福寺の通天橋と哲学の道はどんな感じ？
  must_include_coverage: 1.0
  must_exclude_violations: 1
stage2:
  top_file_ids:
  - xv2nmOPUTJvi
  file_recall@5: 0.5
  file_recall@10: 0.5
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.2
stage3_runs:
- answer: 東福寺の通天橋からの眺めは、京都のもみじが絶景です。また、栄感道のもみじも有名で、夜にはライトアップされます。鉄画区の道を歩けば、もみじがりを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 5265
- answer: 東福寺の通天橋からの眺めは、京都のもみじが絶景です。また、栄感道のもみじも有名で、夜にはライトアップされます。鉄画区の道を歩けば、もみじがりを楽しむことができます。
  citation_file_ids:
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 5302

---
```
</details>

## Pair comparison vs phase_efgh_run1.json

- baseline: `/eval-data/reports/phase_efgh_run1.json`
- common cases: 8

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 1: must_include_coverage | 0 | 0 | 8 |
| Stage 1: must_exclude_violations | 0 | 0 | 8 |
| Stage 2: recall@5 (file) | 0 | 0 | 8 |
| Stage 2: recall@10 (file) | 0 | 0 | 8 |
| Stage 2: segment recall@5 | 0 | 0 | 8 |
| Stage 2: MRR | 0 | 0 | 8 |
| Stage 3: must_mention (median) | 0 | 0 | 8 |
| Stage 3: citation_in_ground_truth (median) | 0 | 0 | 8 |
| Stage 3: citation_segment_match (median) | 0 | 0 | 8 |
| Stage 3: citation_in_retrieved (median) | 0 | 0 | 8 |
