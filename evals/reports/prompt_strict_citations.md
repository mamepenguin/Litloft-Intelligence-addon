# Eval Report: prompt: ban [N] markers, require citations

- date: 2026-04-14T12:55:34Z
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
- runs: stage1=1 stage2=1 stage3=3
- epsilon: 0.1
- drive: eval-drive
- total_cases: 8

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 1.00 |
| Stage 1: must_exclude_violations (sum) | 4 |
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
| 001_proper_noun_kyoto | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | seg-recall<1 |
| 002_question_word_noise | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | seg-recall<1 |
| 003_recipe_segment | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 004_segment_overlap | 1.00 | 1.00 | 0.50 | 1.00 | 0.00 | - |
| 005_cross_modal_image_text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| 006_no_answer | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | - |
| 007_blocklist_question_word | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - |
| 008_proper_noun_preservation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都 紅葉'
- top_10: ['✓xv2nmOPUTJvi', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫 共通点'
- top_10: ['✓ZmxTegJvjxAy', '✓5z3uW83PgOb1', '✗j_UPRviAs1Mm']
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
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、鉄画区の道を歩きながらもみじがりを楽しむこともできます。家族の思い出ノートによると、家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは本当に絶景で、橋の下一面に広がる紅葉の海に息をのんだとあります。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せると記載されています。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8079
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、鉄画区の道を歩きながらもみじがりを楽しむこともできます。家族の思い出ノートによると、家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは本当に絶景で、橋の下一面に広がる紅葉の海に息をのんだとあります。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せると記載されています。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8514
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、鉄画区の道を歩きながらもみじがりを楽しむこともできます。家族の思い出ノートによると、家族で京都の紅葉を見に行った際、東福寺の通天橋からの眺めは本当に絶景で、橋の下一面に広がる紅葉の海に息をのんだとあります。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せると記載されています。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8547

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
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 0.0
  segment_recall@10: 0.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: 黒猫の共通点として、遺伝的に病気に強いと言われていること、そして性格が人懐こく賢い個体が多いという点が挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7102
- answer: 黒猫の共通点として、遺伝的に病気に強いと言われていること、そして性格が人懐こく賢い個体が多いという点が挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7183
- answer: 黒猫の共通点として、遺伝的に病気に強いと言われていること、そして性格が人懐こく賢い個体が多いという点が挙げられます。
  citation_file_ids:
  - 5z3uW83PgOb1
  - ZmxTegJvjxAy
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7189

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
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めることでダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7481
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めることでダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7538
- answer: カレーの作り方として、玉ねぎをみじん切りにしてしっかり飴色になるまで炒めるのがコツで、鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込みます。最後にガラムマサラを少し振ると本格的な味になります。また、ルーを入れる前に一度火を止めることでダマにならないという知恵もあります。
  citation_file_ids:
  - qf--MrL_g_Gt
  - 6ZUevJjXSBVz
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7565

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
  mrr: 0.5
  precision@5: 0.2
stage3_runs:
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6572
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6611
- answer: 永観堂の紅葉は夜のライトアップで昼とは違う表情を見せると述べられています。
  citation_file_ids:
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 6635

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
  - j_UPRviAs1Mm
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.4
stage3_runs:
- answer: '提供されたファイルには、黒猫の写真や日記が直接含まれていません。しかし、ファイル情報から以下の内容を提供できます。


    黒猫ミケの日記によると、黒猫ミケは人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出すなど、人間に懐いている様子が記されています。


    また、家族の思い出ノートには「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。


    黒猫に関する事実として、黒猫は遺伝的に病気に強いと言われていることや、日本では幸運の象徴とされてきたことなどが記載されています。


    画像ファイルとして「black…'
  citation_file_ids:
  - j_UPRviAs1Mm
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8144
- answer: '提供されたファイルには、黒猫の写真や日記が直接含まれていません。しかし、ファイル情報から以下の内容を提供できます。


    黒猫ミケの日記によると、黒猫ミケは人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出すなど、人間に懐いている様子が記されています。


    また、家族の思い出ノートには「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。


    黒猫に関する事実として、黒猫は遺伝的に病気に強いと言われていることや、日本では幸運の象徴とされてきたことなどが記載されています。


    画像ファイルとして「black…'
  citation_file_ids:
  - j_UPRviAs1Mm
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8279
- answer: '提供されたファイルには、黒猫の写真や日記が直接含まれていません。しかし、ファイル情報から以下の内容を提供できます。


    黒猫ミケの日記によると、黒猫ミケは人懐こく賢い性格で、朝は窓辺で日向ぼっこをしたり、新しいおもちゃを渡すとすぐに遊び方を編み出すなど、人間に懐いている様子が記されています。


    また、家族の思い出ノートには「黒猫のミケが押し入れで子猫のように丸まっていた」という記述があります。


    黒猫に関する事実として、黒猫は遺伝的に病気に強いと言われていることや、日本では幸運の象徴とされてきたことなどが記載されています。


    画像ファイルとして「black…'
  citation_file_ids:
  - j_UPRviAs1Mm
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 8263

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
  took_ms: 1223
- answer: null
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1209
- answer: null
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 1221

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
  file_recall@5: 1.0
  file_recall@10: 1.0
  segment_recall@5: 1.0
  segment_recall@10: 1.0
  mrr: 1.0
  precision@5: 0.6
stage3_runs:
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の具体的な共通点についての直接的な情報は得られません。ファイルには機械学習の入門に関する情報と、京都の紅葉に関する情報がそれぞれ含まれています。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 11226
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の具体的な共通点についての直接的な情報は得られません。ファイルには機械学習の入門に関する情報と、京都の紅葉に関する情報がそれぞれ含まれています。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 11372
- answer: 提供されたファイル情報からは、機械学習と京都の紅葉動画の具体的な共通点についての直接的な情報は得られません。ファイルには機械学習の入門に関する情報と、京都の紅葉に関する情報がそれぞれ含まれています。
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 11280

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
- answer: 東福寺の通天橋からの眺めは紅葉の海に広がり、絶景であると述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできた、という経験が家族の思い出として語られています。
  citation_file_ids:
  - qf--MrL_g_Gt
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7399
- answer: 東福寺の通天橋からの眺めは紅葉の海に広がり、絶景であると述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできた、という経験が家族の思い出として語られています。
  citation_file_ids:
  - qf--MrL_g_Gt
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7491
- answer: 東福寺の通天橋からの眺めは紅葉の海に広がり、絶景であると述べられています。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできた、という経験が家族の思い出として語られています。
  citation_file_ids:
  - qf--MrL_g_Gt
  - xv2nmOPUTJvi
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: null
  citation_in_retrieved: 1.0
  took_ms: 7511

---
```
</details>

## Pair comparison vs quote-aware + median fix, runs=5 (variance check)

- baseline: `/eval-data/reports/quote_aware_runs5.json`
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
| Stage 3: citation_in_ground_truth (median) | 1 | 1 | 6 |
| Stage 3: citation_segment_match (median) | 1 | 1 | 6 |
| Stage 3: citation_in_retrieved (median) | 1 | 0 | 7 |
