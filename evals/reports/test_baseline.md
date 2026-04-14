# Eval Report

- date: 2026-04-14T10:02:45Z
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
- blocklists: TODO (Phase F)
- runs: stage1=1 stage2=1 stage3=3
- epsilon: 0.1
- drive: eval-drive
- total_cases: 3

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 1.00 |
| Stage 1: must_exclude_violations (sum) | 3 |
| Stage 2: file recall@5 (median) | 1.00 |
| Stage 2: file recall@10 (median) | 1.00 |
| Stage 2: segment recall@5 (median) | 0.00 |
| Stage 2: MRR (median) | 1.00 |
| Stage 3: must_mention_coverage (median) | 1.00 |
| Stage 3: citation_in_ground_truth (median) | 1.00 |
| Stage 3: citation_segment_match (median) | 1.00 |
| Stage 3: citation_in_retrieved (median) | 1.00 |

## Per-case summary

| id | recall@5 | seg-recall@5 | MRR | must_mention | citations | flags |
|---|---|---|---|---|---|---|
| 001_proper_noun_kyoto | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | seg-recall<1 |
| 002_question_word_noise | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | seg-recall<1 |
| 003_recipe_segment | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

### 001_proper_noun_kyoto

- query: '京都の紅葉について何を言ってた？'
- keywords: '京都の紅葉について何を言ってた？'
- top_10: ['✓xv2nmOPUTJvi', '✓qf--MrL_g_Gt']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

### 002_question_word_noise

- query: '黒猫の共通点は？'
- keywords: '黒猫の共通点は？'
- top_10: ['✓ZmxTegJvjxAy', '✓5z3uW83PgOb1', '✗j_UPRviAs1Mm']
- Stage 3 must_mention values: [1.0, 1.0, 1.0]

<details>
<summary>Full appendix (raw runs)</summary>

```yaml
case: 001_proper_noun_kyoto
query: 京都の紅葉について何を言ってた？
stage1:
  keywords: 京都の紅葉について何を言ってた？
  must_include_coverage: 1.0
  must_exclude_violations: 1
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
  took_ms: 7356
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7440
- answer: 京都の紅葉について、東福寺の通天橋からの眺めは絶景であり、栄感道のもみじも有名で夜にはライトアップされます。また、哲学の道を散歩しながら紅葉狩りを楽しむこともできます。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とは違う表情を見せます。
  citation_file_ids:
  - xv2nmOPUTJvi
  - qf--MrL_g_Gt
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7382

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
  took_ms: 8539
- answer: 提供されたファイルの情報に基づくと、黒猫の共通点として、遺伝的に病気に強いとされること、そして性格が人懐こく賢い個体が多いという点が挙げられます。[2]また、日本では幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。[1]（ただし、これは特定の個体に関する情報や一般的な傾向として言及されています。）
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 8620
- answer: 提供されたファイルの情報に基づくと、黒猫の共通点として、遺伝的に病気に強いとされること、そして性格が人懐こく賢い個体が多いという点が挙げられます。[2]また、日本では幸運の象徴とされていますが、欧米では迷信的に避けられることもあるようです。[1]（ただし、これは特定の個体に関する情報や一般的な傾向として言及されています。）
  citation_file_ids: []
  must_mention_coverage: 1.0
  citation_in_ground_truth: 0.0
  citation_segment_match: null
  citation_in_retrieved: 0.0
  took_ms: 8766

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
  took_ms: 8132
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
  took_ms: 8271
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
  took_ms: 8244

---
```
</details>
