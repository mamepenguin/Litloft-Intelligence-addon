# Eval Report: case 011: transcript-faithful query

- date: 2026-04-14T13:50:16Z
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
- total_cases: 1

## Aggregate

| metric | value |
|---|---|
| Stage 1: must_include_coverage (median) | 0.50 |
| Stage 1: must_exclude_violations (sum) | 0 |
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
| 011_long_video_atashinchi_game | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |

## Failures

(no failures)

<details>
<summary>Full appendix (raw runs)</summary>

```yaml
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
- answer: 半年かけて溜めた小豆が落ちたまでようやく手に入れたこのゲームと共に、今日この日曜日だけはゆいぎな一日にしたい、という話がありました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 7964
- answer: 半年かけて溜めた小豆が落ちたまでようやく手に入れたこのゲームと共に、今日この日曜日だけはゆいぎな一日にしたい、という話がありました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8102
- answer: 半年かけて溜めた小豆が落ちたまでようやく手に入れたこのゲームと共に、今日この日曜日だけはゆいぎな一日にしたい、という話がありました。
  citation_file_ids:
  - E91YCBdvjlOT
  must_mention_coverage: 1.0
  citation_in_ground_truth: 1.0
  citation_segment_match: 1.0
  citation_in_retrieved: 1.0
  took_ms: 8065

---
```
</details>

## Pair comparison vs long videos + coverage-based segment match

- baseline: `/eval-data/reports/long_video_with_coverage.json`
- common cases: 1
- only in baseline: ['001_proper_noun_kyoto', '002_question_word_noise', '003_recipe_segment', '004_segment_overlap', '005_cross_modal_image_text', '006_no_answer', '007_blocklist_question_word', '008_proper_noun_preservation', '009_long_video_olive', '010_long_video_chiikawa_taste']

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 1: must_include_coverage | 1 | 0 | 0 |
| Stage 1: must_exclude_violations | 1 | 0 | 0 |
| Stage 2: recall@5 (file) | 0 | 0 | 1 |
| Stage 2: recall@10 (file) | 0 | 0 | 1 |
| Stage 2: segment recall@5 | 1 | 0 | 0 |
| Stage 2: MRR | 1 | 0 | 0 |
| Stage 3: must_mention (median) | 1 | 0 | 0 |
| Stage 3: citation_in_ground_truth (median) | 1 | 0 | 0 |
| Stage 3: citation_segment_match (median) | 1 | 0 | 0 |
| Stage 3: citation_in_retrieved (median) | 1 | 0 | 0 |
