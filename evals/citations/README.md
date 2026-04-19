# detailed_summary Citation Eval

`detailed_summary` の citation リンカ（各段落・箇条書き・表行が原文のどのチャンクと結びつくか）を
ゴールデンケースで評価するための開発者向けツール。

Ask (RAG) 用の eval (`../README.md`) と同じスナップショット DB を読むが、ハーネスは別建て。

## いつ使うか

- `app/citations.py` のロジックを変更した（hybrid retrieval の重み、
  BM25 パラメータ、table row の pooling 戦略、閾値など）
- `app/summary_parser.py` のセグメント分割を変更した
- `search-config.yml` の citation 関連設定をチューニングしたい
- 「ぴったり合わない」と感じたケースを再現 + 定点観測したい

## セットアップ

既存の Ask eval と同じスナップショットを使う:

```bash
# スナップショット生成（Ask eval の README を参照）
cd addons/intelligence
# … snapshot 生成手順 …

# ケース実行
python -m app.evals_citations \
    --cases ../evals/citations/cases/ \
    --snapshot ../evals/test-drive/snapshot/search.db \
    --drive eval-drive \
    --label "hybrid v1"
```

出力: `../evals/citations/reports/<timestamp>.md` と `<timestamp>.json`（sidecar）。

## ベースライン比較

```bash
# 変更前ベースラインを保存
python -m app.evals_citations --label baseline --output reports/baseline.md

# コード変更後、--baseline で比較
python -m app.evals_citations --label phase1 --baseline reports/baseline.json
```

レポートの末尾に `## Baseline comparison` セクションが入り、`top1_accuracy` / `recall_at_3` /
`has_citation_precision` の変化が表で出る。

## ケース YAML スキーマ

```yaml
id: 001_example_recipe_table          # 一意な識別子
file_path: audio/recipe_curry.mp3     # snapshot 内の相対パス
notes: |
  Optional free-form notes.

expectations:
  # 段落に対する期待
  - section_path: "全体像/0"
    # いずれかを指定（複数指定は chunk_ids > hint の順で評価）
    chunk_ids: ["transcript:0", "transcript:1"]
    segment_hint:
      time_range: [0.0, 30.0]
    must_have_citation: true   # ⚠ だったら case として失敗扱い

  # 表の行に対する期待
  - section_path: "重要ポイントまとめ/row/0"
    segment_hint:
      time_range: [60.0, 90.0]
    must_have_citation: true

  # 文書ファイルなら page で
  - section_path: "主要な章・場面/3"
    segment_hint:
      page: 5
```

**必須**: `section_path` と `chunk_ids` / `segment_hint` / `must_have_citation`
の少なくとも一つ。

## 指標

| 指標 | 意味 |
|---|---|
| `top1_accuracy` | 各 expected segment の top-1 chunk が合っている割合 |
| `recall_at_3` | 各 expected segment の top-3 の中にひとつでも合う chunk があるか |
| `has_citation_precision` | `has_citation = True` のうち top-1 が正解だった割合 |
| `missing_required_citations` | `must_have_citation: true` だが ⚠ になった数 |
| `by_segment_type` | paragraph / bullet / bullet (row) 別の top-1 / recall@3 |

## 現状の既知の観測限界

- **短尺素材の segment-level 偏り**: `raw/` の test-drive が数十秒クラスの素材中心で、
  table row と paragraph の出現頻度が低い。長尺（講義動画、100p PDF、料理レシピ動画）を
  追加しないと table row / paragraph の精度改善を計測しきれない
- **detailed_summary 非生成ファイル**: snapshot に `file_summaries.detailed_summary` 行が
  無いファイルはケースに書けない。先に intelligence 本体で `detailed` モードを使って生成 +
  snapshot 更新が必要
- **Whisper グループ化の副作用**: chunk が統合されてしまうと segment_hint の IoU が
  0 or 1 の boolean になる（Ask eval と同じ制約）

## ケース追加のワークフロー

1. 実運用中に「外れた」citation を見つける（UI で ⚠ か、明らかに違うハイライト）
2. `GET /api/addons/intelligence/summaries/{file_id}/citations` で現状の citation 出力を確認
3. 期待する chunk を DB から引いて（`SELECT chunk_index, timestamp_start, timestamp_end
   FROM transcript_chunks WHERE file_id = '...'`）、case yml に書く
4. `python -m app.evals_citations --filter <case_id>` で当該ケースだけ実行して再現確認
5. コード調整 → `--baseline` 付きで前後比較

## アンチパターン

- **大量のケースを一度に追加する**: まず 5-10 ケースで現状を可視化し、何が効くかを
  見てから増やす。先に「100 ケース書かないと始まらない」状態にすると、
  改善サイクルが回らない
- **自動生成された summaries を「正解」として使う**: detailed_summary 自体が
  機械生成物なので、GT は「人間が読んで妥当な出典」であるべき。自動生成を正解化
  するとテストは常にパスして何も測れなくなる
