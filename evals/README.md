# Ask 評価ハーネス (intelligence dev-time eval)

`intelligence` アドオンの Ask (RAG) パイプラインを「変更前 / 変更後」で bulk
比較するための開発者向けツール。本番には影響せず、CI 統合も持たない。
詳細設計は `docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md`。

`raw/README.md` がテストドライブ素材の来歴と再生成手順、こちらのファイルが
ハーネス本体の使い方。

---

## 1. 概要

- 対象パイプライン: `app.rag.*` (`query_transform` → `search` → `answer_question`)
- 実行単位: YAML で書かれたゴールデン case (`cases/*.yml`)
- 出力: `reports/<name>.md` + `reports/<name>.json` (sidecar)
- 誰が使うか: intelligence を触る開発者。プロンプトや retrieval パラメータを
  変える前後で md を diff / `--baseline` で比較する

**スコープ外**: LLM-as-judge 自動採点、本番 admin UI、CI ゲート。

---

## 2. セットアップ

### 前提

```bash
docker compose up -d
```

- `docker-compose.override.yml` がハーネス用のマウントを定義:
  - `./addons/intelligence/evals:/eval-data` (cases + reports + snapshot)
  - `./addons/intelligence/evals/test-drive/raw:/drives/eval-drive:ro`
    (本体 backend / intelligence の両方に同じパスでマウント)
- `drives.json` に `eval-drive` を readonly で登録しておく
  (設定例は `docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md` §"drives.json への登録")

### snapshot の入手

test-drive の派生 index は `snapshot/search.db` として git 管理されている。
`raw/` 素材がなくても retrieval / Stage 1〜3 は完走する (スナップショットに
全 embedding / transcript が入っているため)。素材が必要なのは `re-index`
するときだけ (§7)。

---

## 3. case 作成

### YAML スキーマ (抜粋)

```yaml
id: 007_blocklist_question_word
query: "機械学習と京都の紅葉動画の共通点は？"

expected_keywords:
  must_include: ["機械学習", "紅葉"]
  must_exclude: ["共通点"]   # グローバル blocklist に加える case 個別分のみ

ground_truth_files:
  - path: "videos/kyoto_autumn.mp4"
    segment_hint:
      time_range: [0.0, 20.0]   # 動画/音声 (秒)
  - path: "docs/tech_paper.md"
  - path: "docs/chapter3.md"
    segment_hint:
      page: 3                   # 文書 (ページ)

must_mention: []

notes: |
  hako O_lh_jKTbQBcMKs23483Z の再発防止ケース。
```

### segment_hint の使い方

- `time_range`: 正解が含まれる区間 (秒)。IoU ≥ 0.3 で segment_recall ヒット判定
- `page`: 文書の正解ページ。完全一致で判定
- 省略すれば **file 粒度** だけで評価される。まずは省略、必要になったら付ける

### グローバル blocklist との関係

- `app/evals/blocklists/question_words.txt` と `file_type_words.txt` は
  **全 case 共通**で `must_exclude` に加算される (NFKC + ひらがな/カタカナ
  統一 + 部分一致)
- case 個別の `must_exclude` には「このケース固有の除外語」のみ書く
- blocklist の sha256 はレポートの meta に記録されるので、blocklist 更新の
  影響だけを diff で追える

### 良い例 / 悪い例

```yaml
# GOOD: 固有名詞 + 具体的な問い + 最低限の must_mention
query: "永観堂のライトアップについて何て言ってた？"
expected_keywords:
  must_include: ["永観堂", "ライトアップ"]
must_mention: ["ライトアップ"]

# BAD: must_mention が絞りすぎ (LLM の言い換えで即 fail)
must_mention: ["夜はライトアップされて紅葉が映えます"]

# BAD: must_include に一般語だけ → 何もテストしていない
expected_keywords:
  must_include: ["紅葉"]
```

---

## 4. runner 実行

### 基本

```bash
docker compose exec intelligence python -m app.evals \
  --cases /eval-data/cases/ \
  --snapshot /eval-data/test-drive/snapshot/search.db \
  --output /eval-data/reports/my_run.md
```

### よく使うフラグ

| フラグ | 役割 |
|---|---|
| `--runs N` | Stage 3 の試行回数 (default 3) |
| `--filter <substr>` | case id 部分一致で絞り込み (単体デバッグ用) |
| `--top-k N` | RAG `top_k` を config から上書き |
| `--baseline <path>` | 既存 sidecar json との差分を md に追記 |
| `--label "<s>"` | レポート見出しに任意ラベル |
| `--epsilon <f>` | improved/regressed/tied の閾値 (default 0.1) |

出力は `<output>.md` と `<output>.json` (sidecar) の 2 点。sidecar はパース
フリーで `--baseline` の入力に使える。

---

## 5. 結果の読み方

### Aggregate セクション

case 全体の中央値 / 合計。Stage 1 の `must_exclude_violations` は**合計**
(全 case にわたるグローバル blocklist 違反の総数)。

### Per-case summary

1 行 = 1 case。`flags`:
- `recall<1`: Stage 2 file recall@5 が 1.0 未満
- `seg-recall<1`: segment_hint 指定があるのに IoU マッチしなかった
- `unstable`: Stage 3 の N 回で max-min が ε を超えた

### Failures

`recall<1` / `seg-recall<1` / `unstable` のいずれかを満たす case を抜粋。
`top_10` は `✓`/`✗` で ground truth ヒット状況、`Stage 3 must_mention`
values は N 回分の生値。

### Appendix

`<details>` 内の raw runs YAML。LLM の回答全文と citation 生データを含む。
「なぜ失敗した？」を調べるときはこれを読む。

---

## 6. ペア比較 (`--baseline`)

```bash
docker compose exec intelligence python -m app.evals \
  --cases /eval-data/cases/ \
  --snapshot /eval-data/test-drive/snapshot/search.db \
  --output /eval-data/reports/after_change.md \
  --baseline /eval-data/reports/before_change.json
```

`--baseline` には **sidecar json** を渡すのが推奨 (md でも twin の .json を
探す)。レポート末尾に下記セクションが追記される:

```
## Pair comparison vs before_change.json

| stage / metric | improved | regressed | tied |
|---|---|---|---|
| Stage 2: recall@5 (file) | 3 | 1 | 8 |
| Stage 3: must_mention (median) | 5 | 0 | 7 |
```

- `improved/regressed` の判定閾値は `--epsilon` (default 0.1)
- case が baseline / current で増減した場合は `only in baseline` / `only in
  current` に表示される
- 同一条件 2 回実行すれば全 tied になる (sanity check)

---

## 7. snapshot 再生成

### いつ必要か

- `raw/` の素材を差し替えた / 追加した
- Whisper / CLIP / BLIP のモデルバージョンを上げた
- indexer / chunker のロジックを変えた

### フロー

```bash
# 1. raw/ を差し替え
# 2. drives.json で eval-drive を一時的に readonly:false に
# 3. docker compose up -d --force-recreate intelligence
#    (scan 開始 → Whisper / CLIP / BLIP 完了を待つ)
# 4. scripts/snapshot.sh eval-drive snapshot/
# 5. drives.json を readonly に戻す
# 6. snapshot/search.db + manifest.json を git commit
```

snapshot を git commit する理由は、retrieval 評価の再現性を確保するため。
モデル更新で embedding が変われば recall@k の数値も変わるので、snapshot の
sha256 とモデルバージョンを report に記録して紐付けてある。

---

## 8. セキュリティ留意

- **test-drive のコンテンツは LLM API に送信される** (Stage 1 query_transform
  と Stage 3 answer_question で)
- 外部 LLM (OpenAI, DeepSeek 等) を使うときは `raw/` に機密を置かない
- プライバシー重視ならローカル LLM (ollama) を `search-config.yml` で指定
- eval-drive は `passwords.json` に登録しない (公開ドライブ扱い)。開発者環境
  特有の設定であり本番デプロイ手順とは分離する
- 関連 hako: `gn49jY8F6OCR75WgjdoIg` (LLM プロバイダ選択とプライバシー)

---

## 9. 既知の制限

- **短い動画の segment_recall**: test-drive の動画は数十秒の短尺で、Whisper
  が 1 segment group にまとめることが多い。この場合 IoU は事実上 0 or 1 の
  bool になり、segment 粒度評価が骨抜きになる。snapshot に長尺素材を足すか、
  chunk 分割粒度を変えたときに意味が出る
- **`json_object` 非対応 LLM**: `gemma4:e2b` などは Stage 1 の
  `transform_query` で raw fallback を踏み、keywords が原文そのままになる。
  これは retrieval にはむしろ助かる (FTS5 が自然文でも動く) が、Stage 1
  metrics は「LLM が動いていない」ことを示す。`must_exclude_violations` が
  高い値になる場合はまずここを疑う
- **citation_segment_match**: citation の `segment_location` が `"m:ss"` 形式
  で来る前提のパース。LLM が別形式 (秒数だけ、別区切り) を返すと null に倒れる
- **N=3 の弱統計**: Stage 3 の試行回数は標準 3。完全な分散を見たい場合は
  `--runs 10` 程度に増やす (ただしコストも線形に増える)

---

## 関連

- spec: `docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md`
- RAG 本体の設計: `docs/superpowers/specs/2026-04-11-intelligence-rag-redesign.md`
- hako `O_lh_jKTbQBcMKs23483Z`: "共通点は？" 0 件回帰 (case 007 の元ネタ)
- hako `2O3vVFFie6y66EpoIpGH3`: auto_tags の「LLM 提案は常に正しくない」前提
