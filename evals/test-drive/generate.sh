#!/usr/bin/env bash
# Generate synthetic eval content into raw/.
#
# Host-only script (macOS): uses /usr/bin/say for TTS and ffmpeg for media
# synthesis. Idempotent: wipes and regenerates target files each run.
#
# Layout produced (total < 100MB):
#   raw/videos/{kyoto_autumn,black_cat_facts,ml_intro}.mp4
#   raw/audio/recipe_curry.mp3
#   raw/docs/{family_notes,tech_paper,cat_diary}.md
#   raw/images/{sunset,blackcat,autumn_leaves}.jpg
#
# Spec: docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md
# Phase: B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw"
TMP_DIR="$(mktemp -d -t hv-eval-gen.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

SAY_BIN="${SAY_BIN:-/usr/bin/say}"
FFMPEG_BIN="${FFMPEG_BIN:-/opt/homebrew/bin/ffmpeg}"
SAY_VOICE="${SAY_VOICE:-Kyoko}"

if [[ ! -x "$SAY_BIN" ]]; then
  echo "generate.sh: $SAY_BIN not executable (macOS host required)" >&2
  exit 1
fi
if [[ ! -x "$FFMPEG_BIN" ]]; then
  echo "generate.sh: $FFMPEG_BIN not executable" >&2
  exit 1
fi

echo "generate.sh: RAW_DIR=$RAW_DIR"
echo "generate.sh: wiping previous content"
rm -rf "$RAW_DIR/videos" "$RAW_DIR/audio" "$RAW_DIR/docs" "$RAW_DIR/images"
mkdir -p "$RAW_DIR/videos" "$RAW_DIR/audio" "$RAW_DIR/docs" "$RAW_DIR/images"

# -----------------------------------------------------------------------------
# Helper: synthesise a narrated mp4 with a solid colour background.
# args: out_path, spoken_text, bg_color, overlay_text
# -----------------------------------------------------------------------------
make_video() {
  local out="$1"
  local text="$2"
  local color="$3"
  local overlay="$4"

  local aiff="$TMP_DIR/$(basename "$out" .mp4).aiff"
  local bg="$TMP_DIR/$(basename "$out" .mp4)_bg.jpg"

  echo "  say -> $aiff"
  "$SAY_BIN" -v "$SAY_VOICE" -o "$aiff" "$text"

  echo "  bg   -> $bg"
  "$FFMPEG_BIN" -y -loglevel error \
    -f lavfi -i "color=c=$color:s=640x360:d=1" \
    -frames:v 1 "$bg"

  echo "  mux  -> $out"
  # overlay text is advisory for humans; drawtext is unavailable on the
  # default Homebrew ffmpeg build (no freetype), so we fall back to a
  # plain coloured background. The spoken audio carries the semantic
  # content that Whisper will transcribe.
  _=$overlay
  "$FFMPEG_BIN" -y -loglevel error \
    -loop 1 -i "$bg" -i "$aiff" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p \
    -c:a aac -b:a 96k -shortest "$out"
}

# -----------------------------------------------------------------------------
# Helper: synthesise an m4a from TTS only.
# -----------------------------------------------------------------------------
make_audio() {
  local out="$1"
  local text="$2"

  local aiff="$TMP_DIR/$(basename "$out" .mp3).aiff"
  echo "  say -> $aiff"
  "$SAY_BIN" -v "$SAY_VOICE" -o "$aiff" "$text"

  echo "  enc -> $out"
  "$FFMPEG_BIN" -y -loglevel error -i "$aiff" -c:a libmp3lame -b:a 96k "$out"
}

# -----------------------------------------------------------------------------
# Helper: synthesise a jpg with solid bg + overlay text.
# -----------------------------------------------------------------------------
make_image() {
  local out="$1"
  local color="$2"
  local overlay="$3"

  # drawtext unavailable in default Homebrew ffmpeg build (no freetype).
  # BLIP/CLIP will caption the colour itself; overlay text is advisory.
  _=$overlay
  "$FFMPEG_BIN" -y -loglevel error \
    -f lavfi -i "color=c=$color:s=1024x768:d=1" \
    -frames:v 1 "$out"
}

# -----------------------------------------------------------------------------
# Videos
# -----------------------------------------------------------------------------
echo "generate.sh: building videos"

make_video \
  "$RAW_DIR/videos/kyoto_autumn.mp4" \
  "京都の紅葉について。東福寺の通天橋からの眺めは絶景です。永観堂の紅葉も有名で、夜にはライトアップされます。哲学の道を歩きながら紅葉狩りを楽しめます。" \
  "darkred" \
  "Kyoto Autumn"

make_video \
  "$RAW_DIR/videos/black_cat_facts.mp4" \
  "黒猫について解説します。黒猫は実は遺伝的に病気に強いとされています。日本では幸運の象徴ですが、欧米では迷信的に避けられることもあります。性格は人懐こく賢い個体が多いです。" \
  "black" \
  "Black Cat Facts"

make_video \
  "$RAW_DIR/videos/ml_intro.mp4" \
  "機械学習入門。教師あり学習、教師なし学習、強化学習の三つに大別されます。代表的なアルゴリズムには線形回帰、決定木、ニューラルネットワークなどがあります。" \
  "navy" \
  "ML Intro"

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------
echo "generate.sh: building audio"

make_audio \
  "$RAW_DIR/audio/recipe_curry.mp3" \
  "カレーレシピ。玉ねぎ二個をみじん切りにして、よく炒めます。鶏肉を加えて表面を焼き、水とカレールーを入れて二十分煮込みます。最後にガラムマサラを振ると本格的な味になります。"

# -----------------------------------------------------------------------------
# Docs
# -----------------------------------------------------------------------------
echo "generate.sh: building docs"

cat > "$RAW_DIR/docs/family_notes.md" <<'EOF'
# 家族の思い出ノート

## はじめに

このノートは家族の何気ない日々を書き留めたものです。特別な日だけでなく、普段の会話やちょっとした出来事も残しておきたくて書いています。

## お姉ちゃんが妹にカレーを教えた日

ある日の夕方、お姉ちゃんが妹にカレーの作り方を教えていました。玉ねぎをみじん切りにして、しっかり飴色になるまで炒めるのがコツなのだそうです。鶏肉を加えて表面を焼き、水とカレールーを入れて二十分ほど煮込む。最後にガラムマサラを少し振ると、家庭のカレーがぐっと本格的な味になると妹に説明していました。妹は真剣にメモを取っていて、その横顔がずいぶん大人びて見えた日でした。

ちなみに、お姉ちゃん曰く「ルーを入れる前に一度火を止めるとダマにならない」とのこと。これは祖母から受け継いだ小さな知恵らしいです。

## 家族旅行で京都の紅葉を見に行った話

去年の秋、家族で京都の紅葉を見に行きました。東福寺の通天橋からの眺めは本当に絶景で、橋の下一面に広がる紅葉の海に皆で息をのみました。永観堂の紅葉も素晴らしく、夜のライトアップでは昼とはまったく違う表情を見せてくれます。哲学の道を散歩しながら紅葉狩りを楽しんだ時間は、家族全員の良い思い出になりました。

旅の帰り道、妹が「来年もまた来たいね」と言ったのが印象に残っています。紅葉は毎年少しずつ違う顔をするので、同じ場所でも飽きることがありません。

## その他の日々

- 父が新しいコーヒーミルを買ってきて、朝の香りが変わった
- 母の日に妹がカレーを作ってくれた（お姉ちゃんに教わった成果）
- 黒猫のミケが押し入れで子猫のように丸まっていた

こうした小さな出来事こそ、あとで読み返すと一番沁みます。
EOF

cat > "$RAW_DIR/docs/tech_paper.md" <<'EOF'
# 機械学習による画像分類の概説

## 概要

本稿は、機械学習を用いた画像分類の主要手法を俯瞰する技術ノートである。教師あり学習、教師なし学習、強化学習の三領域のうち、画像分類は主に教師あり学習の枠組みで扱われる。

## 代表的アーキテクチャ

### ResNet

残差接続（residual connection）を導入し、極めて深いネットワークの学習を可能にした。ImageNet ベンチマークで長く基準となった古典的モデル。

### Vision Transformer (ViT)

画像をパッチ系列に分割し、Transformer で直接処理する。大規模事前学習を前提に ResNet 系を上回る精度を示した。

### CLIP

画像とテキストを対照学習で同一埋め込み空間に射影するモデル。ゼロショット分類やマルチモーダル検索の基盤技術として広く使われている。

## 学習アルゴリズムの類型

- 線形回帰 / ロジスティック回帰
- 決定木 / ランダムフォレスト / 勾配ブースティング
- ニューラルネットワーク（MLP, CNN, Transformer）

## 評価指標

top-1 / top-5 accuracy, F1, AUC などが標準。近年は retrieval 系タスクで recall@k や MRR も併用される。

## まとめ

ResNet から ViT, CLIP へと至る流れは、帰納バイアスを減らし大規模事前学習に依存する方向への推移と読める。実運用では計算コストと精度のトレードオフを考慮してアーキテクチャを選択する必要がある。
EOF

cat > "$RAW_DIR/docs/cat_diary.md" <<'EOF'
# 黒猫ミケの日記

## プロフィール

- 名前: ミケ（黒猫なのにミケ）
- 毛色: 真っ黒
- 性格: 人懐こく、とても賢い

## 黒猫という存在について

黒猫は遺伝的に病気に強いと言われている。日本では古くから幸運の象徴とされてきたが、欧米では迷信的に避けられることもあるらしい。ミケはそんな歴史など意に介さず、今日もソファの上で堂々と眠っている。

## 最近の様子

- 朝は必ず窓辺で日向ぼっこ。黒い毛が太陽を吸い込んでいく
- 新しいおもちゃを渡すと、数分で遊び方を編み出す。賢い
- 来客に対しても警戒が薄く、むしろ膝に乗りにいく人懐こさ

## 小さな事件

先日、押し入れの奥で丸まって眠っていたところを発見された。黒い毛が暗闇に溶けていて、危うく踏みかけた。黒猫あるある、とも言う。

## 考察

性格的特徴として、人懐こく賢い個体が多いという一般論はミケに関しては完全に当てはまる。他の黒猫も同様なのかは、他の家の子を知らないのでわからない。
EOF

# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------
echo "generate.sh: building images"

make_image "$RAW_DIR/images/sunset.jpg"        "orange" "Sunset over the mountains"
make_image "$RAW_DIR/images/blackcat.jpg"      "black"  "A black cat sitting in the window"
make_image "$RAW_DIR/images/autumn_leaves.jpg" "red"    "Autumn leaves in Kyoto"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "generate.sh: done. Contents:"
find "$RAW_DIR" -type f -not -name '.gitignore' -not -name 'README.md' \
  -exec du -h {} + | sort -k2
