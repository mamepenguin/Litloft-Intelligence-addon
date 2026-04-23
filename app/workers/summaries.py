"""AI summaries worker using LLM.

Generates two-layer summaries (1-sentence short + paragraph long) for
videos, audio files, and documents. Runs as a dedicated async queue
processing one file at a time.

Unlike auto_tags, summaries do not have an approve/dismiss workflow:
the generated summary is stored in the intelligence DB and displayed
directly. The host Litloft DB is never touched. Users can "hide" a
summary (status='hidden') or "regenerate" it (delete + re-enqueue).

Long content (transcripts, document text) that exceeds the configured
threshold is sampled from beginning/middle/end windows rather than
truncated from the front — this preserves coverage of the full file
without requiring expensive map-reduce over many LLM calls.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.llm import LLMClient
from app.models import IndexedFile, TranscriptChunk, generate_insight_id
from app.workers.whisper import HVLINK_MIME
from app.text_utils import trim_to_sentence_boundary

logger = logging.getLogger(__name__)


async def _emit_ws_event(event: str, data: dict) -> None:
    """Best-effort WebSocket event emission via the host's internal API.

    Mirrors ``app.workers.refine._emit_ws_event``. The host forwards
    the posted JSON to its WebSocket broadcaster; delivery failures
    are swallowed so citation calculation never fails a detailed-
    summary generation. Tests monkeypatch this function.
    """
    import os

    logger.info("summaries-event %s %s", event, data)

    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/addon-events"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(url, json={"event": event, "data": data})
    except Exception:
        # Host endpoint is optional; never fail the worker on WS.
        return


async def _recalculate_citations(file_id: str, summary_text: str) -> None:
    """Recompute detailed_summary_citations for ``file_id``.

    Executes the synchronous ``calculate_and_store`` in the default
    thread executor so the worker event loop stays responsive during
    embedding generation (CPU-bound). Emits
    ``intelligence.detailed_summary.citations_ready`` on completion.

    Silently swallows failures — citations are a best-effort layer on
    top of the detailed summary, and a citation crash must never
    invalidate the summary itself.
    """
    if not summary_text or not summary_text.strip():
        return
    try:
        import asyncio
        from app.citations import calculate_and_store

        loop = asyncio.get_running_loop()
        with_count, without_count = await loop.run_in_executor(
            None, calculate_and_store, file_id, summary_text
        )
        await _emit_ws_event(
            "intelligence.detailed_summary.citations_ready",
            {
                "file_id": file_id,
                "citation_count": with_count,
                "no_citation_count": without_count,
            },
        )
    except Exception as e:  # noqa: BLE001 — never fail the caller
        logger.warning(
            "Citation recalculation failed for %s: %s", file_id, e
        )


# Re-export the shared helper under its legacy private name so existing
# callers (and the summaries tests) that imported `_trim_to_sentence_boundary`
# from this module keep working after the helper moved to app.text_utils.
_trim_to_sentence_boundary = trim_to_sentence_boundary

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- 要約は日本語で生成すること\n",
    "en": "- Summaries must be in English\n",
}

# Language instructions for the detailed (Markdown long-form) summary.
# Unlike the short/long prompt above these spell out the required writing
# style because the detailed output is user-facing prose, not a caption.
_DETAILED_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": (
        "- 日本語で、自然で読みやすい文体。各セクションの見出しを使用\n"
    ),
    "en": (
        "- Write in English with a natural, readable style. "
        "Use headings for each section\n"
    ),
}

# Context types this worker produces summaries for.
# Images are intentionally excluded — BLIP captions already fill
# that role and running a second LLM pass adds no value.
_SUPPORTED_CONTEXT_TYPES: frozenset[str] = frozenset({"video", "audio", "document"})

# Separator inserted between sampled windows in truncated contexts.
_WINDOW_SEPARATOR = "\n\n[...中略...]\n\n"

# Proper-noun handling rules shared by short/long and detailed prompts.
# The design accepts that plausible-substitution errors (e.g. "堀井雄二" →
# "堀江雄三") can't be detected from the transcript alone. Instead we:
#   1. Fix what we CAN verify — names that appear in filename/title/description
#   2. Leave the rest alone rather than guess (LLM "correction" tends to
#      invent new errors when no trusted anchor exists)
#   3. (detailed only) Surface low-confidence names via a dedicated annotation
#      section so the user knows where to manually verify
# See hako: "LLM による要約... 固有名詞を間違うこと" (2026-04-17).
_COMMON_RULES = (
    "## トピック境界の厳守\n"
    "\n"
    "原文が複数の独立したトピック（別レシピ、別製品、別章、別エピソード等）を"
    "扱っている場合、各トピックのセクションには、原文でそのトピックが扱われている"
    "範囲内の情報のみを記載すること。\n"
    "\n"
    "- あるトピックで述べられた特徴・効能・ポイントを、別のトピックのセクションに"
    "転記しないこと（隣接トピック間の情報混線を避ける）\n"
    "- 判断に迷った場合は、該当情報を省略するか、トピック横断的なセクション"
    "（「全体を通じたポイント」等）を別途設けて記載する\n"
    "\n"
    "## 固有名詞の扱い\n"
    "\n"
    "このコンテンツはトランスクリプト（音声認識やテキスト抽出）由来のため、"
    "固有名詞の誤認識が含まれる可能性があります。誤認識はカタカナ化だけでなく、"
    "漢字の取り違え、もっともらしい別語への置換など多様な形で現れます。\n"
    "\n"
    "### 積極的に修正すべきもの\n"
    "- ファイル名・タイトル・説明文に登場する固有名詞について、"
    "トランスクリプト中で異なる表記（カタカナ化、区切り違い、漢字違い、音素類似など）"
    "がされている場合は、ファイル名・説明文側の表記に統一する\n"
    "- 同一対象と明確に判断できる複数の表記が登場する場合、最も妥当な一つに統一する\n"
    "- 明らかに日本語として意味不明な表記（例: 「雷にゃく」「鼻字に入れて」等の、"
    "日本語として成立しない語句）で、文脈から一般名詞・一般的表現として"
    "妥当な復元候補が明確な場合は、復元した表記を使用する\n"
    "  - 復元候補が複数あり絞り込めない場合、または一般的な語として定着していない"
    "可能性がある場合は、原文表記のまま残す\n"
    "\n"
    "### 慎重に扱うもの\n"
    "- ファイル名・タイトル・説明文に登場しない固有名詞は、"
    "トランスクリプトの表記を原則そのまま使う\n"
    "- 元の表記が誤っている可能性はあっても、"
    "推測で別の漢字や読みに置き換えないこと（別の誤りを生むリスクのほうが高い）\n"
    "- 人名・地名・商品名・ブランド名など、誤った復元が実害につながりうる"
    "固有名詞は、意味不明であっても安易に復元せず原文表記を保持する\n"
    "\n"
    "## 数値情報の扱い\n"
    "\n"
    "原文中の数値（分量、時間、温度、保存期間、価格、距離、年数等）は、"
    "原則として原文の表記をそのまま維持すること。\n"
    "\n"
    "- 数値を勝手に変更・修正・丸めてはならない\n"
    "- 単位の換算や表記統一も行わない（原文が「小さじ2/3」なら「小さじ2/3」のまま）\n"
    "- ただし、数値そのものは変えずに、不確実性の注記を添えることは許容する\n"
    "  - トランスクリプト由来で、数値が音声認識エラーの可能性が高いと"
    "判断される場合（例: 葉物野菜の保存期間が「45日」となっている等、"
    "一般常識と大きく乖離する場合）、以下の形式で注記を添えてよい:\n"
    "    「保存期間の目安はおよそ45日程度である（※原文表記。"
    "音声認識エラーの可能性あり）」\n"
    "  - 注記を添える基準は保守的に。迷う程度なら注記は不要。"
    "明確に常識的範囲を逸脱している場合のみに限定する\n"
    "\n"
    "## 種別固有の注意点\n"
    "\n"
    "### 手順型（レシピ、チュートリアル、攻略）の場合\n"
    "- 材料名・分量・調理時間・保存期間・温度などの具体値は特に正確に記載する\n"
    "- 工程の順序を入れ替えない\n"
    "- あるレシピ・手順のポイントを別のレシピ・手順のセクションに混入させない"
    "（トピック境界の厳守を特に意識する）\n"
    "- 料理名・技名・操作名など、音声認識で崩れがちだが一般名詞として"
    "復元可能な用語は、固有名詞ルールに従い妥当な表記に修正する\n"
    "\n"
    "### 物語型（アニメ、ドラマ、小説、映画）の場合\n"
    "- 登場人物名の表記ゆれに注意し、同一人物は一つの表記に統一する\n"
    "- 時系列と因果関係を崩さない\n"
    "- ネタバレの扱いは原文の提示順に従う（結末を冒頭に持ってこない）\n"
    "\n"
    "### 対話型（インタビュー、対談、座談会）の場合\n"
    "- 誰の発言かを必ず明示する\n"
    "- 発言者間で意見が対立している箇所は、対立構造を残す"
    "（片方の主張だけを採用して整理しない）\n"
    "\n"
    "### 評価型（レビュー、比較、感想）の場合\n"
    "- 評価は必ず誰による評価かを明示する\n"
    "- 評価軸（価格、性能、使いやすさ等）を可能な限り明確に区別する\n"
    "\n"
    "### 情報伝達型（ニュース、速報、まとめ）の場合\n"
    "- 情報源が複数ある場合、各情報の出典を区別する\n"
    "- 確定情報と未確定情報（「〜と報じられている」「〜の可能性がある」等）を区別する\n"
    "\n"
    "### 解説型（講義、技術記事、ハウツー）の場合\n"
    "- 主張と根拠の対応関係を崩さない\n"
    "- 前提条件や適用範囲（「〜の場合に限る」等）を省略しない\n"
    "\n"
    "### 文書型（報告書、論文、契約書）の場合\n"
    "- 章構成をできる限り原文の構造に合わせる\n"
    "- 定義された用語は初出時の定義を尊重し、勝手に言い換えない\n"
)

# Annotation section (modification history) was attempted in two designs
# (2026-04-17) and both failed. v1 asked the LLM to list "low-confidence
# names" but LLMs report high confidence on the context-driven recoveries
# we most want humans to verify. v2 switched to "list every modification"
# to bypass self-confidence judgment, but the LLM produced entries that
# did not match what it actually wrote in the body (hallucinated history).
# Conclusion: self-reflective tasks on proper-noun edits cannot be
# reliably embedded in the same prompt that produces the summary. A
# two-pass implementation (separate LLM call diffing summary against
# transcript) would be more trustworthy but is out of scope here. The
# correction rules themselves still work — anchor-based normalization
# and conservative preservation do fix the core issue — so we keep them
# and drop only the annotation. See hako: yIKMeQpNJXNGw1TYaiPN2 → A.

# Per-request token budget for detailed summaries. Generous because the
# output is a full Markdown document (intro + bullets + table + conclusion)
# and can legitimately run past the default 2048 cap. 4096 leaves head
# room for Japanese output where each character costs one token.
_DETAILED_MAX_TOKENS = 4096

# detailed_status transitions: None → "generating" → "generated" | "failed".
# "generating" is the only non-terminal state; transitions through it are
# taken while a background task is working on the file.
DETAILED_STATUS_GENERATING = "generating"
DETAILED_STATUS_GENERATED = "generated"
DETAILED_STATUS_FAILED = "failed"


def _build_system_prompt() -> str:
    """Build the system prompt with language instruction from config."""
    lang = settings.llm.output_language
    lang_line = _LANGUAGE_INSTRUCTIONS.get(lang, "")

    return (
        "あなたはファイル管理システムの要約アシスタントです。\n"
        "以下のコンテンツを読んで、短いサマリーと段落要約を生成してください。\n"
        "\n"
        "## 出力形式\n"
        "\n"
        'JSON形式で返すこと: {"short": "1文サマリー", "long": "段落要約"}\n'
        "JSONのみ返し、他のテキストは含めないこと\n"
        "\n"
        "## short(1文サマリー)の規則\n"
        "\n"
        "- 1文（30-80文字）で最も重要な要点を表す\n"
        "- 原文が複数トピックを扱う場合、最も中心的なテーマを選ぶか、"
        "「〜など複数の話題」のように明示する\n"
        "- 断定的な誇張を避ける。原文が推測・予想なら"
        "「〜の可能性」「〜の見通し」などの表現を保持する\n"
        "- 原文にない評価語（「画期的」「衝撃の」等）を加えない\n"
        "\n"
        "## long(段落要約)の規則\n"
        "\n"
        "- 3-5文（200-400文字）で主要内容を説明する\n"
        "- 原文の流れ（時系列または論理順）に沿って記述する\n"
        "- 複数トピックがある場合は、重要度順または原文の順序で触れる\n"
        "- 語り手の温度感は維持するが、視点は観察者（三人称）で書く\n"
        "  例: 「〜と述べている」「〜と推測している」「〜への期待を示している」\n"
        "- 事実・発言・推測・評価を区別する\n"
        "  - 発言を紹介する場合、可能な限り発言者を明示する\n"
        "  - 推測は「〜と推測している」「〜の可能性を示唆している」と明記する\n"
        "  - 評価語（「神ゲー」「傑作」「素晴らしい」等）を含める場合は、"
        "必ず誰による評価かを明示する（「語り手は〜と評している」「投稿者が〜と呼ぶ」等）。"
        "無帰属の評価語は要約者視点と誤解されるため避ける\n"
        "- 原文の不確実性マーカー（「かもしれない」「〜ではないか」）を"
        "字数削減のために省略しないこと\n"
        "- 原文にないニュアンス・一般化・総括を付け加えないこと\n"
        "\n"
        f"{_COMMON_RULES}"
        f"{lang_line}"
    )


def _build_detailed_system_prompt() -> str:
    """Build the system prompt for detailed (Markdown long-form) summaries.

    The user-visible output is a structured Markdown document with four
    sections (intro / detailed bullets / key-points table / conclusion).
    Language style is selected from ``settings.llm.output_language``;
    ``"auto"`` omits the style line so the model mirrors the source.
    """
    lang = settings.llm.output_language
    lang_line = _DETAILED_LANGUAGE_INSTRUCTIONS.get(lang, "")

    return (
        "あなたはファイル管理システムの要約アシスタントです。\n"
        "以下のコンテンツを読んで、長文の構造化要約を Markdown 形式で生成してください。\n"
        "\n"
        "## 手順\n"
        "\n"
        "**ステップ1: コンテンツの種別を見極める**\n"
        "冒頭を読んで、以下のどれに近いかを判断してください（内部処理のみ、出力不要）:\n"
        "- 情報伝達型（ニュース、速報、まとめ）: 複数の出典が混在、事実と推測の区別が重要\n"
        "- 解説型（講義、技術記事、ハウツー）: 論理展開と因果関係が重要\n"
        "- 手順型（レシピ、チュートリアル、攻略）: 順序と具体値が重要\n"
        "- 評価型（レビュー、比較、感想）: 評価軸と主観の帰属が重要\n"
        "- 対話型（インタビュー、対談、座談会）: 発言者と立場の区別が重要\n"
        "- 物語型（アニメ、ドラマ、小説、映画）: 時系列と因果、登場人物の関係が重要\n"
        "- 文書型（報告書、論文、契約書）: 章構成と論旨が重要\n"
        "- その他: 原文の構造に従う\n"
        "\n"
        "**ステップ2: 種別に応じて構造を調整する**\n"
        "下記の基本構成をベースに、種別に合わせて見出しや粒度を柔軟に変えてください。\n"
        "例えば手順型なら「材料/手順」、物語型なら「あらすじ/主要な展開/登場人物」、\n"
        "対話型なら「論点/各発言者の主張」など、コンテンツに最も適した構成を選びます。\n"
        "\n"
        "**ステップ3: 種別固有の注意点を適用する**\n"
        "「## 種別固有の注意点」セクションに、種別ごとの追加ルールを記載しています。\n"
        "ステップ1で判定した種別に該当する注意点を、要約作成時に必ず適用してください。\n"
        "\n"
        "## 基本構成\n"
        "\n"
        "1. **導入**(1-2文): 全体像と主要テーマを簡潔に\n"
        "2. **詳細内容**: 原文の流れ（時系列が明確ならその順、"
        "そうでなければ論理・章構成）に沿って整理。\n"
        "   並列な情報は箇条書き、因果・順序・対比が重要な部分は文章で記述する。\n"
        "   箇条書きの強制ではなく、内容に応じて使い分けること。\n"
        "3. **重要ポイントまとめ**: 数値・比較・対応関係が実際に存在する場合のみ "
        "Markdown 表で整理する。該当する内容がなければこのセクションは省略する"
        "（無理に表を作らない）\n"
        "4. **結論**(1-2文): 原文で示されている結論・締めくくり・示唆を要約する。"
        "原文にない提案や一般化を加えないこと。\n"
        "\n"
        "## 共通規則\n"
        "\n"
        "- 出力は Markdown のみ。JSON や他のラッパーで包まないこと\n"
        "- 語り手の温度感・熱量は維持するが、視点は観察者（三人称）で書く\n"
        "  例: 「〜と述べている」「〜と推測している」「〜と強い期待を示している」\n"
        "- 事実・発言・推測・評価を区別する\n"
        "  - 事実: 「〜が発表された」「〜が実施されている」\n"
        "  - 発言: 「誰それは〜と述べた」（可能なら出典も明示）\n"
        "  - 推測: 「語り手は〜と推測している」「〜の可能性を示唆している」\n"
        "  - 評価: 「語り手は〜と評価している」「〜を好意的に捉えている」\n"
        "  - 評価語（「神ゲー」「傑作」「素晴らしい」等）を本文に含める場合は、"
        "必ず誰による評価かを明示する（「語り手は〜と評している」「投稿者が〜と呼ぶ」等）。"
        "無帰属の評価語は要約者視点と誤解されるため避ける\n"
        "- 原文の不確実性マーカー（「かもしれない」「〜ではないか」「おそらく」）を"
        "省略せず保持する\n"
        "- 原文にない事実・一般化・総括・提案を付け加えないこと\n"
        "- 複数の情報源や発言者が登場する場合、誰の発言・情報かを明示する\n"
        "\n"
        f"{_COMMON_RULES}"
        f"{lang_line}"
    )


def _build_detailed_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    was_truncated: bool,
) -> str:
    """Build the user prompt for detailed-summary generation.

    Shares the same header layout (filename / type / title / description
    / truncation notice / content marker) as ``_build_user_prompt`` so
    the LLM sees a consistent structure across both summary variants.
    Labels stay Japanese regardless of output language — they are model
    instructions, not output, and the model does not mirror them.
    """
    parts: list[str] = [
        f"ファイル名: {indexed_file['filename']}",
        f"タイプ: {context_type}",
    ]

    title = indexed_file.get("title") or ""
    if title and title != indexed_file["filename"]:
        parts = [*parts, f"タイトル: {title}"]

    description = indexed_file.get("description") or ""
    if description:
        parts = [*parts, f"説明: {description}"]

    if was_truncated:
        parts = [
            *parts,
            "\n注: 以下は長いコンテンツの抜粋です。冒頭・中盤・終盤から取得しています。",
        ]

    parts = [*parts, "\n--- コンテンツ ---", context]

    return "\n".join(parts)


def _sample_windows(text: str, window_chars: int, window_count: int) -> str:
    """Sample head/middle/tail windows from a long text and join them.

    Uses endpoint distribution: the first window starts at the beginning,
    the last window ends at the end, and middle windows are centered at
    evenly-spaced fractions between. For window_count=3 this places
    windows at 0%, 50%, and 100% of the text — guaranteeing head and
    tail coverage, which an "even centers" approach would miss.

    Adjacent windows are merged if they overlap so duplicated content
    doesn't waste LLM context. Each window is then trimmed to sentence
    boundaries before being joined with a separator marker.

    Args:
        text: Source text to sample from.
        window_chars: Width of each window in characters.
        window_count: Number of windows to take (use odd numbers for
            symmetric head/middle/tail coverage).

    Returns:
        Joined windows with a separator marker between non-adjacent spans.
    """
    total = len(text)
    if window_count <= 0 or window_chars <= 0 or total == 0:
        return text

    # Compute (start, end) spans. For n=1 we take a single head window;
    # for n>1 we distribute centers at i/(n-1) so windows 0 and n-1
    # are pinned to the head and tail respectively.
    spans: list[tuple[int, int]] = []
    for i in range(window_count):
        if window_count == 1:
            center = 0
        else:
            center = int(total * i / (window_count - 1))
        half = window_chars // 2
        start = max(0, center - half)
        end = min(total, start + window_chars)
        # If we hit the tail, shift start back so the window stays full-width.
        # The shifted start can overlap the previous window — that's fine
        # because the merge pass below dedupes overlapping spans.
        if end == total:
            start = max(0, end - window_chars)
        spans.append((start, end))

    # Merge overlapping spans so we don't duplicate text between windows.
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged = [*merged, span]

    snippets = [trim_to_sentence_boundary(text[s:e]) for s, e in merged]
    return _WINDOW_SEPARATOR.join(snippets)


def _prepare_context(
    text: str,
    *,
    max_chars: int | None = None,
    window_chars: int | None = None,
    window_count: int | None = None,
) -> tuple[str, bool]:
    """Prepare the context text for the LLM, sampling windows if needed.

    Args:
        text: Raw context text (full transcript, full document, etc.).
        max_chars: Full-text threshold override. Falls back to
            ``settings.summaries.max_context_chars`` when None.
        window_chars: Per-window width override for the sampling
            fallback. Falls back to ``settings.summaries.window_chars``.
        window_count: Window-count override for the sampling fallback.
            Falls back to ``settings.summaries.window_count``.

    Returns:
        Tuple of (prepared_text, was_truncated). was_truncated is True
        when the input exceeded the threshold and windows were sampled.
    """
    cfg = settings.summaries
    effective_max = cfg.max_context_chars if max_chars is None else max_chars
    effective_window_chars = (
        cfg.window_chars if window_chars is None else window_chars
    )
    effective_window_count = (
        cfg.window_count if window_count is None else window_count
    )
    if len(text) <= effective_max:
        return (text, False)

    sampled = _sample_windows(
        text, effective_window_chars, effective_window_count
    )
    return (sampled, True)


def _classify_file_type(file_type: str, mime_type: str | None = None) -> str | None:
    """Map a raw file_type to a summaries context type, or None if unsupported.

    HVLink files (external video references like YouTube) are classified as
    "video" so their VTT-derived transcripts feed the same summary path as
    local videos — their host-side file_type is "other" by MIME heuristics.
    """
    if mime_type == HVLINK_MIME:
        return "video"
    if file_type == "video":
        return "video"
    if file_type == "audio":
        return "audio"
    if file_type in ("document", "text"):
        return "document"
    return None


def classify_missing_reason(file_id: str) -> str:
    """Explain why a file does not currently have a stored summary.

    Called by the router when the file_summaries table has no row for
    a given file, so the frontend can render the right UI state rather
    than always offering a "Generate" button that would silently skip.

    Return values:
        "file_not_found"        — no indexed_files row at all
        "unsupported_type"      — image/archive/etc.
        "insufficient_content"  — supported type but below threshold
        "not_generated"         — ready to generate, waiting for user
    """
    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        return "file_not_found"

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
    if context_type is None:
        return "unsupported_type"

    # Reuse _build_context so the threshold check is applied exactly
    # the same way the worker would apply it — no second source of truth.
    if _build_context(indexed_file, context_type) is None:
        return "insufficient_content"

    return "not_generated"


def _get_indexed_file(file_id: str) -> dict | None:
    """Fetch basic indexed-file info for a file, or None if not indexed."""
    with get_search_db() as session:
        f = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if f is None:
            return None
        return {
            "file_id": f.file_id,
            "filename": f.filename,
            "file_type": f.file_type,
            "mime_type": f.mime_type,
            "title": f.title,
            "description": f.description,
        }


def _get_full_transcript(file_id: str) -> str:
    """Load the entire Whisper transcript for a file as one string."""
    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        if not chunks:
            return ""
        return " ".join(c.text for c in chunks if c.text)


def _get_full_document_text(file_id: str) -> str:
    """Load the entire extracted document text for a file as one string.

    Reads from the fts_text_content FTS5 table rather than the embeddings
    table — the latter only stores 200-char content_previews (see
    app/workers/metadata.py), while fts_text_content holds the full
    chunk text written during indexing. Chunks are concatenated in
    numeric chunk_index order.
    """
    with get_search_db() as session:
        # chunk_index is stored as a string in the FTS5 table, so we cast
        # to INTEGER for correct ordering (otherwise "10" sorts before "2").
        rows = session.execute(
            sql_text(
                "SELECT text FROM fts_text_content "
                "WHERE file_id = :fid "
                "ORDER BY CAST(chunk_index AS INTEGER)"
            ),
            {"fid": file_id},
        ).fetchall()
        if not rows:
            return ""
        return "\n\n".join(row[0] for row in rows if row[0])


def _build_context(indexed_file: dict, context_type: str) -> str | None:
    """Build raw context text for a file, or None if no content is available.

    Enforces settings.summaries.min_context_chars: any file whose usable
    content (after stripping whitespace) is shorter than that threshold
    returns None and is skipped by the worker. This guards against the
    LLM hallucinating a summary from only the filename when Whisper
    produces a trivial transcript (e.g., "you" on a silent piano video)
    or a document extractor yields a near-empty text layer.

    Args:
        indexed_file: File info dict from _get_indexed_file.
        context_type: "video" | "audio" | "document".

    Returns:
        The raw (untruncated) context text, or None if the file has no
        usable transcript / document text to summarize.
    """
    file_id = indexed_file["file_id"]
    raw: str = ""

    if context_type in ("video", "audio"):
        raw = _get_full_transcript(file_id)
    elif context_type == "document":
        raw = _get_full_document_text(file_id)

    if not raw:
        return None

    min_chars = settings.summaries.min_context_chars
    if len(raw.strip()) < min_chars:
        return None

    return raw


def _build_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    was_truncated: bool,
) -> str:
    """Build the user prompt for LLM summary generation."""
    parts: list[str] = [
        f"ファイル名: {indexed_file['filename']}",
        f"タイプ: {context_type}",
    ]

    title = indexed_file.get("title") or ""
    if title and title != indexed_file["filename"]:
        parts = [*parts, f"タイトル: {title}"]

    description = indexed_file.get("description") or ""
    if description:
        parts = [*parts, f"説明: {description}"]

    if was_truncated:
        parts = [
            *parts,
            "\n注: 以下は長いコンテンツの抜粋です。冒頭・中盤・終盤から取得しています。",
        ]

    parts = [*parts, "\n--- コンテンツ ---", context]

    return "\n".join(parts)


def _has_summary(file_id: str) -> bool:
    """True if a summary (in any status) already exists for this file."""
    with get_search_db() as session:
        row = session.execute(
            sql_text("SELECT 1 FROM file_summaries WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _save_summary(
    *,
    file_id: str,
    short_summary: str,
    long_summary: str,
    model: str,
    context_type: str,
    context_chars: int,
    was_truncated: bool,
) -> None:
    """Insert or replace the summary record for a file."""
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR REPLACE INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:file_id, :short_summary, :long_summary, :model, "
                ":context_type, :context_chars, :was_truncated, 'generated', "
                ":created_at)"
            ),
            {
                "file_id": file_id,
                "short_summary": short_summary,
                "long_summary": long_summary,
                "model": model,
                "context_type": context_type,
                "context_chars": context_chars,
                "was_truncated": 1 if was_truncated else 0,
                "created_at": now,
            },
        )


def _has_detailed_summary(file_id: str) -> bool:
    """True if a detailed summary exists for this file (any status).

    ``generating`` rows count as "has" — the router uses this to 409 a
    second generation request while one is already in flight.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT 1 FROM file_summaries "
                "WHERE file_id = :fid AND detailed_status IS NOT NULL"
            ),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _set_detailed_status(
    file_id: str,
    status: str,
    *,
    error: str | None = None,
    model: str | None = None,
) -> None:
    """Upsert the row and set detailed_status.

    On first write we may land on a file that has no file_summaries row
    at all (short/long never generated). INSERT OR IGNORE establishes
    the row with empty short/long placeholders, then UPDATE writes the
    detailed-related columns. Using placeholders keeps the NOT NULL
    constraints on short_summary/long_summary happy without lying about
    their presence — the status column is the source of truth for
    "does detailed exist".
    """
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR IGNORE INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', :now)"
            ),
            {"fid": file_id, "now": now},
        )
        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "detailed_status = :status, "
                "detailed_error = :error, "
                "detailed_model = COALESCE(:model, detailed_model) "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "status": status,
                "error": error,
                "model": model,
            },
        )


def _save_detailed_summary(
    *,
    file_id: str,
    detailed_summary: str,
    model: str,
    context_chars: int,
    was_truncated: bool,
) -> None:
    """Write the generated Markdown summary and transition to 'generated'.

    Also logs the event in ``file_insights``:
    - Any existing active insight for this file is marked ``superseded``
    - A new row is inserted with ``status='active'``,
      ``created_by='intelligence'``, and generation metadata.
    Both writes happen in the same session so the history log and the
    summary body can never drift out of sync.
    """
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "detailed_summary = :detailed_summary, "
                "detailed_status = :status, "
                "detailed_model = :model, "
                "detailed_generated_at = :generated_at, "
                "detailed_context_chars = :context_chars, "
                "detailed_was_truncated = :was_truncated, "
                "detailed_error = NULL "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "detailed_summary": detailed_summary,
                "status": DETAILED_STATUS_GENERATED,
                "model": model,
                "generated_at": now,
                "context_chars": context_chars,
                "was_truncated": 1 if was_truncated else 0,
            },
        )
        _supersede_and_insert_insight(
            session,
            file_id=file_id,
            kind="detailed_summary",
            content=detailed_summary,
            created_by="intelligence",
            metadata={
                "model": model,
                "context_chars": context_chars,
                "was_truncated": was_truncated,
            },
            created_at=now,
        )


def _supersede_and_insert_insight(
    session,
    *,
    file_id: str,
    kind: str,
    content: str,
    created_by: str,
    metadata: dict,
    created_at: str,
) -> None:
    """Append an insight event: mark existing active row superseded, insert new.

    Helper shared by the worker (new generation) and the router
    (manual edit, revert). Caller is responsible for committing the
    session — this function only queues statements.
    """
    session.execute(
        sql_text(
            "UPDATE file_insights SET status = 'superseded' "
            "WHERE file_id = :fid AND kind = :kind AND status = 'active'"
        ),
        {"fid": file_id, "kind": kind},
    )
    session.execute(
        sql_text(
            "INSERT INTO file_insights "
            "(id, file_id, kind, content, metadata_json, "
            " status, created_by, created_at) "
            "VALUES (:id, :fid, :kind, :c, :m, 'active', :cb, :ca)"
        ),
        {
            "id": generate_insight_id(),
            "fid": file_id,
            "kind": kind,
            "c": content,
            "m": json.dumps(metadata) if metadata else None,
            "cb": created_by,
            "ca": created_at,
        },
    )


def _get_detailed_summary(file_id: str) -> dict | None:
    """Fetch the detailed-summary record for a file.

    Returns a dict with the user-facing fields, or None if no detailed
    work has been started yet (status column NULL).
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT detailed_summary, detailed_status, detailed_model, "
                "detailed_generated_at, detailed_context_chars, "
                "detailed_was_truncated, detailed_error, "
                "detailed_original, detailed_edited_at "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is None or row[1] is None:
            return None
        return {
            "detailed_summary": row[0],
            "detailed_status": row[1],
            "detailed_model": row[2],
            "detailed_generated_at": row[3],
            "detailed_context_chars": row[4],
            "detailed_was_truncated": (
                bool(row[5]) if row[5] is not None else None
            ),
            "detailed_error": row[6],
            "detailed_original": row[7],
            "detailed_edited_at": row[8],
        }


def _delete_detailed_summary(file_id: str) -> bool:
    """Clear all detailed-* columns for a file.

    Returns True when at least one row matched, False when the file had
    no summary row at all. The row itself is preserved when short/long
    are still present; if it has no other content, the row is deleted
    entirely so repeat generation starts from a clean slate.

    Always drops the file's ``detailed_summary_citations`` rows — the
    citation set is derived from the body text, so it's meaningless
    after the body is cleared. No-op when no citations exist.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT short_summary, long_summary FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is None:
            return False

        short_val = row[0] or ""
        long_val = row[1] or ""
        if short_val or long_val:
            session.execute(
                sql_text(
                    "UPDATE file_summaries SET "
                    "detailed_summary = NULL, "
                    "detailed_status = NULL, "
                    "detailed_model = NULL, "
                    "detailed_generated_at = NULL, "
                    "detailed_context_chars = NULL, "
                    "detailed_was_truncated = NULL, "
                    "detailed_error = NULL, "
                    "detailed_original = NULL, "
                    "detailed_edited_at = NULL "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )
        else:
            # Placeholder row created by _set_detailed_status on a file
            # that never had short/long — drop it to return to the
            # pristine "no summary" state.
            session.execute(
                sql_text(
                    "DELETE FROM file_summaries WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )

        # Always drop citations: their lifetime is pegged to the
        # summary body.
        session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )

        # Drop the full ``file_insights`` history for this file+kind:
        # regenerate is an explicit "clean slate" action (user
        # confirmed via the UI flow — see hako ``Mt4TJ9joamta7G0mt-Ifo``),
        # so retaining prior superseded rows is not desired. The next
        # ``_save_detailed_summary`` will insert a fresh active row.
        session.execute(
            sql_text(
                "DELETE FROM file_insights "
                "WHERE file_id = :fid AND kind = 'detailed_summary'"
            ),
            {"fid": file_id},
        )
        return True


def classify_detailed_missing_reason(file_id: str) -> str:
    """Explain why no detailed summary exists for a file.

    Mirrors :func:`classify_missing_reason` for the detailed column so
    the router can tell the frontend why a "Generate" button would be
    a no-op. Return values match the short/long variant:

    * ``"file_not_found"``       — no indexed_files row
    * ``"unsupported_type"``     — image / archive / other
    * ``"insufficient_content"`` — supported type but below threshold
    * ``"not_generated"``        — ready to generate
    """
    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        return "file_not_found"

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
    if context_type is None:
        return "unsupported_type"

    if _build_context(indexed_file, context_type) is None:
        return "insufficient_content"

    return "not_generated"


async def generate_detailed_summary(
    file_id: str,
    llm_client: LLMClient,
) -> None:
    """Generate a detailed Markdown summary for a single file.

    Intended to be scheduled via FastAPI BackgroundTasks from the router
    — runs synchronously against the configured LLM and takes tens of
    seconds to a few minutes for long transcripts on local ollama.

    Lifecycle:
    1. Policy gate: skip if ``features.detailed_summaries`` is off.
    2. LLM gate: skip if the client is disabled.
    3. Pre-checks via :func:`classify_detailed_missing_reason` — any
       non ``"not_generated"`` result is a silent skip (router has
       already rejected the request in that case, this is a defence).
    4. Status transitions: ``generating`` → ``generated`` or ``failed``
       (with the error message stored in ``detailed_error``).

    Raises nothing — all failure modes land in the ``failed`` row so
    the frontend polling surface is uniform. The background-task
    harness only sees a clean return.
    """
    if settings.features.detailed_summaries == "false":
        return

    if not llm_client.enabled:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="LLM provider is disabled",
        )
        return

    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        # Router would have returned 404 already; defence in depth.
        return

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
    if context_type not in _SUPPORTED_CONTEXT_TYPES:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error=f"Unsupported file type: {indexed_file['file_type']}",
        )
        return

    raw_context = _build_context(indexed_file, context_type)
    if not raw_context:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="No usable content to summarize",
        )
        return

    prepared, was_truncated = _prepare_context(
        raw_context,
        max_chars=settings.summaries.detailed_max_context_chars,
        window_count=settings.summaries.detailed_window_count,
    )
    system_prompt = _build_detailed_system_prompt()
    user_prompt = _build_detailed_user_prompt(
        indexed_file, context_type, prepared, was_truncated
    )

    _set_detailed_status(
        file_id,
        DETAILED_STATUS_GENERATING,
        model=settings.llm.model,
    )

    try:
        raw = await llm_client.generate(
            system_prompt,
            user_prompt,
            max_tokens_override=_DETAILED_MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001 - surface any LLM error uniformly
        logger.exception(
            "Detailed summary generation crashed for %s", file_id
        )
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error=f"LLM error: {e}",
        )
        return

    if not raw or not raw.strip():
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="LLM returned empty output",
        )
        return

    saved_text = raw.strip()
    _save_detailed_summary(
        file_id=file_id,
        detailed_summary=saved_text,
        model=settings.llm.model,
        context_chars=len(prepared),
        was_truncated=was_truncated,
    )
    logger.info(
        "Detailed summary: saved for %s (%s, %d chars, truncated=%s)",
        file_id, context_type, len(prepared), was_truncated,
    )

    # Post-generation hook: compute citations so the UI can overlay
    # source badges onto the freshly generated summary. Failures are
    # logged + swallowed inside ``_recalculate_citations`` — they must
    # not mark the summary itself as failed.
    await _recalculate_citations(file_id, saved_text)


class SummariesWorker:
    """Async worker that processes summary generation requests via a queue."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the summaries queue.

        The queue is shared between the short/long path and the
        detailed-summary chain, so a file is accepted when either
        per-drive policy allows its respective feature. Drives that
        disable both features drop out here. Fails open on transient
        internal-API failure (`is_file_feature_enabled` returns True).
        """
        from app.policy_client import is_file_feature_enabled
        if await is_file_feature_enabled(file_id, "summaries"):
            await self._queue.put(file_id)
            return
        # Short/long blocked — still enqueue if detailed is enabled for
        # this drive so the detailed chain in _process_file can run.
        if (
            settings.features.detailed_summaries == "on_index"
            and await is_file_feature_enabled(file_id, "detailed_summaries")
        ):
            await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files needing summary work and enqueue them.

        Walks two gaps in the search DB:
        - short/long: files with no ``file_summaries`` row at all
        - detailed:   files that have short/long but no detailed yet
          (``detailed_status IS NULL``)

        The short/long gap is only walked when
        ``features.summaries == "on_index"``; the detailed gap only
        when ``features.detailed_summaries == "on_index"``. Per-drive
        policy for each feature is consulted once per drive so
        opted-out drives don't generate work.

        Returns:
            Number of files queued (after policy filtering).
        """
        from app.policy_client import is_feature_enabled

        want_short = settings.features.summaries == "on_index"
        want_detailed = settings.features.detailed_summaries == "on_index"
        if not want_short and not want_detailed:
            return 0

        pending: list[tuple[str, str, str]] = []  # (file_id, drive, feature)
        with get_search_db() as session:
            if want_short:
                rows = session.execute(
                    sql_text(
                        "SELECT f.file_id, f.drive FROM indexed_files f "
                        "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                        "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                        "AND f.file_id NOT IN (SELECT file_id FROM file_summaries)"
                    )
                ).fetchall()
                pending.extend(
                    (file_id, drive, "summaries") for file_id, drive in rows
                )
            if want_detailed:
                rows = session.execute(
                    sql_text(
                        "SELECT f.file_id, f.drive FROM indexed_files f "
                        "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                        "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                        "AND f.file_id NOT IN ("
                        "  SELECT file_id FROM file_summaries "
                        "  WHERE detailed_status IS NOT NULL"
                        ")"
                    )
                ).fetchall()
                pending.extend(
                    (file_id, drive, "detailed_summaries")
                    for file_id, drive in rows
                )

        # De-dupe file_ids: a file may appear in both gaps. Keep the
        # first occurrence (short/long if both modes are on, detailed
        # otherwise) — _process_file handles each layer independently.
        seen: set[str] = set()
        allowed_cache: dict[tuple[str, str], bool] = {}
        count = 0
        for file_id, drive, feature in pending:
            if file_id in seen:
                continue
            key = (drive, feature)
            allowed = allowed_cache.get(key)
            if allowed is None:
                allowed = await is_feature_enabled(drive, feature)
                allowed_cache[key] = allowed
            if not allowed:
                continue
            await self._queue.put(file_id)
            seen.add(file_id)
            count += 1
        return count

    async def run(self) -> None:
        """Main worker loop. Processes one file at a time."""
        while True:
            try:
                file_id = await self._queue.get()
                await self._process_file(file_id)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Summaries worker error: %s", e)

    async def _process_file(self, file_id: str) -> None:
        """Generate summaries for a single file.

        Handles both the short/long summary and (when configured) the
        detailed summary. Each layer has its own feature gate and
        existence check, so callers can enable them independently:

        - ``features.summaries``: ``"manual"`` or ``"on_index"`` runs the
          short/long generation when no file_summaries row exists yet
        - ``features.detailed_summaries``: only ``"on_index"`` triggers
          automatic detailed generation from this worker; ``"manual"``
          is handled by the router's BackgroundTasks route instead
        """
        if not self._llm_client.enabled:
            return

        want_short = (
            settings.features.summaries != "false"
            and not _has_summary(file_id)
        )
        want_detailed = (
            settings.features.detailed_summaries == "on_index"
            and not _has_detailed_summary(file_id)
        )
        if not want_short and not want_detailed:
            return

        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        context_type = _classify_file_type(
            indexed_file["file_type"], indexed_file.get("mime_type")
        )
        if context_type not in _SUPPORTED_CONTEXT_TYPES:
            return

        raw_context = _build_context(indexed_file, context_type)
        if not raw_context:
            logger.debug(
                "Summaries: no context available for %s (%s)",
                file_id, context_type,
            )
            return

        if want_short:
            await self._generate_short_long(
                file_id, indexed_file, context_type, raw_context
            )

        if want_detailed:
            # Per-drive policy for "detailed_summaries" gates independently
            # of "summaries" so operators can opt individual drives in/out
            # without disabling the short/long path.
            from app.policy_client import is_file_feature_enabled
            if await is_file_feature_enabled(file_id, "detailed_summaries"):
                await generate_detailed_summary(file_id, self._llm_client)

    async def _generate_short_long(
        self,
        file_id: str,
        indexed_file: dict,
        context_type: str,
        raw_context: str,
    ) -> None:
        """Run the short/long generation path and persist the result."""
        prepared, was_truncated = _prepare_context(raw_context)
        user_prompt = _build_user_prompt(
            indexed_file, context_type, prepared, was_truncated
        )

        parsed = await self._llm_client.generate_json(
            _build_system_prompt(), user_prompt
        )

        if not isinstance(parsed, dict):
            logger.warning(
                "Summaries LLM returned non-dict for %s, skipping", file_id
            )
            return

        short_raw = parsed.get("short")
        long_raw = parsed.get("long")
        if not isinstance(short_raw, str) or not isinstance(long_raw, str):
            logger.warning(
                "Summaries LLM response missing short/long fields for %s", file_id
            )
            return

        short_summary = short_raw.strip()
        long_summary = long_raw.strip()
        if not short_summary or not long_summary:
            logger.warning(
                "Summaries LLM produced empty short/long for %s", file_id
            )
            return

        _save_summary(
            file_id=file_id,
            short_summary=short_summary,
            long_summary=long_summary,
            model=settings.llm.model,
            context_type=context_type,
            context_chars=len(prepared),
            was_truncated=was_truncated,
        )
        logger.info(
            "Summaries: saved summary for %s (%s, %d chars, truncated=%s)",
            file_id, context_type, len(prepared), was_truncated,
        )
