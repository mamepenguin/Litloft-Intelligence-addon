"""TF-IDF similarity for similar file search.

Uses word-level tokenization (Janome) to compute TF-IDF vectors from
transcript text and filenames, then finds similar files by cosine
similarity.  Filenames are a high-density source of topic keywords
(especially proper nouns that Whisper may miss), so their tokens are
boosted relative to transcript tokens.

The tokenizer is loaded lazily and released after use to avoid holding
~100MB of memory when not indexing.
"""

import logging
import math
import os
import re
import threading
import time
from collections import Counter

from app.database import get_search_db, get_search_db_read
from app.models import IndexedFile, TranscriptChunk

logger = logging.getLogger(__name__)

# --- Corpus IDF cache for single-file keyword extraction ---
# The cache is invalidated on a time-based TTL. Rebuilding IDF
# means tokenizing every indexed file's transcript+filename, so a
# per-call rebuild is prohibitive during batch auto-tagging. A 10
# minute window gives new files a chance to appear in IDF without
# pinning stale counts across long-running sessions.
_IDF_CACHE_TTL_SECONDS = 600

_corpus_cache_lock = threading.Lock()
#: drive -> (idf, n_docs, built_at). Document frequency is a corpus
#: statistic, and a drive is a security boundary, so one drive's
#: contents must never decide which words survive in another. The
#: cache is keyed by drive for the same reason the counts are.
_corpus_idf_by_drive: dict[str, tuple[dict[str, float], int, float]] = {}

# Keep only nouns — topic-relevant words are predominantly nouns.
# Verbs/adjectives/adverbs describe actions/states, not topics.
_CONTENT_POS = frozenset({"名詞"})
# Skip non-informative noun subtypes
_SKIP_NOUN_SUBTYPES = frozenset({
    "非自立", "代名詞", "数", "接尾",
    "副詞可能",      # 今日, 最初, 全部 等 — 時間・程度表現
    "形容動詞語幹",  # 簡単, 重要, 必要 等 — 汎用的な形容表現
})
# Tokens consisting entirely of symbols/punctuation (全角・半角とも)
_SYMBOL_RE = re.compile(
    r"^[\s\d"
    r"\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"  # ASCII symbols
    r"\u3000-\u303F"   # CJK symbols & punctuation (、。「」等)
    r"\uFF01-\uFF60"   # fullwidth forms (！＃＄等)
    r"\u2000-\u206F"   # general punctuation
    r"\u2190-\u21FF"   # arrows
    r"\u2500-\u257F"   # box drawing
    r"\u25A0-\u25FF"   # geometric shapes
    r"\u2600-\u26FF"   # miscellaneous symbols
    r"\u2700-\u27BF"   # dingbats
    r"\uFE30-\uFE4F"   # CJK compatibility forms
    r"]+$",
)
# ASCII-only tokens that are 5+ chars with no vowels are likely gibberish
# from random filenames (e.g. "fsdjh", "krfk49"), not real words.
# Short tokens (≤4) are allowed as abbreviations (DQ, HD, FF14).
# Tokens with vowels are allowed as English words (Game, Pokemon, Review).
_ASCII_GIBBERISH_RE = re.compile(
    r"^[a-zA-Z0-9]{5,}$",
)
_HAS_VOWEL_RE = re.compile(r"[aeiouAEIOU]")

# Topic-independent nouns that appear frequently in spoken transcripts.
# Single-char words are already filtered by len(surface) <= 1.
# 代名詞/副詞可能/形容動詞語幹 are filtered by _SKIP_NOUN_SUBTYPES.
_STOPWORDS = frozenset({
    # 形式名詞・断片になりやすいもの
    "こと", "もの", "ところ", "とき", "ため", "よう", "はず",
    "わけ", "つもり", "くらい", "ぐらい", "ほど", "せい", "おかげ",
    "ほう", "とこ", "以上", "以下",
    "感じ", "部分", "向け", "単位",
    # 指示・参照（代名詞フィルタを通過するもの）
    "こちら", "そちら", "あちら", "こっち", "そっち", "あっち", "どっち",
    "ここ", "そこ", "あそこ", "あたり", "へん",
    "あれ", "これ", "それ",
    # 誤分割で名詞として出やすい口語の断片
    "じゃん", "じゃない", "みたい", "っぽい",
    "なん", "なの", "なんか", "なんで", "なんと",
    "てか", "ていう", "という", "とか",
    # 文字起こし特有の誤分割（句読点なしで動詞語尾が名詞化）
    "ます", "です", "ました", "でした", "ません",
    "あと", "もう", "ほんとに", "マジで", "ガチで",
    # 時間・順序
    "今回", "今度", "今後", "今週", "今月", "最初", "最後", "最近", "途中",
    "以前", "以降", "直前", "直後", "現在", "将来", "過去", "当時",
    "次回", "前回", "初回", "毎回", "後半", "前半",
    "タイミング", "段階", "時期", "時点", "時間", "期間",
    # 抽象・構造
    "雰囲気", "流れ", "展開", "パターン", "ケース", "場合",
    "箇所", "観点", "視点", "角度",
    "形式", "レベル", "規模",
    "方法", "手段", "仕組み", "方針", "方向", "目的", "理由", "原因", "結果",
    "影響", "効果", "状況", "状態", "場面", "場所", "機会",
    "可能性", "必要性", "重要性",
    "内容", "概要", "詳細", "項目", "要素", "事例",
    # 行為の名詞化（サ変接続、話題に依存しないもの）
    "確認", "対応", "実施", "実行", "検討", "判断", "決定", "選択",
    "設定", "変更", "修正", "更新", "追加", "削除", "登録",
    "管理", "運用", "利用", "使用", "活用", "適用", "導入", "達成", "解決",
    "共有", "連携", "統合", "分析", "評価", "比較", "検証", "調査",
    "把握", "整理", "準備", "作成", "生成", "提供",
    "送信", "受信", "取得", "収集", "保存", "表示", "出力", "入力", "処理",
    "説明", "紹介",
    # 口語・文字起こし頻出
    "やつ", "マジ", "ガチ", "ほんと",
    "感想", "印象", "イメージ", "ネタ", "話題",
    # 動画・配信文脈（メタ語）
    "動画", "配信", "放送", "収録", "本編",
    "チャンネル", "コメント", "シリーズ", "企画", "テーマ", "タイトル",
    "サムネ", "概要欄", "コメント欄", "スパチャ", "切り抜き", "コラボ", "ゲスト",
    "リンク", "shorts",
    # 人称・人物
    "みんな", "皆さん", "みなさん", "自分", "相手",
    "チーム", "メンバー", "担当", "担当者",
    "ユーザー", "スタッフ", "視聴者", "リスナー", "ファン", "フォロワー",
    # English stopwords (Janome classifies unknown English words as 名詞,一般
    # so they pass through the POS filter; IDF naturally downweights the
    # most common ones, but explicit stopwords catch the rest)
    "the", "be", "to", "of", "and", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "most", "us", "very", "much", "really", "thing", "things",
    "something", "actually", "basically", "kind", "stuff", "right",
    "going", "gonna", "got", "yeah", "okay", "here",
})

# Brackets, symbols, and punctuation used as visual separators in titles.
# These are replaced with spaces to split the title into meaningful segments.
_TITLE_SEPARATOR_RE = re.compile(
    r"["
    r"【】\[\]「」『』（）\(\)＜＞<>｛｝\{\}〈〉《》〔〕"  # brackets
    r"／＼/\\｜\|"                     # slashes & pipes
    r"※★☆◆◇■□▲△▼▽●○"               # decorative markers
    r"♪♫♬♩"                           # music notes
    r"＾～〜"                          # wave/circumflex
    r"！？!?＊\*＃#"                   # exclamation, question, etc.
    r"⧸"                              # fraction slash (U+29F8)
    r"]+",
)

# Filename-specific patterns to strip before tokenization
_FILENAME_NOISE_RE = re.compile(
    r"(?:"
    r"(?:19|20)\d{2}[.\-/]?\d{2}[.\-/]?\d{2}"  # dates: 20240101, 2024-01-01
    r"|\b(?:ep|episode|vol|part|ch)\.?\s*\d+"  # ep01, episode 3, vol.2
    r"|\b第?\d{1,4}話\b"              # 第332話, 12話
    r"|\b\d{1,2}月\d{1,2}日\b"        # 4月14日
    r"|\bAm?\d|Pm?\d"                 # Am7, P3 (time fragments)
    r"|\b\d{1,2}時間\b"               # 3時間
    r"|\d+週間"                       # 1週間
    r"|\d+個"                         # 1個
    r"|\d+円"                         # 220円
    r"|\(\d+\)|\[\d+\]"              # (1), [2]
    r"|公式|限定配信|限定"             # meta labels
    r")",
    re.IGNORECASE,
)

# How many times to repeat filename tokens to boost their TF weight.
# Filenames are short but information-dense; this ensures they contribute
# meaningfully against a long transcript.  A typical transcript has
# hundreds of tokens, so a high multiplier is needed for filename
# keywords to rank competitively.
_FILENAME_BOOST = 10

# Raw segments from filenames (potential proper nouns) that survive
# Janome splitting get an extra boost on top of _FILENAME_BOOST,
# since they are very likely to be the core topic identifier.
_RAW_SEGMENT_EXTRA_BOOST = 3

# Reject raw segments that contain Japanese particles or punctuation —
# these are sentence fragments, not proper nouns.
_SEGMENT_JUNK_RE = re.compile(
    r"[、。，．\s]"                     # punctuation inside segment
    r"|[はがをのにでとへもからまでより]$"  # trailing particles
    r"|^[はがをのにでとへもからまでより]"  # leading particles
)


def _tokenize_filename(filename: str, tokenizer) -> list[str]:
    """Extract topic tokens from a filename.

    Two-pass strategy:
    1. Raw segments: split by separators (_, -, space, etc.) to preserve
       proper nouns unknown to Janome (e.g. "ちいかわ", "FF14").
    2. Janome pass: morphological analysis to decompose compound segments
       (e.g. "グッズ紹介" → "グッズ", "紹介").

    Both are merged (deduplicated) so unknown proper nouns survive even
    when Janome splits them into single characters.
    """
    name = os.path.splitext(filename)[0]
    # Replace all separator types with spaces (ASCII + fullwidth + brackets)
    name = re.sub(r"[_\-.]", " ", name)
    name = _TITLE_SEPARATOR_RE.sub(" ", name)
    # Remove noise patterns (dates, episode numbers, meta labels)
    name = _FILENAME_NOISE_RE.sub(" ", name)
    name = name.strip()
    if not name:
        return []

    # Pass 1: raw segments (preserves unknown proper nouns)
    # Max length caps at typical proper noun length; longer segments are
    # sentence fragments that dilute the TF-IDF vector without matching.
    max_segment_len = 10
    raw_segments = []
    for seg in name.split():
        seg = seg.strip()
        if len(seg) <= 1 or len(seg) > max_segment_len:
            continue
        if _SYMBOL_RE.match(seg):
            continue
        if _SEGMENT_JUNK_RE.search(seg):
            continue
        if _ASCII_GIBBERISH_RE.match(seg) and not _HAS_VOWEL_RE.search(seg):
            continue
        if seg in _STOPWORDS:
            continue
        raw_segments.append(seg)

    # Pass 2: Janome morphological analysis
    janome_tokens = _word_tokenize(name, tokenizer)

    # Merge: raw segments get extra boost (likely proper nouns),
    # Janome tokens fill in the rest.
    seen = set(raw_segments)
    merged = raw_segments * _RAW_SEGMENT_EXTRA_BOOST
    for tok in janome_tokens:
        if tok not in seen:
            seen.add(tok)
            merged.append(tok)

    return merged


def _word_tokenize(text: str, tokenizer) -> list[str]:
    """Tokenize text into content words using Janome."""
    tokens = []
    for tok in tokenizer.tokenize(text):
        pos_parts = tok.part_of_speech.split(",")
        pos = pos_parts[0]
        subtype = pos_parts[1] if len(pos_parts) > 1 else ""

        if pos not in _CONTENT_POS:
            continue
        if pos == "名詞" and subtype in _SKIP_NOUN_SUBTYPES:
            continue

        surface = tok.surface
        if len(surface) <= 1:
            continue
        if _SYMBOL_RE.match(surface):
            continue
        if _ASCII_GIBBERISH_RE.match(surface) and not _HAS_VOWEL_RE.search(surface):
            continue
        if surface in _STOPWORDS:
            continue
        tokens.append(surface)

    return tokens


def _get_transcript_text(session, file_id: str) -> str:
    """Get full transcript text for a file."""
    chunks = (
        session.query(TranscriptChunk.text)
        .filter(TranscriptChunk.file_id == file_id)
        .order_by(TranscriptChunk.chunk_index)
        .all()
    )
    return " ".join(c.text for c in chunks).strip()


def _compute_idf(
    doc_tokens: dict[str, list[str]],
) -> dict[str, float]:
    """Compute IDF scores across all documents."""
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return {}

    doc_freq: Counter = Counter()
    for tokens in doc_tokens.values():
        for token in set(tokens):
            doc_freq[token] += 1

    return {
        token: math.log(n_docs / df) + 1.0
        for token, df in doc_freq.items()
    }


def _tfidf_vector(
    tokens: list[str],
    idf: dict[str, float],
) -> dict[str, float]:
    """Compute TF-IDF vector for a single document."""
    tf = Counter(tokens)
    total = sum(tf.values())
    if total == 0:
        return {}
    return {
        token: (count / total) * idf.get(token, 1.0)
        for token, count in tf.items()
    }


def _cosine_similarity(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0

    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _top_keywords_with_scores(
    vec: dict[str, float],
    k: int = 30,
) -> list[dict[str, object]]:
    """Return top-k keywords with TF-IDF scores."""
    sorted_items = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"word": word, "score": score} for word, score in sorted_items]


def _shared_keywords_with_scores(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
    idf: dict[str, float],
    k: int = 10,
) -> list[dict[str, object]]:
    """Return top-k shared keywords with relevance scores.

    Uses IDF-weighted product to favor topic-specific shared words
    over common language patterns.
    """
    shared = set(vec_a) & set(vec_b)
    if not shared:
        return []
    scored = [
        {
            "word": w,
            "source_tfidf": vec_a[w],
            "target_tfidf": vec_b[w],
            "relevance": idf.get(w, 1.0) * vec_a[w] * vec_b[w],
        }
        for w in shared
    ]
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:k]


def find_similar_by_tfidf(
    file_id: str,
    limit: int,
    drive: str,
) -> tuple[list[dict], list[dict]]:
    """Find similar files using TF-IDF cosine similarity on transcripts + filenames.

    Filename tokens are boosted (repeated _FILENAME_BOOST times) so that
    topic-specific proper nouns in filenames — which Whisper often
    mis-transcribes — have a strong influence on similarity scoring.

    Loads Janome tokenizer on demand and releases it after computation.

    Args:
        file_id: Source file ID.
        limit: Max results to return.
        drive: Only this drive's files are candidates.

    Returns:
        Tuple of (results, source_keywords).
        results: List of dicts with file info, similarity score, and shared_keywords.
        source_keywords: Top TF-IDF keywords for the source file.
    """
    empty: tuple[list[dict], list[dict]] = ([], [])

    with get_search_db() as session:
        source_text = _get_transcript_text(session, file_id)

        # Get source filename
        source_file = (
            session.query(IndexedFile)
            .filter(IndexedFile.file_id == file_id, IndexedFile.active.is_(True))
            .first()
        )
        if source_file is None:
            return empty
        source_filename = source_file.filename

        # Need at least a transcript or a meaningful filename
        if len(source_text) < 20 and not source_filename:
            logger.debug(
                "Skipping TF-IDF similar for %s: no transcript or filename",
                file_id,
            )
            return empty

        candidate_files = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.active.is_(True),
                IndexedFile.file_type == "video",
                IndexedFile.file_id != file_id,
                IndexedFile.drive == drive,
            )
            .all()
        )
        if not candidate_files:
            return empty

        # Gather transcripts and filenames for all candidates
        candidate_texts: dict[str, str] = {}
        candidate_filenames: dict[str, str] = {}
        candidate_info: dict[str, IndexedFile] = {}
        for f in candidate_files:
            text = _get_transcript_text(session, f.file_id)
            candidate_filenames[f.file_id] = f.filename
            candidate_info[f.file_id] = f
            candidate_texts[f.file_id] = text

    # Tokenize (load Janome, then release)
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()

    # Build token lists: transcript tokens + boosted filename tokens
    source_fn_tokens = _tokenize_filename(source_filename, tokenizer)
    source_transcript_tokens = (
        _word_tokenize(source_text, tokenizer) if len(source_text) >= 20 else []
    )
    all_tokens: dict[str, list[str]] = {
        file_id: source_transcript_tokens + source_fn_tokens * _FILENAME_BOOST,
    }

    for fid, text in candidate_texts.items():
        transcript_tokens = (
            _word_tokenize(text, tokenizer) if len(text) >= 20 else []
        )
        fn_tokens = _tokenize_filename(candidate_filenames[fid], tokenizer)
        combined = transcript_tokens + fn_tokens * _FILENAME_BOOST
        if combined:
            all_tokens[fid] = combined

    del tokenizer  # release ~100MB

    # Need at least the source to have tokens
    if not all_tokens.get(file_id):
        return empty

    # Compute IDF across all documents (source + candidates)
    idf = _compute_idf(all_tokens)

    # Build TF-IDF vectors
    source_vec = _tfidf_vector(all_tokens[file_id], idf)
    if not source_vec:
        return empty

    # Source topic keywords: top TF-IDF terms with scores
    source_keywords = _top_keywords_with_scores(source_vec, k=30)

    # Compute similarities and shared keywords
    similarities: list[tuple[str, float, list[dict]]] = []
    for fid, tokens in all_tokens.items():
        if fid == file_id:
            continue
        vec = _tfidf_vector(tokens, idf)
        sim = _cosine_similarity(source_vec, vec)
        if sim > 0.0:
            shared = _shared_keywords_with_scores(source_vec, vec, idf, k=10)
            similarities.append((fid, sim, shared))

    similarities.sort(key=lambda x: x[1], reverse=True)

    # Build results
    results: list[dict] = []
    for fid, score, shared in similarities[:limit]:
        f = candidate_info.get(fid)
        if f is None:
            continue
        results.append({
            "file_id": f.file_id,
            "drive": f.drive,
            "filename": f.filename,
            "file_type": f.file_type,
            "mime_type": f.mime_type,
            "score": score,
            "shared_keywords": shared,
        })

    return results, source_keywords


# ---------------------------------------------------------------------------
# Single-file keyword extraction for auto-tag candidate generation
# ---------------------------------------------------------------------------


def _tokenize_file(
    file_id: str,
    filename: str,
    tokenizer,
) -> list[str]:
    """Tokenize a single file's transcript + filename (boosted).

    Mirrors the token construction used inside find_similar_by_tfidf
    so keyword extraction sees the same vocabulary distribution.
    """
    with get_search_db_read() as session:
        transcript = _get_transcript_text(session, file_id)

    fn_tokens = _tokenize_filename(filename, tokenizer)
    transcript_tokens = (
        _word_tokenize(transcript, tokenizer) if len(transcript) >= 20 else []
    )
    return transcript_tokens + fn_tokens * _FILENAME_BOOST


def _build_corpus_idf(drive: str) -> tuple[dict[str, float], int]:
    """Build IDF over one drive's active files (transcripts + filenames).

    Covers every indexable file type (video/audio/document), not just
    video, so the resulting IDF is meaningful for any caller. Janome
    is loaded once and released at the end — the tokenizer is ~100MB
    and keeping it alive between auto-tag runs would dwarf the savings.

    Args:
        drive: Only this drive's files contribute to the counts.

    Returns:
        Tuple of (idf dict, number of documents that contributed
        tokens). The document count is needed downstream to convert
        a minimum-document-frequency filter into an IDF upper bound.
    """
    with get_search_db_read() as session:
        files = (
            session.query(
                IndexedFile.file_id, IndexedFile.filename
            )
            .filter(
                IndexedFile.active.is_(True),
                IndexedFile.drive == drive,
            )
            .all()
        )

    if not files:
        return {}, 0

    from janome.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    corpus_tokens: dict[str, list[str]] = {}
    try:
        for fid, fname in files:
            tokens = _tokenize_file(fid, fname, tokenizer)
            if tokens:
                corpus_tokens[fid] = tokens
    finally:
        del tokenizer

    return _compute_idf(corpus_tokens), len(corpus_tokens)


def _get_corpus_idf(
    drive: str,
    force_reload: bool = False,
) -> tuple[dict[str, float], int]:
    """Return a cached per-drive IDF + doc count, rebuilding if stale.

    Rebuilding tokenizes every active file with Janome and can take
    tens of seconds on large libraries. We deliberately build
    *outside* the cache lock so concurrent auto-tag jobs do not
    serialize on the rebuild — each job races to produce a new IDF,
    and only the final assignment is guarded. The cost is at most
    redundant computation (a few concurrent rebuilds), which is far
    cheaper than a multi-minute stall for every waiting worker.
    """
    now = time.monotonic()
    with _corpus_cache_lock:
        cached = _corpus_idf_by_drive.get(drive)
        if (
            cached is not None
            and not force_reload
            and (now - cached[2]) < _IDF_CACHE_TTL_SECONDS
        ):
            return cached[0], cached[1]

    logger.info(
        "Rebuilding corpus IDF for TF-IDF keyword extraction (drive=%s)", drive
    )
    new_idf, new_n_docs = _build_corpus_idf(drive)

    with _corpus_cache_lock:
        _corpus_idf_by_drive[drive] = (new_idf, new_n_docs, time.monotonic())
        return new_idf, new_n_docs


def reset_corpus_idf_cache() -> None:
    """Reset the cached IDF for every drive (primarily for tests)."""
    with _corpus_cache_lock:
        _corpus_idf_by_drive.clear()


def _idf_upper_bound_from_min_df(n_docs: int, min_doc_freq: int) -> float | None:
    """Convert a min-document-frequency filter to an IDF upper bound.

    A word appearing in ``df`` documents has IDF = log(n_docs / df) + 1.
    Keeping only words that appear in ``>= min_doc_freq`` documents
    means rejecting anything whose IDF exceeds ``log(n/min_df) + 1``.

    Returns None when the filter is a no-op (corpus too small for the
    filter to meaningfully apply, or min_doc_freq <= 1).
    """
    if min_doc_freq <= 1 or n_docs <= 0:
        return None
    # For tiny corpora the formula produces a threshold that rejects
    # almost every token — skip it until there's real data to learn
    # from. 10 documents is the "first meaningful corpus" heuristic.
    if n_docs < 10:
        return None
    return math.log(n_docs / min_doc_freq) + 1.0


def extract_top_keywords(
    text: str,
    filename: str,
    *,
    drive: str,
    k: int = 30,
) -> list[str]:
    """Extract top-k TF-IDF keywords from text + filename without a DB fetch.

    Used by the whisper/VTT indexing path to build the tfidf_keywords
    embedding at index time. Accepts the transcript text and filename
    directly so the caller does not need to re-fetch from DB.

    Falls back to raw TF (log-normalised term frequency) when the
    drive's IDF cache is not yet populated (e.g. first few files).

    Args:
        text: Full transcript text.
        filename: File's display name (used for topic-keyword boost).
        drive: The file's drive, whose IDF weights the terms. The
            keywords end up in that drive's search index, so the
            statistics behind them stay inside it too.
        k: Maximum number of keywords to return.

    Returns:
        List of keyword strings sorted by descending TF-IDF score.
        Empty list when the text is too short or tokenisation yields nothing.
    """
    if len(text.strip()) < 20:
        return []

    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    try:
        transcript_tokens = _word_tokenize(text, tokenizer)
        fn_tokens = _tokenize_filename(filename, tokenizer)
    finally:
        del tokenizer

    tokens = transcript_tokens + fn_tokens * _FILENAME_BOOST
    if not tokens:
        return []

    idf, _ = _get_corpus_idf(drive)

    if idf:
        vec = _tfidf_vector(tokens, idf)
    else:
        # IDF not ready yet — use log-normalised TF as fallback.
        tf = Counter(tokens)
        total = sum(tf.values())
        vec = {
            token: math.log(1 + count / total)
            for token, count in tf.items()
        }

    if not vec:
        return []

    sorted_items = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:k]
    return [word for word, _ in sorted_items]


def get_tfidf_keywords_for_file(
    file_id: str,
    *,
    drive: str,
    k: int = 10,
    min_word_length: int = 2,
    idf_min: float = 0.0,
    idf_max: float | None = None,
    min_doc_freq: int = 1,
) -> list[dict[str, object]]:
    """Extract top TF-IDF keywords for a single file as tag candidates.

    Uses the same tokenization pipeline as similar-file search (Janome
    + filename boost + stopwords + gibberish filter) and the cached
    IDF of the file's own drive. Additional filters tame the Whisper
    noise problem:

    - ``min_word_length``: drop tokens shorter than this.
    - ``idf_min``: drop words whose IDF is below this value (too
      common to be useful — usually stopwords the explicit list
      missed). Default 0.0 keeps every surviving token.
    - ``min_doc_freq``: drop words that appear in fewer than this
      many documents within the drive. A classic Whisper mis-transcription
      like "フジヤシフ" shows up in exactly one file, so requiring
      ``min_doc_freq=2`` kills most of that noise without needing to
      tune raw IDF thresholds by hand. Automatically disabled for
      corpora too small to be meaningful.
    - ``idf_max``: manual IDF upper bound. Overrides whatever
      ``min_doc_freq`` would compute. Exists for tests and advanced
      tuning; normal callers should use ``min_doc_freq`` instead.

    Args:
        file_id: The file ID to extract keywords from.
        drive: The file's drive. Both the corpus statistics and the
            file lookup are confined to it, so keyword selection never
            depends on what another drive happens to contain.
        k: Max number of keywords to return.
        min_word_length: Drop tokens shorter than this.
        idf_min: Minimum IDF score to keep a token.
        idf_max: Explicit maximum IDF score; None means "derive from
            ``min_doc_freq``".
        min_doc_freq: Minimum document frequency; used when idf_max
            is not explicitly set.

    Returns:
        List of {"word": str, "score": float} dicts, highest first.
    """
    with get_search_db_read() as session:
        file = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
                IndexedFile.drive == drive,
            )
            .first()
        )
        if file is None:
            return []
        filename = file.filename

    idf, n_docs = _get_corpus_idf(drive)
    if not idf:
        return []

    # Resolve the effective IDF upper bound.
    effective_idf_max = idf_max
    if effective_idf_max is None:
        effective_idf_max = _idf_upper_bound_from_min_df(n_docs, min_doc_freq)

    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    try:
        tokens = _tokenize_file(file_id, filename, tokenizer)
    finally:
        del tokenizer

    if not tokens:
        return []

    # Filter by length before scoring to avoid noise dominating TF.
    tokens = [t for t in tokens if len(t) >= min_word_length]
    if not tokens:
        return []

    vec = _tfidf_vector(tokens, idf)
    if not vec:
        return []

    # Apply IDF bounds (noise reduction from Whisper mis-transcriptions).
    if idf_min > 0.0 or effective_idf_max is not None:
        vec = {
            w: s for w, s in vec.items()
            if idf.get(w, 1.0) >= idf_min
            and (effective_idf_max is None or idf.get(w, 1.0) <= effective_idf_max)
        }
        if not vec:
            return []

    return _top_keywords_with_scores(vec, k=k)
