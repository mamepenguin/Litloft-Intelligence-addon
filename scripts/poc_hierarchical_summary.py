"""PoC: paragraph-boundary map-reduce vs 3-window summarisation.

One-off experiment to evaluate whether multi-stage summarisation
beats the current 3-window sampling for long PDFs whose Markdown
rendering exceeds ``detailed_max_context_chars`` (24000).

Three modes:

- ``window3``     baseline; reuses ``_prepare_context`` (head/middle/tail
  8000-char windows) and feeds the result to the existing detailed
  summary prompt in a single LLM call.
- ``paragraphs``  paragraph-greedy map-reduce: split markdown on blank
  lines, greedily pack paragraphs into chunks of up to ``chunk_size``
  chars, summarise each chunk, then merge into a 3-4k-char abstract.
- ``structured``  paragraph splitter + per-chunk leaf summaries, but
  the final merge step is replaced with a structural concatenation —
  each chunk's leaf summary becomes one numbered section with a title
  derived from the chunk body. No additional aggressive compression.
  Output length scales sub-linearly with input (≈15% of original).

A pure heading-based splitter was rejected after a survey on the
indexed PDFs: PyMuPDF4LLM tags small style cues (bold runs, OCR
fragments) as ``######`` and rarely emits high-level ``#``/``##``
headings, so heading boundaries do not correspond to real chapters.
Paragraph (blank-line) boundaries survive the round-trip far better.

Output is a JSON dump with per-stage prompts, responses, and timings,
written to ``/tmp/poc_<file_id>_<mode>.json``. Stdout shows progress
plus the final summary so the run can be inspected as it goes.

Usage (inside the intelligence container)::

    docker compose exec intelligence python -m scripts.poc_hierarchical_summary \
        <file_id> --mode paragraphs

Notes:

- This script does NOT touch the production ``file_summaries`` table.
  detailed_summary state is not modified; this is purely observational.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field

from sqlalchemy import text as sql_text

logger = logging.getLogger("poc_hierarchical")


_LEAF_SYSTEM_PROMPT = (
    "あなたは長文の章節要約アシスタントです。\n"
    "以下に与えられたコンテンツの一部分（章 / セクション）を要約してください。\n"
    "\n"
    "## 規則\n"
    "- 与えられた範囲内の事実・主張・数値・固有名詞をできるだけ網羅する\n"
    "- 範囲外の知識を補完しない\n"
    "- 評価語は誰による評価かを明示する\n"
    "- 出力は Markdown の箇条書き。冒頭に「## <セクションタイトル>」を付ける\n"
    "- 表・数値・対応関係はできる限り保持する（小さな表は Markdown 表で残す）\n"
    "- 推測語（「かもしれない」「おそらく」）は省略せず保持する\n"
    "- 原文に存在しない結論・提案を加えないこと\n"
)


_INTRO_SYSTEM_PROMPT = "あなたは要約アシスタントです。"

_MERGE_SYSTEM_PROMPT = (
    "あなたはファイル管理システムの要約アシスタントです。\n"
    "以下に複数の章節要約が連結された入力が与えられます。\n"
    "これらを統合して、ファイル全体の構造化要約を Markdown 形式で生成してください。\n"
    "\n"
    "## 重要な制約\n"
    "- 入力は既に各章の要約済みテキストです。**事実関係を改変しないこと**。\n"
    "- 重複する内容は削減してよいが、数値・固有名詞・主張内容は変更しない。\n"
    "- 章節要約に登場しない事実を新たに加えない（推測で埋めない）。\n"
    "- 章節要約に書かれた「推測」「不確実性マーカー」はそのまま保持する。\n"
    "\n"
    "## 出力構成\n"
    "1. **導入**(1-2文): 全体像と主要テーマを簡潔に\n"
    "2. **詳細内容**: 章の流れに沿って整理。並列情報は箇条書き、"
    "因果・順序・対比は文章で。\n"
    "3. **重要ポイントまとめ**: 数値・比較・対応関係が複数章にまたがって"
    "存在する場合のみ Markdown 表で整理。該当なしならこのセクションは省略。\n"
    "4. **結論**(1-2文): 各章節要約から読み取れる結論・締めくくり。\n"
    "\n"
    "出力は Markdown のみ。JSON や他のラッパーで包まないこと。\n"
)


@dataclass
class StageResult:
    """One LLM invocation's input/output and timing."""

    label: str
    char_count: int
    response: str | None
    elapsed_s: float
    error: str | None = None


@dataclass
class RunResult:
    """The full PoC output for one mode."""

    mode: str
    file_id: str
    model: str
    markdown_chars: int
    leaf_stages: list[StageResult] = field(default_factory=list)
    merge_stage: StageResult | None = None
    final_summary: str | None = None
    total_elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Markdown loading
# ---------------------------------------------------------------------------


def _load_markdown(file_id: str) -> str | None:
    """Fetch pdf_markdown.markdown for the given file_id.

    Returns None when no row exists (fitz_fallback PDFs or non-PDF).
    """
    from app.database import get_search_db

    with get_search_db() as session:
        row = session.execute(
            sql_text("SELECT markdown FROM pdf_markdown WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchone()
    if row is None:
        return None
    md = row[0]
    return md if md else None


def _file_label(file_id: str) -> str:
    """Best-effort filename lookup for log readability."""
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as session:
        f = (
            session.query(IndexedFile)
            .filter(IndexedFile.file_id == file_id)
            .first()
        )
        if f is None:
            return file_id
        return f"{file_id} ({f.filename})"


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


_PARAGRAPH_SEP = re.compile(r"\n\s*\n+")


def _split_oversized_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Hard-split a single paragraph that exceeds chunk_size.

    Tries Japanese full stops first (``。``), then ASCII periods, then
    falls back to a fixed-width cut. Used only for paragraphs that are
    themselves larger than the target chunk — typically OCR-fragmented
    walls of glued-together text.
    """
    if len(paragraph) <= chunk_size:
        return [paragraph]
    parts: list[str] = []
    for sep in ("。", "."):
        if sep in paragraph:
            buf = ""
            for piece in paragraph.split(sep):
                candidate = (buf + piece + sep) if piece else buf
                if len(candidate) > chunk_size and buf:
                    parts.append(buf)
                    buf = piece + sep if piece else ""
                else:
                    buf = candidate
            if buf:
                parts.append(buf)
            if all(len(p) <= chunk_size for p in parts):
                return parts
    # Last resort: fixed-width cut.
    return [
        paragraph[i : i + chunk_size]
        for i in range(0, len(paragraph), chunk_size)
    ]


_LEADING_HEADING_RE = re.compile(r"\A\s*#{1,6}\s+(.+?)\s*$", re.MULTILINE)


_SENTENCE_TERMINATORS = "。．.!?！？"


def _clean_truncated_tail(text: str) -> str:
    """Trim a mid-sentence cut-off back to the last completed unit.

    The LLM can hit ``max_tokens`` mid-sentence and leave a leaf body
    ending with ``"...必要な事項の細則"`` (no terminator) or even mid-
    bullet (``"*   **文書管理者:**\\n    *"``). For the structured
    deliverable we'd rather show a clean ending than a dangling
    fragment, so we walk back to the last sentence terminator. If the
    last line is itself an incomplete bullet (``*`` / ``-`` / ``#`` with
    no body) we drop that line entirely. Returns the original text
    unchanged when it already ends cleanly.
    """
    stripped = text.rstrip()
    if not stripped:
        return text
    last_char = stripped[-1]
    # Already ends on a recognised terminator or markdown structure char.
    if last_char in _SENTENCE_TERMINATORS or last_char in "])｝》」』）":
        return stripped
    # Find last terminator anywhere in the text.
    last_idx = -1
    for term in _SENTENCE_TERMINATORS:
        idx = stripped.rfind(term)
        if idx > last_idx:
            last_idx = idx
    if last_idx < 0:
        return stripped
    cleaned = stripped[: last_idx + 1]
    # Drop any trailing bullet/heading marker that has nothing after it.
    lines = cleaned.splitlines()
    while lines:
        last_line = lines[-1].rstrip()
        bare = last_line.lstrip(" \t*-•#")
        if not bare:
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def _split_leaf_title_and_body(leaf_response: str) -> tuple[str, str]:
    """Split a leaf summary into ``(title, body_without_title)``.

    The leaf prompt instructs the LLM to lead with ``## <section title>``,
    so when present we consume that line as the section heading and
    strip it from the body to avoid duplication in the assembled output.
    Returns ``("", original)`` when the model didn't follow the format.
    """
    if not leaf_response:
        return "", ""
    m = _LEADING_HEADING_RE.match(leaf_response)
    if not m:
        return "", leaf_response.strip()
    title = m.group(1).strip()
    body = leaf_response[m.end():].lstrip("\n")
    return title, body


def _assemble_structured(
    *,
    file_label: str,
    chunks: list[tuple[str, str]],
    leaf_results: list[StageResult],
    intro: str | None = None,
) -> str:
    """Concatenate leaf summaries into a structured Markdown deliverable.

    The output preserves a 1:1 mapping between sections and original
    chunks so the reader can still locate any point in the source PDF
    by section number. Section titles are derived from chunk content
    via :func:`_derive_section_title`.
    """
    parts: list[str] = [f"# {file_label} 構造化要約\n"]
    if intro:
        parts.append(intro.strip() + "\n")
    skipped = 0
    section_no = 0
    for leaf in leaf_results:
        if leaf.error or not (leaf.response or "").strip():
            # Empty / failed leaves are dropped from the output rather than
            # rendered as title-only stubs. Section numbers are reassigned
            # densely so the reader sees a continuous list.
            skipped += 1
            continue
        title, body = _split_leaf_title_and_body(leaf.response or "")
        body_clean = _clean_truncated_tail(body.strip())
        if not title and not body_clean:
            skipped += 1
            continue
        section_no += 1
        if not title:
            title = f"セクション {section_no}"
        parts.append(f"## {section_no}. {title}\n")
        parts.append(body_clean + "\n")
    if skipped:
        parts.append(
            f"\n---\n\n*注: 元文書の {len(leaf_results)} chunk のうち "
            f"{skipped} chunk は要約生成に失敗または空応答のため除外している。*\n"
        )
    return "\n".join(parts)


def _split_by_paragraphs(
    markdown: str, chunk_size: int = 8000
) -> list[tuple[str, str]]:
    """Greedy map-reduce splitter that respects paragraph (blank-line) boundaries.

    Concatenates paragraphs until adding the next one would exceed
    ``chunk_size``, at which point the accumulated buffer is flushed
    into a chunk. A paragraph that already exceeds ``chunk_size`` is
    pre-split via :func:`_split_oversized_paragraph` so the greedy loop
    sees only ``<= chunk_size`` units.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SEP.split(markdown) if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            units.append(p)
        else:
            units.extend(_split_oversized_paragraph(p, chunk_size))

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
            continue
        candidate = buf + "\n\n" + unit
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            chunks.append(buf)
            buf = unit
    if buf:
        chunks.append(buf)

    n = len(chunks)
    return [(f"chunk-{i + 1}/{n}", body) for i, body in enumerate(chunks)]


def _three_window_extract(markdown: str, max_chars: int = 24000) -> str:
    """Reuse the production 3-window extractor for the baseline mode."""
    from app.workers.summaries import _prepare_context

    prepared, _ = _prepare_context(markdown, max_chars=max_chars, window_count=3)
    return prepared


# ---------------------------------------------------------------------------
# LLM stages
# ---------------------------------------------------------------------------


async def _generate_with_empty_retry(
    llm_client,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float | None = None,
    max_attempts: int = 2,
) -> str | None:
    """Call ``llm_client.generate`` and retry once on empty response.

    The production LLMClient only retries on transport / status errors —
    a successful HTTP 200 with ``content=""`` (which ollama / gemma4:e4b
    occasionally returns) flows straight through and shows up as a leaf
    summary that's just an empty string. For the PoC we treat empty
    output as a soft failure and try once more before giving up.
    """
    last: str | None = None
    for attempt in range(max_attempts):
        last = await llm_client.generate(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens,
            temperature=temperature,
        )
        if last and last.strip():
            return last
    return last


async def _summarise_leaf(
    llm_client, title: str, body: str, max_tokens: int
) -> StageResult:
    """One leaf section/chunk → bullet-point summary."""
    user_prompt = f"## {title}\n\n--- コンテンツ ---\n{body}"
    start = time.monotonic()
    try:
        response = await _generate_with_empty_retry(
            llm_client,
            _LEAF_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
        )
        elapsed = time.monotonic() - start
        return StageResult(
            label=title, char_count=len(body), response=response, elapsed_s=elapsed
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return StageResult(
            label=title,
            char_count=len(body),
            response=None,
            elapsed_s=elapsed,
            error=repr(exc),
        )


async def _generate_intro(
    llm_client, leaf_results: list[StageResult]
) -> StageResult:
    """One-shot intro paragraph (1-2 sentences) from existing leaf titles.

    The intro only needs the document's overall topic, so we feed the
    LLM each leaf's heading + its first bullet line rather than the full
    summary text. This keeps the input under a few thousand characters
    and avoids exceeding ollama's default 8k context window — feeding
    the full concatenated leaves (10-30k chars) reliably produces empty
    responses on gemma4:e4b.
    """
    digests: list[str] = []
    for leaf in leaf_results:
        if not leaf.response:
            continue
        title, body = _split_leaf_title_and_body(leaf.response)
        title = title or "(無題)"
        first_bullet = ""
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            first_bullet = line.lstrip("*-• ").strip()
            break
        digests.append(
            f"- {title}: {first_bullet[:150]}"
            if first_bullet
            else f"- {title}"
        )
    # Empirically, gemma4:e4b silently returns empty content when the
    # prompt asks for an "introduction" with constraints attached. The
    # phrasing below ("何が書かれているか") and a slightly higher
    # temperature reliably produce a complete 2-sentence answer.
    user_prompt = (
        "次の文書について、何が書かれているかを 1-2 文でまとめてください。\n\n"
        "章節:\n" + "\n".join(digests) + "\n\n回答:"
    )
    start = time.monotonic()
    try:
        response = await _generate_with_empty_retry(
            llm_client,
            _INTRO_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=1024,
            temperature=0.5,
        )
        return StageResult(
            label="intro",
            char_count=len(user_prompt),
            response=response,
            elapsed_s=time.monotonic() - start,
        )
    except Exception as exc:
        return StageResult(
            label="intro",
            char_count=len(user_prompt),
            response=None,
            elapsed_s=time.monotonic() - start,
            error=repr(exc),
        )


async def _merge_leaves(
    llm_client, leaf_results: list[StageResult], max_tokens: int
) -> StageResult:
    """Concatenate leaf summaries and run the final merge prompt."""
    valid = [r for r in leaf_results if r.response]
    body = "\n\n".join(r.response for r in valid if r.response)
    user_prompt = f"--- 章節要約の集合 ---\n{body}"
    start = time.monotonic()
    try:
        response = await llm_client.generate(
            _MERGE_SYSTEM_PROMPT,
            user_prompt,
            max_tokens_override=max_tokens,
        )
        elapsed = time.monotonic() - start
        return StageResult(
            label="merge",
            char_count=len(body),
            response=response,
            elapsed_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return StageResult(
            label="merge",
            char_count=len(body),
            response=None,
            elapsed_s=elapsed,
            error=repr(exc),
        )


async def _run_window3(
    llm_client, markdown: str, file_id: str, model: str
) -> RunResult:
    """Baseline: existing 3-window extract → existing detailed prompt."""
    from app.workers.summaries import (
        _DETAILED_MAX_TOKENS,
        _build_detailed_system_prompt,
    )

    result = RunResult(
        mode="window3",
        file_id=file_id,
        model=model,
        markdown_chars=len(markdown),
    )
    extracted = _three_window_extract(markdown)
    user_prompt = (
        "ファイル名: (PoC)\nタイプ: document\n\n"
        "注: 以下は長いコンテンツの抜粋です。冒頭・中盤・終盤から取得しています。\n"
        "\n--- コンテンツ ---\n"
        f"{extracted}"
    )

    started = time.monotonic()
    start = time.monotonic()
    try:
        response = await llm_client.generate(
            _build_detailed_system_prompt(),
            user_prompt,
            max_tokens_override=_DETAILED_MAX_TOKENS,
        )
        result.merge_stage = StageResult(
            label="window3",
            char_count=len(extracted),
            response=response,
            elapsed_s=time.monotonic() - start,
        )
        result.final_summary = response
    except Exception as exc:
        result.merge_stage = StageResult(
            label="window3",
            char_count=len(extracted),
            response=None,
            elapsed_s=time.monotonic() - start,
            error=repr(exc),
        )
    result.total_elapsed_s = time.monotonic() - started
    return result


async def _run_multistage(
    llm_client,
    markdown: str,
    file_id: str,
    model: str,
    *,
    mode: str,
    sections: list[tuple[str, str]],
    leaf_max_tokens: int,
    merge_max_tokens: int,
    file_label: str,
    skip_merge: bool = False,
) -> RunResult:
    """Run leaf summaries serially (local LLM-friendly), then merge.

    When ``skip_merge`` is True (``--mode structured``) the leaf
    summaries are concatenated as numbered sections instead of being
    consolidated by a final LLM call, preserving 1:1 mapping between
    output sections and source chunks.
    """
    result = RunResult(
        mode=mode,
        file_id=file_id,
        model=model,
        markdown_chars=len(markdown),
    )
    started = time.monotonic()

    n = len(sections)
    for i, (title, body) in enumerate(sections, 1):
        sys.stdout.write(
            f"  [{i}/{n}] {title[:40]} ({len(body)} chars)... "
        )
        sys.stdout.flush()
        stage = await _summarise_leaf(llm_client, title, body, leaf_max_tokens)
        result.leaf_stages.append(stage)
        if stage.error:
            sys.stdout.write(f"ERROR ({stage.elapsed_s:.1f}s)\n")
        else:
            sys.stdout.write(f"ok ({stage.elapsed_s:.1f}s)\n")

    if skip_merge:
        sys.stdout.write("  [intro] generating overview... ")
        sys.stdout.flush()
        intro_stage = await _generate_intro(llm_client, result.leaf_stages)
        if intro_stage.error:
            sys.stdout.write(f"ERROR ({intro_stage.elapsed_s:.1f}s)\n")
        else:
            sys.stdout.write(f"ok ({intro_stage.elapsed_s:.1f}s)\n")
        result.merge_stage = intro_stage
        result.final_summary = _assemble_structured(
            file_label=file_label,
            chunks=sections,
            leaf_results=result.leaf_stages,
            intro=intro_stage.response,
        )
    else:
        sys.stdout.write(f"  [merge] consolidating {n} section summaries... ")
        sys.stdout.flush()
        merge = await _merge_leaves(
            llm_client, result.leaf_stages, merge_max_tokens
        )
        result.merge_stage = merge
        if merge.error:
            sys.stdout.write(f"ERROR ({merge.elapsed_s:.1f}s)\n")
        else:
            sys.stdout.write(f"ok ({merge.elapsed_s:.1f}s)\n")
        result.final_summary = merge.response
    result.total_elapsed_s = time.monotonic() - started
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _to_jsonable(result: RunResult) -> dict:
    return {
        "mode": result.mode,
        "file_id": result.file_id,
        "model": result.model,
        "markdown_chars": result.markdown_chars,
        "total_elapsed_s": round(result.total_elapsed_s, 2),
        "leaf_stages": [
            {
                "label": s.label,
                "char_count": s.char_count,
                "elapsed_s": round(s.elapsed_s, 2),
                "error": s.error,
                "response": s.response,
            }
            for s in result.leaf_stages
        ],
        "merge_stage": (
            None
            if result.merge_stage is None
            else {
                "label": result.merge_stage.label,
                "char_count": result.merge_stage.char_count,
                "elapsed_s": round(result.merge_stage.elapsed_s, 2),
                "error": result.merge_stage.error,
                "response": result.merge_stage.response,
            }
        ),
        "final_summary": result.final_summary,
    }


async def _reassemble_from_json(
    json_path: str,
    *,
    markdown: str,
    sections: list[tuple[str, str]],
    file_id: str,
    file_label: str,
    llm_client,
) -> RunResult:
    """Rebuild a structured RunResult from an existing paragraphs JSON dump.

    Used to re-evaluate the same leaves under a different assembly
    strategy without spending another minute per chunk on the LLM.
    Validates that the chunk count from the splitter still matches the
    saved leaves (a config drift between runs would silently mismatch
    section bodies and summaries otherwise).
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    saved_leaves = data.get("leaf_stages") or []
    if len(saved_leaves) != len(sections):
        raise RuntimeError(
            f"chunk count mismatch: split produced {len(sections)} sections "
            f"but saved JSON has {len(saved_leaves)} leaves. Refusing to "
            f"misalign — re-run --mode paragraphs to regenerate."
        )
    leaves = [
        StageResult(
            label=s["label"],
            char_count=s["char_count"],
            response=s["response"],
            elapsed_s=s["elapsed_s"],
            error=s.get("error"),
        )
        for s in saved_leaves
    ]
    result = RunResult(
        mode="structured",
        file_id=file_id,
        model=data.get("model", "unknown"),
        markdown_chars=len(markdown),
        leaf_stages=leaves,
        merge_stage=None,
        total_elapsed_s=0.0,
    )
    intro_text: str | None = None
    if llm_client is not None and llm_client.enabled:
        sys.stdout.write("  [intro] generating overview from saved leaves... ")
        sys.stdout.flush()
        intro_stage = await _generate_intro(llm_client, leaves)
        if intro_stage.error:
            sys.stdout.write(f"ERROR ({intro_stage.elapsed_s:.1f}s)\n")
        else:
            sys.stdout.write(f"ok ({intro_stage.elapsed_s:.1f}s)\n")
        result.merge_stage = intro_stage
        intro_text = intro_stage.response
    result.final_summary = _assemble_structured(
        file_label=file_label,
        chunks=sections,
        leaf_results=leaves,
        intro=intro_text,
    )
    return result


async def _amain(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.database import init_search_db
    from app.llm import LLMClient

    init_search_db()

    markdown = _load_markdown(args.file_id)
    if markdown is None:
        logger.error(
            "No pdf_markdown row for %s — non-PDF or fitz_fallback. Aborting.",
            args.file_id,
        )
        return 2

    label = _file_label(args.file_id)
    logger.info(
        "%s — markdown %d chars, mode=%s%s",
        label,
        len(markdown),
        args.mode,
        " (reusing leaves)" if args.reuse_leaves else "",
    )

    llm_client = LLMClient(settings.llm)
    if not llm_client.enabled:
        logger.error("LLM client is disabled in config; cannot run PoC.")
        return 3

    if args.mode == "window3":
        result = await _run_window3(
            llm_client, markdown, args.file_id, settings.llm.model
        )
    elif args.mode in ("paragraphs", "structured"):
        sections = _split_by_paragraphs(markdown, chunk_size=args.chunk_size)
        if len(sections) > args.max_sections:
            logger.error(
                "%s mode produced %d chunks (limit %d). Increase "
                "--chunk-size or --max-sections.",
                args.mode,
                len(sections),
                args.max_sections,
            )
            return 4
        logger.info("paragraph split → %d chunks", len(sections))
        if args.reuse_leaves:
            result = await _reassemble_from_json(
                args.reuse_leaves,
                markdown=markdown,
                sections=sections,
                file_id=args.file_id,
                file_label=label,
                llm_client=llm_client,
            )
        else:
            result = await _run_multistage(
                llm_client,
                markdown,
                args.file_id,
                settings.llm.model,
                mode=args.mode,
                sections=sections,
                leaf_max_tokens=args.leaf_max_tokens,
                merge_max_tokens=args.merge_max_tokens,
                file_label=label,
                skip_merge=(args.mode == "structured"),
            )
    else:
        logger.error("unknown mode: %s", args.mode)
        return 5

    output_path = (
        args.output
        or f"/tmp/poc_{args.file_id}_{args.mode}.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(result), fh, ensure_ascii=False, indent=2)
    logger.info(
        "wrote %s (total %.1fs)", output_path, result.total_elapsed_s
    )

    if result.final_summary:
        sys.stdout.write("\n=== FINAL SUMMARY ===\n")
        sys.stdout.write(result.final_summary)
        sys.stdout.write("\n=== END ===\n")
    else:
        sys.stdout.write("\n(no final summary — see JSON for errors)\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id",
        dest="file_id",
        required=True,
        help="indexed_files.file_id of a PDF (use --id=-fvB...) for ids "
        "starting with a dash to avoid argparse treating them as flags",
    )
    parser.add_argument(
        "--mode",
        choices=("window3", "paragraphs", "structured"),
        default="paragraphs",
        help="summarisation strategy (default: paragraphs)",
    )
    parser.add_argument(
        "--reuse-leaves",
        default=None,
        help="path to an existing paragraphs JSON dump; reassemble the "
        "structured output from its saved leaves without calling the LLM. "
        "Only valid with --mode structured (or paragraphs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="paragraph-greedy chunk target size (default 8000)",
    )
    parser.add_argument(
        "--leaf-max-tokens",
        type=int,
        default=2048,
        help="max_tokens for per-section LLM calls (default 2048; "
        "gemma4:e4b expands to fill any cap so raising this just makes "
        "the deliverable longer, not safer — assembly-time post-hoc "
        "cleanup handles mid-sentence truncations cosmetically)",
    )
    parser.add_argument(
        "--merge-max-tokens",
        type=int,
        default=4096,
        help="max_tokens for the final merge call (default 4096)",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=50,
        help="abort when split count exceeds this (default 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output JSON path (default /tmp/poc_<file_id>_<mode>.json)",
    )
    args = parser.parse_args(argv)

    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
