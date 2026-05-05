"""RAG prompt builders.

Pure string functions — no LLM calls, no DB access. Kept separate
from ``service.py`` so unit tests can assert on the exact wording
without mocking anything.
"""

from app.prompt_loader import render
from app.rag.context import FileContext


_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- Answer in Japanese\n",
    "en": "- Answers must be in English\n",
}


def build_system_prompt(output_language: str) -> str:
    """Return the system prompt, optionally pinned to a language.

    ``output_language`` values:

    * ``"auto"`` — let the model mirror the query language (no line).
    * ``"ja"`` — explicitly force Japanese output.
    * ``"en"`` — explicitly force English output.
    * anything else — treated as ``"auto"`` (no instruction line).

    The JSON schema hint and anti-fabrication rules are always
    included regardless of language.
    """
    lang_line = _LANGUAGE_INSTRUCTIONS.get(output_language, "")
    # citations は file_id と location の2フィールドのみ。
    # 引用文（quote / text 等）の生成は LLM 出力末尾のレイテンシを
    # 5-10 秒延ばすため厳禁。実際の引用文はバックエンドが
    # コンテキスト snippet から取得して表示するので、LLM は生成不要。
    return render(
        "rag/answer_system.jinja2",
        language_instruction=lang_line,
    )


def _serialize_file_context(ctx: FileContext) -> str:
    """Serialize a single FileContext into the prompt's per-file block.

    Each file is wrapped in ``[file_id: ...] ... [/file_id]`` markers so
    the LLM can reference it by the file_id the retriever assigned.
    Filename, title, description, and every snippet body are
    included as plain text; locations are rendered inline with each
    snippet when present.
    """
    lines: list[str] = [f"[file_id: {ctx.file_id}]"]

    if ctx.filename:
        lines = [*lines, f"filename: {ctx.filename}"]
    # Normalise the type label the LLM sees. .loft files are stored as
    # file_type="other" in older intelligence DB rows (before the core
    # classify() update promoted them to "video").  When the context has
    # transcript snippets the file is effectively a video — tell the LLM
    # so it emits m:ss location markers instead of verbatim text.
    display_type = ctx.file_type
    if display_type not in ("video", "audio") and any(
        s.source.startswith("transcript") for s in ctx.snippets
    ):
        display_type = "video"
    if display_type:
        lines = [*lines, f"type: {display_type}"]
    if ctx.title:
        lines = [*lines, f"title: {ctx.title}"]
    if ctx.description:
        lines = [*lines, f"description: {ctx.description}"]

    if ctx.snippets:
        lines = [*lines, "---"]
        for snippet in ctx.snippets:
            if snippet.location:
                lines = [
                    *lines,
                    f"[{snippet.source} @ {snippet.location}]",
                ]
            else:
                lines = [*lines, f"[{snippet.source}]"]
            lines = [*lines, snippet.text]

    lines = [*lines, "[/file_id]"]
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    contexts: list[FileContext],
) -> str:
    """Serialize retrieved contexts + the user query into a prompt string.

    Format:

        質問: <query>

        コンテキスト:
        [file_id: abc123]
        ...
        [/file_id]
        [file_id: def456]
        ...
        [/file_id]

    An empty ``contexts`` list still emits a well-formed prompt (the
    system instructions tell the model to answer "no information" in
    that case).
    """
    context_blocks = [_serialize_file_context(ctx) for ctx in contexts]
    return render(
        "rag/answer_user.jinja2",
        query=query,
        context_blocks=context_blocks,
    )
