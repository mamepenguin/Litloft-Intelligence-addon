"""RAG prompt builders.

Pure string functions — no LLM calls, no DB access. Kept separate
from ``service.py`` so unit tests can assert on the exact wording
without mocking anything.
"""

from app.rag.context import FileContext


_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- 回答は日本語で書くこと\n",
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
    return (
        "あなたはファイル管理システムの質問応答アシスタントです。\n"
        "与えられたファイル情報を元に、ユーザーの質問に答えてください。\n"
        "\n"
        "規則:\n"
        '- JSON形式で返すこと: {"answer": "...", "citations": '
        '[{"file_id": "...", "location": "..."}]}\n'
        "- answer: 回答本文。番号 [1][2] の使用は禁止 — マーカーを書かないこと\n"
        "- citations: 回答の根拠として実際に参照した箇所をすべて列挙する。\n"
        "  同じファイルの別の箇所を参照した場合は、それぞれを別 citation として列挙してよい。\n"
        "  ただし **同じ (file_id, location) の組み合わせは1回だけ** 列挙すること — 重複禁止。\n"
        "  answer に何らかの情報を書いたなら、その出典を1件以上含めること。\n"
        "  情報が無くて答えられない場合のみ空配列 [] を返してよい\n"
        "- 各 citation は **file_id と location の2フィールドのみ** を含める。\n"
        "  quote / text / snippet / content / relevance など他のフィールドは"
        "一切出力してはいけない（出力すると処理が遅延する）\n"
        "- file_id: ファイル情報ブロックの [file_id: ...] 値をそのまま使うこと\n"
        "- location: 参照した箇所のマーカー（例: 動画は '0:45'、"
        "ドキュメントは 'page 3'）。[transcript @ 0:45] 等の形で"
        "コンテキストに書かれているものをそのまま使う。該当がなければ空文字\n"
        f"{lang_line}"
        "- 与えられたファイル情報に答えがない場合は、その旨を正直に書くこと\n"
        "- JSONのみ返し、他のテキストは含めないこと"
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
    if ctx.file_type:
        lines = [*lines, f"type: {ctx.file_type}"]
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
    parts: list[str] = [f"質問: {query}", "", "コンテキスト:"]

    if not contexts:
        parts = [*parts, "(no context available)"]
    else:
        for ctx in contexts:
            parts = [*parts, _serialize_file_context(ctx)]

    return "\n".join(parts)
